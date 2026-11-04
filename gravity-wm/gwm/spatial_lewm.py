"""Semigroup-JEPA variants whose latent can carry the object's location.

The constraint of this study acts on the latent the predictor outputs, so the
position probe is only as good as the position information that survives into
that latent.  Measured on a pretrained Semigroup-JEPA checkpoint of this task,
it does not survive.  With the encoder trained for nine epochs, a probe of the
ViT's *spatial* token map locates the object to 6.2 pixels (variance explained
0.97) while the same probe of the projected ``[CLS]`` token -- the latent --
reaches 0.02, i.e. chance.  The gap is not the probe's: it is unchanged by a
36x larger probe (9.4M parameters), by probing an eight-frame window instead of
one frame, and by per-dimension standardisation of the latent.  A randomly
initialised encoder gives chance on both, so training is what teaches the ViT
where the object is, and the read-out is what discards it.

Why pooling discards it.  A pooled patch feature ``sum_p alpha_p t_p`` carries
location through the position embeddings in ``t_p``, but only if the weights
``alpha_p`` depend on which patches the object occupies.  Uniform mean pooling
averages every position embedding into the same constant, so location cancels
exactly -- which is why mean-pooled patch features locate the object no better
than the ``[CLS]`` token does, and why DINO-WM's mean-pooled features would
have the same problem.

Two replacements for the ``[CLS]`` read-out are provided, and neither changes
anything else: same encoder, same ``embed_dim``, same projector shape, same
predictor, same action conditioning, same SIGReg and rollout losses.

``attn_pool`` learns a single pooling query, which *can* concentrate on the
    object and so keep location.  Measured, it does not: started from the
    uniform weights it is initialised to, the training objective does not drive
    it to concentrate, and its latent stays at chance -- the same failure as
    ``[CLS]``, for the same reason.  Kept as the record of that result.

``spatial_map`` compresses each patch token to a few channels and keeps the
    spatial axis, so nothing is pooled away.  Grafted onto the trained encoder
    with an untrained projector it already reaches 12.6 pixels (variance
    explained 0.87), which is what makes it worth training.

Neither read-out decodes coordinates and feeds them in; that would hand the
probe the answer and make the comparison meaningless.  Both produce an ordinary
learned summary, and whether location ends up in it is still decided by the
training objective.  ``spatial_map`` is reported as a modification of the
published model, not as the published model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from einops import rearrange
from sg_jepa.config import WorldModelConfig
from sg_jepa.models.factory import build_vit_encoder
from sg_jepa.models.lewm import (
    MLP,
    ActionConditionedGRUPredictor,
    ActionConditionedSSMPredictor,
    Embedder,
    LeWM,
    Predictor,
)
from torch import nn

__all__ = [
    "READOUTS",
    "AttentionPool",
    "SpatialMapReadout",
    "SpatialLeWM",
    "build_readout_world_model",
    "read_readout_sidecar",
    "write_readout_sidecar",
]

READOUTS = ("cls", "attn_pool", "spatial_map")
SIDECAR_NAME = "readout.json"


class AttentionPool(nn.Module):
    """Single-query attention pooling over patch tokens.

    ``alpha = softmax_p(<q, W_k t_p> / sqrt(d))`` and the output is
    ``sum_p alpha_p W_v t_p``.  With a uniform ``alpha`` this reduces to mean
    pooling, which destroys location; the query exists so that ``alpha`` can
    concentrate on the patches the object occupies, and the pooled position
    embeddings then say where it is.

    The query is initialised to zero so that pooling starts *exactly* at the
    mean over patches.  Training therefore begins from the uninformative
    pooling and has to learn to concentrate, which keeps the variant from
    getting a head start that the ``[CLS]`` baseline does not have.
    """

    def __init__(self, hidden: int, *, temperature: float = 1.0) -> None:
        super().__init__()
        if hidden <= 0:
            raise ValueError(f"hidden must be positive, got {hidden}")
        self.hidden = int(hidden)
        self.scale = float(temperature) / float(hidden) ** 0.5
        self.query = nn.Parameter(torch.zeros(hidden))
        self.to_keys = nn.Linear(hidden, hidden, bias=False)
        self.to_values = nn.Linear(hidden, hidden, bias=False)
        self.norm = nn.LayerNorm(hidden)
        nn.init.eye_(self.to_keys.weight)
        nn.init.eye_(self.to_values.weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """``tokens`` is ``(N, P, D)``; returns ``(N, D)``."""

        if tokens.ndim != 3:
            raise ValueError(f"expected tokens of shape (N, P, D), got {tuple(tokens.shape)}")
        keys = self.to_keys(self.norm(tokens))
        logits = torch.einsum("npd,d->np", keys, self.query) * self.scale
        weights = torch.softmax(logits, dim=-1)
        return torch.einsum("np,npd->nd", weights, self.to_values(tokens))

    @torch.no_grad()
    def attention(self, tokens: torch.Tensor) -> torch.Tensor:
        """The pooling weights, for inspecting what the pool attends to."""

        keys = self.to_keys(self.norm(tokens))
        logits = torch.einsum("npd,d->np", keys, self.query) * self.scale
        return torch.softmax(logits, dim=-1)


class SpatialMapReadout(nn.Module):
    """Compress each patch token to ``channels`` numbers and keep the map.

    This is the read-out the measurements point at.  With the trained encoder, a
    linear probe of the per-patch channel mean locates the object to 6.2 pixels
    (variance explained 0.97), while the same probe of the projected ``[CLS]``
    token manages 0.02 -- and no amount of probe capacity, window length or
    feature normalisation recovers the difference.  Keeping the spatial axis and
    compressing the channels preserves what pooling destroys, at the cost of a
    latent that is a (small) feature map rather than a single vector.

    Spatially structured latents are the norm for world models, so this is a
    change of read-out rather than a change of kind.  It is reported as a
    modification of the published model, not as the published model.
    """

    def __init__(self, hidden: int, patches: int, *, channels: int = 2) -> None:
        super().__init__()
        if hidden <= 0 or patches <= 0 or channels <= 0:
            raise ValueError("hidden, patches and channels must be positive")
        self.hidden = int(hidden)
        self.patches = int(patches)
        self.channels = int(channels)
        self.norm = nn.LayerNorm(hidden)
        self.compress = nn.Linear(hidden, int(channels))

    @property
    def output_dim(self) -> int:
        return self.patches * self.channels

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """``tokens`` is ``(N, P, D)``; returns ``(N, P * channels)``."""

        if tokens.ndim != 3:
            raise ValueError(f"expected tokens of shape (N, P, D), got {tuple(tokens.shape)}")
        if tokens.shape[1] != self.patches:
            raise ValueError(
                f"expected {self.patches} patches, got {tokens.shape[1]}; the read-out is "
                "sized for one image resolution"
            )
        return self.compress(self.norm(tokens)).reshape(tokens.shape[0], -1)


class SpatialLeWM(LeWM):
    """``LeWM`` with the ``[CLS]`` read-out replaced by an attention pool.

    Only :meth:`encode` differs from the base class, and only in how the
    per-frame summary is taken from the encoder output.  ``info["emb"]`` keeps
    its shape and meaning, so the trainer, the rollout loss, the latent cache,
    the probes and the constraint all apply unchanged.
    """

    def __init__(self, *args: Any, pool: nn.Module, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.pool = pool

    def encode(self, info: dict) -> dict:
        pixels = info["pixels"].to(next(self.encoder.parameters()).dtype)
        batch_size = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        tokens = output.last_hidden_state
        summary = self.pool(tokens[:, 1:])
        if getattr(self, "store_cls_raw", False):
            info["cls_raw"] = rearrange(tokens[:, 0], "(b t) d -> b t d", b=batch_size)
        if getattr(self, "store_pool_summary", False):
            info["pool_summary"] = rearrange(summary, "(b t) d -> b t d", b=batch_size)
        emb = self.projector(summary)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=batch_size)
        if getattr(self, "store_patch_tokens", False):
            info["patch_tokens"] = rearrange(
                tokens[:, 1:], "(b t) p d -> b t p d", b=batch_size
            )
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info


def _build_predictor(config: WorldModelConfig) -> nn.Module:
    kwargs = dict(
        num_frames=config.history_size,
        input_dim=config.embed_dim,
        hidden_dim=config.predictor_hidden_dim,
        output_dim=config.embed_dim,
        depth=config.predictor_depth,
        heads=config.predictor_heads,
        mlp_dim=config.predictor_mlp_dim,
        dim_head=config.predictor_dim_head,
        dropout=config.predictor_dropout,
        emb_dropout=config.predictor_emb_dropout,
        cond_dim=config.embed_dim,
    )
    if config.predictor_kind == "transformer":
        return Predictor(**kwargs)
    if config.predictor_kind == "gru":
        return ActionConditionedGRUPredictor(
            **kwargs,
            conditioning=config.predictor_conditioning,
            residual_init=config.predictor_residual_init,
        )
    return ActionConditionedSSMPredictor(
        **kwargs,
        conditioning=config.predictor_conditioning,
        residual_init=config.predictor_residual_init,
        ssm_state_dim=config.ssm_state_dim,
        ssm_conv_kernel=config.ssm_conv_kernel,
        ssm_expand=config.ssm_expand,
        ssm_dt_rank=config.ssm_dt_rank,
    )


def build_readout_world_model(config: WorldModelConfig, readout: str = "cls") -> LeWM:
    """Build the stock model or the attention-pool variant.

    ``readout='cls'`` returns exactly what ``sg_jepa.models.build_world_model``
    returns, so the baseline path is untouched.
    """

    if readout not in READOUTS:
        raise ValueError(f"readout must be one of {READOUTS}, got {readout!r}")
    if readout == "cls":
        from sg_jepa.models import build_world_model

        return build_world_model(config)

    encoder = build_vit_encoder(config)
    hidden = int(encoder.config.hidden_size)
    patches = (config.image_size // config.patch_size) ** 2
    if readout == "attn_pool":
        pool: nn.Module = AttentionPool(hidden)
        summary_dim = hidden
    else:
        pool = SpatialMapReadout(hidden, patches, channels=2)
        summary_dim = pool.output_dim
    return SpatialLeWM(
        encoder=encoder,
        predictor=_build_predictor(config),
        action_encoder=Embedder(input_dim=config.action_dim, emb_dim=config.embed_dim),
        projector=MLP(
            input_dim=summary_dim,
            output_dim=config.embed_dim,
            hidden_dim=config.projector_hidden_dim,
            norm_fn=nn.BatchNorm1d,
        ),
        pred_proj=MLP(
            input_dim=config.embed_dim,
            output_dim=config.embed_dim,
            hidden_dim=config.projector_hidden_dim,
            norm_fn=nn.BatchNorm1d,
        ),
        pool=pool,
    )


def write_readout_sidecar(output_dir: str | Path, readout: str) -> Path:
    """Record which read-out a run used, next to its checkpoints.

    ``WorldModelConfig`` rejects unknown keys, so the choice cannot live in the
    experiment YAML; keeping it beside the checkpoint means every downstream
    script rebuilds the right architecture without being told.
    """

    if readout not in READOUTS:
        raise ValueError(f"readout must be one of {READOUTS}, got {readout!r}")
    path = Path(output_dir) / SIDECAR_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"readout": readout}, indent=2) + "\n")
    return path


def read_readout_sidecar(checkpoint_path: str | Path) -> str:
    """The read-out recorded beside a checkpoint; ``'cls'`` when absent."""

    directory = Path(checkpoint_path)
    directory = directory.parent if directory.is_file() or directory.suffix else directory
    sidecar = directory / SIDECAR_NAME
    if not sidecar.is_file():
        return "cls"
    value = json.loads(sidecar.read_text()).get("readout", "cls")
    if value not in READOUTS:
        raise ValueError(f"sidecar {sidecar} names an unknown read-out: {value!r}")
    return str(value)
