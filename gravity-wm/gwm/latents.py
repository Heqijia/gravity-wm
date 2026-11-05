"""Encode the dataset once and cache the latents.

Post-training updates only the predictor; the encoder and the projector are
frozen, so ``z_t = E(x_t)`` never changes after pretraining.  Encoding every
episode once and caching the result makes the rest of the study cheap: the
position probe, the tolerance calibration, every post-training variant and
every evaluation then read latents instead of running a ViT over pixels.  It
also removes a source of discrepancy, because the JEPA branch, the constraint
branch and the evaluation all consume byte-identical latents.

The cache is keyed by the checkpoint it came from and by the model
configuration, and loading verifies both, so a cache can never be silently
paired with a different encoder.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sg_jepa.config import WorldModelConfig
from sg_jepa.data import ActionStatistics, open_trajectory_store
from sg_jepa.train_utils import load_training_checkpoint
from torch import nn

from gwm.spatial_lewm import build_readout_world_model, read_readout_sidecar

__all__ = [
    "LatentCache",
    "build_latent_cache",
    "load_latent_cache",
    "load_world_model",
    "set_posttraining_modes",
    "trainable_parameters",
]


def load_world_model(
    checkpoint_path: str | Path, *, device: str | torch.device = "cpu"
) -> tuple[nn.Module, WorldModelConfig, ActionStatistics, dict[str, Any]]:
    """Rebuild a pretrained world model and its action normalization."""

    payload = load_training_checkpoint(checkpoint_path)
    if payload.get("kind") != "native_world_model_training":
        raise ValueError(f"not a Semigroup-JEPA training checkpoint: {checkpoint_path}")
    model_config = WorldModelConfig(**payload["config"]["model"])
    # Which latent read-out the run used cannot live in WorldModelConfig,
    # which rejects unknown keys.  Post-trained checkpoints carry it in their
    # own metadata block; pretraining runs record it in a sidecar beside the
    # checkpoint.  Prefer the payload, which travels with the file.
    stored = (payload.get("gwm_post_training") or {}).get("readout")
    readout = str(stored) if stored else read_readout_sidecar(checkpoint_path)
    model = build_readout_world_model(model_config, readout)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device)
    statistics = ActionStatistics.from_dict(payload["action_statistics"])
    config = dict(payload["config"])
    config["readout"] = readout
    return model, model_config, statistics, config


def set_posttraining_modes(model: nn.Module, trainable: tuple[str, ...]) -> dict[str, Any]:
    """Freeze everything but ``trainable``, and make every module deterministic.

    Three points matter beyond ``requires_grad``.

    *Normalisation and dropout are held in eval mode everywhere.*  The projector
    and ``pred_proj`` contain ``BatchNorm1d``.  In train mode they would use
    batch statistics, so the same latent would decode differently in the JEPA
    branch and the constraint branch, the running statistics of nominally frozen
    modules would drift, and the per-sample violation ``c`` would stop being a
    function of that sample alone.  Dropout in the ViT and in the GRU predictor
    would inject noise straight into ``C``, whose expectation the algorithm
    estimates.

    *Trainable modules are otherwise left in train mode.*  cuDNN refuses to run
    an RNN backward pass on a module in eval mode, and the GRU predictor is
    exactly what post-training differentiates.  Train mode is harmless for it
    once dropout is pinned to eval, since nothing else in the predictor depends
    on the mode.

    *The encoder and projector stay frozen* even though they are part of the
    latent pathway, because the position probe is trained on the projector's
    output and the JEPA targets must carry no gradient.
    """

    allowed = {"predictor", "pred_proj", "action_encoder", "projector", "encoder"}
    unknown = set(trainable) - allowed
    if unknown:
        raise ValueError(f"unknown trainable module names: {sorted(unknown)}")
    if "encoder" in trainable or "projector" in trainable:
        raise ValueError(
            "the encoder and projector must stay frozen: the position probe is trained on "
            "their output and the JEPA targets must carry no gradient"
        )
    model.eval()
    model.requires_grad_(False)
    counts: dict[str, int] = {}
    for name in trainable:
        module = getattr(model, name)
        module.requires_grad_(True)
        module.train()
        counts[name] = sum(p.numel() for p in module.parameters())
    stochastic = 0
    for module in model.modules():
        if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.Dropout)):
            module.eval()
            stochastic += 1
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {
        "trainable_modules": list(trainable),
        "trainable_parameters": counts,
        "trainable_total": sum(counts.values()),
        "frozen_total": frozen,
        "normalisation_and_dropout_in_eval": stochastic,
        "frozen_modules_in_eval_mode": all(
            not getattr(model, name).training
            for name in ("encoder", "projector")
            if hasattr(model, name)
        ),
    }


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Parameters in a stable order, so that flat gradients line up across steps."""

    return [parameter for _name, parameter in sorted(model.named_parameters()) if parameter.requires_grad]


@dataclass
class LatentCache:
    """Per-episode latents and labels, all aligned on the episode time axis.

    ``latents`` is ``(E, T, D)``, ``state`` is ``(E, T, S)`` in world units,
    ``action`` is ``(E, T, A)`` already z-scored with the training statistics,
    ``gravity`` is ``(E,)`` and ``split`` is ``0`` for development episodes and
    ``1`` for held-out ones.
    """

    latents: torch.Tensor
    state: torch.Tensor
    action: torch.Tensor
    gravity: torch.Tensor
    split: torch.Tensor
    source_id: torch.Tensor
    # ``(E, T)``: True when a contact falls between frame ``t`` and ``t+1``.
    # All False for the contact-free datasets, whose scenes cannot produce one.
    # The constraint needs it because the free-fall identity of Lemma 1 is
    # false across a contact -- measured at 12.4 px against a 1.008 px signal --
    # so such a window has to be excluded rather than counted as a violation.
    contact: torch.Tensor
    dt: float
    pixels_per_unit: float
    action_mean: torch.Tensor
    action_std: torch.Tensor
    metadata: dict[str, Any]

    def to(self, device: str | torch.device) -> LatentCache:
        return LatentCache(
            latents=self.latents.to(device),
            state=self.state.to(device),
            action=self.action.to(device),
            gravity=self.gravity.to(device),
            split=self.split.to(device),
            source_id=self.source_id.to(device),
            contact=self.contact.to(device),
            dt=self.dt,
            pixels_per_unit=self.pixels_per_unit,
            action_mean=self.action_mean.to(device),
            action_std=self.action_std.to(device),
            metadata=self.metadata,
        )

    @property
    def episodes(self) -> int:
        return int(self.latents.shape[0])

    @property
    def frames(self) -> int:
        return int(self.latents.shape[1])

    @property
    def latent_dim(self) -> int:
        return int(self.latents.shape[2])

    def select(self, mask: torch.Tensor | np.ndarray) -> LatentCache:
        index = (
            mask.detach()
            if torch.is_tensor(mask)
            else torch.as_tensor(np.asarray(mask))
        ).to(self.latents.device)
        if index.dtype == torch.bool:
            index = index.nonzero(as_tuple=True)[0]
        return LatentCache(
            latents=self.latents[index],
            state=self.state[index],
            action=self.action[index],
            gravity=self.gravity[index],
            split=self.split[index],
            source_id=self.source_id[index],
            contact=self.contact[index],
            dt=self.dt,
            pixels_per_unit=self.pixels_per_unit,
            action_mean=self.action_mean,
            action_std=self.action_std,
            metadata=self.metadata,
        )

    def windows(self, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Every contiguous window of ``length`` frames, as ``(episode, start)``."""

        if length > self.frames:
            raise ValueError(f"window length {length} exceeds episode length {self.frames}")
        starts = torch.arange(self.frames - length + 1, device=self.latents.device)
        episodes = torch.arange(self.episodes, device=self.latents.device)
        grid_e, grid_s = torch.meshgrid(episodes, starts, indexing="ij")
        return grid_e.reshape(-1), grid_s.reshape(-1)

    def gather_windows(
        self, episode_index: torch.Tensor, start: torch.Tensor, length: int
    ) -> dict[str, torch.Tensor]:
        """Materialise a batch of windows as a dict the constraint can consume."""

        offsets = torch.arange(int(length), device=self.latents.device)
        time = start.unsqueeze(-1) + offsets.unsqueeze(0)
        episode = episode_index.unsqueeze(-1).expand_as(time)
        return {
            "latents": self.latents[episode, time],
            "state": self.state[episode, time],
            "action": self.action[episode, time],
            "contact": self.contact[episode, time],
            "gravity": self.gravity[episode_index],
            "episode_index": episode_index,
            "start": start,
        }


def manifest_range(dataset_path: str | Path) -> list[int] | None:
    """The generator's recorded source range for held-out training-gravity episodes."""

    directory = Path(dataset_path)
    manifest = directory / "generation_manifest.json"
    if not manifest.is_file():
        return None
    value = json.loads(manifest.read_text()).get("heldout_train_gravity_source_range")
    if isinstance(value, list) and len(value) == 2:
        return [int(value[0]), int(value[1])]
    return None


def read_contact_flags(dataset_path: str | Path, *, episodes: int, frames: int) -> torch.Tensor:
    """``(E, T)`` contact flags, or all False for a dataset that has no column.

    Read straight from Lance rather than through the upstream trajectory store,
    which exposes a fixed set of fields.  A dataset without the column is one
    whose scene cannot produce a contact, so all False is the right value and
    not a silent default standing in for missing data.
    """

    import lance

    dataset = lance.dataset(str(Path(dataset_path) / "data.lance"))
    if "contact" not in dataset.schema.names:
        return torch.zeros((episodes, frames), dtype=torch.bool)
    table = dataset.to_table(columns=["episode_idx", "step_idx", "contact"]).to_pydict()
    flags = torch.zeros((episodes, frames), dtype=torch.bool)
    rows = np.asarray(table["episode_idx"]), np.asarray(table["step_idx"])
    values = np.asarray(table["contact"], dtype=bool)
    keep = (rows[0] < episodes) & (rows[1] < frames)
    flags[torch.from_numpy(rows[0][keep]).long(), torch.from_numpy(rows[1][keep]).long()] = (
        torch.from_numpy(values[keep])
    )
    return flags


@torch.no_grad()
def build_latent_cache(
    model: nn.Module,
    dataset_path: str | Path,
    statistics: ActionStatistics,
    *,
    device: str | torch.device = "cuda",
    batch_episodes: int = 32,
    dtype: torch.dtype = torch.float32,
) -> LatentCache:
    """Encode every episode of a dataset with the frozen encoder."""

    store = open_trajectory_store(dataset_path)
    lengths = np.unique(store.lengths)
    if len(lengths) != 1:
        raise ValueError(f"expected equal-length episodes, got lengths {lengths}")
    frames = int(lengths[0])
    was_training = model.training
    model.eval()
    image_mean = torch.tensor((0.485, 0.456, 0.406)).reshape(1, 3, 1, 1)
    image_std = torch.tensor((0.229, 0.224, 0.225)).reshape(1, 3, 1, 1)
    action_mean = torch.tensor(statistics.mean, dtype=torch.float32)
    action_std = torch.tensor(statistics.std, dtype=torch.float32)

    latents: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    gravities: list[float] = []
    source_ids: list[int] = []
    for begin in range(0, len(store.lengths), batch_episodes):
        end = min(begin + batch_episodes, len(store.lengths))
        pixel_chunk, state_chunk, action_chunk = [], [], []
        for index in range(begin, end):
            episode = store.episode(index)
            pixels = np.moveaxis(np.asarray(episode.pixels), -1, 1)
            tensor = torch.from_numpy(np.ascontiguousarray(pixels)).float().div_(255.0)
            pixel_chunk.append((tensor - image_mean) / image_std)
            state_chunk.append(torch.from_numpy(np.asarray(episode.state, dtype=np.float32)))
            raw_action = torch.from_numpy(np.asarray(episode.action, dtype=np.float32))
            action_chunk.append((raw_action - action_mean) / action_std)
            gravities.append(float(episode.gravity[0]))
            source_ids.append(int(episode.episode_id))
        pixel_batch = torch.stack(pixel_chunk).to(device)
        encoded = model.encode({"pixels": pixel_batch})["emb"]
        latents.append(encoded.to(dtype).cpu())
        states.append(torch.stack(state_chunk))
        actions.append(torch.stack(action_chunk))
    if was_training:
        model.train()

    contact = read_contact_flags(dataset_path, episodes=len(store.lengths), frames=frames)
    metadata = dict(store.metadata)
    scene = metadata.get("scene") or {}
    dt = float(metadata["dt"])
    view_height = float(scene.get("view_height", metadata.get("view_height", 1.0)))
    image_size = int(metadata["pixels_shape"][0])
    return LatentCache(
        latents=torch.cat(latents),
        state=torch.cat(states),
        action=torch.cat(actions),
        gravity=torch.tensor(gravities, dtype=torch.float32),
        split=torch.from_numpy(np.asarray(store.split_id, dtype=np.int64)),
        source_id=torch.tensor(source_ids, dtype=torch.long),
        contact=contact,
        dt=dt,
        pixels_per_unit=image_size / view_height,
        action_mean=action_mean,
        action_std=action_std,
        metadata={
            "dataset": str(Path(dataset_path).resolve()),
            "frames": frames,
            "dt": dt,
            "view_height": view_height,
            "image_size": image_size,
            "scene": scene,
            "test_gravities": metadata.get("test_gravities"),
            "train_gravity": metadata.get("train_gravity"),
            # Source-episode range of the held-out episodes drawn from the
            # *training* gravity distribution.  The counterfactual protocol
            # needs its real windows from that distribution, so selecting them
            # by range is exact where a gravity filter would also admit the
            # nearby points of the wide held-out grid.
            "heldout_train_gravity_source_range": manifest_range(dataset_path),
        },
    )


def save_latent_cache(cache: LatentCache, path: str | Path, fingerprint: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "gwm_latent_cache",
            "schema_version": 2,
            "fingerprint": fingerprint,
            "latents": cache.latents,
            "state": cache.state,
            "action": cache.action,
            "gravity": cache.gravity,
            "split": cache.split,
            "source_id": cache.source_id,
            "contact": cache.contact,
            "dt": cache.dt,
            "pixels_per_unit": cache.pixels_per_unit,
            "action_mean": cache.action_mean,
            "action_std": cache.action_std,
            "metadata": cache.metadata,
        },
        destination,
    )
    destination.with_suffix(".json").write_text(
        json.dumps({"fingerprint": fingerprint, "metadata": cache.metadata}, indent=2, sort_keys=True)
    )


def load_latent_cache(
    path: str | Path, *, fingerprint: dict[str, Any] | None = None
) -> LatentCache:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("kind") != "gwm_latent_cache":
        raise ValueError(f"not a gwm latent cache: {path}")
    if fingerprint is not None and payload["fingerprint"] != fingerprint:
        raise ValueError(
            "latent cache was built with a different encoder or dataset:\n"
            f"  cache:    {payload['fingerprint']}\n  requested: {fingerprint}"
        )
    return LatentCache(
        latents=payload["latents"],
        state=payload["state"],
        action=payload["action"],
        gravity=payload["gravity"],
        split=payload["split"],
        source_id=payload["source_id"],
        # Caches written before the bouncing scene existed have no contact
        # column; their datasets are contact-free by construction, so all
        # False is the correct value rather than a fallback.
        contact=payload.get(
            "contact", torch.zeros(payload["state"].shape[:2], dtype=torch.bool)
        ),
        dt=float(payload["dt"]),
        pixels_per_unit=float(payload["pixels_per_unit"]),
        action_mean=payload["action_mean"],
        action_std=payload["action_std"],
        metadata=payload["metadata"],
    )


__all__ += ["manifest_range", "save_latent_cache"]
