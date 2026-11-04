"""The attention-pool read-out must be a strict, neutral replacement for [CLS]."""

from __future__ import annotations

import json

import pytest
import torch
from sg_jepa.config import WorldModelConfig

from gwm.spatial_lewm import (
    READOUTS,
    AttentionPool,
    SpatialLeWM,
    build_readout_world_model,
    read_readout_sidecar,
    write_readout_sidecar,
)


def small_config() -> WorldModelConfig:
    return WorldModelConfig(image_size=64, patch_size=16, history_size=8, embed_dim=64)


def test_pool_starts_exactly_at_mean_pooling():
    """A zero query makes the weights uniform, so the variant gets no head start.

    Mean pooling is the uninformative starting point -- it averages every
    position embedding into the same constant and so carries no location -- and
    the variant has to learn its way out of it just as [CLS] does.
    """

    torch.manual_seed(0)
    pool = AttentionPool(32)
    tokens = torch.randn(5, 49, 32)
    torch.testing.assert_close(pool(tokens), tokens.mean(dim=1), atol=1e-5, rtol=1e-5)
    weights = pool.attention(tokens)
    torch.testing.assert_close(weights, torch.full_like(weights, 1.0 / 49), atol=1e-6, rtol=1e-6)


def test_pool_can_express_a_location():
    """There must *exist* a query for which pooling is location-selective.

    This is the capability mean pooling lacks, and the thing training has to
    find.  A random query does not have it -- measured, it puts *less* than
    uniform weight on the patch that differs -- so the test aims the query at
    that patch's key and checks that the pooled features then separate the
    locations, which they do by a margin of order the position code itself.
    """

    torch.manual_seed(1)
    hidden, patches = 16, 25
    pool = AttentionPool(hidden)
    position = torch.randn(patches, hidden)           # stands in for position embeddings
    content = torch.randn(hidden) * 4.0               # how an occupied patch differs
    outputs, selected = [], []
    for occupied in range(patches):
        tokens = position.clone()
        tokens[occupied] = tokens[occupied] + content
        with torch.no_grad():
            key = pool.to_keys(pool.norm(tokens.unsqueeze(0)))[0, occupied]
            pool.query.copy_(key / key.norm() * 40.0)
            outputs.append(pool(tokens.unsqueeze(0)))
            selected.append(float(pool.attention(tokens.unsqueeze(0))[0, occupied]))
    assert min(selected) > 0.9, "the pool cannot concentrate on a chosen patch"
    stacked = torch.cat(outputs)
    off_diagonal = torch.cdist(stacked, stacked) + torch.eye(patches) * 1.0e9
    assert float(off_diagonal.min()) > 0.5, "the pool maps different locations to one feature"


def test_pool_keys_ignore_a_channel_uniform_offset():
    """The LayerNorm on the key path is not neutral, so state it.

    ``LayerNorm`` removes each token's mean, so adding the same value to every
    channel of a token leaves its key unchanged and the pooling weights with
    it.  A patch is distinguished by the *direction* of its token, not by a
    constant offset -- worth knowing before reading attention maps.
    """

    torch.manual_seed(3)
    pool = AttentionPool(16)
    tokens = torch.randn(1, 25, 16)
    shifted = tokens.clone()
    shifted[0, 7] = shifted[0, 7] + 5.0
    with torch.no_grad():
        pool.query.copy_(torch.randn(16))
    torch.testing.assert_close(pool.attention(tokens), pool.attention(shifted), atol=1e-6, rtol=1e-6)


def test_uniform_pooling_cannot_express_a_location():
    """The reason the [CLS]/mean read-out loses position, as a direct check.

    Mean pooling averages every position code into the same constant, so two
    scenes differing only in *which* patch the object occupies collapse to
    features a factor ``patches`` closer together than attention pooling keeps
    them.
    """

    torch.manual_seed(2)
    patches, hidden = 25, 16
    position = torch.randn(patches, hidden)
    content = torch.randn(hidden) * 4.0
    mean_out, attn_out = [], []
    pool = AttentionPool(hidden)
    with torch.no_grad():
        pool.query.copy_(torch.randn(hidden))
    for occupied in range(patches):
        tokens = position.clone()
        tokens[occupied] = tokens[occupied] + content
        mean_out.append(tokens.mean(dim=0, keepdim=True))
        attn_out.append(pool(tokens.unsqueeze(0)))
    mean_spread = float(torch.cdist(torch.cat(mean_out), torch.cat(mean_out)).max())
    attn_spread = float(torch.cdist(torch.cat(attn_out), torch.cat(attn_out)).max())
    assert attn_spread > 5.0 * mean_spread


@pytest.mark.parametrize("readout", READOUTS)
def test_both_readouts_share_the_latent_contract(readout):
    config = small_config()
    model = build_readout_world_model(config, readout)
    batch = {"pixels": torch.randn(2, 10, 3, 64, 64), "action": torch.randn(2, 10, 3)}
    info = model.encode(dict(batch))
    assert info["emb"].shape == (2, 10, config.embed_dim)
    assert info["act_emb"].shape == (2, 10, config.embed_dim)
    predicted = model.predict(info["emb"][:, :8], info["act_emb"][:, :8])
    assert predicted.shape == (2, 8, config.embed_dim)


def test_attention_pool_variant_is_a_small_addition():
    config = small_config()
    stock = build_readout_world_model(config, "cls")
    variant = build_readout_world_model(config, "attn_pool")
    stock_count = sum(p.numel() for p in stock.parameters())
    variant_count = sum(p.numel() for p in variant.parameters())
    assert isinstance(variant, SpatialLeWM)
    assert variant_count > stock_count
    assert variant_count < stock_count * 1.05, "the variant must not change model capacity"


def test_readout_sidecar_round_trip(tmp_path):
    assert read_readout_sidecar(tmp_path) == "cls", "absent sidecar must mean the stock model"
    write_readout_sidecar(tmp_path, "attn_pool")
    assert read_readout_sidecar(tmp_path) == "attn_pool"
    assert read_readout_sidecar(tmp_path / "checkpoint.pt") == "attn_pool"
    (tmp_path / "readout.json").write_text(json.dumps({"readout": "nonsense"}))
    with pytest.raises(ValueError, match="unknown read-out"):
        read_readout_sidecar(tmp_path)
    with pytest.raises(ValueError, match="readout must be one of"):
        write_readout_sidecar(tmp_path, "nonsense")


def test_variant_state_dict_reloads_strictly():
    """The trainer reloads with strict=True, so the keys must match exactly."""

    config = small_config()
    model = build_readout_world_model(config, "attn_pool")
    reloaded = build_readout_world_model(config, "attn_pool")
    reloaded.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    reloaded.eval()
    batch = {"pixels": torch.randn(1, 9, 3, 64, 64)}
    with torch.no_grad():
        torch.testing.assert_close(
            model.encode(dict(batch))["emb"], reloaded.encode(dict(batch))["emb"]
        )


def test_spatial_map_keeps_the_spatial_axis():
    """The read-out must preserve *which* patch, not just how much.

    Two scenes differing only in which patch the object occupies have to map to
    different latents.  This is the property both ``[CLS]`` and mean pooling
    lack, and it is why this read-out exists.
    """

    from gwm.spatial_lewm import SpatialMapReadout

    torch.manual_seed(4)
    patches, hidden = 16, 12
    readout = SpatialMapReadout(hidden, patches, channels=2)
    assert readout.output_dim == patches * 2
    base = torch.randn(patches, hidden)
    content = torch.randn(hidden) * 4.0
    outputs = []
    for occupied in range(patches):
        tokens = base.clone()
        tokens[occupied] = tokens[occupied] + content
        outputs.append(readout(tokens.unsqueeze(0)))
    stacked = torch.cat(outputs)
    off_diagonal = torch.cdist(stacked, stacked) + torch.eye(patches) * 1.0e9
    assert float(off_diagonal.min()) > 1.0e-2


def test_spatial_map_rejects_a_mismatched_resolution():
    """The read-out is sized for one patch count; a silent reshape would be worse."""

    from gwm.spatial_lewm import SpatialMapReadout

    readout = SpatialMapReadout(12, 16, channels=2)
    with pytest.raises(ValueError, match="expected 16 patches"):
        readout(torch.randn(1, 25, 12))


def test_spatial_map_latent_dimension_is_unchanged():
    """Only the read-out changes; the latent the constraint sees keeps its size."""

    config = small_config()
    for readout in READOUTS:
        model = build_readout_world_model(config, readout)
        info = model.encode({"pixels": torch.randn(1, 4, 3, 64, 64)})
        assert info["emb"].shape == (1, 4, config.embed_dim)


def test_posttrained_checkpoint_is_self_describing(tmp_path):
    """A post-trained checkpoint must rebuild without its sidecar.

    The evaluation loads the checkpoint the post-training stage writes, so if
    the read-out is not carried forward the wrong architecture is built and
    ``load_state_dict(strict=True)`` fails -- which it did, after the whole
    post-training run had already been paid for.
    """

    from gwm.latents import load_world_model

    config = small_config()
    model = build_readout_world_model(config, "attn_pool")
    payload = {
        "schema_version": 2,
        "kind": "native_world_model_training",
        "step": 0,
        "cursor": {"next_epoch": 0, "next_microbatch": 0, "completed_epochs": 0},
        "model": model.state_dict(),
        "config": {"name": "t", "model": vars(config) | {}},
        "action_statistics": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
        "gwm_post_training": {"readout": "attn_pool", "variant": "counterfactual"},
    }
    payload["config"]["model"] = {
        key: getattr(config, key) for key in config.__dataclass_fields__
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(payload, path)
    assert not (tmp_path / "readout.json").exists(), "the sidecar must not be needed"
    rebuilt, _config, _statistics, info = load_world_model(path, device="cpu")
    assert info["readout"] == "attn_pool"
    assert isinstance(rebuilt, SpatialLeWM)
