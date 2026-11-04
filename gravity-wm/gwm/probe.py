"""The frozen position probe ``h`` and its accuracy budget.

The probe is a small network trained on top of the frozen encoder to regress
object positions, then frozen.  Two requirements from the method note shape
this module.

*The probe must be accurate relative to the signal.*  The constraint is a
second difference of decoded positions, so a position error ``delta`` enters
the acceleration with weight ``4/dt_eff^2`` (Lemma 2: ``E[c_real] <= 16
delta^2 / dt_eff^4``).  What matters is therefore ``delta`` measured against
the one-step gravity displacement ``g dt_eff^2``.  :func:`probe_report`
reports exactly that ratio, in world units and in pixels, so the viability of
a scene can be judged before any post-training is run.

*A second, independently trained probe is reserved for evaluation*, so that a
predictor which satisfies the constraint only by exploiting the training probe
is detected.  :func:`train_position_probes` trains both from one feature
extraction pass, on disjoint halves of the labelled episodes and from
different initialisations, and the evaluation probe is never shown to
post-training.

The probe decodes a *single* latent, as the theory requires: ``p_{k,tau}`` is
the position decoded from the latent at step ``tau``, whether that latent is
encoded from a real frame or predicted.  A multi-frame probe window would mix
real and predicted latents inside one decoded position and break the
correspondence with Lemma 1.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

__all__ = [
    "PositionProbe",
    "ProbeConfig",
    "ProbeReport",
    "build_position_probe",
    "load_position_probe",
    "probe_report",
    "save_position_probe",
    "train_position_probes",
]


@dataclass(frozen=True)
class ProbeConfig:
    hidden: tuple[int, int] = (512, 256)
    input_norm: bool = True
    dropout: float = 0.0
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-6
    batch_size: int = 512
    max_epochs: int = 400
    patience: int = 25
    seed: int = 1234

    def __post_init__(self) -> None:
        if self.max_epochs < 1 or self.patience < 1 or self.batch_size < 1:
            raise ValueError("max_epochs, patience and batch_size must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("invalid probe optimizer settings")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_position_probe(
    feature_dim: int, config: ProbeConfig, *, generator: torch.Generator | None = None
) -> nn.Module:
    """An MLP from one latent to one planar position.

    Dropout defaults to zero: the probe is evaluated inside a differentiable
    constraint, and stochastic dropout would inject noise directly into the
    quantity whose expectation defines ``C``.  The width follows the paper's
    state probe so that probe capacity is not what distinguishes this study.

    ``input_norm`` controls the leading ``LayerNorm``, which the paper's state
    probe has.  It is configurable because it is not harmless here: it removes
    the per-sample mean and scale of the latent, so any position information
    carried by the latent's norm is discarded before the probe sees it.  The
    probe's accuracy is the quantity the whole constraint rests on, so this is
    worth measuring rather than assuming.
    """

    first, second = config.hidden
    layers: list[nn.Module] = []
    if config.input_norm:
        layers.append(nn.LayerNorm(int(feature_dim)))
    layers += [nn.Linear(int(feature_dim), first), nn.GELU()]
    if config.dropout > 0.0:
        layers.append(nn.Dropout(config.dropout))
    layers += [nn.Linear(first, second), nn.GELU()]
    if config.dropout > 0.0:
        layers.append(nn.Dropout(config.dropout))
    layers.append(nn.Linear(second, 2))
    probe = nn.Sequential(*layers)
    if generator is not None:
        for module in probe.modules():
            if isinstance(module, nn.Linear):
                bound = 1.0 / float(np.sqrt(module.in_features))
                with torch.no_grad():
                    module.weight.uniform_(-bound, bound, generator=generator)
                    module.bias.uniform_(-bound, bound, generator=generator)
    return probe


class PositionProbe(nn.Module):
    """A frozen probe plus the normalization that puts it in world units.

    The network regresses standardized positions; this wrapper rescales them,
    so callers always work in the world units in which ``dt`` and ``g`` are
    expressed and in which the free-fall targets are exact.
    """

    def __init__(self, network: nn.Module, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.network = network
        self.register_buffer("target_mean", mean.reshape(1, 2).clone())
        self.register_buffer("target_std", std.reshape(1, 2).clone())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = self.network(features.to(self.target_mean.dtype))
        return normalized * self.target_std + self.target_mean

    def freeze(self) -> PositionProbe:
        self.eval()
        self.requires_grad_(False)
        return self


@dataclass(frozen=True)
class ProbeReport:
    """Measured probe accuracy and the signal it has to resolve."""

    rmse_world: float
    rmse_per_axis: tuple[float, float]
    median_abs_world: float
    p95_abs_world: float
    rmse_pixels: float
    gravity_step_world: float
    gravity_step_pixels: float
    signal_to_noise: float
    lemma2_bound_on_c_real: float
    samples: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def probe_report(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    dt_effective: float,
    gravity: float,
    pixels_per_unit: float,
) -> ProbeReport:
    """Summarize probe accuracy against the gravity displacement it must resolve.

    ``dt_effective`` is ``stride * dt``, the interval actually used by the
    finite difference, so ``gravity_step_world = gravity * dt_effective**2`` is
    the displacement one step of gravity contributes at that spacing.  The
    reported ``lemma2_bound_on_c_real`` is Lemma 2's ``16 delta^2 /
    dt_effective^4``, an upper bound on ``E[c_real]`` and hence on the scale of
    the tolerance.
    """

    error = (predicted - target).double()
    squared = error.pow(2).sum(dim=-1)
    delta = float(squared.mean().sqrt())
    step = float(gravity) * float(dt_effective) ** 2
    return ProbeReport(
        rmse_world=delta,
        rmse_per_axis=(
            float(error[:, 0].pow(2).mean().sqrt()),
            float(error[:, 1].pow(2).mean().sqrt()),
        ),
        median_abs_world=float(error.abs().median()),
        p95_abs_world=float(torch.quantile(error.abs().flatten(), 0.95)),
        rmse_pixels=delta * float(pixels_per_unit),
        gravity_step_world=step,
        gravity_step_pixels=step * float(pixels_per_unit),
        signal_to_noise=step / delta if delta > 0.0 else float("inf"),
        lemma2_bound_on_c_real=16.0 * delta**2 / float(dt_effective) ** 4,
        samples=int(error.shape[0]),
    )


def _fit_one(
    network: nn.Module,
    train: tuple[torch.Tensor, torch.Tensor],
    validation: tuple[torch.Tensor, torch.Tensor],
    config: ProbeConfig,
    *,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Train one probe with early stopping on held-out standardized MSE."""

    features, targets = train
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)
    loader = DataLoader(
        TensorDataset(features, targets),
        batch_size=min(config.batch_size, len(features)),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(seed)),
        drop_last=False,
    )
    best = float("inf")
    best_state = {key: value.detach().clone() for key, value in network.state_dict().items()}
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(config.max_epochs):
        network.train()
        total = 0.0
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(network(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(x)
        scheduler.step()
        network.eval()
        with torch.no_grad():
            value = float(
                F.mse_loss(network(validation[0].to(device)), validation[1].to(device))
            )
        history.append({"epoch": epoch + 1, "train_mse": total / len(features), "val_mse": value})
        if value < best - 1.0e-9:
            best = value
            best_state = {k: v.detach().clone() for k, v in network.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    network.load_state_dict(best_state)
    network.eval()
    return {"best_val_mse_standardized": best, "epochs_run": len(history), "history": history}


def train_position_probes(
    features: torch.Tensor,
    positions: torch.Tensor,
    episode_ids: np.ndarray,
    *,
    config: ProbeConfig,
    device: torch.device,
    validation_fraction: float = 0.15,
) -> tuple[PositionProbe, PositionProbe, dict[str, Any]]:
    """Train the training probe and the independent evaluation probe.

    The two probes are made independent in the two ways that matter for the
    audit they support: they see **disjoint halves of the labelled episodes**
    and they start from **different initialisations**.  A predictor that
    reduced the constraint by exploiting idiosyncrasies of ``h`` would then
    still be caught by ``h_eval``, which shares no training frame with it.

    Splitting is by episode, never by frame, so that neighbouring frames of
    one episode cannot leak a position across the split.
    """

    if features.ndim != 2 or positions.shape != (features.shape[0], 2):
        raise ValueError(
            f"expected features (N, D) and positions (N, 2); got {tuple(features.shape)} "
            f"and {tuple(positions.shape)}"
        )
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must lie in (0, 0.5)")
    episodes = np.unique(episode_ids)
    if len(episodes) < 8:
        raise ValueError("need at least eight labelled episodes to split two probes")
    rng = np.random.default_rng(config.seed)
    shuffled = episodes.copy()
    rng.shuffle(shuffled)
    halves = np.array_split(shuffled, 2)

    standardize_mean = positions.mean(dim=0)
    standardize_std = positions.std(dim=0).clamp_min(1.0e-6)
    probes: list[PositionProbe] = []
    reports: list[dict[str, Any]] = []
    for index, half in enumerate(halves):
        validation_count = max(1, int(round(len(half) * validation_fraction)))
        val_episodes = set(int(value) for value in half[:validation_count])
        train_episodes = set(int(value) for value in half[validation_count:])
        if not train_episodes:
            raise ValueError("probe split left no training episodes")
        train_mask = np.isin(episode_ids, list(train_episodes))
        val_mask = np.isin(episode_ids, list(val_episodes))
        standardized = (positions - standardize_mean) / standardize_std
        generator = torch.Generator().manual_seed(config.seed + 7919 * index)
        network = build_position_probe(features.shape[1], config, generator=generator).to(device)
        report = _fit_one(
            network,
            (features[train_mask], standardized[train_mask]),
            (features[val_mask], standardized[val_mask]),
            config,
            device=device,
            seed=config.seed + index,
        )
        report.update(
            {
                "role": "train" if index == 0 else "eval",
                "train_frames": int(train_mask.sum()),
                "val_frames": int(val_mask.sum()),
                "train_episodes": len(train_episodes),
                "val_episodes": len(val_episodes),
            }
        )
        reports.append(report)
        probes.append(
            PositionProbe(
                network, standardize_mean.to(device), standardize_std.to(device)
            ).to(device).freeze()
        )
    overlap = set(int(v) for v in halves[0]) & set(int(v) for v in halves[1])
    summary = {
        "config": config.to_dict(),
        "feature_dim": int(features.shape[1]),
        "episodes": int(len(episodes)),
        "frames": int(features.shape[0]),
        "episode_overlap_between_probes": len(overlap),
        "target_mean": [float(value) for value in standardize_mean],
        "target_std": [float(value) for value in standardize_std],
        "probes": reports,
    }
    if overlap:
        raise RuntimeError("the two probes must not share labelled episodes")
    return probes[0], probes[1], summary


def save_position_probe(probe: PositionProbe, path: str | Path, metadata: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "gwm_position_probe",
            "schema_version": 1,
            "state_dict": probe.state_dict(),
            "metadata": metadata,
        },
        destination,
    )
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True))


def load_position_probe(
    path: str | Path, *, feature_dim: int, config: ProbeConfig, device: str | torch.device = "cpu"
) -> PositionProbe:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("kind") != "gwm_position_probe":
        raise ValueError(f"not a gwm position probe: {path}")
    network = build_position_probe(feature_dim, config)
    probe = PositionProbe(network, torch.zeros(2), torch.ones(2))
    probe.load_state_dict(payload["state_dict"], strict=True)
    return probe.to(device).freeze()
