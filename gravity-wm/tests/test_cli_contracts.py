"""Contracts between the CLI scripts and the library they drive."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def test_evaluation_takes_the_constraint_form_from_the_run() -> None:
    """A position-form run must not be evaluated against the acceleration default.

    The two forms of ``C`` share a zero set but not a scale, so a tolerance
    calibrated in one is meaningless against a violation measured in the other.
    Evaluating the position-form runs with the acceleration default reported
    ``C/eps = 240`` and ``feasible = False`` for runs that were inside their own
    tolerance.
    """

    spec = importlib.util.spec_from_file_location(
        "gwm_eval_cli", Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    report = {"command": {"constraint_form": "position"}}
    none = argparse.Namespace(constraint_form=None)
    assert module.constraint_form_for(report, none) == "position"
    assert module.constraint_form_for(None, none) == "acceleration"
    assert module.constraint_form_for({"command": {}}, none) == "acceleration"
    assert (
        module.constraint_form_for(report, argparse.Namespace(constraint_form="position"))
        == "position"
    )
    with pytest.raises(ValueError, match="contradicts the post-training report"):
        module.constraint_form_for(report, argparse.Namespace(constraint_form="acceleration"))


def test_gravity_scale_overrides_only_the_gravity_coordinate() -> None:
    """``--gravity-scale`` must fix g's scale and leave the others measured.

    With a constant training gravity the measured scale collapses to the 1e-6
    floor, so a counterfactual ``g' = 9`` would reach 5e6 in normalised units.
    On the narrow dataset it reaches +100 against a training range of +-4.
    """

    from sg_jepa.data.dataset import ActionStatistics

    spec = importlib.util.spec_from_file_location(
        "gwm_pretrain_cli", Path(__file__).resolve().parents[1] / "scripts" / "pretrain.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    # gravity constant at 4.0; the other two coordinates genuinely vary
    action = np.array([[[0.0, 1.0, 4.0], [2.0, 3.0, 4.0]]])
    measured = ActionStatistics.fit(action)
    assert measured.std[-1] == pytest.approx(1.0e-6)
    assert abs((9.0 - measured.mean[-1]) / measured.std[-1]) > 1.0e6

    undo = module.fix_gravity_scale(2.309)
    try:
        fixed = ActionStatistics.fit(action)
        assert fixed.std[-1] == pytest.approx(2.309)
        assert fixed.std[:-1] == pytest.approx(measured.std[:-1])
        assert fixed.mean == pytest.approx(measured.mean)
        assert abs((9.0 - fixed.mean[-1]) / fixed.std[-1]) < 3.0
    finally:
        undo()
    assert ActionStatistics.fit(action).std[-1] == pytest.approx(1.0e-6)


def test_epsilon_override_pins_the_tolerance_across_stages() -> None:
    """A staged schedule must not let a stage relax its successor's tolerance.

    ``epsilon`` is normally ``(1 + rho)`` times the *current* checkpoint's
    factual rollout violation.  In a staged run that is self-defeating:
    measured on ball_ax, stage one took the factual violation from 0.0113 to
    8.711 and stage two then calibrated ``eps`` at 10.89 -- 773x the original --
    so the constraint was vacuous and stage two's OPTIMISE steps undid stage
    one entirely (horizontal response 0.421 -> 0.014).
    """

    from gwm.posttrain import PostTrainConfig

    assert PostTrainConfig().epsilon_override is None
    pinned = PostTrainConfig(epsilon_override=0.0141)
    assert pinned.epsilon_override == pytest.approx(0.0141)
    # the scale still composes, so a staged schedule can tighten deliberately
    assert PostTrainConfig(epsilon_override=0.0141, epsilon_scale=8.0).epsilon_scale == 8.0
    with pytest.raises(ValueError, match="epsilon_override must be positive"):
        PostTrainConfig(epsilon_override=0.0)
