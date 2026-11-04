"""Gravity-constrained post-training of JEPA world models.

Implements the algorithm analysed in ``WM_gravity_theory``: post-train the
predictor of a pretrained Semigroup-JEPA world model so that, for any gravity
it is conditioned on, the object motion encoded in its predicted latents obeys
free fall.  Physics enters through a single constraint evaluated with a frozen
position probe on rollouts under counterfactual gravities.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
