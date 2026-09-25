"""
Causal agent scaffold for CausalGame.

Three pieces, kept deliberately separate so the causal-learning core can be
swapped without touching data collection:

- session.py    count-based data collection (immune to survivorship censoring)
- design.py     factorial / information-driven allocation of the 10 deployments
- estimator.py  pooled Bayesian logistic regression over design features

The scaffold never reads experiments/*.json. Ground truth (trap_design,
structural formulas, expected_performance) lives there, so anything that reads
it is cheating. All knowledge enters through the HTTP API.
"""

from .session import Arm, CausalSession, CensoredDataError
from .design import Factor, DesignSpace, fractional_factorial, FeatureMap
from .estimator import BinomialLogit, BetaBinomial

__all__ = [
    "Arm",
    "CausalSession",
    "CensoredDataError",
    "Factor",
    "DesignSpace",
    "fractional_factorial",
    "FeatureMap",
    "BinomialLogit",
    "BetaBinomial",
]
