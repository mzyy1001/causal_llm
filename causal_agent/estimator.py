"""
Outcome models over deployment counts.

Both estimators consume (successes, trials) pairs with a known denominator.
Neither ever sees the per-drone survivor table - see session.CausalSession for
why that table cannot be used for outcome modelling.

numpy only: scipy is not a dependency of this repo, so the logistic fit is a
hand-rolled Newton/IRLS with a Laplace approximation for the posterior, and
tail probabilities come from Monte Carlo rather than incomplete-beta calls.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


# ----------------------------------------------------------------------
# Model-free, per-arm
# ----------------------------------------------------------------------

@dataclass
class BetaBinomial:
    """
    Independent Beta posterior per arm. No pooling, no functional form.

    This is the honest baseline: it makes no structural assumption at all, so
    it is the reference the pooled model has to beat. With ~20 drones per arm
    its intervals are wide, which is precisely the budget problem.
    """

    prior_a: float = 1.0
    prior_b: float = 1.0
    rng: Any = None

    def __post_init__(self):
        if self.rng is None:
            self.rng = np.random.default_rng(0)

    def posterior(self, survived: int, deployed: int) -> tuple:
        return self.prior_a + survived, self.prior_b + (deployed - survived)

    def mean(self, survived: int, deployed: int) -> float:
        a, b = self.posterior(survived, deployed)
        return a / (a + b)

    def interval(self, survived: int, deployed: int, mass: float = 0.9,
                 draws: int = 20000) -> tuple:
        a, b = self.posterior(survived, deployed)
        s = self.rng.beta(a, b, size=draws)
        lo = (1.0 - mass) / 2.0
        return float(np.quantile(s, lo)), float(np.quantile(s, 1.0 - lo))

    def prob_at_least(self, survived: int, deployed: int, threshold: float,
                      draws: int = 20000) -> float:
        a, b = self.posterior(survived, deployed)
        return float(np.mean(self.rng.beta(a, b, size=draws) >= threshold))


# ----------------------------------------------------------------------
# Pooled
# ----------------------------------------------------------------------

class BinomialLogit:
    """
    Bayesian binomial logistic regression, MAP + Laplace posterior.

    Pooling is the whole point: every drone informs every coefficient, so an
    8-arm factorial with 200 drones estimates a main effect far more precisely
    than a 2-arm contrast with the same 200 drones.

    Args:
        prior_sd: Gaussian prior scale on the coefficients (weakly informative;
            the intercept gets `intercept_sd`). Also acts as the ridge that
            keeps the fit defined when arms are fewer than features.
        prior_mean: optional per-coefficient prior mean (length p, including the
            intercept slot). This is how a hypothesis' sign priors enter: a
            mechanism believed protective gets a small positive mean, believed
            harmful a small negative one. Zero (default) is agnostic.
    """

    def __init__(self, prior_sd: float = 2.5, intercept_sd: float = 10.0,
                 max_iter: int = 100, tol: float = 1e-9, rng: Any = None,
                 prior_mean: Optional[np.ndarray] = None):
        self.prior_sd = prior_sd
        self.intercept_sd = intercept_sd
        self.max_iter = max_iter
        self.tol = tol
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.prior_mean = None if prior_mean is None else np.asarray(prior_mean, dtype=float)
        self.beta: Optional[np.ndarray] = None
        self.cov: Optional[np.ndarray] = None
        self.names: Optional[List[str]] = None
        self.converged: bool = False
        self.n_iter: int = 0
        self.log_evidence: Optional[float] = None

    def _prior_precision(self, p: int) -> np.ndarray:
        sd = np.full(p, self.prior_sd, dtype=float)
        sd[0] = self.intercept_sd
        return np.diag(1.0 / sd ** 2)

    def fit(
        self,
        X: np.ndarray,
        successes: Sequence[int],
        trials: Sequence[int],
        names: Optional[Sequence[str]] = None,
    ) -> "BinomialLogit":
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(successes, dtype=float)
        n = np.asarray(trials, dtype=float)
        if X.shape[0] != len(y) or len(y) != len(n):
            raise ValueError("X, successes and trials must have the same length")
        if np.any(y > n) or np.any(y < 0):
            raise ValueError("successes must lie in [0, trials]")

        p = X.shape[1]
        prior_prec = self._prior_precision(p)
        m0 = np.zeros(p) if self.prior_mean is None else self.prior_mean
        if len(m0) != p:
            raise ValueError(f"prior_mean has length {len(m0)}, expected {p}")
        beta = m0.copy()

        for it in range(self.max_iter):
            eta = X @ beta
            mu = sigmoid(eta)
            grad = X.T @ (y - n * mu) - prior_prec @ (beta - m0)
            w = np.clip(n * mu * (1.0 - mu), 1e-10, None)
            hess = X.T @ (X * w[:, None]) + prior_prec
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hess, grad, rcond=None)[0]
            # Damped Newton: the MAP is well behaved but arms with 0/n or n/n
            # counts can otherwise overshoot badly.
            beta_new = beta + np.clip(step, -4.0, 4.0)
            if np.max(np.abs(beta_new - beta)) < self.tol:
                beta = beta_new
                self.converged = True
                self.n_iter = it + 1
                break
            beta = beta_new
        else:
            self.n_iter = self.max_iter

        eta = X @ beta
        mu = sigmoid(eta)
        w = np.clip(n * mu * (1.0 - mu), 1e-10, None)
        hess = X.T @ (X * w[:, None]) + prior_prec
        self.beta = beta
        self.cov = np.linalg.inv(hess)
        self.names = list(names) if names is not None else [f"x{i}" for i in range(p)]

        # Laplace approximation to the marginal likelihood:
        #   log p(D|h) ~ loglik(b^) + log prior(b^) + (p/2) log 2pi + (1/2) log|Cov|
        # The binomial coefficient log C(n, y) is constant across hypotheses on
        # the same data and is dropped, so only *differences* in log_evidence
        # between hypotheses are meaningful.
        mu_c = np.clip(mu, 1e-12, 1 - 1e-12)
        loglik = float(np.sum(y * np.log(mu_c) + (n - y) * np.log(1.0 - mu_c)))
        resid = beta - m0
        log_prior = float(
            -0.5 * resid @ prior_prec @ resid
            - 0.5 * p * np.log(2.0 * np.pi)
            + 0.5 * np.linalg.slogdet(prior_prec)[1]
        )
        sign, logdet_cov = np.linalg.slogdet(self.cov)
        occam = 0.5 * logdet_cov if sign > 0 else -np.inf
        self.log_evidence = loglik + log_prior + 0.5 * p * np.log(2.0 * np.pi) + occam
        return self

    # -- inference ------------------------------------------------------

    def _check_fitted(self):
        if self.beta is None:
            raise RuntimeError("call fit() first")

    def coefficients(self) -> Dict[str, Dict[str, float]]:
        """Posterior mean and sd per coefficient, on the log-odds scale."""
        self._check_fitted()
        sd = np.sqrt(np.diag(self.cov))
        return {
            name: {
                "mean": float(self.beta[i]),
                "sd": float(sd[i]),
                "z": float(self.beta[i] / sd[i]) if sd[i] > 0 else 0.0,
            }
            for i, name in enumerate(self.names)
        }

    def sample_beta(self, draws: int = 4000) -> np.ndarray:
        self._check_fitted()
        # Cholesky of the Laplace covariance; jitter if it is near-singular.
        cov = self.cov
        for jitter in (0.0, 1e-10, 1e-8, 1e-6):
            try:
                L = np.linalg.cholesky(cov + jitter * np.eye(cov.shape[0]))
                break
            except np.linalg.LinAlgError:
                continue
        else:
            L = np.linalg.cholesky(np.diag(np.diag(cov)) + 1e-8 * np.eye(cov.shape[0]))
        z = self.rng.standard_normal((draws, len(self.beta)))
        return self.beta + z @ L.T

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Posterior-mean survival probability (plug-in)."""
        self._check_fitted()
        return sigmoid(np.atleast_2d(np.asarray(X, dtype=float)) @ self.beta)

    def predictive_samples(self, X: np.ndarray, draws: int = 4000) -> np.ndarray:
        """
        Posterior draws of the survival probability.

        Returns:
            (draws, n_rows) array.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return sigmoid(self.sample_beta(draws) @ X.T)

    def prob_at_least(self, X: np.ndarray, threshold: float, draws: int = 4000) -> np.ndarray:
        """P(true survival rate >= threshold) for each row of X."""
        return np.mean(self.predictive_samples(X, draws) >= threshold, axis=0)
