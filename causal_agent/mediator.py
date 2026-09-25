"""
C3-lite: censored-mediator decomposition of deployment counts.

Some scenarios assign each drone a hidden stratum (a deployment zone) that
strongly moderates survival. The stratum is never reported for deployments,
but two side channels identify it:

  * the seeded observational history reports the stratum label together with
    a numeric proxy (e.g. altitude_band with temperature), which calibrates
    P(stratum | proxy);
  * per-drone survivor records report the proxy, so each survivor's stratum
    is inferable - only the *deaths'* strata are missing.

Given survivor strata (soft counts) and total deaths per arm, an EM over the
shared assignment mix pi recovers per-arm per-stratum survival - the censored
likelihood P(M|do(d)) * P(surv|M,d) / P(surv|d) from the design doc, in its
simplest usable form. Downstream, each arm becomes one pseudo-arm per stratum
with the stratum as an environment variable, so the existing conditioning and
worst-case-over-environments decision machinery apply unchanged.

Why this matters: when stage 2 draws a different stratum mix than stage 1,
marginal stage-1 rates are systematically wrong for stage 2 (measured: arms at
70% observed collapse to ~50% on the fleet). Only stratum-specific rates
transfer.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .session import Arm


@dataclass
class StratumProxy:
    """P(stratum | numeric proxy) calibrated from history, per-level Gaussian."""

    cat_key: str
    proxy_key: str
    levels: List[str]
    means: np.ndarray
    sds: np.ndarray
    counts: np.ndarray

    def infer(self, value: float) -> np.ndarray:
        """Posterior over strata for one proxy value (uniform level prior)."""
        z = (value - self.means) / self.sds
        log_lik = -0.5 * z ** 2 - np.log(self.sds)
        w = np.exp(log_lik - log_lik.max())
        return w / w.sum()


def calibrate_stratum_proxy(history: Sequence[Dict[str, Any]],
                            min_levels: int = 2, max_levels: int = 5,
                            min_separation: float = 1.5,
                            allowed_proxies: Optional[set] = None
                            ) -> Optional[StratumProxy]:
    """
    Find a categorical env variable in history plus a numeric proxy that
    separates its levels. Returns None when no reliable pair exists (then the
    mediator machinery stays off and counts are used as-is).
    """
    env_rows = [h.get("environment") or {} for h in history]
    env_rows = [e for e in env_rows if e]
    if len(env_rows) < 8:
        return None
    keys = set().union(*(e.keys() for e in env_rows))
    cat_keys = [k for k in sorted(keys)
                if any(isinstance(e.get(k), str) for e in env_rows)]
    num_keys = [k for k in sorted(keys)
                if all(isinstance(e.get(k), (int, float)) for e in env_rows
                       if k in e)]
    if allowed_proxies is not None:
        # Only proxies that deployment records actually expose are usable:
        # a perfect separator that survivors never report assigns nothing.
        num_keys = [k for k in num_keys if k in allowed_proxies]
    best = None
    for ck in cat_keys:
        levels = sorted({e[ck] for e in env_rows if isinstance(e.get(ck), str)})
        if not (min_levels <= len(levels) <= max_levels):
            continue
        for nk in num_keys:
            vals = {lv: [e[nk] for e in env_rows
                         if e.get(ck) == lv and isinstance(e.get(nk), (int, float))]
                    for lv in levels}
            if any(len(v) < 2 for v in vals.values()):
                continue
            means = np.array([np.mean(vals[lv]) for lv in levels])
            sds = np.array([max(np.std(vals[lv]), 1e-3) for lv in levels])
            counts = np.array([len(vals[lv]) for lv in levels], dtype=float)
            # separation: adjacent-level mean gaps vs within-level spread
            order = np.argsort(means)
            gaps = np.diff(means[order])
            spread = np.array([max(sds[order][i], sds[order][i + 1])
                               for i in range(len(gaps))])
            if len(gaps) and np.min(gaps / spread) >= min_separation:
                score = float(np.min(gaps / spread))
                if best is None or score > best[0]:
                    best = (score, StratumProxy(ck, nk, levels, means, sds, counts))
    return best[1] if best else None


def decompose_arms(
    arms: Sequence[Arm],
    arm_records: Sequence[Sequence[Dict[str, Any]]],
    proxy: StratumProxy,
    n_iter: int = 200,
    smooth: float = 1.0,
    pi_prior_strength: float = 40.0,
) -> Tuple[List[Arm], np.ndarray]:
    """
    EM over the shared stratum-assignment mix pi and per-arm per-stratum
    survival, given survivors' (soft) strata and censored deaths.

    Returns (pseudo_arms, pi): one pseudo-arm per (arm, stratum) with
    fractional survived/deployed counts and environment[cat_key] = level, plus
    the estimated assignment mix.

    Identifiability note: from survivor-side data alone, (pi, p) is a
    one-parameter family for any single design (n*pi_z*p_z is all the data
    constrains). pi is anchored by a uniform-centred Dirichlet prior
    (pi_prior_strength pseudo-drones) - the causal reading is that stratum
    assignment is the environment's own randomisation, not a function of the
    design; arms with different designs then sharpen pi through sharing.
    """
    K = len(proxy.levels)
    A = len(arms)
    n = np.array([a.deployed for a in arms], dtype=float)
    # per-arm per-record proxy likelihoods f_z(x) (rows sum arbitrary); records
    # without the proxy get a flat likelihood
    F: List[np.ndarray] = []
    n_unproxied: List[int] = []
    for arm, recs in zip(arms, arm_records):
        rows, missing = [], 0
        for r in recs:
            env = r.get("environment") or {}
            v = env.get(proxy.proxy_key)
            if isinstance(v, (int, float)):
                rows.append(proxy.infer(float(v)))
            else:
                missing += 1
        missing += max(arm.survived - len(recs), 0)
        F.append(np.vstack(rows) if rows else np.zeros((0, K)))
        n_unproxied.append(missing)

    pi = np.full(K, 1.0 / K)
    p = np.full((A, K), 0.5)

    def survivor_counts() -> np.ndarray:
        # correct survivor responsibilities: P(z|x, survived) ∝ pi_z p_z f_z(x)
        S = np.zeros((A, K))
        for i in range(A):
            w = pi[None, :] * p[i][None, :]
            if len(F[i]):
                r = F[i] * w
                r = r / np.clip(r.sum(axis=1, keepdims=True), 1e-12, None)
                S[i] = r.sum(axis=0)
            if n_unproxied[i] > 0:
                flat = w[0] / np.clip(w[0].sum(), 1e-12, None)
                S[i] += n_unproxied[i] * flat
        return S

    for _ in range(n_iter):
        S = survivor_counts()
        deaths = np.maximum(n - S.sum(axis=1), 0.0)
        # deaths' responsibilities: P(z | died) ∝ pi_z (1 - p_z)
        w = pi[None, :] * (1.0 - p)
        w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)
        D = deaths[:, None] * w
        N = S + D
        p_new = np.clip((S + smooth * pi[None, :]) / (N + smooth), 0.01, 0.99)
        alpha = pi_prior_strength / K
        pi_new = (N.sum(axis=0) + alpha) / np.clip(N.sum() + pi_prior_strength,
                                                   1e-12, None)
        delta = max(float(np.max(np.abs(pi_new - pi))),
                    float(np.max(np.abs(p_new - p))))
        pi, p = pi_new, p_new
        if delta < 1e-9:
            break
    S = survivor_counts()
    deaths = np.maximum(n - S.sum(axis=1), 0.0)

    N = S + deaths[:, None] * (pi[None, :] * (1.0 - p)
                               / np.clip((pi[None, :] * (1.0 - p)).sum(axis=1,
                                                                       keepdims=True),
                                         1e-12, None))
    pseudo: List[Arm] = []
    for i, arm in enumerate(arms):
        for z, level in enumerate(proxy.levels):
            if N[i, z] < 0.5:
                continue
            env = dict(arm.environment)
            env[proxy.cat_key] = level
            pseudo.append(Arm(
                label=f"{arm.label}:{level}",
                design=dict(arm.design),
                equipment=dict(arm.equipment),
                deployed=int(round(N[i, z])),
                survived=int(round(min(S[i, z], N[i, z]))),
                environment=env,
            ))
    return pseudo, pi
