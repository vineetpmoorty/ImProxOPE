'''
Copied from ShadOPE (Wei, Qu, Miao, ICML 2026), https://github.com/NAIVlab/ShadOPE @ 4231ba5, file src/policies/behavior_policy.py.
Changes from the original: import path; removed fallback import.
No license is published for ShadOPE; do not redistribute without the authors' permission.

Behavior policy π_b for the simulated MDP.

Form
----
P(A_t = +1 | S_t = s) = sigmoid( 0.3 + [0.8, -0.3]^T s ),
where s ∈ R^2 and A_t ∈ {-1, +1}.

Notes
-----
- This module provides both probability evaluation and action sampling.
- Use `sample` for stochastic action during data generation.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from improxope.sim.utils import NumpyRNG, sigmoid


@dataclass(frozen=True)
class BehaviorPolicy:
    '''
    Logistic behavior policy over the current state s ∈ R^2.

    Parameters
    ----------
    bias : float, default=0.3
        Intercept term in the logit.
    w : tuple[float, float], default=(0.8, -0.3)
        Linear coefficients for (s1, s2).
    seed : Optional[int], default=None
        Seed for the internal RNG used by `sample`.

    Methods
    -------
    prob_a_plus(s) -> float
        Return P(A=+1 | s).
    prob_a_plus_from_obs(obs) -> float
        `obs` is the 3D vector (s1, s2, o_prev) from the env; o_prev is ignored.
    sample(s, rng=None) -> int
        Sample an action in {-1, +1}.
    sample_from_obs(obs, rng=None) -> int
        Sample directly from the env observation (s1, s2, o_prev).
    greedy(s) -> int
        Deterministic action: +1 if prob ≥ 0.5, else -1.
    sample_with_prob(s, rng=None) -> tuple[int, float]
        Convenience: return (action, prob_plus).
    '''

    bias: float = 0.3
    w: Tuple[float, float] = (0.8, -0.3)
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        # Internal RNG for sampling; callers may override per-call via `rng=...`.
        object.__setattr__(self, "_rng", NumpyRNG(self.seed))

    # ------------------------- public API ------------------------- #
    def prob_a_plus(self, s: np.ndarray) -> float:
        '''
        Return P(A=+1 | s).
        
        Parameters
        ----------
        s : np.ndarray of shape (2,)
            Current state vector (s1, s2).

        Returns
        -------
        float
            Probability in (0, 1).
        '''
        s = np.asarray(s, dtype=float).reshape(2)
        logit = self.bias + self.w[0] * s[0] + self.w[1] * s[1]
        p = float(sigmoid(logit))
        return float(np.clip(p, 1e-8, 1 - 1e-8))

    def prob_a_plus_from_obs(self, obs: np.ndarray) -> float:
        '''
        Return P(A=+1 | obs=(s1, s2, o_prev)); o_prev is ignored for behavior policy.
        '''
        obs = np.asarray(obs, dtype=float).reshape(3)
        return self.prob_a_plus(obs[:2])

    def sample(self, s: np.ndarray, *, rng: Optional[NumpyRNG] = None) -> int:
        '''
        Sample a stochastic action in {-1, +1}.
        
        Parameters
        ----------
        s : np.ndarray, shape (2,)
            Current state.
        rng : Optional[NumpyRNG], default=None
            RNG to use for this call. If None, the internal RNG is used.

        Returns
        -------
        int
            +1 with probability prob_a_plus(s); -1 otherwise.
        '''
        p = self.prob_a_plus(s)
        g = self._rng if rng is None else rng
        return +1 if g.bernoulli(p) == 1 else -1

    def sample_from_obs(self, obs: np.ndarray, *, rng: Optional[NumpyRNG] = None) -> int:
        '''
        Sample an action directly from an env observation (s1, s2, o_prev).
        '''
        return self.sample(np.asarray(obs, float).reshape(3)[:2], rng=rng)

    def greedy(self, s: np.ndarray) -> int:
        '''Return a deterministic action using threshold 0.5.'''
        return +1 if self.prob_a_plus(s) >= 0.5 else -1

    def sample_with_prob(self, s: np.ndarray, *, rng: Optional[NumpyRNG] = None) -> Tuple[int, float]:
        '''
        Return (action, prob_plus) for convenience.
        '''
        p = self.prob_a_plus(s)
        a = self.sample(s, rng=rng)
        return a, p
    
    def act(self, obs: np.ndarray, *, rng: Optional[NumpyRNG] = None) -> int:
        '''
        Return one action in {-1,+1} given obs=(s1, s2, o_prev).
        Prefer `sample_from_obs`; if unavailable, fall back to `sample(s)`.
        '''
        try:
            return self.sample_from_obs(obs, rng=rng)
        except AttributeError:
            s = np.asarray(obs, dtype=float).reshape(3)[:2]
            return self.sample(s, rng=rng)

