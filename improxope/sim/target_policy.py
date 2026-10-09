"""Adapted from ShadOPE (Wei, Qu, Miao, ICML 2026), https://github.com/NAIVlab/ShadOPE @ 4231ba5, file src/policies/target_policy.py."""

from __future__ import annotations
import torch
from torch import Tensor
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from improxope.sim.utils import NumpyRNG, sigmoid


@dataclass(frozen=True)
class TargetPolicy:
    '''
    Logistic target policy depending on (s, o_prev).

    Parameters
    ----------
    d : tuple[float, float], default=(3.0, 0.9)
        Linear coefficients for (s1, s2).
    d_o : float, default=-2.4
        Coefficient for (2*o_prev - 1), mapping {0,1} -> {-1,+1}.
    bias : float, default=1.5
        Optional intercept term.
    seed : Optional[int], default=None
        Seed for the internal RNG used by `sample`.

    Methods
    -------
    prob_a_plus(s, o_prev) -> float
        Return P(A=+1 | s, o_prev).
    prob_a_plus_from_obs(obs) -> float
        `obs` is the 3D vector (s1, s2, o_prev) from the env.
    sample(s, o_prev, rng=None) -> int
        Stochastic action in {-1, +1}.
    sample_from_obs(obs, rng=None) -> int
        Stochastic action directly from env observation.
    greedy(s, o_prev) -> int
        Deterministic action (threshold 0.5).
    sample_with_prob(s, o_prev, rng=None) -> tuple[int, float]
        Convenience wrapper returning both action and probability.
    '''

    d: Tuple[float, float] = (3.0, 0.9)
    d_o: float = -2.4
    bias: float = 1.5
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "_rng", NumpyRNG(self.seed))

    # ------------------------- public API ------------------------- #
    def prob_a_plus(self, s: np.ndarray, o_prev: int) -> float:
        '''
        Return P(A=+1 | s, o_prev).
        
        Parameters
        ----------
        s : np.ndarray of shape (2,)
            Current state vector (s1, s2).
        o_prev : int in {0,1}
            Previous missingness indicator from the env.

        Returns
        -------
        float
            Probability in (0, 1).
        '''
        s = np.asarray(s, dtype=float).reshape(2)
        o_pm = 2 * int(o_prev) - 1  # {0,1} -> {-1,+1}
        logit = self.bias + self.d[0] * s[0] + self.d[1] * s[1] + self.d_o * o_pm
        p = float(sigmoid(logit))
        return float(np.clip(p, 1e-8, 1 - 1e-8))
    
    def prob_a_plus_batch(self, S: Tensor, Ominus: Tensor) -> Tensor:
        """
        Vectorized P(A=+1 | S, O_prev) for a batch.

        Parameters
        ----------
        S : (n, 2) torch.Tensor
            States (s1, s2) on any device (cpu/cuda).
        Ominus : (n,) or (n,1) torch.Tensor
            Previous missingness indicator {0,1} on any device.

        Returns
        -------
        probs_plus : (n,) torch.Tensor
            Probabilities P(A=+1 | S, O_prev) on the same device as S.
        """
        device = S.device
        S = S.to(dtype=torch.float32, device=device)
        Ominus = Ominus.to(dtype=torch.float32, device=device).reshape(-1)

        s1 = S[:, 0]
        s2 = S[:, 1]
        o_pm = 2.0 * Ominus - 1.0  # {0,1} -> {-1,+1}

        d1, d2 = self.d
        d_o = self.d_o
        bias = self.bias

        logits = bias + d1 * s1 + d2 * s2 + d_o * o_pm  # same as scalar version
        p = torch.sigmoid(logits)
        # Match scalar version's clipping
        p = torch.clamp(p, 1e-8, 1.0 - 1e-8)
        return p


    def prob_a_plus_from_obs(self, obs: np.ndarray) -> float:
        '''
        Return P(A=+1 | obs=(s1, s2, o_prev)).
        '''
        obs = np.asarray(obs, dtype=float).reshape(3)
        s = obs[:2]
        o_prev = int(round(obs[2]))
        return self.prob_a_plus(s, o_prev)

    def sample(self, s: np.ndarray, o_prev: int, *, rng: Optional[NumpyRNG] = None) -> int:
        '''
        Sample a stochastic action in {-1, +1} given (s, o_prev).
        '''
        p = self.prob_a_plus(s, o_prev)
        g = self._rng if rng is None else rng
        return +1 if g.bernoulli(p) == 1 else -1

    def sample_from_obs(self, obs: np.ndarray, *, rng: Optional[NumpyRNG] = None) -> int:
        '''
        Sample an action directly from an env observation (s1, s2, o_prev).
        '''
        obs = np.asarray(obs, dtype=float).reshape(3)
        return self.sample(obs[:2], int(round(obs[2])), rng=rng)

    def greedy(self, s: np.ndarray, o_prev: int) -> int:
        '''
        Return a deterministic action using threshold 0.5.
        '''
        return +1 if self.prob_a_plus(s, o_prev) >= 0.5 else -1

    def sample_with_prob(
        self, s: np.ndarray, o_prev: int, *, rng: Optional[NumpyRNG] = None
    ) -> Tuple[int, float]:
        '''
        Return (action, prob_plus) for convenience.
        '''
        p = self.prob_a_plus(s, o_prev)
        a = self.sample(s, o_prev, rng=rng)
        return a, p
    
    def act(self, obs: np.ndarray, *, rng: Optional[NumpyRNG] = None) -> int:
        '''
        Return one action in {-1,+1} given obs=(s1, s2, o_prev).
        Prefer `sample_from_obs`; if unavailable, fall back to `sample(s, o_prev)`.
        '''
        try:
            return self.sample_from_obs(obs, rng=rng)
        except AttributeError:
            arr = np.asarray(obs, dtype=float).reshape(3)
            s = arr[:2]
            o_prev = int(round(arr[2]))
            return self.sample(s, o_prev, rng=rng)