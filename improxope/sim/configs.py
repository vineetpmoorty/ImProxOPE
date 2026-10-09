"""Adapted from ShadOPE (Wei, Qu, Miao, ICML 2026), https://github.com/NAIVlab/ShadOPE @ 4231ba5, file src/configs.py."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class EnvConfig:
    '''
    Configuration for the MNARMDP environment.

    Parameters
    ----------
    horizon : int, default=20
        Episode length T (must be ≥ 1).
    sigma_s : float, default=0.1
        Std dev for Gaussian transition noise on S_{t+1}.
    sigma_r : float, default=0.1
        Std dev for Gaussian reward noise on R_t.
    init_mean : np.ndarray, shape (2,), default=zeros(2)
        Mean of the initial state S_1.
    init_std : float, default=1.0
        Std dev (isotropic) for the initial state S_1.
    seed : Optional[int], default=42
        RNG seed for reproducibility.
    gamma : float, default=1.0
        Discount factor for future rewards.
    mnar_c0 : float, default=1.0
        Intercept in the MNAR logit: logit(O_t) = c0 - 0.1*A + 0.2*[1,-2]^T S + 2.5*R.
        Lower values increase the missing rate.
    reward_type : str, default='sigmoid'
        Reward generation mechanism: 'sigmoid', 'linear', or 'interaction'.

    Notes
    -----
    - The dataclass is frozen to avoid accidental mutation during experiments.
    - `init_mean` uses a default_factory to avoid shared mutable defaults.
    '''

    horizon: int = 20
    sigma_s: float = 0.1
    sigma_r: float = 0.1
    init_mean: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    init_std: float = 1.0
    seed: Optional[int] = 42
    gamma: float = 1.0
    mnar_c0: float = 1.0
    reward_type: str = 'sigmoid'
    reward_latent_sd: float = 0.0   # beta: R_t += beta * xi_t, xi_t ~ N(0,1), not revealed by S_{t+1}
    latent_read_sd: float = 0.0     # tau: shadow reading W_t = xi_t + tau * eta_t, eta_t ~ N(0,1)
    mnar_curve: float = 0.0         # kappa: MNAR logit += kappa * (R_t - 0.5)^2

    def __post_init__(self) -> None:
        # Basic validation on shapes and ranges.
        if self.init_mean.shape != (2,):
            raise ValueError(f"init_mean must have shape (2,), got {self.init_mean.shape}.")
        if not (self.horizon >= 1):
            raise ValueError("horizon must be ≥ 1.")
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError("gamma must be in [0,1].")
        if not (self.sigma_s >= 0.0 and self.sigma_r >= 0.0):
            raise ValueError("sigma_s and sigma_r must be non-negative.")
        if self.reward_latent_sd < 0.0 or self.latent_read_sd < 0.0:
            raise ValueError("reward_latent_sd and latent_read_sd must be non-negative.")
        if self.reward_type not in ('sigmoid', 'linear', 'interaction'):
            raise ValueError(f"reward_type must be 'sigmoid', 'linear', or 'interaction', got '{self.reward_type}'.")