"""Patient-by-time arrays shared by every estimator in this package.

A Panel holds N trajectories of a fixed horizon T. The decision state is X_t = (S_t, O_{t-1});
Z_t = S_{t+1} is the next state (at t = T the terminal state). The shadow variable for R_t is Z_t,
or W_t when set: a reduced proxy (some columns of Z_t) used only by the bridge and the recording
model, while Q-function targets still use the full next state Z_t. Target-policy probabilities are stored for X_t and X_{t+1} so that deterministic and
stochastic targets are handled the same way.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Optional

import numpy as np


@dataclass
class Panel:
    ids: np.ndarray        # (N,)       trajectory identifiers
    S: np.ndarray          # (N, T, d)  state at t
    Z: np.ndarray          # (N, T, d)  next state S_{t+1}; terminal state at t = T
    o_prev: np.ndarray     # (N, T)     previous recording indicator O_{t-1}
    A: np.ndarray          # (N, T)     logged action index in 0..K-1
    M: np.ndarray          # (N, T)     recording indicator O_t
    R_obs: np.ndarray      # (N, T)     observed reward, 0 where M = 0
    pi: np.ndarray         # (N, T, K)  target probabilities at X_t
    pi_next: np.ndarray    # (N, T, K)  target probabilities at X_{t+1}; zeros at t = T
    R_true: Optional[np.ndarray] = None  # (N, T) evaluation only; never used for fitting
    W: Optional[np.ndarray] = None       # (N, T, d_w) shadow variable if not the full next state

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def shadow(self) -> np.ndarray:
        return self.Z if self.W is None else self.W

    @property
    def horizon(self) -> int:
        return self.S.shape[1]

    @property
    def n_actions(self) -> int:
        return self.pi.shape[2]

    def subset(self, rows: np.ndarray) -> "Panel":
        return replace(self, **{f.name: None if getattr(self, f.name) is None else getattr(self, f.name)[rows]
                                for f in fields(self)})

    def validate(self) -> None:
        N, T = self.A.shape
        assert self.S.shape[:2] == (N, T) and self.Z.shape == self.S.shape
        assert self.W is None or self.W.shape[:2] == (N, T)
        assert self.pi.shape == self.pi_next.shape and self.pi.shape[:2] == (N, T)
        assert np.allclose(self.pi.sum(-1), 1.0), "target probabilities must sum to 1"
        assert np.allclose(self.pi_next[:, :-1].sum(-1), 1.0) and np.all(self.pi_next[:, -1] == 0)
        assert set(np.unique(self.M)) <= {0, 1} and set(np.unique(self.o_prev)) <= {0, 1}
        assert np.all(self.R_obs[self.M == 0] == 0), "missing rewards must be stored as 0"
        assert self.A.min() >= 0 and self.A.max() < self.n_actions


def one_hot(index: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros(index.shape + (k,), dtype=np.float32)
    np.put_along_axis(out, index[..., None].astype(np.int64), 1.0, axis=-1)
    return out
