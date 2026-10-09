'''
Copied from ShadOPE (Wei, Qu, Miao, ICML 2026), https://github.com/NAIVlab/ShadOPE @ 4231ba5, file src/utils.py.
Changes from the original: none.
No license is published for ShadOPE; do not redistribute without the authors' permission.

General utilities for the simulation project.

It provides:
  - `sigmoid`: a numerically-stable logistic function.
  - `to_signed_action`: robust mapping from {0,1} or {-1,+1} encodings to {-1,+1}.
  - `NumpyRNG`: a thin RNG wrapper with a convenient `bernoulli(p)` method.

'''

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------
def sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    '''
    Numerically-stable logistic sigmoid.

    Parameters
    ----------
    x : float or ndarray
        Input value(s).

    Returns
    -------
    float or ndarray
        Elementwise 1 / (1 + exp(-x)) with overflow-safe branches.
    '''
    x_arr = np.asarray(x)
    # Two-branch implementation avoids overflow when |x| is large.
    pos = 1.0 / (1.0 + np.exp(-x_arr))
    neg = np.exp(x_arr) / (1.0 + np.exp(x_arr))
    out = np.where(x_arr >= 0, pos, neg)
    # Preserve scalar semantics for scalar-like input
    return float(out) if out.ndim == 0 else out


def to_signed_action(action: Union[int, float, np.ndarray]) -> int:
    '''
    Map an action encoding to {-1, +1}.

    Accepts:
      - integers/floats in {-1, +1}
      - integers in {0, 1} (Discrete(2)): 0→-1, 1→+1
      - scalar-shaped NumPy arrays containing one of the above

    Parameters
    ----------
    action : int | float | ndarray
        Raw action encoding.

    Returns
    -------
    int
        The signed action in {-1, +1}.

    Raises
    ------
    ValueError
        If the input cannot be interpreted as {-1,+1} or {0,1}.
    '''
    if isinstance(action, np.ndarray):
        if action.size != 1:
            raise ValueError("Action array must be scalar-like.")
        action = float(action.reshape(()))
    if action in (-1, 1):
        return int(action)
    if action in (0, 1):
        return -1 if int(action) == 0 else +1
    raise ValueError(f"Invalid action {action!r}; expected -1/+1 or 0/1.")


# ---------------------------------------------------------------------
# RNG helper
# ---------------------------------------------------------------------
class NumpyRNG:
    '''
    Tiny wrapper around NumPy's Generator with a `bernoulli` method.

    This class exposes only the minimal surface used by envs/policies:
      - normal(loc=..., scale=..., size=...)
      - random(size=None)   # Uniform(0,1)
      - bernoulli(p)        # return 1 w.p. p, else 0

    Parameters
    ----------
    seed : Optional[int], default=None
        Seed for the underlying generator.
    '''

    def __init__(self, seed: Optional[int] = None) -> None:
        self._gen = np.random.default_rng(seed)

    # --- core draws ---------------------------------------------------
    def normal(
        self,
        *,
        loc: float = 0.0,
        scale: float = 1.0,
        size: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> Union[float, np.ndarray]:
        return self._gen.normal(loc=loc, scale=scale, size=size)

    def random(self, size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[float, np.ndarray]:
        '''Draw Uniform(0,1) samples.'''
        return self._gen.random(size=size)
    
    def uniform(self, low=0.0, high=1.0, size=None):
        return self._gen.uniform(low, high, size)

    def bernoulli(self, p: float) -> int:
        '''
        Return 1 with probability p, else 0.

        Parameters
        ----------
        p : float
            Probability in [0,1].

        Returns
        -------
        int
            1 with prob. p; 0 otherwise.

        Raises
        ------
        ValueError
            If p is outside [0,1].
        '''
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Bernoulli prob must be in [0,1], got {p}.")
        return int(self._gen.random() < p)