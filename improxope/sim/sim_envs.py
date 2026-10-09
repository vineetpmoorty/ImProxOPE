'''
Copied from ShadOPE (Wei, Qu, Miao, ICML 2026), https://github.com/NAIVlab/ShadOPE @ 4231ba5, file src/envs/sim_envs.py.
Changes from the original: import paths; removed fallback imports; optional reward_latent_sd,
latent_read_sd and mnar_curve (EnvConfig, all 0 by default; with reward_latent_sd > 0 the step also
returns the shadow reading info['w']): with 0, the random draws and results are unchanged.
No license is published for ShadOPE; do not redistribute without the authors' permission.

Gymnasium environment for a 2D-state MDP with MNAR (censored) rewards.

Observation
-----------

obs[t] = [s1_t, s2_t, o_{t-1}], where o_{t-1} ∈ {0,1} is the previous
missingness indicator. This implements the "extended MDP" idea so that
target policies π(a | s, o_prev) can consume o_prev directly.

Action space
------------
gymnasium.spaces.Discrete(2), with index-to-signed mapping: 0 ↦ -1, 1 ↦ +1.
For convenience, the environment also accepts raw actions in {-1, +1}.

Reward
------
The Gym reward is the observed reward r_obs = O_t * R_t. The true uncensored
reward R_t is placed in info["r_true"].

Notes
-----
Transition:
    S_{t+1} = diag(0.9, 0.9) * S_t + 0.2 * A_t + ε_s,   ε_s ~ N(0, σ_s^2 I_2).
Reward:
    R_t = expit( (0.9 - 0.6 A_t, -0.7)^T S_t + (1.3, 2)^T S_{t+1} - 0.4 A_t ) + U_t,
        U_t ~ Uniform[-0.1, 0.1].
MNAR mechanism:
    O_t ~ Bernoulli( sigmoid( 1 - 0.1 A_t + 0.2 * [1,-2]^T S_t + 2.5 R_t ) ).
'''

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces




# --- import configs and utils ------------------------------------
from improxope.sim.configs import EnvConfig
from improxope.sim.utils import NumpyRNG, sigmoid



class MNARMDP(gym.Env):
    '''
    A 2D-state MDP with MNAR rewards and {−1,+1} actions.

    Parameters
    ----------
    config : EnvConfig
        Dataclass holding horizon/noise/discount/initialization/seed.

    Attributes
    ----------
    metadata : dict
        Declares supported render modes (here only "human").
    observation_space : gymnasium.spaces.Box
        3-dimensional Box for (s1, s2, o_prev), dtype=float32.
    action_space : gymnasium.spaces.Discrete
        Discrete(2) with built-in mapping {0,1}→{-1,+1}.
    '''

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: EnvConfig, render_mode: Optional[str] = None) -> None:
        super().__init__()
        self.cfg = config
        self.render_mode = render_mode

        # Expose gamma so OPE modules can read env.gamma if needed.
        self.gamma: float = float(self.cfg.gamma)
        
        # Build spaces
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, 0.0], dtype=np.float32),
            high=np.array([np.inf, np.inf, 1.0], dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)

        # Internal state (initialized in reset)
        self._t: int = 0
        self._s: np.ndarray = np.zeros(2, dtype=np.float32)  # current S_t
        self._o_prev: int = 0                                # O_{t-1}
        self._rng = self._make_rng(self.cfg.seed)



    # Gymnasium API
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        '''
        Reset the episode; return (initial_obs, info).

        Returns
        -------
        obs : np.ndarray, shape (3,), dtype float32
            Concatenation of (s1, s2, o_prev) with o_prev = 0 by convention.
        info : dict
            Contains {"t": 1, "s": S_1 copy, "o_prev": 0}.
        '''
        # Optional per-episode reseeding
        if seed is not None:
            self._rng = self._make_rng(seed)

        self._t = 1
        self._s = self._sample_initial_state()
        self._o_prev = 0  # define O_0 := 0

        obs = self._pack_obs(self._s, self._o_prev)
        info = {"t": self._t, "s": self._s.copy(), "o_prev": self._o_prev}
        return obs, info



    def step(
        self, action: int | float | np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        '''
        Simulate one step; return (obs_next, reward_obs, terminated, truncated, info).
        '''
        a = self._to_action_scalar(action)  # a ∈ {−1,+1}

        # 1. Transition kernel
        eps_s = self._rng.normal(loc=0.0, scale=self.cfg.sigma_s, size=2)
        s_next = np.array([0.9, 0.9]) * self._s + 0.2 * a + eps_s

        # 2. True reward: R_t (depends on reward_type)
        r_true = self._compute_reward(self._s, a, s_next)
        w = 0.0
        if self.cfg.reward_latent_sd > 0.0:     # component of R_t that S_{t+1} does not reveal
            xi = float(self._rng.normal(loc=0.0, scale=1.0, size=None))
            r_true += self.cfg.reward_latent_sd * xi
            # recorded reading of xi (extra shadow input); eta is drawn even when tau = 0 so that
            # all tau levels share the same xi draws
            w = xi + self.cfg.latent_read_sd * float(self._rng.normal(loc=0.0, scale=1.0, size=None))

        # 3. MNAR missingness
        linear_s = np.array([1.0, -2.0]) @ self._s
        logit_o = self.cfg.mnar_c0 - 0.1 * a + 0.2 * linear_s + 2.5 * r_true
        if self.cfg.mnar_curve != 0.0:          # recording not logistic-linear in R_t
            logit_o += self.cfg.mnar_curve * (r_true - 0.5) ** 2
        p_o = float(sigmoid(logit_o))
        o_t = int(self._rng.bernoulli(p_o))

        # Observed reward for Gym API
        r_obs = float(o_t * r_true)

        # Prepare outputs
        obs_next = self._pack_obs(s_next, o_t)
        terminated = self._t >= self.cfg.horizon
        truncated = False

        info = {
            "t": self._t,
            "s": self._s.copy(),
            "s_next": s_next.copy(),
            "a": int(a),
            "o_t": o_t,
            "o_prob": p_o,
            "r_true": r_true,
            "r_obs": r_obs,
            "w": w,
        }

        # Update internal state
        self._s = s_next
        self._o_prev = o_t
        self._t += 1

        return obs_next, r_obs, terminated, truncated, info



    def render(self) -> None:
        if self.render_mode == "human":
            print(f"[t={self._t}] s={self._s}, o_prev={self._o_prev}")

    def close(self) -> None:
        return



    # Helpers
    def _make_rng(self, seed: Optional[int]):
        '''
        Create a RNG object with a Bernoulli method.
        '''
        return NumpyRNG(seed)

    def _sample_initial_state(self) -> np.ndarray:
        '''
        Draw S_1 ~ N(init_mean, init_std^2 I).
        '''
        z = self._rng.normal(loc=0.0, scale=self.cfg.init_std, size=2)
        return self.cfg.init_mean + z

    def _compute_reward(self, s: np.ndarray, a: int, s_next: np.ndarray) -> float:
        '''
        Compute R_t given (S_t, A_t, S_{t+1}) based on cfg.reward_type.

        All variants depend on S_{t+1} to satisfy the shadow variable
        relevance condition (Assumption 4.2).
        '''
        rt = self.cfg.reward_type
        if rt == 'sigmoid':
            eps_r = float(self._rng.uniform(low=-0.1, high=0.1, size=None))
            w_s = np.array([0.9 - 0.6 * a, -0.7])
            w_sp = np.array([1.3, 2.0])
            return float(sigmoid(w_s @ s + w_sp @ s_next - 0.4 * a) + eps_r)
        elif rt == 'linear':
            eps_r = float(self._rng.normal(loc=0.0, scale=0.1, size=None))
            val = (np.array([0.5, -0.3]) @ s
                   + np.array([0.8, 0.6]) @ s_next
                   - 0.3 * a + eps_r)
            return float(np.clip(val, -1.0, 1.0))
        else:
            raise ValueError(f"Unknown reward_type: {rt}")

    @staticmethod
    def _pack_obs(s: np.ndarray, o_prev: int) -> np.ndarray:
        '''
        Pack (s1, s2, o_prev) into a float32 vector.
        '''
        return np.array([s[0], s[1], float(o_prev)], dtype=np.float32)

    @staticmethod
    def _to_action_scalar(action: int | float | np.ndarray) -> int:
        '''
        Map inputs {0,1} or {-1,+1} or scalar arrays to {-1,+1}.
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