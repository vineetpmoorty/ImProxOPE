from __future__ import annotations

import time
import zlib
from functools import lru_cache
from typing import Dict, Optional

import numpy as np
import torch

from improxope.fqe import (ImputeFQE, KernelRidgeCV, NaiveFQE, ProxFQE, SCOPE, WeightedFQE, _policy_probs_target,
                         _stack_cols, _to32, group_by_t)
from improxope.learners import RecordingInverse
from improxope.sim.behavior_policy import BehaviorPolicy
from improxope.sim.configs import EnvConfig
from improxope.sim.generate_data import collect_episodes
from improxope.sim.sim_envs import MNARMDP
from improxope.sim.simulation import compute_true_value_via_target_rollout
from improxope.sim.target_policy import TargetPolicy

GAMMA = 1.0
ACTIONS = (-1, +1)
LAM_GRID = np.logspace(-7, 1, 30).tolist()
METHODS = ('naive', 'prox', 'ipw', 'impute', 'scope', 'ours')
# ShadOPE's intercepts for ~20/40/60/80% missing (run_simulation.sh)
SHADOPE_C0 = {0.2: 0.3, 0.4: -0.7, 0.6: -1.5, 0.8: -2.8}


def bridge_kwargs(device):
    return dict(delta_scale=5.0, delta_exp=0.4, gamma_f='auto', gamma_hs='auto', n_gamma_hs=30, cv=5, device=device)


def krr_kwargs(device):
    return dict(lam_grid=LAM_GRID, folds=5, device=device)


def env_config(T, seed, c0, beta=0.0, tau=0.0, kappa=0.0) -> EnvConfig:
    return EnvConfig(horizon=T, seed=seed, gamma=GAMMA, mnar_c0=c0, reward_type='sigmoid',
                     reward_latent_sd=float(beta), latent_read_sd=float(tau), mnar_curve=float(kappa))


@lru_cache(maxsize=4)
def dataset(T: int, n: int, seed: int, c0: float, beta: float = 0.0, tau: float = 0.0,
            kappa: float = 0.0) -> Dict[str, np.ndarray]:
    ds = collect_episodes(MNARMDP(env_config(T, seed, c0, beta, tau, kappa)), BehaviorPolicy(seed=seed + 11),
                          n_episodes=n, seed=seed)
    return ds[0] if isinstance(ds, tuple) else ds


def true_value(T: int, seed: int, c0: float, beta: float = 0.0, tau: float = 0.0, kappa: float = 0.0,
               n_eval: int = 5000) -> float:
    return compute_true_value_via_target_rollout(T, GAMMA, seed, n_eval=n_eval, mnar_c0=c0, reward_type='sigmoid',
                                                 reward_latent_sd=float(beta), latent_read_sd=float(tau),
                                                 mnar_curve=float(kappa))


class CorrectedProxFQE(ProxFQE):
    """ShadOPE's ProxFQE with the corrected reward U = b + M h (R - b); b and h cross-fitted."""

    def __init__(self, n_folds: int = 3, recording: Optional[Dict] = None, fold_seed: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.n_folds, self.fold_seed = int(n_folds), int(fold_seed)
        self.recording = {**dict(cap=20.0, max_iter=500, ridge=1e-6), **(recording or {})}

    def _v(self, Q, S, Ominus, target_policy):
        probs = _policy_probs_target(target_policy, S, Ominus)
        out = torch.zeros(S.shape[0], dtype=torch.float32, device=self.device)
        for j, a in enumerate(self.action_list):
            a_col = torch.full((S.shape[0], 1), float(a), dtype=torch.float32, device=self.device)
            out += probs[:, j] * Q.predict(_stack_cols(S, a_col)).reshape(-1)
        return out

    def fit(self, dataset: Dict[str, np.ndarray], target_policy) -> 'CorrectedProxFQE':
        by_t = group_by_t(dataset)
        T = max(by_t.keys())
        t_idx, ep = np.asarray(dataset['t']), np.asarray(dataset['ep'])
        episodes = np.unique(ep)
        fold_of = dict(zip(np.random.RandomState(self.fold_seed).permutation(episodes),
                           np.arange(len(episodes)) % self.n_folds))
        self.Q = {'corrected': {}, 'bridge': {}}
        self.info_ = {'mean_Mh': [], 'cap_fraction': []}
        next_Q = {'corrected': None, 'bridge': None}
        for t in reversed(range(1, T + 1)):
            pack = by_t[t]
            S = _to32(pack['S']).to(self.device)
            A = _to32(pack['A']).to(self.device)
            R = _to32(pack['R']).to(self.device).reshape(-1)
            O = _to32(pack['O']).to(self.device).reshape(-1)
            Sp = _to32(pack['Sp']).to(self.device)
            W = _to32(pack['W']).to(self.device)          # shadow: S_{t+1} (and the reading w, if any)
            fold = torch.as_tensor(np.array([fold_of[e] for e in ep[t_idx == t]]), device=self.device)
            XH_all = _stack_cols(W, S, A)
            b = torch.zeros_like(R)
            h = torch.ones_like(R)
            S_np, A_np, R_np, O_np, W_np = (x.cpu().numpy().astype(np.float64) for x in (S, A, R, O, W))
            a_idx = ((A_np.reshape(-1) + 1) / 2).astype(np.int64)        # -1/+1 -> 0/1
            for k in range(self.n_folds):
                te = fold == k
                tr = ~te if self.n_folds > 1 else te
                obs = tr & (O == 1.0)
                if obs.sum() <= 3:
                    raise RuntimeError(f'[CorrectedProxFQE] too few O=1 samples at t={t}, fold {k}')
                bridge = self.bridge_cfg.make_estimator().fit(
                    XH_all[obs], R[obs].reshape(-1, 1), _stack_cols(R[obs].reshape(-1, 1), S[obs], A[obs]))
                b[te] = bridge.predict(XH_all[te]).reshape(-1)
                tr_np, te_np = tr.cpu().numpy(), te.cpu().numpy()
                rec = RecordingInverse(n_actions=2, device=str(self.device), **self.recording).fit(
                    R_np[tr_np], S_np[tr_np], a_idx[tr_np], W_np[tr_np], O_np[tr_np])
                h[te] = torch.as_tensor(rec.predict(R_np[te_np], S_np[te_np], a_idx[te_np]),
                                        dtype=torch.float32, device=self.device)
                self.info_['mean_Mh'].append(rec.info_['mean_Mh'])
                self.info_['cap_fraction'].append(rec.info_['cap_fraction'])
            U = {'corrected': b + O * h * (R - b), 'bridge': b + O * (R - b)}
            On = _to32(pack['Onext']).to(self.device).reshape(-1)
            for v in U:
                y = U[v] if next_Q[v] is None else U[v] + self.gamma * self._v(next_Q[v], Sp, On, target_policy)
                Q_t = KernelRidgeCV(**self.krr_kwargs).fit(_stack_cols(S, A), y.reshape(-1, 1))
                self.Q[v][t] = Q_t
                next_Q[v] = Q_t
        self._target = target_policy
        return self

    def values(self, S1: np.ndarray, O0: np.ndarray) -> Dict[str, float]:
        S1_t = torch.as_tensor(S1, dtype=torch.float32, device=self.device)
        O0_t = torch.as_tensor(O0, dtype=torch.float32, device=self.device).reshape(-1)
        return {v: float(self._v(self.Q[v][1], S1_t, O0_t, self._target).mean().item()) for v in self.Q}


def seed_everything(*parts) -> None:
    s = zlib.crc32(repr(parts).encode()) % (2 ** 31)
    torch.manual_seed(s)
    np.random.seed(s)


def run_method(method: str, T: int, n: int, seed: int, c0: float, beta: float, tau: float, kappa: float,
               device: str, n_folds: int = 3, recording: Optional[Dict] = None) -> Dict:
    """One estimator on one dataset, with ShadOPE's eval_grid.py settings. Returns {name: value}.
    CPU only: the kernel estimators solve near-singular float32 systems (ridge down to 1e-7), and on
    GPUs these solves are sometimes badly wrong (e.g. ProxFQE error 1.87 on GPU vs 0.056 on CPU for the
    same data; ShadOPE's own results were computed on CPU and match ours on CPU)."""
    if str(device) != 'cpu':
        raise ValueError(f'sim_ope.run_method must run on CPU (got {device}); see the docstring')
    t0 = time.time()
    seed_everything('sim', method, T, n, seed, c0, beta, tau, kappa)
    ds = dataset(T, n, seed, c0, beta, tau, kappa)
    tvec = ds['t'].astype(int)
    S1 = ds['obs'][tvec == 1, :2]
    O0 = np.zeros((S1.shape[0],), dtype=np.float32)
    pi = TargetPolicy()
    if method == 'naive':
        out = {'naive': NaiveFQE(ACTIONS, GAMMA, krr_kwargs(device), device).fit(ds, pi).value(S1, O0)}
    elif method == 'prox':
        out = {'prox': ProxFQE(ACTIONS, GAMMA, bridge_kwargs(device), krr_kwargs(device), device).fit(ds, pi).value(S1, O0)}
    elif method == 'ipw':
        est = WeightedFQE(ACTIONS, GAMMA, bridge_kwargs(device), krr_kwargs(device), logit_l2=1e-3,
                          logit_max_iter=200, pmin=1e-2, w_cap=50.0, device=device)
        out = {'ipw': est.fit(ds, pi).value(S1, O0)}
    elif method == 'impute':
        out = {'impute': ImputeFQE(ACTIONS, GAMMA, krr_kwargs(device), device).fit(ds, pi).value(S1, O0)}
    elif method == 'scope':
        est = SCOPE(gamma=GAMMA, frac_shape=0.3, krr_kwargs=krr_kwargs(device), w_cap=50.0, device=device)
        out = {'scope': est.fit(ds, pi, BehaviorPolicy(seed=seed + 11)).value()}
    elif method == 'ours':
        est = CorrectedProxFQE(n_folds=n_folds, recording=recording, fold_seed=seed, action_list=ACTIONS, gamma=GAMMA,
                               bridge_cv_kwargs=bridge_kwargs(device), krr_kwargs=krr_kwargs(device),
                               device=device).fit(ds, pi)
        v = est.values(S1, O0)
        out = {'ours': v['corrected'], 'ours_bridge': v['bridge'],
               'mean_Mh': float(np.mean(est.info_['mean_Mh'])), 'cap_fraction': float(np.mean(est.info_['cap_fraction']))}
    else:
        raise ValueError(method)
    out.update(missing=float(1.0 - np.mean(ds['o'])), seconds=time.time() - t0)
    return out
