"""Task functions for the corrected estimator and the ShadOPE baselines.

Every task takes a `source` dict identifying one dataset (MIMIC: miss rate, seed; synthetic: an
.npz file) and loads it through a per-process cache, so only small arguments and prediction
arrays cross process boundaries.

The fit stays (ShadOPE's 60% split) are used according to a cross-fitting design:
- a design is a list of chains; each chain fits Q (and the ratios) on its `q_rows` and produces
  per-test-stay evaluation terms; chains are averaged;
- a chain's corrected scores on `q_rows` come from units; each unit fits the bridge on
  `bridge_rows` and the recording model on `recording_rows`, and scores `score_rows` (the unit
  score rows partition `q_rows`). Test-stay scores are averaged over a chain's units.

Designs (`design['name']`, with `n_folds` folds of the fit stays):
- rotate: chain r: Q on fold r+2; bridge on fold r; recording on fold r+1 (fold r if not separate)
- full:   one chain; bridge, recording and Q all on every fit stay (ShadOPE's data use)
- oof:    one chain; Q on every fit stay; unit k scores fold k with bridge and recording fitted on
          the other folds (separate: bridge on fold k+1, recording on fold k+2)

Target policies: `target=None` is the dataset's own (ShadOPE's DQN) target. Otherwise `target`
(e.g. {'eps': 0.5, 'tau': 0.05}, see improxope.data.target_policy) replaces the target probabilities
for the tasks that depend on the target (ratios, Q-function, baselines); the bridge and recording
model do not, so their fits are shared across targets.

Evaluation on the held-out test stays:
  psi_i = V_1(X_i1) + sum_t w_t(X_it, A_it) * (U_it + V_{t+1}(X_i,t+1) - Q_t(X_it, A_it)),
  U = b + M h (R - b);  "corrected FQE" reports V_1 alone (no ratio layer).
"""
from __future__ import annotations

import time
import zlib
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import torch

from improxope import data as D
from improxope.learners import RatioModel, RecordingInverse
from improxope.nn_bridge import NNBridge
from improxope.nn_fqe import NNQFunction
from improxope.panel import Panel


# --------------------------------------------------------------------------- data access

class Dataset:
    """A loaded source: full panel plus fit and test rows (indices into the panel)."""

    def __init__(self, panel: Panel, fit_ids, test_ids, seed: int, extra=None):
        self.panel = panel
        pos = {sid: i for i, sid in enumerate(panel.ids)}
        self.fit_rows = np.array([pos[s] for s in fit_ids])
        self.test_rows = np.array([pos[s] for s in test_ids])
        self.fit_ids, self.test_ids, self.seed = fit_ids, test_ids, seed
        self.extra = extra or {}

    def folds(self, n_folds: int) -> List[np.ndarray]:
        pos = {sid: i for i, sid in enumerate(self.panel.ids)}
        return [np.array([pos[s] for s in part]) for part in D.crossfit_parts(self.fit_ids, self.seed, n_folds)]


def _key(source: Dict) -> tuple:
    return tuple(sorted(source.items()))


@lru_cache(maxsize=2)
def _load(key: tuple) -> Dataset:
    s = dict(key)
    if s['kind'] == 'mimic':
        df, term = D.load_mimic(s['data_dir'], s['miss_rate'], s['seed'])
        panel = D.mimic_panel(df, term, terminal_fix=s['terminal_fix'], shadow=s.get('shadow', 'full'))
        fit_ids, test_ids = D.shadope_split(df, s['seed'], s['train_frac'])
        return Dataset(panel, fit_ids, test_ids, s['seed'], {'df': df, 'term': term})
    if s['kind'] == 'sim':
        from improxope.sim.target_policy import TargetPolicy
        ds = dict(np.load(s['path']))
        panel = D.sim_panel(ds, TargetPolicy())
        ids = np.sort(panel.ids)
        rng = np.random.RandomState(s['seed'])
        rng.shuffle(ids)
        n_fit = int(len(ids) * s['train_frac'])
        return Dataset(panel, ids[:n_fit], ids[n_fit:], s['seed'])
    raise ValueError(f"unknown source kind {s['kind']}")


def load(source: Dict) -> Dataset:
    return _load(_key(source))


@lru_cache(maxsize=2)
def _target_panel(key: tuple, target_key: tuple) -> Panel:
    ds, s, target = _load(key), dict(key), dict(target_key)
    T, K = ds.panel.horizon, ds.panel.n_actions
    pi = D.target_policy(ds.extra['df'], target, s['data_dir']).reshape(ds.panel.n, T, K)
    pi_next = np.zeros_like(pi)
    pi_next[:, :-1] = pi[:, 1:]
    return replace(ds.panel, pi=pi, pi_next=pi_next)


def target_panel(source: Dict, target: Optional[Dict]) -> Panel:
    """The dataset's panel with target probabilities for `target` (None: the dataset's own)."""
    if target is None:
        return load(source).panel
    return _target_panel(_key(source), tuple(sorted(target.items())))


# --------------------------------------------------------------------------- cross-fitting plans

@dataclass
class Unit:
    bridge_rows: np.ndarray
    recording_rows: np.ndarray
    score_rows: np.ndarray      # subset of the chain's q_rows


@dataclass
class Chain:
    q_rows: np.ndarray
    units: List[Unit]


def plan(ds: Dataset, design: Dict) -> List[Chain]:
    name, K, separate = design['name'], design.get('n_folds', 3), design.get('separate', True)
    if name == 'full':
        fit = ds.fit_rows
        return [Chain(fit, [Unit(fit, fit, fit)])]
    folds = ds.folds(K)
    if name == 'rotate':
        chains = []
        for r in range(design.get('rotations', K)):
            b = folds[r % K]
            h = folds[(r + 1) % K] if separate else b
            q = folds[(r + 2) % K]
            chains.append(Chain(q, [Unit(b, h, q)]))
        return chains
    if name == 'oof':
        units = []
        for k in range(K):
            if separate:
                b, h = folds[(k + 1) % K], folds[(k + 2) % K]
            else:
                b = h = np.concatenate([folds[j] for j in range(K) if j != k])
            units.append(Unit(b, h, folds[k]))
        return [Chain(ds.fit_rows, units)]
    raise ValueError(f'unknown design {name}')


def design_shape(design: Dict) -> List[int]:
    """Number of units per chain, without loading data (for building the task graph)."""
    K = design.get('n_folds', 3)
    if design['name'] == 'full':
        return [1]
    if design['name'] == 'rotate':
        return [1] * design.get('rotations', K)
    if design['name'] == 'oof':
        return [K]
    raise ValueError(design['name'])


def _plan(source, design) -> List[Chain]:
    return plan(load(source), design)


def seed_everything(*parts) -> None:
    s = zlib.crc32(repr(parts).encode()) % (2 ** 31)
    torch.manual_seed(s)
    np.random.seed(s)


# --------------------------------------------------------------------------- features

def _a(p: Panel, t):
    return p.A[:, t:t + 1].astype(np.float32)


def bridge_inputs(p: Panel, t: int):
    """ShadOPE's bridge inputs: hypothesis (shadow_t, S_t, A_t), critic (R_t, S_t, A_t); the shadow
    variable is S_{t+1} unless the source reduces it (`shadow`)."""
    XH = np.concatenate([p.shadow[:, t], p.S[:, t], _a(p, t)], axis=1)
    XF = np.concatenate([p.R_obs[:, t:t + 1], p.S[:, t], _a(p, t)], axis=1)
    return XH, XF


def ratio_inputs(p: Panel):
    """Decision state X_t = (S_t, O_{t-1})."""
    return np.concatenate([p.S, p.o_prev[..., None].astype(np.float32)], axis=2)


# --------------------------------------------------------------------------- tasks: nuisances

def task_bridge(source, design, chain, unit, t, cfg, device):
    t0 = time.time()
    ds = load(source)
    seed_everything('bridge', _key(source), repr(sorted(design.items())), chain, unit, t)
    u = _plan(source, design)[chain].units[unit]
    train = ds.panel.subset(u.bridge_rows)
    obs = train.M[:, t] == 1
    XH, XF = bridge_inputs(train, t)
    model = NNBridge(device=device, **cfg['bridge']).fit(XH[obs], train.R_obs[obs, t], XF[obs])
    out = {}
    for name, rows in (('score', u.score_rows), ('test', ds.test_rows)):
        XH_p, _ = bridge_inputs(ds.panel.subset(rows), t)
        out[name] = model.predict(XH_p).cpu().numpy()
    out['n_obs'] = int(obs.sum())
    out['seconds'] = time.time() - t0
    return out


def task_recording(source, design, chain, unit, t, cfg, device):
    t0 = time.time()
    ds = load(source)
    u = _plan(source, design)[chain].units[unit]
    tr = ds.panel.subset(u.recording_rows)
    model = RecordingInverse(n_actions=tr.n_actions, device=device, **cfg['recording']).fit(
        tr.R_obs[:, t], tr.S[:, t], tr.A[:, t], tr.shadow[:, t], tr.M[:, t])
    out = {}
    for name, rows in (('score', u.score_rows), ('test', ds.test_rows)):
        p = ds.panel.subset(rows)
        out[name] = model.predict(p.R_obs[:, t], p.S[:, t], p.A[:, t]).astype(np.float32)
    out.update(model.info_)
    out['seconds'] = time.time() - t0
    return out


def _quantiles(x: np.ndarray) -> Dict[str, List[float]]:
    """Per-t quantiles of an (n, T) array (coverage diagnostics)."""
    return {name: np.quantile(x, q, axis=0).round(4).tolist()
            for name, q in (('q50', 0.5), ('q90', 0.9), ('q99', 0.99), ('max', 1.0))}


def task_ratios(source, design, chain, cfg, device, target=None):
    t0 = time.time()
    ds = load(source)
    panel = target_panel(source, target)
    seed_everything('ratios', _key(source), repr(sorted(design.items())), chain)
    tr = panel.subset(_plan(source, design)[chain].q_rows)
    model = RatioModel(n_actions=tr.n_actions, device=device, **cfg['ratios']).fit(
        tr.S, ratio_inputs(tr), tr.A, tr.pi)
    te = panel.subset(ds.test_rows)
    w = model.predict(te.S, ratio_inputs(te), te.A, te.pi)
    action_ratio = model.action_ratio(te.S, te.A, te.pi)      # per-step pi / mu on test stays, uncapped
    return {'test': w, 'train_mean': model.info_['mean_w'], 'objective': model.info_['objective'],
            'cap_fraction': float(np.mean(w >= cfg['ratios']['cap'] - 1e-6)),
            'w_quantiles': _quantiles(w), 'action_ratio_quantiles': _quantiles(action_ratio),
            'cum_action_ratio_quantiles': _quantiles(np.cumprod(action_ratio, axis=1)),
            'seconds': time.time() - t0}


# --------------------------------------------------------------------------- task: Q chain + evaluation

def corrected_scores(p: Panel, b: np.ndarray, h: np.ndarray) -> np.ndarray:
    """U = b + M h (R - b); h and R are only used where M = 1."""
    return b + p.M * h * (p.R_obs - b)


def assemble(source, design, chain, unit_scores: List[np.ndarray]) -> np.ndarray:
    """Place per-unit score-row predictions (each (n_u, T)) into the chain's q_rows order."""
    c = _plan(source, design)[chain]
    pos = {row: i for i, row in enumerate(c.q_rows)}
    out = np.empty((len(c.q_rows),) + unit_scores[0].shape[1:], dtype=np.float32)
    filled = np.zeros(len(c.q_rows), dtype=bool)
    for u, s in zip(c.units, unit_scores):
        idx = np.array([pos[r] for r in u.score_rows])
        out[idx] = s
        filled[idx] = True
    assert filled.all(), 'unit score rows must cover the chain q_rows'
    return out


def _q_values(q: NNQFunction, S: np.ndarray, device) -> torch.Tensor:
    return q.predict_all(torch.as_tensor(S, dtype=torch.float32, device=device))


def task_q_eval(source, design, chain, b_units, h_units, b_test, h_test, w_test, cfg, device, target=None):
    """Backward FQE on corrected scores (chain q_rows), then per-stay evaluation terms on test stays.
    b_units/h_units: per-unit (n_u, T) score-row predictions; b_test/h_test/w_test: (n_test, T)."""
    t0 = time.time()
    ds = load(source)
    seed_everything('q', _key(source), repr(sorted(design.items())), chain)
    panel = target_panel(source, target)
    train = panel.subset(_plan(source, design)[chain].q_rows)
    test = panel.subset(ds.test_rows)
    U_train = corrected_scores(train, assemble(source, design, chain, b_units),
                               assemble(source, design, chain, h_units))
    U_test = corrected_scores(test, b_test, h_test)
    T, K = train.horizon, train.n_actions
    dev = torch.device(device)

    Q_sa = np.zeros((test.n, T), dtype=np.float32)
    V_next = np.zeros((test.n, T), dtype=np.float32)
    q_next: Optional[NNQFunction] = None
    for t in reversed(range(T)):
        y = U_train[:, t].copy()
        if q_next is not None:
            y = y + (_q_values(q_next, train.Z[:, t], dev).cpu().numpy() * train.pi_next[:, t]).sum(1)
            V_next[:, t] = (_q_values(q_next, test.Z[:, t], dev).cpu().numpy() * test.pi_next[:, t]).sum(1)
        q_t = NNQFunction(state_dim=train.S.shape[2], n_actions=K, device=device, **cfg['q']).fit(
            torch.as_tensor(train.S[:, t], device=dev), torch.as_tensor(train.A[:, t], device=dev),
            torch.as_tensor(y, device=dev))
        Q_sa[:, t] = _q_values(q_t, test.S[:, t], dev).cpu().numpy()[np.arange(test.n), test.A[:, t]]
        q_next = q_t
    V1 = (_q_values(q_next, test.S[:, 0], dev).cpu().numpy() * test.pi[:, 0]).sum(1)

    correction = w_test * (U_test + V_next - Q_sa)
    psi = V1 + correction.sum(1)
    return {'psi': psi, 'fqe': V1, 'mean_correction_by_t': correction.mean(0).tolist(),
            'seconds': time.time() - t0}


# --------------------------------------------------------------------------- task: ShadOPE baselines

BASELINES = ('OracleFQE', 'NaiveFQE', 'ImputeFQE', 'IPW-FQE', 'SCOPE', 'ProxFQE')


def task_baseline(source, method, cfg, device, target=None):
    """One ShadOPE method with the settings of its eval_ope.py, fitted on the 60% fit stays and
    evaluated on the 40% test stays. SCOPE is an importance-sampling estimate computed inside
    fit() on its own internal split of the fit stays (its value() ignores the data argument)."""
    from improxope.nn_fqe import (NNImputeFQE, NNIPWFQE, NNNaiveFQE, NNOracleFQE, NNProxFQE, NNSCOPE)
    t0 = time.time()
    ds = load(source)
    seed_everything('baseline', _key(source), method)
    df, term = ds.extra['df'], ds.extra['term']
    term = term if source['terminal_fix'] else None
    shadow = source.get('shadow', 'full')
    fit_data = D.shadope_dict(df[df['icustayid'].isin(set(ds.fit_ids))].copy(), term, shadow,
                              target, source['data_dir'])
    test_data = D.shadope_dict(df[df['icustayid'].isin(set(ds.test_ids))].copy(), term, shadow,
                               target, source['data_dir'])
    sd = fit_data['states'].shape[1]
    s = cfg['shadope']
    gamma = 1.0
    bridge_kw = dict(n_steps=s['bridge_steps'], batch_size=s['batch_size'], device=device,
                     h_hidden=tuple(s['bridge_hidden']), f_hidden=tuple(s['bridge_hidden']))
    q_kw = dict(n_steps=s['q_steps'], batch_size=s['batch_size'], device=device, state_dim=sd,
                n_actions=25, hidden=tuple(s['q_hidden']))
    aux = dict(hidden=tuple(s['aux_hidden']), n_steps=s['bridge_steps'], lr=1e-3,
               batch_size=s['batch_size'], device=device)
    if method == 'OracleFQE':
        est = NNOracleFQE(state_dim=sd, gamma=gamma, q_kwargs=q_kw, device=device)
    elif method == 'NaiveFQE':
        est = NNNaiveFQE(state_dim=sd, gamma=gamma, q_kwargs=q_kw, device=device)
    elif method == 'ImputeFQE':
        est = NNImputeFQE(state_dim=sd, gamma=gamma, q_kwargs=q_kw, impute_kwargs=aux, device=device)
    elif method == 'IPW-FQE':
        est = NNIPWFQE(state_dim=sd, gamma=gamma, bridge_kwargs=bridge_kw, q_kwargs=q_kw,
                       prop_kwargs=aux, device=device)
    elif method == 'SCOPE':
        beh_kw = dict(n_actions=25, hidden=tuple(s['q_hidden']), n_steps=s['q_steps'], lr=1e-3,
                      batch_size=s['batch_size'], device=device)
        est = NNSCOPE(state_dim=sd, gamma=gamma, phi_kwargs=aux, behavior_kwargs=beh_kw, device=device)
    elif method == 'ProxFQE':
        est = NNProxFQE(state_dim=sd, gamma=gamma, bridge_kwargs=bridge_kw, q_kwargs=q_kw, device=device)
    else:
        raise ValueError(method)
    est.fit(fit_data)
    value, se = est.value(test_data)
    return {'value': float(value), 'se': float(se), 'seconds': time.time() - t0}
