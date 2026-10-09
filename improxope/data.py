from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

from improxope.panel import Panel, one_hot

T_MIMIC = 10
N_ACTIONS_MIMIC = 25
# Columns of the masked files that are not state features (ShadOPE's eval_ope.py drop list).
NON_STATE = ['icustayid', 'vaso_input', 'iv_input', 'reward',
             'o_t', 'r_obs', 'r_true', 'o_prev', 'vaso_target', 'iv_target']


# Next-state columns removed from the shadow variable at each level.
SOFA_COMPONENTS = [
    'SOFA',
    'PaO2_FiO2', 'paO2', 'FiO2_1', 'mechvent',       # respiration
    'Platelets_count',                                # coagulation
    'Total_bili',                                     # liver
    'MeanBP', 'SysBP', 'DiaBP', 'Shock_Index',        # cardiovascular (blood pressure)
    'GCS',                                            # central nervous system
    'Creatinine', 'output_4hourly', 'output_total',   # renal
]
# Intermediate levels hide SOFA plus the organ systems that drive its 4-hour changes, in the order
# that weakens the shadow most (chosen by held-out R^2 of the reward from (shadow, S_t, A_t),
# scripts/shadow_organs.py: no_sofa 0.92, +respiratory 0.77, +neurological 0.62, +renal 0.46,
# all six systems 0.32).
_RESPIRATORY = ['PaO2_FiO2', 'paO2', 'FiO2_1', 'mechvent']
_NEUROLOGICAL = ['GCS']
_RENAL = ['Creatinine', 'output_4hourly', 'output_total']
SHADOW_DROP = {
    'full': [],
    'no_sofa': ['SOFA'],
    'no_sofa_resp': ['SOFA'] + _RESPIRATORY,
    'no_sofa_resp_neuro': ['SOFA'] + _RESPIRATORY + _NEUROLOGICAL,
    'no_sofa_resp_neuro_renal': ['SOFA'] + _RESPIRATORY + _NEUROLOGICAL + _RENAL,
    'no_sofa_components': SOFA_COMPONENTS,
}


def shadow_columns(state_cols, shadow: str):
    """Indices (into state_cols) of the next-state columns kept in the shadow variable."""
    drop = SHADOW_DROP[shadow]
    missing = [c for c in drop if c not in state_cols]
    assert not missing, f'unknown columns {missing}'
    return [i for i, c in enumerate(state_cols) if c not in drop]


def _dose_rule(a: np.ndarray) -> np.ndarray:
    """ShadOPE's conservative rule after a missing reward: vaso and iv levels each lowered by 1."""
    return np.maximum(a // 5 - 1, 0) * 5 + np.maximum(a % 5 - 1, 0)


@lru_cache(maxsize=2)
def _clinician(path: str):
    z = np.load(path)
    key = z['icustayid'].astype(np.int64) * 100 + z['bloc'].astype(np.int64)
    order = np.argsort(key)
    return key[order], z['clin'][order], z['q_dqn'][order]


def target_policy(df: pd.DataFrame, target: Dict, data_dir: str) -> np.ndarray:
    """Target probabilities (rows of df, N_ACTIONS) of pi_eps = (1 - eps) * clinician + eps * DQN'.

    clinician: the clone from scripts/data/train_clinician_policy.py. DQN': ShadOPE's DQN target
    (including its lower-dose rule after a missing reward, from the masked file) where the clone
    gives that action probability >= tau; elsewhere the DQN's highest-valued action among those
    with clone probability >= tau, with the same dose rule applied when the result stays above
    tau. So every target action is one clinicians take in similar states (coverage), while the
    target still differs from them at every step for eps > 0. tau = 0 gives the plain mixture, and
    eps = 1, tau = 0 ShadOPE's DQN target exactly."""
    eps, tau = float(target['eps']), float(target.get('tau', 0.0))
    keys, clin_all, q_all = _clinician(os.path.join(data_dir, 'clinician_policy.npz'))
    key = df['icustayid'].values.astype(np.int64) * 100 + df['bloc'].values.astype(np.int64)
    pos = np.searchsorted(keys, key)
    assert np.all(keys[np.minimum(pos, len(keys) - 1)] == key), 'rows missing from clinician_policy.npz'
    clin, q = clin_all[pos].astype(np.float64), q_all[pos]
    rows = np.arange(len(df))
    a_dqn = (df['vaso_target'].values * 5 + df['iv_target'].values).astype(np.int64)
    supported = clin >= np.minimum(tau, clin.max(1, keepdims=True))   # never empty
    a_best = np.where(supported, q, -np.inf).argmax(1)
    a_rule = _dose_rule(a_best)
    missed = df['o_prev'].values == 0
    a_best = np.where(missed & supported[rows, a_rule], a_rule, a_best)
    a_star = np.where(supported[rows, a_dqn], a_dqn, a_best)
    pi = (1.0 - eps) * clin
    pi[rows, a_star] += eps
    return (pi / pi.sum(1, keepdims=True)).astype(np.float32)


def mimic_paths(data_dir: str, miss_rate: float, seed: int) -> Tuple[str, str]:
    """Masked dataset for (miss_rate, seed) and the terminal-state file."""
    masked = os.path.join(data_dir, 'mnar', f'seed{seed}', f'sepsis_T10_mnar{round(miss_rate * 100)}_ope.csv')
    terminal = os.path.join(data_dir, 'sepsis_T10_terminal.csv')
    return masked, terminal


@lru_cache(maxsize=2)
def load_mimic(data_dir: str, miss_rate: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    masked, terminal = mimic_paths(data_dir, miss_rate, seed)
    df = pd.read_csv(masked)
    term = pd.read_csv(terminal)
    df = df.sort_values(['icustayid', 'bloc']).reset_index(drop=True)
    counts = df.groupby('icustayid')['bloc'].agg(['count', 'min', 'max'])
    assert (counts['count'] == T_MIMIC).all() and (counts['min'] == 1).all() and (counts['max'] == T_MIMIC).all()
    term = term.set_index('icustayid').loc[df['icustayid'].unique()].reset_index()
    return df, term


def shadope_split(df: pd.DataFrame, seed: int, train_frac: float = 0.6):
    """Fit/test stay ids exactly as ShadOPE's eval_ope.load_and_split."""
    ids = np.sort(df['icustayid'].unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)
    n_fit = int(len(ids) * train_frac)
    return ids[:n_fit], ids[n_fit:]


def crossfit_parts(fit_ids: np.ndarray, seed: int, n_parts: int):
    """Split the fitting stays into n_parts disjoint blocks for cross-fitting."""
    ids = np.array(fit_ids, copy=True)
    np.random.RandomState(seed + 1000).shuffle(ids)
    return [np.sort(part) for part in np.array_split(ids, n_parts)]


def state_columns(df: pd.DataFrame):
    return [c for c in df.columns if c not in NON_STATE]


def shadope_dict(df: pd.DataFrame, terminal: pd.DataFrame = None, shadow: str = 'full',
                 target: Dict = None, data_dir: str = None) -> Dict:
    """ShadOPE's eval_ope._df_to_data (https://github.com/NAIVlab/ShadOPE @ 4231ba5), unchanged,
    plus the optional terminal fix: if `terminal` is given, rows at t = T use the stay's terminal
    state as next state instead of the current state. If `shadow` is not 'full', a reduced
    'shadow' array is added, which ShadOPE's bridges use instead of 'next_states'. If `target` is
    given, target probabilities 'pi_target' (current row) and 'pi_target_next' (next row of the
    stay, as for at_joint_next) are added, which ShadOPE's estimators use instead of the DQN
    actions."""
    state_cols = state_columns(df)
    states = df[state_cols].values.astype(np.float32)

    T = T_MIMIC
    next_states = np.zeros_like(states)
    dones = np.zeros(len(df), dtype=np.float32)
    bloc = df['bloc'].values
    icuids = df['icustayid'].values

    for i in range(len(df) - 1):
        if bloc[i] < T and icuids[i] == icuids[i + 1]:
            next_states[i] = states[i + 1]
        else:
            next_states[i] = states[i]
            dones[i] = 1.0
    next_states[-1] = states[-1]
    dones[-1] = 1.0

    if terminal is not None:
        last = bloc == T
        term = terminal.set_index('icustayid').loc[icuids[last], state_cols]
        next_states[last] = term.values.astype(np.float32)

    a_joint = (df['vaso_input'].values * 5 + df['iv_input'].values).astype(np.int64)
    at_joint = (df['vaso_target'].values * 5 + df['iv_target'].values).astype(np.int64)

    at_joint_next = np.zeros_like(at_joint)
    for i in range(len(df) - 1):
        if bloc[i] < T and icuids[i] == icuids[i + 1]:
            at_joint_next[i] = at_joint[i + 1]
        else:
            at_joint_next[i] = at_joint[i]
    at_joint_next[-1] = at_joint[-1]

    out = {
        'states': states,
        'next_states': next_states,
        'a_joint': a_joint,
        'at_joint': at_joint,
        'at_joint_next': at_joint_next,
        'r_true': df['r_true'].values.astype(np.float32),
        'r_obs': df['r_obs'].values.astype(np.float32),
        'o_t': df['o_t'].values.astype(np.int8),
        'o_prev': df['o_prev'].values.astype(np.int8),
        'dones': dones,
        'bloc': bloc.astype(np.int32),
        'state_cols': state_cols,
    }
    if shadow != 'full':
        out['shadow'] = next_states[:, shadow_columns(state_cols, shadow)]
    if target is not None:
        pi = target_policy(df, target, data_dir)
        nxt = np.arange(len(df))
        same = (bloc[:-1] < T) & (icuids[:-1] == icuids[1:])
        nxt[:-1][same] += 1
        out['pi_target'], out['pi_target_next'] = pi, pi[nxt]
    return out


def mimic_panel(df: pd.DataFrame, terminal: pd.DataFrame, terminal_fix: bool = True,
                shadow: str = 'full', target: Dict = None, data_dir: str = None) -> Panel:
    """Panel view of a masked MIMIC dataset (rows sorted by icustayid, bloc; 10 rows per stay)."""
    T, K = T_MIMIC, N_ACTIONS_MIMIC
    ids = df['icustayid'].values.reshape(-1, T)[:, 0]
    cols = state_columns(df)

    def grid(col):
        return df[col].values.reshape(-1, T)

    S = df[cols].values.astype(np.float32).reshape(len(ids), T, len(cols))
    Z = np.empty_like(S)
    Z[:, :-1] = S[:, 1:]
    if terminal_fix:
        Z[:, -1] = terminal.set_index('icustayid').loc[ids, cols].values.astype(np.float32)
    else:
        Z[:, -1] = S[:, -1]
    if target is None:
        pi = one_hot((grid('vaso_target') * 5 + grid('iv_target')).astype(np.int64), K)
    else:
        pi = target_policy(df, target, data_dir).reshape(len(ids), T, K)
    pi_next = np.zeros_like(pi)
    pi_next[:, :-1] = pi[:, 1:]
    panel = Panel(
        ids=ids,
        S=S,
        Z=Z,
        o_prev=grid('o_prev').astype(np.int8),
        A=(grid('vaso_input') * 5 + grid('iv_input')).astype(np.int64),
        M=grid('o_t').astype(np.int8),
        R_obs=grid('r_obs').astype(np.float32),
        pi=pi,
        pi_next=pi_next,
        R_true=grid('r_true').astype(np.float32),
        W=None if shadow == 'full' else Z[..., shadow_columns(cols, shadow)],
    )
    panel.validate()
    return panel


def sim_panel(ds: Dict[str, np.ndarray], target_policy) -> Panel:
    """Panel view of a dataset from improxope.sim.generate_data.collect_episodes.

    obs = (s1, s2, o_prev); actions -1/+1 map to indices 0/1; the stochastic target policy gives
    pi(+1 | s, o_prev)."""
    ep, t = ds['ep'], ds['t']
    T = int(t.max())
    n_ep = len(np.unique(ep))
    assert len(ep) == n_ep * T, "episodes must all have the full horizon"
    order = np.lexsort((t, ep))

    def grid(x):
        x = np.asarray(x)[order]
        return x.reshape((n_ep, T) + x.shape[1:])

    obs, obs_n = grid(ds['obs']), grid(ds['obs_n'])

    def probs(o):
        flat = torch.tensor(o.reshape(-1, 3), dtype=torch.float32)
        p_plus = target_policy.prob_a_plus_batch(flat[:, :2], flat[:, 2]).numpy().reshape(o.shape[:2])
        return np.stack([1.0 - p_plus, p_plus], axis=-1).astype(np.float32)

    pi = probs(obs)
    pi_next = probs(obs_n)
    pi_next[:, -1] = 0.0
    panel = Panel(
        ids=grid(ep)[:, 0],
        S=obs[..., :2].astype(np.float32),
        Z=obs_n[..., :2].astype(np.float32),
        o_prev=obs[..., 2].astype(np.int8),
        A=((grid(ds['a']) + 1) // 2).astype(np.int64),
        M=grid(ds['o']).astype(np.int8),
        R_obs=grid(ds['r_obs']).astype(np.float32),
        pi=pi,
        pi_next=pi_next,
        R_true=grid(ds['r_true']).astype(np.float32),
    )
    panel.validate()
    return panel
