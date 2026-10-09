"""Calibrate the simulator cells (beta:tau:kappa, see improxope/sim_ope.py) for the synthetic shadow
ladder and the 2x2 misspecification cells.

1. Shadow strength: with the hidden reward component fixed (beta), held-out R^2 of the true reward
   from (shadow = (S_{t+1}, w), S_t, A_t) for a grid of reading noise tau (boosting, as for MIMIC in
   scripts/shadow_strength.py), next to the R^2 from (S_t, A_t) alone. Ladder: tau = 0 (the shadow
   reveals the component) and the tau giving each target R^2 (interpolated, then re-measured).
2. 2x2 cells: the curved recording mechanism (kappa) at tau = 0 (bridge right, recording model
   wrong) and at the middle ladder level (both wrong).
3. Missingness: for every cell and target, the intercept c0 giving that missing rate (bisection on a
   simulated dataset; the missing rate decreases in c0). ShadOPE's cell 0:0:0 keeps its intercepts.
Writes results/sim_levels.csv (aggregate only), read by scripts/run_sim_grid.py.

    python scripts/calibrate_sim.py
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

from improxope import sim_ope as SO

T, CAL_SEED, CAL_N = 8, 999, 2048


def r2(beta, tau):
    ds = SO.dataset(T, CAL_N, CAL_SEED, -0.7, beta, tau, 0.0)    # missingness does not affect rewards
    s, sn, a, r = ds['obs'][:, :2], ds['obs_n'][:, :2], ds['a'].astype(float), ds['r_true']
    shadow = sn if 'w' not in ds else np.c_[sn, ds['w']]
    tr = ds['ep'] < int(0.6 * CAL_N)
    out = {}
    for name, X in (('r2_shadow', np.c_[shadow, s, a]), ('r2_state_only', np.c_[s, a])):
        m = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(X[tr], r[tr])
        out[name] = float(r2_score(r[~tr], m.predict(X[~tr])))
    return out


def missing_rate(c0, beta, tau, kappa):
    return float(1.0 - SO.dataset(T, CAL_N, CAL_SEED, c0, beta, tau, kappa)['o'].mean())


def calibrate_c0(target, beta, tau, kappa, lo=-15.0, hi=15.0, tol=0.002):
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        rate = missing_rate(mid, beta, tau, kappa)
        if abs(rate - target) < tol:
            break
        lo, hi = (mid, hi) if rate > target else (lo, mid)
    return round(mid, 4), rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--beta', type=float, default=0.4)
    ap.add_argument('--taus', default='0,0.25,0.5,0.75,1,1.25,1.5,2,2.5,3,4')
    ap.add_argument('--r2-targets', default='0.8,0.6,0.45')
    ap.add_argument('--kappa', type=float, default=10.0)
    ap.add_argument('--miss-targets', default='0.2,0.4,0.6,0.8')
    ap.add_argument('--out', default='results/sim_levels.csv')
    args = ap.parse_args()
    beta = args.beta

    grid = [(float(t), r2(beta, float(t))) for t in args.taus.split(',')]
    for t, v in grid:
        print(f'beta {beta} tau {t:<5} R^2 shadow {v["r2_shadow"]:.3f}  state only {v["r2_state_only"]:.3f}', flush=True)
    ts = np.array([t for t, _ in grid])
    r2s = np.array([v['r2_shadow'] for _, v in grid])
    taus = [0.0]
    for target in (float(x) for x in args.r2_targets.split(',')):
        assert r2s.min() <= target <= r2s.max(), f'R^2 target {target} outside the tau grid'
        taus.append(round(float(np.interp(-target, -r2s, ts)), 3))       # R^2 decreases in tau
    tau_mid = taus[2]                                                    # the R^2 ~ 0.6 level
    cells = [(0.0, 0.0, 0.0)] + [(beta, t, 0.0) for t in taus] + [(beta, 0.0, args.kappa), (beta, tau_mid, args.kappa)]

    rows = []
    for b, t, k in cells:
        v = r2(b, t)
        role = ('shadope' if b == 0 else 'ladder' if k == 0 else 'recording_wrong' if t == 0 else 'both_wrong')
        for m in (float(x) for x in args.miss_targets.split(',')):
            if b == 0.0 and k == 0.0:
                c0, rate = SO.SHADOPE_C0[m], missing_rate(SO.SHADOPE_C0[m], 0.0, 0.0, 0.0)
            else:
                c0, rate = calibrate_c0(m, b, t, k)
            rows.append(dict(cell=f'{b:g}:{t:g}:{k:g}', role=role, beta=b, tau=t, kappa=k, **v, miss_target=m,
                             c0=c0, missing=round(rate, 4)))
            print(rows[-1], flush=True)
    table = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    table.to_csv(args.out, index=False)
    print('\n' + table.to_string(index=False, float_format=lambda x: f'{x:.3f}'))


if __name__ == '__main__':
    main()
