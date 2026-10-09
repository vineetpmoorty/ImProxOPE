import itertools
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from improxope import data as D
from improxope.panel import one_hot

ORGANS = {
    'respiratory': ['PaO2_FiO2', 'paO2', 'FiO2_1', 'mechvent'],
    'cardiovascular': ['MeanBP', 'SysBP', 'DiaBP', 'Shock_Index'],
    'neurological': ['GCS'],
    'renal': ['Creatinine', 'output_4hourly', 'output_total'],
    'liver': ['Total_bili'],
    'coagulation': ['Platelets_count'],
}
assert sorted(c for cols in ORGANS.values() for c in cols) == sorted(D.SOFA_COMPONENTS[1:])


def main(data_dir='mimic_processed', seed=42, out='results/shadow_organs.csv'):
    df, term = D.load_mimic(os.path.abspath(data_dir), 0.2, seed)   # masks are irrelevant here
    fit_ids, _ = D.shadope_split(df, seed)
    p = D.mimic_panel(df, term)
    cols = D.state_columns(df)
    fit = np.isin(p.ids, fit_ids)
    tr, te = np.repeat(fit, p.horizon), np.repeat(~fit, p.horizon)
    base = np.concatenate([p.S.reshape(-1, p.S.shape[2]),
                           one_hot(p.A, p.n_actions).reshape(-1, p.n_actions)], axis=1)
    y = p.R_true.reshape(-1)
    rows = []
    for k in range(len(ORGANS) + 1):
        for hidden in itertools.combinations(ORGANS, k):
            drop = ['SOFA'] + [c for o in hidden for c in ORGANS[o]]
            keep = [i for i, c in enumerate(cols) if c not in drop]
            X = np.concatenate([p.Z[..., keep].reshape(-1, len(keep)), base], axis=1).astype(np.float64)
            res = dict(n_hidden=k, hidden='+'.join(hidden) or '-', n_shadow_cols=len(keep))
            for name, model in (('linear', make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-6, 3, 10)))),
                                ('boosting', HistGradientBoostingRegressor(max_iter=300, random_state=0))):
                model.fit(X[tr], y[tr])
                res[f'r2_{name}'] = r2_score(y[te], model.predict(X[te]))
            rows.append(res)
            print(f"{res['r2_boosting']:.3f} {res['r2_linear']:.3f}  {res['hidden']}", flush=True)
    table = pd.DataFrame(rows).sort_values('r2_boosting', ascending=False)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    table.to_csv(out, index=False)
    print('\n' + table.to_string(index=False, float_format=lambda x: f'{x:.3f}'))


if __name__ == '__main__':
    main()
