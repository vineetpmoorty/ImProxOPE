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


def features(p, shadow):
    X = [p.S.reshape(-1, p.S.shape[2]), one_hot(p.A, p.n_actions).reshape(-1, p.n_actions)]
    if shadow is not None:
        X.insert(0, shadow.reshape(-1, shadow.shape[2]))
    return np.concatenate(X, axis=1).astype(np.float64)


def main(data_dir='mimic_processed', seed=42, out='results/shadow_strength.csv'):
    df, term = D.load_mimic(os.path.abspath(data_dir), 0.2, seed)   # masks are irrelevant here
    fit_ids, _ = D.shadope_split(df, seed)
    rows = []
    for level in ['none'] + list(D.SHADOW_DROP):
        p = D.mimic_panel(df, term, shadow='full' if level == 'none' else level)
        fit = np.isin(p.ids, fit_ids)
        shadow = None if level == 'none' else p.shadow
        X = features(p, shadow)
        y = p.R_true.reshape(-1)
        tr, te = np.repeat(fit, p.horizon), np.repeat(~fit, p.horizon)
        res = dict(shadow=level, n_shadow_cols=0 if shadow is None else shadow.shape[2])
        for name, model in (('linear', make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-6, 3, 10)))),
                            ('boosting', HistGradientBoostingRegressor(max_iter=300, random_state=0))):
            model.fit(X[tr], y[tr])
            res[f'r2_{name}'] = r2_score(y[te], model.predict(X[te]))
        rows.append(res)
        print(res, flush=True)
    table = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    table.to_csv(out, index=False)
    print('\n' + table.to_string(index=False, float_format=lambda x: f'{x:.3f}'))


if __name__ == '__main__':
    main()
