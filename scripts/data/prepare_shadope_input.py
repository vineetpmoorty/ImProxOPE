"""Build ShadOPE's expected input (sepsis_processed_state_action.csv) from MIMICtable.csv.

MIMICtable.csv is the raw-unit intermediate table written by sepsis_cohort.py
--save_intermediate in https://github.com/microsoft/mimic_sepsis (commit fce1d05, with
the fixes listed in README.md applied). ShadOPE's sepsis/clean_sepsis.py expects that
table plus two discrete action columns, vaso_input and iv_input (levels 0-4 each).
ShadOPE's binning code is not public, so this reproduces the Komorowski/Raghu rule as
implemented in that repository's sepsis_cohort.py lines 1640-1654: level 0 = no drug in
the 4h window, levels 1-4 = quartiles of the nonzero doses over the whole table
(input_4hourly for IV fluid, max_dose_vaso for vasopressors). ShadOPE's joint action is
vaso_input * 5 + iv_input.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats


def iv_levels(input_4hourly):
    a = np.asarray(input_4hourly, dtype=float)
    levels = np.zeros(len(a), dtype=int)
    nonzero = a > 0
    ranks = stats.rankdata(a[nonzero]) / nonzero.sum()
    levels[nonzero] = np.floor((ranks + 0.2499999999) * 4).astype(int)
    return levels


def vaso_levels(max_dose_vaso):
    v = np.asarray(max_dose_vaso, dtype=float)
    levels = np.zeros(len(v), dtype=int)
    nonzero = v != 0
    ranks = stats.rankdata(v[nonzero]) / nonzero.sum()
    q = np.floor((ranks + 0.249999999999) * 4)
    q[q == 0] = 1
    levels[nonzero] = q.astype(int)
    return levels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='mimic_processed/MIMICtable.csv')
    parser.add_argument('--output', default='mimic_processed/sepsis_processed_state_action.csv')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df['iv_input'] = iv_levels(df['input_4hourly'])
    df['vaso_input'] = vaso_levels(df['max_dose_vaso'])

    for col in ('iv_input', 'vaso_input'):
        assert df[col].between(0, 4).all(), col
    print('stays:', df['icustayid'].nunique(), '| rows:', len(df))
    print('iv_input level counts:', df['iv_input'].value_counts().sort_index().to_dict())
    print('vaso_input level counts:', df['vaso_input'].value_counts().sort_index().to_dict())
    joint = df['vaso_input'] * 5 + df['iv_input']
    print('distinct joint actions (vaso*5+iv):', joint.nunique())

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    df.to_csv(args.output, index=False)
    print('saved', args.output)


if __name__ == '__main__':
    main()
