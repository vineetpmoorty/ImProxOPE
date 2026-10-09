"""Clean the ShadOPE input to T=10, also keeping each stay's 11th state as the terminal proxy.

Steps 1-4 follow ShadOPE's sepsis/clean_sepsis.py (https://github.com/NAIVlab/ShadOPE @ 4231ba5),
so sepsis_T10.csv has ShadOPE's format and works with its downstream scripts:
  1. keep stays with >= 11 rows, truncate to the first 11
  2. reward_t = -(SOFA_{t+1} - SOFA_t) for t = 1..10
  3. drop step 11 (no reward) -> T = 10
  4. keep exactly the 48 state features plus icustayid, vaso_input, iv_input, reward

Two additions:
- Step renumbering. The extraction only writes rows for 4-hour windows with recorded data, so
  some stays skip windows and their `bloc` labels have gaps (up to 19 within the first 11 rows).
  ShadOPE's cleaning counts rows but keeps the original labels, while its evaluation groups rows
  by `bloc` value and assumes t = 1..10, so labels with gaps break it. After truncation, `bloc`
  is renumbered 1..11 within each stay: a step is the next recorded window (usually 4 h later,
  sometimes longer). Rewards are unaffected; they were already computed row to row.
- Terminal proxy. ShadOPE discards step 11, so at t = 10 its evaluation code falls back to S_10
  as the "next state" and the final reward has no valid shadow variable. The 11th state of every
  stay is written to sepsis_T10_terminal.csv (icustayid + the same 48 state columns, bloc = 11)
  so that estimators can use S_11 as the t = 10 proxy. It is a separate file because ShadOPE's
  scripts treat every column they do not explicitly drop as a state feature.
"""
import argparse
import os

import pandas as pd

# Identical to DROP_COLS in ShadOPE sepsis/clean_sepsis.py
DROP_COLS = [
    'charttime',
    'presumed_onset',
    'died_in_hosp',
    'died_within_48h_of_out_time',
    'mortality_90d',
    'delay_end_of_record_and_discharge_or_death',
    'input_total',
    'input_4hourly',
    'cumulated_balance',
    'median_dose_vaso',
    'max_dose_vaso',
]
NON_STATE = ['icustayid', 'vaso_input', 'iv_input', 'reward']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='mimic_processed/sepsis_processed_state_action.csv')
    parser.add_argument('--outdir', default='mimic_processed')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    n_before = df['icustayid'].nunique()

    traj_len = df.groupby('icustayid')['bloc'].count()
    keep_ids = traj_len[traj_len >= 11].index
    df = df[df['icustayid'].isin(keep_ids)].copy()
    df = df.sort_values(['icustayid', 'bloc'])
    df = df.groupby('icustayid').head(11).reset_index(drop=True)
    n_after = df['icustayid'].nunique()
    print(f"Patients: {n_before} -> {n_after} (dropped {n_before - n_after} with < 11 steps)")

    step = df.groupby('icustayid').cumcount() + 1
    n_renumbered = df.loc[df['bloc'] != step, 'icustayid'].nunique()
    df['bloc'] = step.astype(df['bloc'].dtype)
    print(f"Renumbered bloc to 1..11 in {n_renumbered} stays with skipped 4-hour windows")

    df['reward'] = -(df.groupby('icustayid')['SOFA'].shift(-1) - df['SOFA'])

    terminal = df[df['reward'].isna()].copy()
    df = df[df['reward'].notna()].copy()
    df['reward'] = df['reward'].astype(float)

    assert df.groupby('icustayid')['bloc'].count().unique().tolist() == [10]
    print(f"After dropping step 11: {len(df)} rows ({n_after} x 10)")

    df = df.drop(columns=DROP_COLS)
    state_cols = [c for c in df.columns if c not in NON_STATE]
    print(f"State features: {len(state_cols)}")
    assert len(state_cols) == 48, f"Expected 48 state features, got {len(state_cols)}: {state_cols}"

    terminal = terminal[['icustayid'] + state_cols].reset_index(drop=True)
    assert len(terminal) == n_after and terminal['icustayid'].is_unique
    assert (terminal['bloc'] == 11).all()
    assert set(terminal['icustayid']) == set(df['icustayid'])

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, 'sepsis_T10.csv')
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    term_path = os.path.join(args.outdir, 'sepsis_T10_terminal.csv')
    terminal.to_csv(term_path, index=False)
    print(f"Saved {term_path} ({len(terminal)} terminal states)")


if __name__ == '__main__':
    main()
