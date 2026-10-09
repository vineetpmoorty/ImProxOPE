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
