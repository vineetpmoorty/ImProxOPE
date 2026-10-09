"""
Copied from ShadOPE (Wei, Qu, Miao, ICML 2026), https://github.com/NAIVlab/ShadOPE @ 4231ba5, file sepsis/apply_mnar.py.
Changes from the original: none.
No license is published for ShadOPE; do not redistribute without the authors' permission.

Apply MNAR missingness to the cleaned sepsis data with DQN target actions.

Pipeline:
    1. Load sepsis_T10_with_targets.csv (clean data + DQN target actions)
    2. Apply MNAR mechanism to rewards -> O_t, r_obs
    3. Compute O_{t-1} (shifted, first step = 1)
    4. Create conservative target policy: where O_{t-1}=0,
       reduce vaso_target and iv_target by 1 (clipped at 0)
    5. Save OPE data for each missing rate
"""

import argparse
import numpy as np
import pandas as pd
import os


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def apply_mnar(df, c0, c_r=2.0, c_s=0.3, c_a=-0.1, seed=42):
    rng = np.random.RandomState(seed)
    reward = df['reward'].values
    r_std = (reward - reward.mean()) / (reward.std() + 1e-8)

    lactate = (df['Arterial_lactate'].values - df['Arterial_lactate'].mean()) / df['Arterial_lactate'].std()
    mbp = (df['MeanBP'].values - df['MeanBP'].mean()) / df['MeanBP'].std()
    s_signal = lactate - 0.5 * mbp

    action_intensity = (df['vaso_input'].values + df['iv_input'].values)
    a_norm = (action_intensity - action_intensity.mean()) / (action_intensity.std() + 1e-8)

    logit = c0 + c_r * r_std + c_s * s_signal + c_a * a_norm
    p_obs = sigmoid(logit)

    o_t = rng.binomial(1, p_obs).astype(np.int8)
    r_obs = reward * o_t
    miss_rate = 1 - o_t.mean()
    return o_t, r_obs, miss_rate, p_obs


def calibrate_c0(df, target_miss_rate, c_r=2.0, c_s=0.3, c_a=-0.1, seed=42):
    lo, hi = -10.0, 10.0
    for _ in range(50):
        mid = (lo + hi) / 2
        _, _, miss_rate, _ = apply_mnar(df, c0=mid, c_r=c_r, c_s=c_s, c_a=c_a, seed=seed)
        if miss_rate < target_miss_rate:
            hi = mid
        else:
            lo = mid
    return mid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='realdata/sepsis_T10_with_targets.csv')
    parser.add_argument('--outdir', type=str, default='realdata')
    parser.add_argument('--miss-rates', type=str, default='0.2,0.4,0.6,0.8',
                        help='Comma-separated target missing rates')
    parser.add_argument('--c_r', type=float, default=2.0)
    parser.add_argument('--c_s', type=float, default=0.3)
    parser.add_argument('--c_a', type=float, default=-0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    assert 'vaso_target' in df.columns, "Input must have DQN target actions. Run train_dqn_sepsis.py first."
    miss_rates = [float(x) for x in args.miss_rates.split(',')]
    os.makedirs(args.outdir, exist_ok=True)

    for target in miss_rates:
        c0 = calibrate_c0(df, target, c_r=args.c_r, c_s=args.c_s, c_a=args.c_a, seed=args.seed)
        o_t, r_obs, actual_rate, p_obs = apply_mnar(
            df, c0=c0, c_r=args.c_r, c_s=args.c_s, c_a=args.c_a, seed=args.seed)

        df_ope = df.copy()
        df_ope['o_t'] = o_t
        df_ope['r_obs'] = r_obs
        df_ope['r_true'] = df['reward']
        df_ope['o_prev'] = df_ope.groupby('icustayid')['o_t'].shift(1).fillna(1).astype(int)

        # Conservative target policy: reduce dose by 1 when O_{t-1}=0
        miss_mask = df_ope['o_prev'] == 0
        df_ope.loc[miss_mask, 'vaso_target'] = (df_ope.loc[miss_mask, 'vaso_target'] - 1).clip(lower=0)
        df_ope.loc[miss_mask, 'iv_target'] = (df_ope.loc[miss_mask, 'iv_target'] - 1).clip(lower=0)

        tag = f"{int(target*100)}"
        ope_path = os.path.join(args.outdir, f'sepsis_T10_mnar{tag}_ope.csv')
        df_ope.to_csv(ope_path, index=False)

        # Stats
        n_patients = df_ope['icustayid'].nunique()
        n_reduced = miss_mask.sum()
        print(f"\n=== Missing rate target={target:.0%} ===")
        print(f"  c0={c0:.3f}, actual={actual_rate:.3f}, {n_patients} patients")
        print(f"  p_obs: mean={p_obs.mean():.3f}, min={p_obs.min():.3f}, max={p_obs.max():.3f}")

        r = df['reward'].values
        for q_lo, q_hi, label in [(0, 25, 'Q1 (worst)'), (25, 50, 'Q2'),
                                   (50, 75, 'Q3'), (75, 100, 'Q4 (best)')]:
            lo_val = np.percentile(r, q_lo)
            hi_val = np.percentile(r, q_hi)
            mask = (r >= lo_val) & (r <= hi_val) if q_lo > 0 else (r <= hi_val)
            miss_q = 1 - o_t[mask].mean()
            print(f"    {label}: miss rate = {miss_q:.3f}")

        print(f"  Conservative adjustment: {n_reduced}/{len(df_ope)} steps with o_prev=0 -> target dose -1")
        print(f"  Saved {ope_path}")


if __name__ == '__main__':
    main()
