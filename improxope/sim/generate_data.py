'''
Copied from ShadOPE (Wei, Qu, Miao, ICML 2026), https://github.com/NAIVlab/ShadOPE @ 4231ba5, file src/generate_data.py.
Changes from the original: import paths; removed fallback imports; episodes also return the shadow
reading 'w' when the environment has reward_latent_sd > 0 (otherwise the output is unchanged).
No license is published for ShadOPE; do not redistribute without the authors' permission.

Generate offline trajectories for the MNARMDP under the behavior policy.

Output schema (one row per time step):
    s1, s2, o_prev, a, o_t, r_obs, r_true, s1_next, s2_next, o_t_next, ep, t

Usage (from repo root):

    # BOTH (default): save .npz (canonical) and .csv (for humans)
    python -m src.generate_data --episodes 500 --seed 123 \
        --out data/simulated/mnar_dataset

    # NPZ only (compressed NumPy archive)
    python -m src.generate_data --episodes 500 --seed 123 \
        --out data/simulated/mnar_dataset.npz --format npz

    # CSV only (human-readable; requires pandas)
    python -m src.generate_data --episodes 500 --seed 123 \
        --out data/simulated/mnar_dataset.csv --format csv
'''

from __future__ import annotations

import argparse
import os
from typing import Dict, Optional

import numpy as np

# Optional dependency only for CSV saving
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None

from improxope.sim.configs import EnvConfig
from improxope.sim.sim_envs import MNARMDP
from improxope.sim.behavior_policy import BehaviorPolicy
from improxope.sim.utils import NumpyRNG


# --------------------------------------------------------------------- #
# Rollout helpers
# --------------------------------------------------------------------- #
def rollout_one_episode(
    env: MNARMDP,
    policy,
    ep_index: int,
    env_seed: Optional[int] = None,
    pol_seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    '''
    Simulate ONE episode; return a dict of arrays for that episode.
    '''
    if env_seed is not None:
        obs, info = env.reset(seed=env_seed)
    else:
        obs, info = env.reset()

    T = env.cfg.horizon
    pi_rng = NumpyRNG(pol_seed) if pol_seed is not None else None

    obs_list, a_list, ro_list, rt_list, o_list, obsn_list, w_list = [], [], [], [], [], [], []
    ep_idx, t_idx = [], []

    for t in range(T):
        a = int(policy.act(obs, rng=pi_rng))
        obs_next, r_obs, terminated, truncated, info = env.step(a)

        obs_list.append(obs.copy())
        a_list.append(a)
        ro_list.append(float(r_obs))
        rt_list.append(float(info["r_true"]))  # true reward for later ground truth use
        o_list.append(int(info["o_t"]))
        obsn_list.append(obs_next.copy())
        w_list.append(float(info.get("w", 0.0)))

        ep_idx.append(ep_index)
        t_idx.append(t + 1)  # 1..T

        obs = obs_next
        if terminated or truncated:
            break

    out = {
        "obs": np.vstack(obs_list).astype(np.float32),      # (L,3)
        "a": np.asarray(a_list, dtype=np.int8),             # (L,)
        "o": np.asarray(o_list, dtype=np.int8),             # (L,)
        "r_obs": np.asarray(ro_list, dtype=np.float32),     # (L,)
        "r_true": np.asarray(rt_list, dtype=np.float32),    # (L,)
        "obs_n": np.vstack(obsn_list).astype(np.float32),   # (L,3)
        "ep": np.asarray(ep_idx, dtype=np.int32),           # (L,)
        "t": np.asarray(t_idx, dtype=np.int16),             # (L,)
    }
    if getattr(env.cfg, "reward_latent_sd", 0.0) > 0.0:
        out["w"] = np.asarray(w_list, dtype=np.float32)    # (L,) shadow reading of the latent reward
    return out


def collect_episodes(
    env: MNARMDP, policy, n_episodes: int, seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    '''
    Repeat `rollout_one_episode` and concatenate row-wise.
    '''
    base = NumpyRNG(seed)
    parts = []
    for ep in range(int(n_episodes)):
        env_seed = int(base.random() * (2**31 - 1))
        pol_seed = int(base.random() * (2**31 - 1))
        parts.append(
            rollout_one_episode(env, policy, ep_index=ep,
                                env_seed=env_seed, pol_seed=pol_seed)
        )
    keys = parts[0].keys()
    out = {k: [] for k in keys}
    for d in parts:
        for k in keys:
            out[k].append(d[k])
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


# --------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------- #
def save_csv(dataset: Dict[str, np.ndarray], out_path: str) -> None:
    '''
    Save the dataset as CSV (requires pandas).
    '''
    if pd is None:
        raise RuntimeError("pandas is required for CSV output. Install `pandas` or use --format npz.")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = _to_dataframe(dataset)
    df.to_csv(out_path, index=False)
    print(f"[saved] {out_path}  (rows={len(df)})")


def save_npz(dataset: Dict[str, np.ndarray], out_path: str) -> None:
    '''
    Save the dataset as compressed NumPy archive (.npz).
    '''
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **dataset)
    n_rows = dataset["a"].shape[0]
    print(f"[saved] {out_path}  (rows={n_rows})")


def save_both(dataset: Dict[str, np.ndarray], out_prefix: str) -> None:
    '''
    Save BOTH .npz (canonical) and .csv (human-readable).
    out_prefix: path without suffix, e.g., "data/simulated/mnar_dataset"
    '''
    base = out_prefix
    low = base.lower()
    if low.endswith(".npz"):
        base = base[:-4]
    elif low.endswith(".csv"):
        base = base[:-4]

    npz_path = base + ".npz"
    save_npz(dataset, npz_path)

    if pd is not None:
        csv_path = base + ".csv"
        save_csv(dataset, csv_path)
    else:
        print(f"[warn] pandas not installed; skipped CSV for {base}.csv")


def save(dataset: Dict[str, np.ndarray], out: str, mode: str = "npz") -> None:
    '''
    Unified saver:
      - mode='npz'  -> write out (ensure .npz suffix)
      - mode='csv'  -> write out (ensure .csv suffix)
      - mode='both' -> write both .npz and .csv using out as prefix or path
    '''
    m = mode.lower()
    if m == "npz":
        path = out if out.lower().endswith(".npz") else (out + ".npz")
        save_npz(dataset, path)
    elif m == "csv":
        path = out if out.lower().endswith(".csv") else (out + ".csv")
        save_csv(dataset, path)
    elif m == "both":
        base = out
        low = base.lower()
        if low.endswith(".npz") or low.endswith(".csv"):
            base = base[:-4]
        save_both(dataset, base)
    else:
        raise ValueError("mode must be one of {'npz','csv','both'}")


def _to_dataframe(dataset: Dict[str, np.ndarray]):
    '''
    Flatten dict-of-arrays to a Pandas DataFrame (one row per time step).
    '''
    return pd.DataFrame(
        {
            "s1":      dataset["obs"][:, 0],
            "s2":      dataset["obs"][:, 1],
            "o_prev":  dataset["obs"][:, 2],
            "a":       dataset["a"],
            "o_t":     dataset["o"],
            "r_obs":   dataset["r_obs"],
            "r_true":  dataset["r_true"],
            "s1_next": dataset["obs_n"][:, 0],
            "s2_next": dataset["obs_n"][:, 1],
            "o_t_next":dataset["obs_n"][:, 2],
            "ep":      dataset["ep"],
            "t":       dataset["t"],
        }
    )




# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate offline MNAR dataset (no metrics).")
    p.add_argument("--episodes", type=int, default=500, help="Number of episodes to simulate.")
    p.add_argument("--seed", type=int, default=0, help="Global RNG seed.")
    p.add_argument(
        "--out",
        type=str,
        # default: prefix (no suffix) so that default --format is BOTH
        default="data/simulated/mnar_dataset",
        help="Output path under `data/`. If no suffix and --format omitted, save BOTH .npz and .csv.",
    )
    p.add_argument(
        "--format",
        choices=["csv", "npz", "both"],
        default=None,
        help="If omitted, inferred from suffix; if no suffix, default to 'both'.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Infer format from suffix if not provided
    fmt = args.format
    if fmt is None:
        low = args.out.lower()
        if low.endswith(".csv"):
            fmt = "csv"
        elif low.endswith(".npz"):
            fmt = "npz"
        else:
            fmt = "both"  # default to saving both when no suffix is given

    # Build env & behavior policy
    cfg = EnvConfig(horizon=10, seed=args.seed, gamma=1.0)
    env = MNARMDP(cfg)
    pi_b = BehaviorPolicy()

    # Rollout and save
    print(f"[info] generating {args.episodes} episodes (horizon={cfg.horizon}) …")
    dataset = collect_episodes(env, pi_b, n_episodes=args.episodes, seed=args.seed)

    # Unified saving
    save(dataset, args.out, mode=fmt)


if __name__ == "__main__":
    main()