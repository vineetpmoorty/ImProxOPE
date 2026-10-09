"""
Copied from ShadOPE (Wei, Qu, Miao, ICML 2026), https://github.com/NAIVlab/ShadOPE @ 4231ba5, file scripts/simulation.py.
Changes from the original: import paths; compute_true_value_via_target_rollout passes extra EnvConfig
fields (**env_kwargs, e.g. reward_latent_sd, mnar_curve) to the environment; removed fallback imports, including the unused np.random.RandomState fallback for NumpyRNG (module docstring added).
No license is published for ShadOPE; do not redistribute without the authors' permission.
"""
import os
import argparse
import numpy as np
import torch

from improxope.fqe import ProxFQE, NaiveFQE, WeightedFQE, ImputeFQE, SCOPE


# -------------------------- dataset helpers --------------------------
def load_npz_dataset(path: str) -> dict:
    """Load a saved dataset (.npz) with the expected keys."""
    data = np.load(path, allow_pickle=True)
    required = ["obs", "obs_n", "a", "o", "r_obs", "t"]
    for k in required:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {path}. Found: {list(data.keys())}")
    return {k: data[k] for k in required}


def compute_true_value_via_target_rollout(T, gamma, seed=1234, n_eval=5000,
                                          mnar_c0=1.0, reward_type='sigmoid', **env_kwargs):
    """
    On-policy rollout of the target policy on MNARMDP.
    """
    try:
        import numpy as np
        from improxope.sim.configs import EnvConfig
        from improxope.sim.sim_envs import MNARMDP
        from improxope.sim.target_policy import TargetPolicy
        from improxope.sim.utils import NumpyRNG
        rng_policy = NumpyRNG(seed + 1)
        bernoulli = lambda p: rng_policy.bernoulli(p)

        cfg = EnvConfig(horizon=T, seed=seed, gamma=gamma,
                        mnar_c0=mnar_c0, reward_type=reward_type, **env_kwargs)
        env = MNARMDP(cfg)
        pi = TargetPolicy()

        vals = []
        for i in range(n_eval):
            # reset: Gymnasium returns (obs, info)
            try:
                obs, info0 = env.reset(seed=seed + i + n_eval)
                o_prev = int(info0.get("o_prev", 0))
            except TypeError:
                # Some envs use reset() -> obs
                obs = env.reset()
                o_prev = 0

            ret = 0.0
            disc = 1.0

            for t in range(T):
                # obs = [s1, s2, o_prev]
                s_vec = np.asarray(obs[:2], dtype=float)
                p_plus = float(pi.prob_a_plus(s_vec, int(o_prev)))
                a_signed = 1 if bernoulli(p_plus) == 1 else -1

                step_out = env.step(a_signed)
                if not isinstance(step_out, tuple):
                    raise RuntimeError("env.step must return a tuple")
                if len(step_out) < 5:
                    raise RuntimeError(f"env.step returned {len(step_out)} items, expected 5")
                obs_next, r_obs, terminated, truncated, info = step_out

                # Use the *true* reward from info
                if "r_true" not in info:
                    raise KeyError("info['r_true'] not found; cannot compute true value")
                r_true = float(info["r_true"])
                if "o_t" not in info:
                    raise KeyError("info['o_t'] not found; cannot track o_prev")
                o_t = int(info["o_t"])

                ret += disc * r_true
                disc *= gamma

                obs = obs_next
                o_prev = o_t

                if terminated or truncated:
                    break

            vals.append(ret)

        return float(np.mean(vals))
    except Exception as e:
        print(f"[true-value] rollout failed: {type(e).__name__}: {e}")
        return np.nan


# -------------------------- main experiment --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/simulated/mnar_dataset.npz",
                        help="Path to .npz dataset. If missing, we will generate.")
    parser.add_argument("--n", type=int, default=500, help="#episodes to generate if needed")
    parser.add_argument("--T", type=int, default=10, help="Horizon")
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1) Load or generate dataset
    from improxope.sim.generate_data import collect_episodes
    from improxope.sim.configs import EnvConfig
    from improxope.sim.sim_envs import MNARMDP
    from improxope.sim.behavior_policy import BehaviorPolicy

    print("Generating dataset via collect_episodes ...")
    cfg = EnvConfig(horizon=args.T, seed=args.seed, gamma=args.gamma)
    env = MNARMDP(cfg)
    pi_b = BehaviorPolicy(seed=args.seed + 11)

    ds = collect_episodes(env, pi_b, n_episodes=args.n, seed=args.seed)


    if isinstance(ds, str) and ds.lower().endswith(".npz"):
        ds = load_npz_dataset(ds)
    if isinstance(ds, tuple) and len(ds) >= 1:
        ds = ds[0]

    raw_target = args.dataset  
    fname = os.path.basename(raw_target)
    if not fname or fname == "_NOFILE_.npz":
        fname = "mnar_dataset.npz"
    outdir = os.path.join("data", "simulated")
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, fname)

    np.savez(out_path, **ds)
    print(f"[saved] {out_path}  (rows={ds['obs'].shape[0]})")

    # 2) Target policy
    from improxope.sim.target_policy import TargetPolicy
    target_pi = TargetPolicy()

    # 3) Train three evaluators
    print("\n[Training] NaiveFQE ...")
    naive = NaiveFQE(
        action_list=(-1, +1),
        gamma=args.gamma,
        krr_kwargs=dict(lam_grid=np.logspace(-7, 1, 30).tolist(), folds=5, device=args.device),
        device=args.device
    ).fit(ds, target_pi)

    print("[Training] ProxFQE ...")
    prox = ProxFQE(
        action_list=(-1, +1),
        gamma=args.gamma,
        bridge_cv_kwargs=dict(
            delta_scale=5.0,
            delta_exp=0.4,  
            gamma_f='auto',
            gamma_hs='auto',
            n_gamma_hs=25,
            cv=5,
            device=args.device
        ),
        krr_kwargs=dict(lam_grid=np.logspace(-7, 1, 30).tolist(),
                         folds=5, device=args.device),
        device=args.device
    ).fit(ds, target_pi)

    print("[Training] IPW-FQE ...")
    ipw = WeightedFQE(
        action_list=(-1, +1), gamma=args.gamma,
        bridge_cv_kwargs=dict(  
            delta_scale=5.0, 
            delta_exp=0.4,
            gamma_f='auto', 
            gamma_hs='auto', 
            n_gamma_hs=25, 
            cv=5, 
            device=args.device
        ),
        krr_kwargs=dict(lam_grid=np.logspace(-7, 1, 30).tolist(),
                         folds=3, device=args.device),
        device=args.device
    ).fit(ds, target_pi)

    print("[Training] ImputeFQE ...")
    impute_fqe = ImputeFQE(
        action_list=(-1, +1),
        gamma=args.gamma,
        krr_kwargs=dict(lam_grid=np.logspace(-7, 1, 30).tolist(),
                         folds=5, device=args.device),
        device=args.device
    ).fit(ds, target_pi)

    print("[Training] SCOPE ...")
    from improxope.sim.behavior_policy import BehaviorPolicy as _BPol
    scope = SCOPE(
        gamma=args.gamma,
        frac_shape=0.3,
        krr_kwargs=dict(lam_grid=np.logspace(-7, 1, 30).tolist(),
                         folds=5, device=args.device),
        w_cap=50.0,
        device=args.device
    ).fit(ds, target_pi, _BPol(seed=args.seed + 11))

    # 4) Aggregate V(pi) over initial states
    t = ds["t"].astype(int)
    S1 = ds["obs"][t == 1, :2]
    O0 = np.zeros((S1.shape[0],), dtype=np.float32)

    v_naive = naive.value(S1, O0)
    v_prox = prox.value(S1, O0)
    v_ipw = ipw.value(S1, O0)
    v_impute = impute_fqe.value(S1, O0)
    v_scope = scope.value()

    # 5) Ground-truth
    v_true = compute_true_value_via_target_rollout(T=args.T, gamma=args.gamma,
                                                   seed=args.seed, n_eval=5000)

    print("\n================= RESULTS =================")
    print(f"NaiveFQE              : {v_naive: .6f}")
    print(f"ImputeFQE             : {v_impute: .6f}")
    print(f"ProxFQE               : {v_prox: .6f}")
    print(f"IPW-FQE               : {v_ipw: .6f}")
    print(f"SCOPE                 : {v_scope: .6f}")
    print(f"True (target rollout) : {v_true: .6f}" if not np.isnan(v_true) else "True (target rollout) : N/A")
    print("===========================================")


if __name__ == "__main__":
    main()