"""Validation on ShadOPE's synthetic MNAR environment, where the true policy value is known.

For each (mnar_c0, seed): generate logged data with ShadOPE's behaviour policy, compute the true
value of its target policy by on-policy rollout, and run our estimator (plus the cross-fitted
bridge variants) with the same pipeline as MIMIC. Outputs go to runs/sim/<timestamp>.

    python scripts/run_synthetic.py
    python scripts/run_synthetic.py mnar_c0s=[1.0,-0.5] seeds=[0] n_episodes=4000
"""
import json
import os
import sys

import hydra
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from improxope import experiment as E
from improxope.config import SimConfig, register
from improxope.parallel import run_tasks

register()
if not any(a.startswith('hydra.run.dir') for a in sys.argv[1:]):
    sys.argv.append('hydra.run.dir=runs/sim/${now:%Y-%m-%d_%H-%M-%S}')


def generate(cfg, c0, seed, path):
    from improxope.sim.behavior_policy import BehaviorPolicy
    from improxope.sim.configs import EnvConfig
    from improxope.sim.generate_data import collect_episodes
    from improxope.sim.sim_envs import MNARMDP
    env = MNARMDP(EnvConfig(horizon=cfg.horizon, seed=seed, gamma=1.0, mnar_c0=c0, reward_type=cfg.reward_type))
    ds = collect_episodes(env, BehaviorPolicy(seed=seed + 11), n_episodes=cfg.n_episodes, seed=seed)
    np.savez(path, **ds)
    return float(1.0 - ds['o'].mean())


@hydra.main(version_base=None, config_name='sim')
def main(cfg: SimConfig) -> None:
    from improxope.sim.simulation import compute_true_value_via_target_rollout
    out_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(os.path.join(out_dir, 'data'), exist_ok=True)
    log_path = os.path.join(out_dir, 'progress.log')

    def log(msg):
        print(msg, flush=True)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

    est = OmegaConf.to_container(cfg.est, resolve=True)
    design = OmegaConf.to_container(cfg.design, resolve=True)
    tasks, datasets = [], []
    for c0 in cfg.mnar_c0s:
        for seed in cfg.seeds:
            prefix = f'c0={c0}/seed{seed}'
            path = os.path.abspath(os.path.join(out_dir, 'data', f'sim_c0{c0}_seed{seed}.npz'))
            miss = generate(cfg, c0, seed, path)
            truth = compute_true_value_via_target_rollout(T=cfg.horizon, gamma=1.0, seed=seed, n_eval=cfg.n_true,
                                                          mnar_c0=c0, reward_type=cfg.reward_type)
            log(f'[data] {prefix}: missing rate {miss:.3f}, true value {truth:.4f}')
            datasets.append((prefix, c0, seed, truth, miss))
            src = dict(kind='sim', path=path, seed=int(seed), train_frac=float(cfg.train_frac))
            tasks += E.corrected_tasks(prefix, src, est, design, cfg.horizon, cfg.use_ratios)
    log(f'[start] {len(tasks)} tasks, {cfg.n_workers} workers -> {out_dir}')
    results = run_tasks(tasks, cfg.n_workers, list(cfg.gpus), os.path.join(out_dir, 'cache'), log)

    rows, diag = [], {}
    for prefix, c0, seed, truth, miss in datasets:
        for r in E.summarize(prefix, results, design, [], cfg.use_ratios):
            rows.append(dict(mnar_c0=c0, seed=seed, missing=miss, truth=truth, **r, error=r['value'] - truth))
        diag[prefix] = E.diagnostics(prefix, results, design, cfg.horizon)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'results_by_seed.csv'), index=False)
    if len(df):
        agg = (df.groupby(['mnar_c0', 'method'], sort=False)
               .agg(missing=('missing', 'mean'), truth=('truth', 'mean'), error_mean=('error', 'mean'),
                    error_sd=('error', 'std'), rmse=('error', lambda e: float(np.sqrt(np.mean(e ** 2)))),
                    n_seeds=('seed', 'nunique'))
               .reset_index())
        agg.to_csv(os.path.join(out_dir, 'results_summary.csv'), index=False)
        log('\n' + agg.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    with open(os.path.join(out_dir, 'diagnostics.json'), 'w') as f:
        json.dump(diag, f, indent=1)


if __name__ == '__main__':
    main()
