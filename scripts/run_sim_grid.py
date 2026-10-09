import json
import os
import pickle
import sys
import time

import hydra
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from improxope import sim_ope as SO
from improxope.config import SimGridConfig, register
from improxope.parallel import Task, _cache_path, run_tasks

register()
if not any(a.startswith('hydra.run.dir') for a in sys.argv[1:]):
    sys.argv.append('hydra.run.dir=runs/sim_grid/${now:%Y-%m-%d_%H-%M-%S}')

# Relative runtimes at n = 2048 (measured: prox ~106 s, ipw ~105 s, ours ~201 s, others 4-10 s).
COST = {'ours': 2.0, 'prox': 1.0, 'ipw': 1.0, 'impute': 0.1, 'naive': 0.05, 'scope': 0.08, 'truth': 0.05}


def task_truth(T, seed, c0, beta, tau, kappa, n_eval, device):
    t0 = time.time()
    return {'truth': SO.true_value(T, seed, c0, beta, tau, kappa, n_eval), 'seconds': time.time() - t0}


def parse_cell(cell: str):
    beta, tau, kappa = (float(x) for x in str(cell).split(':'))
    return beta, tau, kappa


def intercepts(cfg, cell):
    """c0 for each missingness target of a cell: ShadOPE's for '0:0:0', else the calibration file."""
    beta, tau, kappa = parse_cell(cell)
    if beta == 0.0 and kappa == 0.0:
        return {m: SO.SHADOPE_C0[m] for m in cfg.miss_targets}
    levels = pd.read_csv(cfg.levels_file)
    out = {}
    for m in cfg.miss_targets:
        row = levels[np.isclose(levels.beta, beta) & np.isclose(levels.tau, tau) & np.isclose(levels.kappa, kappa)
                     & np.isclose(levels.miss_target, m)]
        assert len(row) == 1, f'no calibrated c0 for cell {cell}, missingness {m} in {cfg.levels_file}'
        out[m] = float(row.c0.iloc[0])
    return out


def published_rows(cfg, results):
    """ShadOPE's released per-seed values on its own simulator, scored against our truth tasks
    (which reproduce its true values exactly)."""
    pub = pd.read_csv(cfg.shadope_published)
    miss_of = {c0: m for m, c0 in SO.SHADOPE_C0.items()}
    rows = []
    for r in pub.itertuples():
        m = miss_of.get(float(r.mnar_c0))
        if m is None or m not in list(cfg.miss_targets) or r.n not in list(cfg.ns) or r.seed not in list(cfg.seeds):
            continue
        truth = results.get(f'cell0:0:0/miss{m:g}/seed{r.seed}/truth')
        if truth is None or not np.isfinite(r.value):
            continue
        assert abs(truth['truth'] - r.true) < 1e-4, f'truth mismatch for seed {r.seed}, c0 {r.mnar_c0}'
        rows.append(dict(cell='0:0:0', beta=0.0, tau=0.0, kappa=0.0, n=int(r.n), miss_target=m, c0=float(r.mnar_c0),
                         seed=int(r.seed), method=f'{r.method}_published', estimator=f'{r.method}_published',
                         value=float(r.value), truth=truth['truth'], error=float(r.value) - truth['truth'],
                         missing=float(r.missing), seconds=np.nan))
    return rows


@hydra.main(version_base=None, config_name='sim_grid')
def main(cfg: SimGridConfig) -> None:
    out_dir = HydraConfig.get().runtime.output_dir
    log_path = os.path.join(out_dir, 'progress.log')

    def log(msg):
        print(msg, flush=True)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

    recording = OmegaConf.to_container(cfg.recording, resolve=True)
    cells = list(cfg.cells)
    if cells == ['calibrated']:        # every calibrated cell except ShadOPE's own
        cells = [c for c in pd.read_csv(cfg.levels_file).cell.unique() if c != '0:0:0']
    tasks, jobs = [], []
    for cell in cells:
        beta, tau, kappa = parse_cell(cell)
        for m, c0 in intercepts(cfg, cell).items():
            for seed in cfg.seeds:
                tkey = f'cell{cell}/miss{m:g}/seed{seed}/truth'
                tasks.append(Task(tkey, task_truth, dict(T=cfg.horizon, seed=seed, c0=c0, beta=beta, tau=tau,
                                                         kappa=kappa, n_eval=cfg.truth_n_eval),
                                  cost=COST['truth'], kind='truth'))
                for n in cfg.ns:
                    for method in cfg.methods:
                        key = f'cell{cell}/n{n}/miss{m:g}/seed{seed}/{method}'
                        tasks.append(Task(key, SO.run_method, dict(method=method, T=cfg.horizon, n=n, seed=seed, c0=c0,
                                                                  beta=beta, tau=tau, kappa=kappa, n_folds=cfg.n_folds,
                                                                  recording=recording),
                                          cost=COST[method] * max(n / 2048, 0.15), kind=method))
                        jobs.append((key, tkey, dict(cell=cell, beta=beta, tau=tau, kappa=kappa, n=n, miss_target=m, c0=c0,
                                                     seed=seed, method=method)))
    log(f'[start] {len(tasks)} tasks, {cfg.n_workers} GPU workers on {list(cfg.gpus)} + {cfg.cpu_workers} CPU '
        f'workers -> {out_dir}')
    if cfg.aggregate_only:      # merged caches from several machines: read, never compute
        results = {}
        for t in tasks:
            path = _cache_path(os.path.join(out_dir, 'cache'), t.key)
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    results[t.key] = pickle.load(f)
        missing = sorted(t.key for t in tasks if t.key not in results)
        log(f'[aggregate] {len(results)} of {len(tasks)} tasks cached; {len(missing)} missing')
        for k in missing:
            log(f'[missing] {k}')
    else:
        results = run_tasks(tasks, cfg.n_workers, list(cfg.gpus), os.path.join(out_dir, 'cache'), log,
                            cpu_workers=cfg.cpu_workers)

    rows = []
    for key, tkey, meta in jobs:
        res, truth = results.get(key), results.get(tkey)
        if res is None or truth is None:
            continue
        for est, value in res.items():
            if est in ('missing', 'seconds', 'mean_Mh', 'cap_fraction'):
                continue
            rows.append(dict(meta, estimator=est, value=value, truth=truth['truth'], error=value - truth['truth'],
                             missing=res['missing'], seconds=res['seconds']))
    if cfg.shadope_published and '0:0:0' in cells:
        rows += published_rows(cfg, results)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'results_by_seed.csv'), index=False)
    if len(df):
        agg = (df.groupby(['cell', 'n', 'miss_target', 'estimator'], sort=False)
               .agg(missing=('missing', 'mean'), truth=('truth', 'mean'), bias=('error', 'mean'),
                    mae=('error', lambda e: float(np.mean(np.abs(e)))),
                    rmse=('error', lambda e: float(np.sqrt(np.mean(np.square(e))))),
                    sd=('value', 'std'), n_seeds=('seed', 'nunique'))
               .reset_index())
        agg.to_csv(os.path.join(out_dir, 'results_summary.csv'), index=False)
        text = agg.to_string(index=False, float_format=lambda x: f'{x:.3f}')
        with open(os.path.join(out_dir, 'results_summary.txt'), 'w') as f:
            f.write(text + '\n')
        log('\n' + text)
    diag = {key: {k: results[key][k] for k in ('mean_Mh', 'cap_fraction')}
            for key, _, meta in jobs if meta['method'] == 'ours' and key in results}
    with open(os.path.join(out_dir, 'diagnostics.json'), 'w') as f:
        json.dump(diag, f, indent=1)


if __name__ == '__main__':
    main()
