"""MIMIC-III sepsis benchmark: our estimator and ShadOPE's baselines on identical data and splits.

All (miss rate, seed) datasets run in one dependency-aware pool over the GPUs. Outputs go to the
Hydra run directory (default runs/mimic/<timestamp>, git-ignored: the task cache holds
patient-level predictions). Rerunning with the same `hydra.run.dir` resumes from the cache.

    python scripts/run_mimic.py                                   # full: 4 rates x 3 seeds
    python scripts/run_mimic.py miss_rates=[0.4] seeds=[42]       # one dataset
    python scripts/run_mimic.py hydra.run.dir=runs/mimic/<old>    # resume
    python scripts/run_mimic.py design.name=full use_ratios=false \
        reuse_cache_from=runs/mimic/main                         # baselines from that run's cache
    python scripts/run_mimic.py shadow=no_sofa_components ...     # weakened shadow (levels: improxope/data.py)
    python scripts/run_mimic.py 'target_eps=[0,0.25,0.5,1]' use_ratios=true \
        reuse_cache_from=runs/mimic/oof_5seeds 'reuse_cache_pattern=[bridge,recording]' ...   # epsilon-mixture targets

`reuse_cache_from` (one run dir or a list) copies cached results matching `reuse_cache_pattern`,
but only those valid for this run, judged from the source run's saved config (.hydra/config.yaml;
see `reusable`): e.g. a weakened-shadow run reuses bridge and recording fits only from runs with the
same shadow level, and from full-shadow runs only the baselines that never use the shadow variable.
"""
import glob
import json
import os
import shutil
import sys

import hydra
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from improxope import experiment as E
from improxope.config import MimicConfig, register
from improxope.data import SHADOW_DROP, T_MIMIC
from improxope.parallel import run_tasks

register()
SHADOW_METHODS = ('ProxFQE', 'IPW-FQE')   # baselines whose bridge uses the shadow variable
if not any(a.startswith('hydra.run.dir') for a in sys.argv[1:]):
    sys.argv.append('hydra.run.dir=runs/mimic/${now:%Y-%m-%d_%H-%M-%S}')


def _settings(cfg, *keys):
    node = cfg
    for k in keys:
        node = node.get(k) if node is not None else None
    return OmegaConf.to_container(node, resolve=True) if OmegaConf.is_config(node) else node


def reusable(name: str, src, cfg) -> bool:
    """Whether cache file `name` from a run with config `src` is valid in this run (config `cfg`).
    Same data, split and target settings are assumed for keys that match; what differs between runs
    with matching keys is checked: the shadow level, design and estimator settings, and the target's
    tau for epsilon-mixture keys."""
    if src is None:
        return False
    same_shadow = src.get('shadow', 'full') == cfg.shadow
    if '_eps' in name and float(src.get('target_tau', 0.05)) != float(cfg.target_tau):
        return False
    if 'baseline_' in name:
        method = name.split('baseline_')[1][:-4]
        return (_settings(src, 'est', 'shadope') == _settings(cfg, 'est', 'shadope')
                and (same_shadow or method not in SHADOW_METHODS))
    needs = {'q_eval': ['bridge', 'recording', 'ratios', 'q'],     # first: q_eval names contain _bridge_
             'ratios': ['ratios'], 'recording': ['recording'], 'bridge': ['bridge']}
    kind = next((k for k in needs if f'_{k}_' in name), None)
    if kind is None or not same_shadow or _settings(src, 'design') != _settings(cfg, 'design'):
        return False
    return all(_settings(src, 'est', k) == _settings(cfg, 'est', k) for k in needs[kind])


def source(cfg, miss_rate, seed):
    src = dict(kind='mimic', data_dir=os.path.abspath(cfg.data_dir), miss_rate=float(miss_rate),
               seed=int(seed), terminal_fix=bool(cfg.terminal_fix), train_frac=float(cfg.train_frac))
    if cfg.shadow != 'full':        # absent for 'full', so earlier runs' seeds and caches stay valid
        src['shadow'] = cfg.shadow
    return src


@hydra.main(version_base=None, config_name='mimic')
def main(cfg: MimicConfig) -> None:
    out_dir = HydraConfig.get().runtime.output_dir
    log_path = os.path.join(out_dir, 'progress.log')

    def log(msg):
        print(msg, flush=True)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

    est = OmegaConf.to_container(cfg.est, resolve=True)
    design = OmegaConf.to_container(cfg.design, resolve=True)
    assert cfg.shadow in SHADOW_DROP, f'shadow must be one of {list(SHADOW_DROP)}'
    sources = cfg.reuse_cache_from
    sources = [sources] if isinstance(sources, str) else list(sources or [])
    patterns = list(cfg.reuse_cache_pattern)
    for run_dir in [d for d in sources if d]:
        os.makedirs(os.path.join(out_dir, 'cache'), exist_ok=True)
        cfg_path = os.path.join(run_dir, '.hydra', 'config.yaml')
        src_cfg = OmegaConf.load(cfg_path) if os.path.exists(cfg_path) else None
        files = sorted({f for p in patterns for f in glob.glob(os.path.join(run_dir, 'cache', f'*{p}*.pkl'))})
        copied = skipped = 0
        for f in files:
            name = os.path.basename(f)
            dest = os.path.join(out_dir, 'cache', name)
            if not reusable(name, src_cfg, cfg):
                skipped += 1
            elif not os.path.exists(dest):
                shutil.copy2(f, dest)
                copied += 1
        log(f'[reuse] {run_dir}: copied {copied}, skipped {skipped} incompatible (shadow '
            f"{src_cfg.get('shadow', 'full') if src_cfg is not None else '?'}) of {len(files)} matching {patterns}")
    methods = list(cfg.baselines)
    targets = ([(f'eps{e:g}', dict(eps=float(e), tau=float(cfg.target_tau))) for e in cfg.target_eps]
               or list(E.OWN_TARGET))
    tasks, datasets = [], []
    for rate in cfg.miss_rates:
        for seed in cfg.seeds:
            prefix = f'mnar{round(rate * 100)}/seed{seed}'
            src = source(cfg, rate, seed)
            datasets += [(prefix, tag, target, rate, seed) for tag, target in targets]
            tasks += E.baseline_tasks(prefix, src, est, methods, targets)
            if cfg.run_ours:
                tasks += E.corrected_tasks(prefix, src, est, design, T_MIMIC, cfg.use_ratios, targets)
    log(f'[start] {len(datasets)} dataset(s), {len(tasks)} tasks, {cfg.n_workers} workers on GPUs '
        f'{list(cfg.gpus)} -> {out_dir}')
    results = run_tasks(tasks, cfg.n_workers, list(cfg.gpus), os.path.join(out_dir, 'cache'), log)

    rows, diag = [], {}
    for prefix, tag, target, rate, seed in datasets:
        tp = E.target_prefix(prefix, tag)
        summary = E.summarize(tp, results, design, methods, cfg.use_ratios)
        oracle = next((r['value'] for r in summary if r['method'] == 'OracleFQE'), float('nan'))
        extra = {} if target is None else dict(eps=target['eps'])
        for r in summary:
            rows.append(dict(miss_rate=rate, seed=seed, **extra, **r, bias_vs_oracle=r['value'] - oracle))
        diag[tp] = E.diagnostics(prefix, results, design, T_MIMIC, tag)
    by_seed = pd.DataFrame(rows)
    by_seed.to_csv(os.path.join(out_dir, 'results_by_seed.csv'), index=False)
    if len(by_seed):
        keys = ['miss_rate'] + (['eps'] if 'eps' in by_seed else []) + ['method']
        agg = (by_seed.groupby(keys, sort=False)
               .agg(value_mean=('value', 'mean'), value_sd=('value', 'std'),
                    bias_mean=('bias_vs_oracle', 'mean'), bias_sd=('bias_vs_oracle', 'std'),
                    abs_bias_mean=('bias_vs_oracle', lambda b: b.abs().mean()), n_seeds=('seed', 'nunique'))
               .reset_index())
        agg.to_csv(os.path.join(out_dir, 'results_summary.csv'), index=False)
        text = agg.to_string(index=False, float_format=lambda x: f'{x:.3f}')
        with open(os.path.join(out_dir, 'results_summary.txt'), 'w') as f:
            f.write(text + '\n')
        log('\n' + text)
    with open(os.path.join(out_dir, 'diagnostics.json'), 'w') as f:
        json.dump(diag, f, indent=1)


if __name__ == '__main__':
    main()
