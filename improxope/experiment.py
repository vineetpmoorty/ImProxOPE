from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from improxope import pipeline as P
from improxope.parallel import Task

# Rough relative runtimes, used for scheduling priority (and the ETA until real durations exist).
# SCOPE: CPU-bound per-step loop, ~30 min per task with a stochastic target, so it is started early.
COST = {'bridge': 1.0, 'recording': 0.2, 'ratios': 1.5, 'q_eval': 1.5,
        'OracleFQE': 1.0, 'NaiveFQE': 1.0, 'ImputeFQE': 1.5, 'SCOPE': 5.0, 'IPW-FQE': 10.0, 'ProxFQE': 10.0}

# Evaluation variants that reuse the same fitted nuisances.
#   corrected: U = b + M h (R - b)  -> our estimator;   bridge: h = 1, U = M R + (1 - M) b
VARIANTS = ('corrected', 'bridge')
OWN_TARGET = ((None, None),)


def target_prefix(prefix: str, tag: Optional[str]) -> str:
    return prefix if tag is None else f'{prefix}/{tag}'


def corrected_tasks(prefix: str, source: Dict, est: Dict, design: Dict, horizon: int,
                    use_ratios: bool = True, targets: Sequence[Tuple] = OWN_TARGET) -> List[Task]:
    tasks: List[Task] = []
    for c, n_units in enumerate(P.design_shape(design)):
        base = dict(source=source, design=design, cfg=est, chain=c)
        bkeys = [[f'{prefix}/bridge/c{c}/u{u}/t{t + 1}' for t in range(horizon)] for u in range(n_units)]
        hkeys = [[f'{prefix}/recording/c{c}/u{u}/t{t + 1}' for t in range(horizon)] for u in range(n_units)]
        for u in range(n_units):
            for t in range(horizon):
                tasks.append(Task(bkeys[u][t], P.task_bridge, dict(base, unit=u, t=t),
                                  cost=COST['bridge'], kind='bridge'))
                tasks.append(Task(hkeys[u][t], P.task_recording, dict(base, unit=u, t=t),
                                  cost=COST['recording'], kind='recording'))
        for tag, target in targets:
            tp = target_prefix(prefix, tag)
            tbase = base if target is None else dict(base, target=target)
            tasks += _target_tasks(tp, tbase, c, bkeys, hkeys, use_ratios)
    return tasks


def _target_tasks(tp, tbase, c, bkeys, hkeys, use_ratios) -> List[Task]:
    """Ratio and Q/evaluation tasks of one chain for one target (keys under `tp`)."""
    tasks: List[Task] = []
    wkey = f'{tp}/ratios/c{c}'
    if use_ratios:
        tasks.append(Task(wkey, P.task_ratios, tbase, cost=COST['ratios'], kind='ratios'))
    for variant in VARIANTS:
        deps = [k for keys in bkeys for k in keys] + ([wkey] if use_ratios else [])
        if variant == 'corrected':
            deps += [k for keys in hkeys for k in keys]

        def make(results, bk=bkeys, hk=hkeys, wk=wkey, v=variant, ratios=use_ratios):
            def stack(keys, field):   # (n, T) for one unit
                return np.stack([results[k][field] for k in keys], axis=1)
            b_units = [stack(keys, 'score') for keys in bk]
            b_test = np.mean([stack(keys, 'test') for keys in bk], axis=0)
            if v == 'corrected':
                h_units = [stack(keys, 'score') for keys in hk]
                h_test = np.mean([stack(keys, 'test') for keys in hk], axis=0)
            else:
                h_units = [np.ones_like(b) for b in b_units]
                h_test = np.ones_like(b_test)
            w_test = results[wk]['test'] if ratios else np.zeros_like(b_test)
            return dict(b_units=b_units, h_units=h_units, b_test=b_test, h_test=h_test, w_test=w_test)

        tasks.append(Task(f'{tp}/q_eval/{variant}/c{c}', P.task_q_eval, tbase, deps=deps,
                          make_kwargs=make, cost=COST['q_eval'], kind='q_eval'))
    return tasks


def baseline_tasks(prefix: str, source: Dict, est: Dict, methods: List[str],
                   targets: Sequence[Tuple] = OWN_TARGET) -> List[Task]:
    tasks = []
    for tag, target in targets:
        kw = dict(source=source, cfg=est) if target is None else dict(source=source, cfg=est, target=target)
        tasks += [Task(f'{target_prefix(prefix, tag)}/baseline/{m}', P.task_baseline, dict(kw, method=m),
                       cost=COST[m], kind=f'baseline:{m}') for m in methods]
    return tasks


def summarize(prefix: str, results: Dict, design: Dict, methods: List[str], use_ratios: bool = True) -> List[Dict]:
    """One row per method: value and SE (SE across test stays, as ShadOPE reports)."""
    rows = []
    for m in methods:
        res = results.get(f'{prefix}/baseline/{m}')
        if res is not None:
            rows.append(dict(method=m, value=res['value'], se=res['se']))
    names = {'corrected': ('Ours: corrected MDR', 'Ours: corrected FQE'),
             'bridge': ('Bridge only (h = 1) MDR', 'Bridge only (h = 1) FQE')}
    n_chains = len(P.design_shape(design))
    for variant in VARIANTS:
        per_chain = [results.get(f'{prefix}/q_eval/{variant}/c{c}') for c in range(n_chains)]
        if any(x is None for x in per_chain):
            continue
        fields = (('psi', 'fqe') if use_ratios else ('fqe',))
        for field in fields:
            name = names[variant][0 if field == 'psi' else 1]
            per_stay = np.mean([x[field] for x in per_chain], axis=0)
            rows.append(dict(method=name, value=float(per_stay.mean()),
                             se=float(per_stay.std(ddof=1) / np.sqrt(len(per_stay)))))
    return rows


def diagnostics(prefix: str, results: Dict, design: Dict, horizon: int, tag: Optional[str] = None) -> Dict:
    """Recording-model diagnostics (dataset level) and ratio diagnostics (for the target `tag`)."""
    out = {}
    rec = [v for k, v in results.items() if k.startswith(f'{prefix}/recording/')]
    if rec:
        out['recording_cap_fraction_mean'] = float(np.mean([x['cap_fraction'] for x in rec]))
        out['recording_mean_h_observed'] = float(np.mean([x['mean_h_observed'] for x in rec]))
        out['recording_mean_Mh'] = float(np.mean([x['mean_Mh'] for x in rec]))  # ~1 if moments hold
    rat = [v for k, v in results.items() if k.startswith(f'{target_prefix(prefix, tag)}/ratios/')]
    if rat:
        out['ratio_cap_fraction_test'] = float(np.mean([x['cap_fraction'] for x in rat]))
        out['ratio_train_mean_by_t'] = np.mean([x['train_mean'] for x in rat], axis=0).round(3).tolist()
        for name in ('w_quantiles', 'action_ratio_quantiles', 'cum_action_ratio_quantiles'):
            if name in rat[0]:
                out[name] = rat[0][name]      # one chain in the oof and full designs
    return out
