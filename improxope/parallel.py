"""Dependency-aware process pool spread over GPUs, with per-task result caching and status/ETA.

Each worker is a separate process pinned to one GPU (round-robin) and limited to one CPU thread,
so many small trainings run side by side. `cpu_workers` adds CPU-only workers (device 'cpu') to
the same pool; every task must then accept device='cpu'. A task runs once its dependencies have finished; its
result is pickled under `cache_dir`, so rerunning with the same cache skips finished tasks.
Priority (larger cost first) orders the ready queue, so long tasks start early.

After every finished task, `status.json` (next to `cache_dir`) records progress and an ETA. The
ETA uses the observed mean wall time per task kind (e.g. 'bridge'), falling back to the `cost`
estimates for kinds not yet seen: remaining work / workers, but never less than the longest
remaining single task. `scripts/status.py` prints it.
"""
from __future__ import annotations

import datetime as dt
import json
import multiprocessing as mp
import os
import pickle
import re
import time
import traceback
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

_DEVICE = 'cpu'


@dataclass
class Task:
    key: str
    fn: Callable
    kwargs: Dict = field(default_factory=dict)
    deps: Sequence[str] = ()
    make_kwargs: Optional[Callable[[Dict], Dict]] = None  # builds extra kwargs from dependency results
    cost: float = 1.0                                       # rough relative runtime, for priority and ETA
    kind: str = 'task'                                      # groups tasks with similar runtimes, for the ETA


def _init_worker(gpu_queue):
    global _DEVICE
    import torch
    torch.set_num_threads(1)
    gpu = gpu_queue.get()
    _DEVICE = f'cuda:{gpu}' if gpu is not None else 'cpu'


def _call(fn, kwargs):
    return fn(device=_DEVICE, **kwargs)


def _cache_path(cache_dir: str, key: str) -> str:
    return os.path.join(cache_dir, re.sub(r'[^A-Za-z0-9_.=-]+', '_', key) + '.pkl')


def _fmt(seconds: float) -> str:
    if seconds != seconds:
        return '?'
    seconds = int(max(seconds, 0))
    return f'{seconds // 3600}h{seconds % 3600 // 60:02d}m' if seconds >= 3600 else f'{seconds // 60}m{seconds % 60:02d}s'


def _now() -> str:
    return dt.datetime.now().isoformat(timespec='seconds')


class _Progress:
    """Tracks per-kind durations and writes status.json."""

    def __init__(self, tasks: List[Task], n_workers: int, path: str):
        self.by_key = {t.key: t for t in tasks}
        self.n_workers, self.path = n_workers, path
        self.durations = defaultdict(list)
        self.items = []            # (key, seconds) of finished tasks
        self.started = time.time()
        self.started_iso = _now()

    def estimate(self, task: Task) -> float:
        seen = self.durations.get(task.kind)
        if seen:
            return sum(seen) / len(seen)
        per_cost = [d / self.by_key[k].cost for k, d in self.items]
        return task.cost * (sum(per_cost) / len(per_cost) if per_cost else 60.0)

    def record(self, task: Task, seconds: float):
        self.durations[task.kind].append(seconds)
        self.items.append((task.key, seconds))

    def eta(self, pending: set, running: Dict[str, float]) -> float:
        now = time.time()
        rest = [self.estimate(self.by_key[k]) for k in pending]
        rest += [max(self.estimate(self.by_key[k]) - (now - t0), 0.0) for k, t0 in running.items()]
        if not rest:
            return 0.0
        return max(sum(rest) / self.n_workers, max(rest))

    def write(self, n_done: int, n_failed: int, pending: set, running: Dict[str, float]):
        elapsed = time.time() - self.started
        eta = self.eta(pending, running)
        kinds = defaultdict(lambda: {'done': 0, 'remaining': 0})
        for k, t in self.by_key.items():
            kinds[t.kind]['remaining' if (k in pending or k in running) else 'done'] += 1
        for kind, d in kinds.items():
            seen = self.durations.get(kind)
            d['mean_seconds'] = round(sum(seen) / len(seen), 1) if seen else None
        status = {
            'started': self.started_iso, 'updated': _now(), 'total': len(self.by_key), 'done': n_done,
            'failed': n_failed, 'running': len(running), 'elapsed_seconds': round(elapsed),
            'eta_seconds': round(eta), 'eta_finish': (dt.datetime.now() + dt.timedelta(seconds=eta)).isoformat(timespec='minutes'),
            'finished': not pending and not running, 'by_kind': dict(kinds),
        }
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(status, f, indent=1)
        os.replace(tmp, self.path)
        return elapsed, eta


def run_tasks(tasks: List[Task], n_workers: int, gpus: Sequence[int], cache_dir: str,
              log: Callable[[str], None] = print, cpu_workers: int = 0) -> Dict[str, object]:
    """Run all tasks; returns {key: result}. Failed tasks (and their dependents) are reported, not raised.
    n_workers are spread over `gpus` (or run on CPU if `gpus` is empty); cpu_workers are added on CPU."""
    os.makedirs(cache_dir, exist_ok=True)
    by_key = {t.key: t for t in tasks}
    assert len(by_key) == len(tasks), 'duplicate task keys'
    for t in tasks:
        missing = [d for d in t.deps if d not in by_key]
        assert not missing, f'{t.key} depends on unknown tasks {missing}'

    results: Dict[str, object] = {}
    for t in tasks:
        path = _cache_path(cache_dir, t.key)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                results[t.key] = pickle.load(f)
    if results:
        log(f'[cache] {len(results)} of {len(tasks)} tasks already done')

    n_total = n_workers + cpu_workers
    progress = _Progress(tasks, n_total, os.path.join(os.path.dirname(os.path.abspath(cache_dir)), 'status.json'))
    failed: Dict[str, str] = {}
    pending = {t.key for t in tasks if t.key not in results}
    running = {}          # future -> task
    started_at = {}       # key -> submit time
    progress.write(len(results), 0, pending, {})

    ctx = mp.get_context('spawn')
    manager = ctx.Manager()  # keep a reference: the queue proxy dies with the manager
    gpu_queue = manager.Queue()
    gpu_list = list(gpus) or [None]
    for i in range(n_workers):
        gpu_queue.put(gpu_list[i % len(gpu_list)])
    for _ in range(cpu_workers):
        gpu_queue.put(None)

    def ready():
        out = [by_key[k] for k in pending if all(d in results for d in by_key[k].deps)]
        return sorted(out, key=lambda t: -t.cost)

    def blocked(key):
        return any(d in failed for d in by_key[key].deps)

    with ProcessPoolExecutor(max_workers=n_total, mp_context=ctx,
                             initializer=_init_worker, initargs=(gpu_queue,)) as pool:
        while pending or running:
            for key in [k for k in pending if blocked(k)]:
                pending.discard(key)
                failed[key] = 'dependency failed'
                log(f'[skip] {key}: dependency failed')
            for t in ready():
                if len(running) >= n_total:
                    break
                kwargs = dict(t.kwargs)
                if t.make_kwargs is not None:
                    kwargs.update(t.make_kwargs(results))
                running[pool.submit(_call, t.fn, kwargs)] = t
                started_at[t.key] = time.time()
                pending.discard(t.key)
            if not running:
                break
            finished, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for fut in finished:
                t = running.pop(fut)
                wall = time.time() - started_at.pop(t.key)
                try:
                    res = fut.result()
                except Exception:
                    failed[t.key] = traceback.format_exc()
                    log(f'[FAIL] {t.key}\n{failed[t.key]}')
                    continue
                results[t.key] = res
                with open(_cache_path(cache_dir, t.key), 'wb') as f:
                    pickle.dump(res, f)
                progress.record(t, wall)
                elapsed, eta = progress.write(len(results), len(failed), pending,
                                              {r.key: started_at[r.key] for r in running.values()})
                log(f'[{len(results)}/{len(tasks)}] {t.key} ({_fmt(wall)}) | elapsed {_fmt(elapsed)} '
                    f'| ETA {_fmt(eta)}')
    manager.shutdown()
    progress.write(len(results), len(failed), pending, {})
    if pending:
        log(f'[done] {len(pending)} task(s) never became ready: {sorted(pending)}')
    if failed:
        log(f'[done] {len(failed)} task(s) failed or skipped: {sorted(failed)}')
    return results
