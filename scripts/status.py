"""Print progress and ETA of running (or finished) runs from their status.json files.

    python scripts/status.py                         # every run under runs/
    python scripts/status.py runs/mimic/rerun_full   # specific run directories
    watch -n 60 python scripts/status.py             # refresh every minute
"""
import datetime as dt
import glob
import json
import os
import sys


def fmt(seconds):
    seconds = int(max(seconds or 0, 0))
    return f'{seconds // 3600}h{seconds % 3600 // 60:02d}m' if seconds >= 3600 else f'{seconds // 60}m{seconds % 60:02d}s'


def show(run_dir):
    path = os.path.join(run_dir, 'status.json')
    with open(path) as f:
        s = json.load(f)
    age = (dt.datetime.now() - dt.datetime.fromisoformat(s['updated'])).total_seconds()
    if s['finished']:
        state = 'FINISHED'
    elif age > 3 * 3600:
        state = f'STALE? (no update for {fmt(age)})'
    else:
        state = 'running'
    print(f"{run_dir}: {state}")
    print(f"  {s['done']}/{s['total']} done, {s['running']} running, {s['failed']} failed | "
          f"elapsed {fmt(s['elapsed_seconds'])}"
          + ('' if s['finished'] else f" | ETA {fmt(s['eta_seconds'])} (~{s['eta_finish'].replace('T', ' ')})"))
    print(f"  last update {s['updated'].replace('T', ' ')} ({fmt(age)} ago)")
    for kind, d in sorted(s['by_kind'].items()):
        mean = f"{d['mean_seconds']:.0f}s/task" if d['mean_seconds'] is not None else '-'
        print(f"    {kind:<22} {d['done']:>4} done {d['remaining']:>4} left   {mean}")
    if s['failed']:
        print('  failures: grep FAIL', os.path.join(run_dir, 'progress.log'))


def main():
    dirs = sys.argv[1:] or sorted({os.path.dirname(p) for p in glob.glob('runs/**/status.json', recursive=True)},
                                  key=lambda d: os.path.getmtime(os.path.join(d, 'status.json')))
    if not dirs:
        print('no status.json found')
    for d in dirs:
        show(d.rstrip('/'))
        print()


if __name__ == '__main__':
    main()
