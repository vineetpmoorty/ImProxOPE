# ImProxOPE

Code for ImProxOPE, off-policy evaluation with rewards missing not at random.

## Setup

Python 3.11. The MIMIC experiments need a CUDA GPU. The synthetic experiments run on CPU.

```sh
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
pip install -e .
```

All commands below run from the repository root.

## MIMIC-III data

MIMIC-III v1.4 requires credentialed access on PhysioNet (https://physionet.org/content/mimiciii/1.4/).

### Step 1: cohort extraction

Extract the sepsis cohort with https://github.com/microsoft/mimic_sepsis (commit `fce1d05`) from a
PostgreSQL build of MIMIC-III (https://github.com/MIT-LCP/mimic-code, `mimic-iii/buildmimic/postgres`).
Follow that repository's instructions (`preprocess.py`, then
`sepsis_cohort.py --save_intermediate --process_raw`). At commit `fce1d05` three fixes are needed:

1. `preprocess.py`: set the database host in the connection string to your server.
2. `sepsis_cohort.py`, line 1582: add the missing line continuation (`\`) in the SOFA sum.
3. `sepsis_cohort.py`, lines 1618 and 1624: remove the whitespace after the line continuations.

Copy the resulting `MIMICtable.csv` to `mimic_processed/`.

### Step 2: target policies and MNAR masks

```sh
python scripts/data/train_target_policy.py --input mimic_processed/sepsis_T10.csv --outdir mimic_processed
python scripts/data/train_clinician_policy.py
for s in 42 43 44 45 46; do
  python scripts/data/apply_mnar.py --input mimic_processed/sepsis_T10_with_targets.csv \
    --outdir mimic_processed/mnar/seed$s --miss-rates 0.2,0.4,0.6,0.8 --seed $s
done
```

## MIMIC experiments

Set `gpus` and `n_workers` for your machine (defaults: 4 GPUs, 20 workers). Rerunning a command with
the same `hydra.run.dir` resumes it. `python scripts/status.py <run dirs>` shows progress.

```sh
OOF=(design.name=oof design.n_folds=3 design.separate=false "seeds=[42,43,44,45,46]")

# 1. Main benchmark
python scripts/run_mimic.py "${OOF[@]}" use_ratios=false hydra.run.dir=runs/mimic/oof_5seeds

# 2. Shadow-strength ladder
for level in no_sofa no_sofa_resp no_sofa_resp_neuro no_sofa_resp_neuro_renal no_sofa_components; do
  python scripts/run_mimic.py "${OOF[@]}" use_ratios=false shadow=$level \
    reuse_cache_from=runs/mimic/oof_5seeds hydra.run.dir=runs/mimic/shadow_$level
done

# 3. Epsilon-mixture target policies
EPS=("target_eps=[0,0.25,0.5,1]" target_tau=0.05 use_ratios=true "baselines=[OracleFQE,NaiveFQE,SCOPE,ProxFQE]"
     "reuse_cache_pattern=[bridge,recording,baseline]")
python scripts/run_mimic.py "${OOF[@]}" "${EPS[@]}" reuse_cache_from=runs/mimic/oof_5seeds \
  hydra.run.dir=runs/mimic/eps_full
python scripts/run_mimic.py "${OOF[@]}" "${EPS[@]}" shadow=no_sofa \
  "reuse_cache_from=[runs/mimic/shadow_no_sofa,runs/mimic/eps_full]" hydra.run.dir=runs/mimic/eps_no_sofa

# 4. Cross-fitting designs
python scripts/run_mimic.py hydra.run.dir=runs/mimic/rotate
python scripts/run_mimic.py design.name=full use_ratios=false \
  reuse_cache_from=runs/mimic/rotate hydra.run.dir=runs/mimic/full
python scripts/run_mimic.py design.name=oof design.n_folds=3 design.separate=false use_ratios=false \
  reuse_cache_from=runs/mimic/rotate hydra.run.dir=runs/mimic/oof

# 5. Shadow strength (R^2) per level
python scripts/shadow_strength.py
python scripts/shadow_organs.py
```

Each run directory gets `results_by_seed.csv`, `results_summary.csv` and `diagnostics.json`.

## Synthetic experiments

Run on CPU only, one thread per worker. Download `results/tables/ope_runs_size_x_missrate.csv` from
https://github.com/NAIVlab/ShadOPE (commit `4231ba5`) to `results/shadope_published/` first.

```sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# 1. Calibrate the simulator cells
python scripts/calibrate_sim.py

# 2. Grid: n = 64..2048, 20/40/60/80% missing, 50 seeds
python scripts/run_sim_grid.py "cells=[0:0:0]" "methods=[ours]" cpu_workers=32 \
  shadope_published=results/shadope_published/ope_runs_size_x_missrate.csv hydra.run.dir=runs/sim_grid/grid

# 3. Shadow-strength ladder and misspecification cells, n = 1024
python scripts/run_sim_grid.py "cells=[calibrated]" "ns=[1024]" cpu_workers=32 hydra.run.dir=runs/sim_grid/cells1024

# 4. Spot check of the baselines
python scripts/run_sim_grid.py "seeds=[321,322,323,324,325]" "cells=[0:0:0]" \
  "methods=[naive,prox,ipw,impute,scope,ours]" cpu_workers=32 \
  shadope_published=results/shadope_published/ope_runs_size_x_missrate.csv hydra.run.dir=runs/sim_grid/spot
```

To split a run over machines, give each its own `seeds=[...]`, copy the `cache/` directories into one
run directory, and rerun the full command there with `aggregate_only=true`.

## Acknowledgements

Files copied from https://github.com/NAIVlab/ShadOPE say so in their docstring.
