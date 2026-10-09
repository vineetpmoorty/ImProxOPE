# ImProxOPE

Code for **ImProxOPE**, an off-policy evaluation (OPE) estimator for the setting where rewards are
missing not at random (MNAR). The estimator combines a shadow-variable bridge b with a model of the
recording mechanism. The corrected reward score is

    U_t = b_t + M_t h_t (R_t - b_t),

where M_t = 1 if the reward is recorded and h_t estimates 1 / P(M_t = 1 | R_t, S_t, A_t). An
optional marginal doubly robust layer uses marginal state-action density ratios w_t:

    psi = V_1(X_1) + sum_t w_t(X_t, A_t) (U_t + V_{t+1}(X_{t+1}) - Q_t(X_t, A_t)).

"Corrected FQE" is the same estimator without the ratio layer (fitted Q-evaluation on U).

The repository contains the estimator, the experiments on MIMIC-III sepsis data, and the experiments
on the synthetic MNAR simulator of ShadOPE (Wei, Qu and Miao, ICML 2026), the closest prior method
and our main baseline. No data or results are included.

## Repository layout

```
improxope/                  Python package
  panel.py, data.py           data format and loaders (MIMIC, simulator); shadow-variable levels
  learners.py                 GMM learners: recording inverse h and marginal ratio w
  pipeline.py                 estimator tasks and cross-fitting designs (rotate, full, oof)
  experiment.py               task graphs per dataset, result summaries and diagnostics
  parallel.py                 process pool over GPUs/CPUs with a per-task result cache and ETA
  config.py                   Hydra configs (Python dataclasses, no YAML files)
  sim_ope.py                  ShadOPE's synthetic benchmark with ImProxOPE added
  nn_bridge.py, nn_fqe.py     from ShadOPE: neural bridge, neural estimators and baselines
  fqe.py, rkhs.py             from ShadOPE: kernel estimators and baselines
  sim/                        from ShadOPE: MNAR simulator, behaviour and target policies, true value
scripts/
  data/                       MIMIC-III: actions, T = 10 format, target policies, MNAR masks
  run_mimic.py                MIMIC experiments (ImProxOPE and six ShadOPE baselines)
  run_sim_grid.py             synthetic experiments on ShadOPE's simulator
  calibrate_sim.py            calibration of the simulator variants (shadow strength, missingness)
  shadow_strength.py          how predictive the shadow variable is at each strength level (MIMIC)
  shadow_organs.py            same, for every subset of hidden SOFA organ systems (MIMIC)
  run_synthetic.py            neural pipeline on ShadOPE's simulator (sanity check)
  status.py                   progress and ETA of running experiments
```

Files copied from ShadOPE (https://github.com/NAIVlab/ShadOPE, commit `4231ba5`) say so in their
module docstring, together with any changes made.

## Setup

Python 3.11. A CUDA GPU is needed for the MIMIC experiments (neural networks). The synthetic
experiments run on CPU only (see below).

```sh
python3.11 -m venv venv            # or: conda create --prefix ./venv python=3.11
source venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
pip install -e .
```

`requirements.txt` pins the exact versions we used (PyTorch 2.9.1 with CUDA 12.6). `pip install -e .`
alone installs compatible versions from `pyproject.toml`. All commands below run from the
repository root.

## MIMIC-III sepsis data

### Access

MIMIC-III v1.4 requires credentialed access on PhysioNet and its data use agreement
(https://physionet.org/content/mimiciii/1.4/). We do not distribute any data, derived tables or
patient-level outputs. Experiment outputs under `runs/` contain per-stay predictions, so treat them
as patient data as well.

### Step 1: cohort extraction

The sepsis cohort is extracted with Microsoft's code, https://github.com/microsoft/mimic_sepsis
(MIT license, commit `fce1d05`), from a PostgreSQL build of MIMIC-III
(https://github.com/MIT-LCP/mimic-code, `mimic-iii/buildmimic/postgres`). Follow that repository's
instructions (`preprocess.py`, then `sepsis_cohort.py --save_intermediate --process_raw`). It needs a
separate Python 3.8 environment. At commit `fce1d05` we had to make three small fixes:

1. `preprocess.py`: set the database host in the connection string to your server (for example
   `localhost`).
2. `sepsis_cohort.py`, line 1582: add the missing line continuation (`\`) in the SOFA sum. This
   keeps all six organ systems in SOFA. Un-indenting the next line instead would silently drop two
   of them.
3. `sepsis_cohort.py`, lines 1618 and 1624: remove the stray whitespace after the line
   continuations (syntax errors).

The extraction writes `MIMICtable.csv`. Copy it to `mimic_processed/` in this repository:

```sh
mkdir -p mimic_processed
cp /path/to/mimic_sepsis/MIMICtable.csv mimic_processed/
```

### Step 2: actions and T = 10 format

```sh
python scripts/data/prepare_shadope_input.py   # -> mimic_processed/sepsis_processed_state_action.csv
python scripts/data/clean_sepsis.py            # -> mimic_processed/sepsis_T10.csv, sepsis_T10_terminal.csv
```

| File | Contents |
|---|---|
| `sepsis_processed_state_action.csv` | `MIMICtable.csv` plus the action levels `vaso_input`, `iv_input` (0 to 4) |
| `sepsis_T10.csv` | ShadOPE's format: 48 state features, action, reward; 10 rows per ICU stay |
| `sepsis_T10_terminal.csv` | the 11th state of each stay (shadow variable of the last reward) |

The reward is SOFA_t - SOFA_{t+1}. Our extraction gives 18,914 sepsis stays, of which 12,945 have at
least 11 four-hour steps. ShadOPE reports 13,943 for the same filter. The extraction is sensitive to
the environment, and ShadOPE's input file is not public.

Departures from ShadOPE's data step:

- **Action levels.** ShadOPE's binning code is not public. We follow the Komorowski/Raghu rule as
  implemented in the extraction code: 0 = no drug in the 4-hour window, 1 to 4 = quartiles of the
  nonzero doses. The joint action is `vaso_input * 5 + iv_input` (25 actions).
- **Step numbering.** The extraction writes rows only for windows with recorded data, so some stays
  skip windows. Steps are renumbered 1 to 11 within each stay, so a step is the next recorded window.
  ShadOPE's evaluation code assumes consecutive labels.
- **Terminal state.** ShadOPE drops the 11th state, so the reward at t = 10 has no next-state shadow
  variable. We keep it in `sepsis_T10_terminal.csv`.

### Step 3: target policies and MNAR masks

```sh
# ShadOPE's Double DQN target policy (trained once on all stays, seed 42, CPU)
python scripts/data/train_target_policy.py --input mimic_processed/sepsis_T10.csv --outdir mimic_processed
# clinician-policy clone, for the epsilon-mixture targets (GPU, about a minute)
python scripts/data/train_clinician_policy.py
# ShadOPE's MNAR recording mechanism at 20/40/60/80% missing rewards, one mask per seed
for s in 42 43 44 45 46; do
  python scripts/data/apply_mnar.py --input mimic_processed/sepsis_T10_with_targets.csv \
    --outdir mimic_processed/mnar/seed$s --miss-rates 0.2,0.4,0.6,0.8 --seed $s
done
```

The target policy is held fixed. The seed sets the mask draw, the 60/40 fit/test split and the
network initializations.

## MIMIC experiments

`scripts/run_mimic.py` runs ImProxOPE and ShadOPE's six baselines (OracleFQE with fully observed
rewards, NaiveFQE, ImputeFQE, IPW-FQE, SCOPE, ProxFQE) on every (missing rate, seed) dataset. Every
field of `MimicConfig` in `improxope/config.py` can be overridden on the command line (Hydra). Set
`gpus` and `n_workers` for your machine (defaults: 4 GPUs, 20 worker processes). Finished tasks are
cached in `<run dir>/cache`, so rerunning a command with the same `hydra.run.dir` resumes it.
`python scripts/status.py <run dirs>` prints progress and an ETA.

Settings shared by all paper runs: ShadOPE's 60/40 split of ICU stays, network sizes and training
lengths, and ShadOPE's baselines run unchanged. Our nuisances use the `oof` design: the Q-function is
fitted on all fit stays, and bridge and recording scores are out-of-fold over 3 folds.

```sh
OOF=(design.name=oof design.n_folds=3 design.separate=false "seeds=[42,43,44,45,46]")

# 1. ShadOPE's benchmark (DQN target; no ratio layer, since the DQN target lacks coverage)
python scripts/run_mimic.py "${OOF[@]}" use_ratios=false hydra.run.dir=runs/mimic/oof_5seeds

# 2. Shadow-strength ladder: SOFA and some of its organ systems hidden from the shadow variable.
#    Baselines that never use the shadow variable are reused from run 1.
for level in no_sofa no_sofa_resp no_sofa_resp_neuro no_sofa_resp_neuro_renal no_sofa_components; do
  python scripts/run_mimic.py "${OOF[@]}" use_ratios=false shadow=$level \
    reuse_cache_from=runs/mimic/oof_5seeds hydra.run.dir=runs/mimic/shadow_$level
done

# 3. Epsilon-mixture targets pi_eps = (1 - eps) clinician clone + eps support-constrained DQN,
#    with the ratio layer. Bridge and recording fits do not depend on the target and are reused.
EPS=("target_eps=[0,0.25,0.5,1]" target_tau=0.05 use_ratios=true "baselines=[OracleFQE,NaiveFQE,SCOPE,ProxFQE]"
     "reuse_cache_pattern=[bridge,recording,baseline]")
python scripts/run_mimic.py "${OOF[@]}" "${EPS[@]}" reuse_cache_from=runs/mimic/oof_5seeds \
  hydra.run.dir=runs/mimic/eps_full
python scripts/run_mimic.py "${OOF[@]}" "${EPS[@]}" shadow=no_sofa \
  "reuse_cache_from=[runs/mimic/shadow_no_sofa,runs/mimic/eps_full]" hydra.run.dir=runs/mimic/eps_no_sofa

# 4. Cross-fitting designs (supplement): rotate (default), full, oof; seeds 42, 43, 44
python scripts/run_mimic.py hydra.run.dir=runs/mimic/rotate
python scripts/run_mimic.py design.name=full use_ratios=false \
  reuse_cache_from=runs/mimic/rotate hydra.run.dir=runs/mimic/full
python scripts/run_mimic.py design.name=oof design.n_folds=3 design.separate=false use_ratios=false \
  reuse_cache_from=runs/mimic/rotate hydra.run.dir=runs/mimic/oof

# Shadow strength: held-out R^2 of the reward from the shadow variable at each level
python scripts/shadow_strength.py   # -> results/shadow_strength.csv
python scripts/shadow_organs.py     # -> results/shadow_organs.csv
```

`reuse_cache_from` copies only cached results that are valid for the new run. It checks each source
run's saved config (`.hydra/config.yaml`): same shadow level, design and estimator settings, and from a
different shadow level only the baselines that never use the shadow variable. Reuse only avoids
refitting models whose inputs and settings are identical. Without it, every run fits everything itself.

Shadow levels (`SHADOW_DROP` in `improxope/data.py`) drop next-state columns from the shadow input of
the bridge and the recording model, and of ShadOPE's ProxFQE and IPW-FQE. Q-function targets keep the
full next state. With the full next state the bridge can read the reward off SOFA_{t+1}.

Outputs in each run directory:

- `results_by_seed.csv`, `results_summary.csv` / `.txt`: estimates, and bias relative to OracleFQE
  (the reference value, fitted with fully observed rewards), with mean and SD over seeds.
  Our rows are `Ours: corrected FQE`, `Ours: corrected MDR` and the bridge-only variants (h = 1).
- `diagnostics.json`: recording-inverse and ratio diagnostics, including the coverage quantiles of
  per-step action ratios, their products, and the marginal ratios.

## Synthetic experiments (ShadOPE's simulator)

`improxope/sim_ope.py` reproduces ShadOPE's synthetic benchmark (its simulator, data generation, true
value from 5000 target-policy rollouts, and its five kernel estimators with the settings of its
`scripts/eval_grid.py`). ImProxOPE there is ShadOPE's kernel ProxFQE with the corrected reward U in
place of M R + (1 - M) b, with the kernel bridge and our recording model cross-fitted over 3 folds
(`ours`; `ours_bridge` is the same with h = 1).

Simulator cells `beta:tau:kappa` (`0:0:0` is ShadOPE's simulator, unchanged):

- `beta`: the reward gets a component the next state does not reveal.
- `tau`: noise of a reading of that component, added to every bridge's shadow input. It sets the
  shadow's strength.
- `kappa`: recording probability curved in the reward, so our recording model is misspecified.

**Run the synthetic experiments on CPU only, one thread per worker.** The kernel estimators' float32
solves with ridge down to 1e-7 are sometimes badly wrong on GPUs, so `run_method` refuses GPU devices.

```sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# 1. Calibrate the shadow-strength ladder and the 2x2 misspecification cells
python scripts/calibrate_sim.py      # -> results/sim_levels.csv

# 2. ShadOPE's grid (n = 64..2048, 20/40/60/80% missing, seeds 321..370): ImProxOPE
python scripts/run_sim_grid.py "cells=[0:0:0]" "methods=[ours]" cpu_workers=32 \
  shadope_published=results/shadope_published/ope_runs_size_x_missrate.csv hydra.run.dir=runs/sim_grid/grid

# 3. Ladder and 2x2 cells at n = 1024: all methods
python scripts/run_sim_grid.py "cells=[calibrated]" "ns=[1024]" cpu_workers=32 hydra.run.dir=runs/sim_grid/cells1024

# 4. Spot check: ShadOPE's methods rerun on the grid next to its published values
python scripts/run_sim_grid.py "seeds=[321,322,323,324,325]" "cells=[0:0:0]" \
  "methods=[naive,prox,ipw,impute,scope,ours]" cpu_workers=32 \
  shadope_published=results/shadope_published/ope_runs_size_x_missrate.csv hydra.run.dir=runs/sim_grid/spot
```

ShadOPE's baselines on its own grid are taken from its released per-seed results:
`results/tables/ope_runs_size_x_missrate.csv` in the ShadOPE repository (commit `4231ba5`). Download
that file to `results/shadope_published/` before steps 2 and 4. Our port reproduces ShadOPE's datasets
and true values exactly, and the script checks the true values against the file. Without
`shadope_published`, step 2 reports ImProxOPE only. Add `"methods=[naive,prox,ipw,impute,scope,ours]"`
to rerun all baselines instead.

To spread the grid over machines, give each machine its own
`seeds=[...]`, copy the `cache/` directories into one run directory, and rerun the full command there
with `aggregate_only=true`, which reads the cache and computes nothing.

Outputs: `results_by_seed.csv`, `results_summary.csv` (bias, MAE and RMSE against the true value, per
cell, n, missing rate and estimator) and `diagnostics.json`.

## Acknowledgements

The MNAR simulator, the ShadOPE baselines, the DQN target policy and the MNAR masking procedure come
from ShadOPE (https://github.com/NAIVlab/ShadOPE). The sepsis cohort is extracted with
https://github.com/microsoft/mimic_sepsis.
