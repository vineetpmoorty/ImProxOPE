from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

from hydra.core.config_store import ConfigStore

SHADOPE_HIDDEN = [512, 512, 256]


def _hidden():
    return list(SHADOPE_HIDDEN)


@dataclass
class BridgeConfig:
    """ShadOPE's NNBridge (min-max bridge), as used by its ProxFQE."""
    h_hidden: List[int] = field(default_factory=_hidden)
    f_hidden: List[int] = field(default_factory=_hidden)
    n_steps: int = 6000
    lr_h: float = 1e-3
    lr_f: float = 1e-3
    lambda_f: float = 0.1
    mu_h: float = 1e-4
    n_critic: int = 5
    batch_size: int = 4096


@dataclass
class RecordingConfig:
    """GMM recording inverse h = 1 + exp(-theta' phi) in [1, cap]. The cap of 20 matches
    ShadOPE's IPW weight cap."""
    cap: float = 20.0
    max_iter: int = 500
    ridge: float = 1e-6        # relative ridge on the instrument Gram matrix (collinear columns)


@dataclass
class BehaviorConfig:
    """ShadOPE's softmax logging-policy network (the one its SCOPE baseline uses). ShadOPE trains
    it for 8000 steps, which memorizes the training stays on MIMIC (held-out log-loss 6.3, worse
    than action frequencies' 2.3); 250 steps minimizes held-out log-loss (1.72, selected in
    scripts/data/train_clinician_policy.py). SCOPE itself keeps ShadOPE's settings."""
    hidden: List[int] = field(default_factory=_hidden)
    n_steps: int = 250
    lr: float = 1e-3
    batch_size: int = 4096


@dataclass
class RatioConfig:
    """Marginal ratio w_t = rho_t(x) pi(a|x) / mu(a|s) in [0, cap]; rho_t by GMM balance."""
    cap: float = 100.0
    max_iter: int = 500
    ridge: float = 1e-6
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)


@dataclass
class QConfig:
    """ShadOPE's NNQFunction (FQE regression)."""
    hidden: List[int] = field(default_factory=_hidden)
    n_steps: int = 8000
    lr: float = 1e-3
    batch_size: int = 4096


@dataclass
class ShadopeConfig:
    """Settings passed to ShadOPE's baselines exactly as in its run_realdata.sh."""
    bridge_steps: int = 6000
    q_steps: int = 8000
    batch_size: int = 4096
    q_hidden: List[int] = field(default_factory=_hidden)
    bridge_hidden: List[int] = field(default_factory=_hidden)
    aux_hidden: List[int] = field(default_factory=lambda: [256, 256])


@dataclass
class DesignConfig:
    """How the fit stays are split among the nuisance models (see improxope/pipeline.py).
    rotate: 3 parts with rotating roles (bridge / recording / Q on different parts)
    full:   every model on all fit stays (ShadOPE's data use)
    oof:    Q on all fit stays; bridge and recording scores out-of-fold over n_folds folds"""
    name: str = 'rotate'
    n_folds: int = 3
    separate: bool = True      # bridge and recording model on disjoint stays (rotate, oof)


@dataclass
class EstimatorConfig:
    bridge: BridgeConfig = field(default_factory=BridgeConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    ratios: RatioConfig = field(default_factory=RatioConfig)
    q: QConfig = field(default_factory=QConfig)
    shadope: ShadopeConfig = field(default_factory=ShadopeConfig)


@dataclass
class MimicConfig:
    data_dir: str = 'mimic_processed'
    miss_rates: List[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8])
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44])
    train_frac: float = 0.6               # ShadOPE's fit/test split
    terminal_fix: bool = True             # S_11 as the t = 10 shadow variable (all methods)
    shadow: str = 'full'                  # shadow-variable strength: a key of improxope.data.SHADOW_DROP
    design: DesignConfig = field(default_factory=DesignConfig)
    use_ratios: bool = True               # ratio layer (MDR); off where the target lacks coverage
    baselines: List[str] = field(default_factory=lambda: [
        'OracleFQE', 'NaiveFQE', 'ImputeFQE', 'IPW-FQE', 'SCOPE', 'ProxFQE'])
    run_ours: bool = True
    reuse_cache_from: Any = ''            # earlier run dir(s), str or list: copy compatible cached results
    reuse_cache_pattern: List[str] = field(default_factory=lambda: ['baseline'])  # substrings of cache file names
    target_eps: List[float] = field(default_factory=list)  # epsilon-mixture targets; empty: ShadOPE's DQN
    target_tau: float = 0.05              # support threshold of the mixture's DQN component
    n_workers: int = 20
    gpus: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    est: EstimatorConfig = field(default_factory=EstimatorConfig)


@dataclass
class SimConfig:
    """Validation on ShadOPE's synthetic MNAR environment (exact truth by target rollout).
    Defaults follow ShadOPE's run_simulation.sh: T = 8, intercepts giving ~20/40/60/80% missing,
    seeds from 321."""
    n_episodes: int = 2048
    horizon: int = 8
    mnar_c0s: List[float] = field(default_factory=lambda: [0.3, -0.7, -1.5, -2.8])
    reward_type: str = 'sigmoid'
    seeds: List[int] = field(default_factory=lambda: [321, 322, 323])
    n_true: int = 20000                   # rollouts for the true value
    train_frac: float = 0.6
    design: DesignConfig = field(default_factory=DesignConfig)
    use_ratios: bool = True
    n_workers: int = 20
    gpus: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    est: EstimatorConfig = field(default_factory=lambda: EstimatorConfig(
        bridge=BridgeConfig(h_hidden=[128, 128], f_hidden=[128, 128], n_steps=2000, batch_size=512),
        ratios=RatioConfig(behavior=BehaviorConfig(hidden=[128, 128], n_steps=3000, batch_size=512)),
        q=QConfig(hidden=[128, 128], n_steps=3000, batch_size=512)))


@dataclass
class SimGridConfig:
    """ShadOPE's synthetic benchmark (improxope/sim_ope.py): its kernel estimators and ours on its
    simulator, on a grid of sample sizes x missingness x seeds, for one or more simulator cells.
    A cell is 'beta:tau:kappa' (hidden reward component; noise of its shadow reading; curved
    recording; see improxope/sim_ope.py). '0:0:0' is ShadOPE's simulator with its intercepts; other
    cells take their intercepts (c0 per missingness target) from `levels_file`, written by
    scripts/calibrate_sim.py."""
    horizon: int = 8
    ns: List[int] = field(default_factory=lambda: [64, 128, 256, 512, 1024, 2048])
    miss_targets: List[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8])
    seeds: List[int] = field(default_factory=lambda: list(range(321, 371)))      # ShadOPE: 321..370
    cells: List[str] = field(default_factory=lambda: ['0:0:0'])
    methods: List[str] = field(default_factory=lambda: ['naive', 'prox', 'ipw', 'impute', 'scope', 'ours'])
    levels_file: str = 'results/sim_levels.csv'
    shadope_published: str = ''           # ShadOPE's released per-seed results, added for cell 0:0:0
    aggregate_only: bool = False          # only read cached results (merged runs); never compute
    n_folds: int = 3                      # cross-fitting folds for our bridge and recording model
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    truth_n_eval: int = 5000              # ShadOPE's eval_grid.py
    n_workers: int = 0                    # GPU workers: must stay 0 (kernel solves are unreliable on GPU)
    cpu_workers: int = 32                 # CPU-only workers (one thread each)
    gpus: List[int] = field(default_factory=list)


def register() -> None:
    cs = ConfigStore.instance()
    cs.store(name='mimic', node=MimicConfig)
    cs.store(name='sim', node=SimConfig)
    cs.store(name='sim_grid', node=SimGridConfig)
