"""Adapted from ShadOPE (Wei, Qu, Miao, ICML 2026), https://github.com/NAIVlab/ShadOPE @ 4231ba5, file src/OPE/fqe.py."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Iterable, List

import math
import numpy as np
import torch
from torch import Tensor


from improxope.rkhs import RKHSIVCV, ApproxRKHSIVCV, pairwise_rbf, Scaler, _compute_M_from_Kf



def _to32(x: Tensor) -> Tensor:
    return x.to(dtype=torch.float32, copy=False)

def _stack_cols(*cols: Tensor) -> Tensor:
    cols = [c if c.ndim == 2 else c.reshape(-1, 1) for c in cols]
    return torch.cat(cols, dim=1)

def _median_bandwidth(X: Tensor) -> float:
    '''Median heuristic bandwidth: 1 / (d * median(||x-x'||^2)).'''
    X = _to32(X)
    D2 = torch.cdist(X, X).pow(2)
    i, j = torch.triu_indices(D2.size(0), D2.size(0), offset=1)
    med = torch.median(D2[i, j])
    med = torch.clamp(med, min=torch.tensor(1e-12, dtype=X.dtype, device=X.device))
    return float((1.0 / (X.shape[1] * med)).cpu())


# ------------------------------- dataset helper -----------------------------

def group_by_t(dataset: Dict[str, np.ndarray]) -> Dict[int, Dict[str, Tensor]]:
    r'''
    Convert a flat dataset dict (from `collect_episodes`) into a step-indexed
    dictionary for FQE.

    Per t we return:
      - S:     S_t (n_t, 2)
      - A:     A_t (n_t, 1)
      - R:     R_obs,t (n_t, 1)
      - O:     O_t (n_t, 1)       -- missingness mask; ONLY for filtering
      - Sp:    S_{t+1} (n_t, 2)
      - Oprev: O_{t-1} (n_t, 1)   -- from obs[:,2]
      - Onext: O_t (n_t, 1)       -- from obs_n[:,2], for V_{t+1}^\pi(S_{t+1}, O_t)
    '''
    obs   = np.asarray(dataset["obs"],   dtype=np.float32)
    a     = np.asarray(dataset["a"],     dtype=np.int8)
    o     = np.asarray(dataset["o"],     dtype=np.int8)
    r_obs = np.asarray(dataset["r_obs"], dtype=np.float32)
    obs_n = np.asarray(dataset["obs_n"], dtype=np.float32)
    t_idx = np.asarray(dataset["t"],     dtype=np.int16)

    w_read = np.asarray(dataset["w"], dtype=np.float32) if "w" in dataset else None

    T = int(t_idx.max())
    out : Dict[int, Dict[str, Tensor]] = {}
    for t in range(1, T+1):
        m = (t_idx == t)
        S    = torch.from_numpy(obs[m, :2]).to(torch.float32)
        A    = torch.from_numpy(a[m].astype(np.float32)).reshape(-1, 1)
        Opr  = torch.from_numpy(obs[m, 2].astype(np.float32)).reshape(-1, 1)   # O_{t-1}
        Ocur = torch.from_numpy(o[m].astype(np.float32)).reshape(-1, 1)        # O_t (mask)
        R    = torch.from_numpy(r_obs[m]).reshape(-1, 1)
        Sp   = torch.from_numpy(obs_n[m, :2]).to(torch.float32)                # S_{t+1}
        Onxt = torch.from_numpy(obs_n[m, 2].astype(np.float32)).reshape(-1, 1) # O_t for next step

        W    = Sp if w_read is None else torch.cat([Sp, torch.from_numpy(w_read[m]).reshape(-1, 1)], dim=1)

        out[t] = {"S": S, "A": A, "R": R, "O": Ocur, "Sp": Sp,
                  "Ocur": Ocur, "Oprev": Opr, "Onext": Onxt, "W": W}
    return out

# --------------------------- Kernel Ridge with CV ---------------------------

class KernelRidgeCV:
    '''
    RBF Kernel ridge with CV.
    '''
    def __init__(self,
                 lam_grid: Optional[Iterable[float]] = None,
                 folds: int = 5,
                 device: str = 'cuda',
                 scale: bool = True):
        self.lam_grid = (list(np.geomspace(1e-6, 5e-3, 12))
                         if lam_grid is None else list(lam_grid))
        self.folds = int(folds)
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.scale = bool(scale)

    def fit(self, X: Tensor, y: Tensor):
        X = _to32(X.to(self.device))
        y = _to32(y.to(self.device)).reshape(-1, 1)

        if self.scale:
            self.scaler_ = Scaler()
            Xs = self.scaler_.fit_transform(X)
        else:
            self.scaler_ = None
            Xs = X

        self.gamma_ = _median_bandwidth(Xs)

        n = Xs.shape[0]
        idx = torch.randperm(n, device=self.device)
        folds = torch.chunk(idx, self.folds)

        best_lam, best_loss = None, float('inf')
        K_full = pairwise_rbf(Xs, gamma=self.gamma_)

        for lam in self.lam_grid:
            lam = float(lam)
            losses = []
            for k in range(self.folds):
                val = folds[k]
                tr  = torch.cat([folds[j] for j in range(self.folds) if j != k], dim=0)

                K_tr  = K_full[tr][:, tr].clone()
                K_tr += lam * torch.eye(K_tr.shape[0], dtype=K_tr.dtype, device=K_tr.device)

                try:
                    alpha = torch.linalg.solve(K_tr, y[tr])
                except RuntimeError:
                    alpha = torch.linalg.lstsq(K_tr, y[tr]).solution

                K_val = K_full[val][:, tr]
                pred  = K_val @ alpha
                mse   = torch.mean((pred - y[val])**2).item()
                losses.append(mse)

            m = float(np.mean(losses)) if len(losses) else float('inf')
            if m < best_loss:
                best_loss, best_lam = m, lam

        K = K_full + best_lam * torch.eye(n, dtype=K_full.dtype, device=K_full.device)
        try:
            alpha = torch.linalg.solve(K, y)
        except RuntimeError:
            alpha = torch.linalg.lstsq(K, y).solution
            
        self.alpha_ = alpha
        self.Xs_ = Xs
        self.KXX_ = K_full # Store the best K (without regularization)
        return self


    def predict(self, X: Tensor) -> Tensor:
        X = _to32(X.to(self.device))
        Xs = self.scaler_.transform(X) if getattr(self, "scaler_", None) is not None else X
        Kx = pairwise_rbf(Xs, self.Xs_, gamma=self.gamma_)
        return Kx @ self.alpha_
    

# ===== Weighted Kernel Ridge with CV (for baseline IPW-FQE) =====

class WeightedKernelRidgeCV:
    '''
    Kernel Ridge Regression with diagonal sample weights (for IPW-FQE).
    '''
    def __init__(self, 
                 lam_grid=None, 
                 folds=5, 
                 device='cuda', 
                 scale=True, 
                 normalize_w=True):
        self.lam_grid = (list(np.geomspace(1e-6, 1e-1, 12))
                         if lam_grid is None else list(lam_grid))
        self.folds = int(folds)
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.scale = bool(scale)
        self.normalize_w = bool(normalize_w)

    def fit(self, X: Tensor, y: Tensor, sample_weight: Optional[Tensor] = None):
        X = _to32(X.to(self.device))
        y = _to32(y.to(self.device)).reshape(-1, 1)

        if sample_weight is None:
            raise ValueError("WeightedKernelRidgeCV.fit requires sample_weight (1D or (n,1)).")
        w = _to32(sample_weight.to(self.device)).reshape(-1)
        if self.normalize_w:
            # stabilize variance: make E[w]≈1 globally (before CV splits)
            m = torch.mean(w)
            w = w / torch.clamp(m, min=torch.tensor(1e-12, dtype=w.dtype, device=w.device))

        if self.scale:
            self.scaler_ = Scaler()
            Xs = self.scaler_.fit_transform(X)
        else:
            self.scaler_ = None
            Xs = X

        self.gamma_ = _median_bandwidth(Xs)
        K_full = pairwise_rbf(Xs, gamma=self.gamma_)

        n = Xs.shape[0]
        idx = torch.randperm(n, device=self.device)
        folds = torch.chunk(idx, self.folds)

        best_lam, best_loss = None, float('inf')

        for lam in self.lam_grid:
            lam = float(lam)
            fold_losses = []
            for k in range(self.folds):
                val = folds[k]
                tr  = torch.cat([folds[j] for j in range(self.folds) if j != k], dim=0)

                Ktr  = K_full[tr][:, tr]
                ytr  = y[tr]
                wtr  = w[tr]
                KWtr = Ktr * wtr.view(-1, 1)
                A = Ktr @ KWtr + lam * Ktr 
                b = Ktr @ (wtr.view(-1, 1) * ytr) 
                try:
                    alpha_tr = torch.linalg.solve(A, b)
                except RuntimeError:
                    alpha_tr = torch.linalg.lstsq(A, b).solution

                Kval = K_full[val][:, tr]
                pred = Kval @ alpha_tr
                wval = w[val]
                resid = pred.reshape(-1) - y[val].reshape(-1)
                denom = torch.clamp(wval.sum(), min=torch.tensor(1e-12, dtype=wval.dtype, device=wval.device))
                wmse  = (wval * resid.pow(2)).sum() / denom
                fold_losses.append(float(wmse.item()))

            avg = float(np.mean(fold_losses)) if len(fold_losses) else float('inf')
            
            if avg < best_loss:
                best_loss = avg
                best_lam = lam

        self.lam_ = float(best_lam)
        
        # Use the stored best Gram matrix
        KW = K_full * w.view(-1, 1)
        A = K_full @ KW + self.lam_ * K_full
        b = K_full @ (w.view(-1, 1) * y)
        try:
            alpha = torch.linalg.solve(A, b)
        except RuntimeError:
            alpha = torch.linalg.lstsq(A, b).solution

        self.alpha_ = alpha
        self.Xs_ = Xs
        self.KXX_ = K_full # Store the best K
        return self

    def predict(self, X: Tensor) -> Tensor:
        X = _to32(X.to(self.device))
        Xs = self.scaler_.transform(X) if getattr(self, "scaler_", None) is not None else X
        Kx = pairwise_rbf(Xs, self.Xs_, gamma=self.gamma_)
        return Kx @ self.alpha_


# ----------------------------- Bridge estimator ----------------------------

@dataclass
class BridgeCVConfig:
    '''Hyperparameters passed to RKHSIVCV/ApproxRKHSIVCV.'''
    use_nystroem: bool = False
    n_components: int = 64           # only used if use_nystroem=True
    gamma_f: str | float = 'auto'    # median heuristic on (R,S,A)
    gamma_hs: str | Iterable[float] = 'auto'  # distance-quantile grid on (S',S,A)
    n_gamma_hs: int = 15

    delta_scale: float = 5.0
    delta_exp: float = 0.4
    alpha_scales: Iterable[float] | str = 'auto'
    n_alphas: int = 16
    cv: int = 5
    device: str = 'cuda'

    def make_estimator(self):
        if self.use_nystroem:
            est = ApproxRKHSIVCV(
                n_components=self.n_components,
                gamma_f=self.gamma_f,
                gamma_hs=self.gamma_hs,
                n_gamma_hs=self.n_gamma_hs,
                delta_scale=self.delta_scale,
                delta_exp=self.delta_exp,
                alpha_scales=self.alpha_scales,
                n_alphas=self.n_alphas,
                cv=self.cv,
                device=self.device,
            )
        else:
            est = RKHSIVCV(
                gamma_f=self.gamma_f,
                gamma_hs=self.gamma_hs,
                n_gamma_hs=self.n_gamma_hs,
                delta_scale=self.delta_scale,
                delta_exp=self.delta_exp,
                alpha_scales=self.alpha_scales,
                n_alphas=self.n_alphas,
                cv=self.cv,
                device=self.device,
            )
        if self.alpha_scales == 'auto':
            est.alpha_scales = [float(x) for x in np.geomspace(0.02, 2.0, self.n_alphas)]
        return est


# ----------------------------- Policy helper -------------------------------

def _policy_probs_target(pi, S: Tensor, Ominus: Tensor) -> Tensor:
    """
    Return a (n,2) tensor with columns [P(a=-1), P(a=+1)] using
    the target policy's batch interface.
    """
    device = S.device
    # Let the policy handle dtype / device conversion internally
    p_plus = pi.prob_a_plus_batch(S, Ominus)  # shape (n,)
    p_plus = torch.clamp(p_plus, 0.0, 1.0)
    p_minus = 1.0 - p_plus
    return torch.stack([p_minus, p_plus], dim=1).to(device=device)


# ------------------------------- Proximal FQE ------------------------------

class ProxFQE:
    '''
    Proximal FQE with an RKHS minimax bridge to impute missing rewards.
    '''
    def __init__(self,
                 action_list: Iterable[int] = (-1, +1),
                 gamma: float = 1.0,
                 bridge_cv_kwargs: Optional[Dict] = None,
                 krr_kwargs: Optional[Dict] = None,
                 device: str = 'cuda'):
        self.action_list = list(action_list)
        self.gamma = float(gamma)
        self.device = device if torch.cuda.is_available() else 'cpu'

        self.bridge_cfg = BridgeCVConfig(device=self.device)
        if bridge_cv_kwargs:
            for k, v in bridge_cv_kwargs.items():
                setattr(self.bridge_cfg, k, v)

        self.krr_kwargs = dict(lam_grid=None, folds=5, device=self.device)
        if krr_kwargs:
            self.krr_kwargs.update(krr_kwargs)

        self.Q: Dict[int, KernelRidgeCV] = {}
        self.V: Dict[int, callable] = {}
        self.q: Dict[int, object] = {}
        self.beta: Dict[int, float] = {}

    def _penalize(self, q_hat: Tensor, beta_t: float) -> Tensor:
        return q_hat

    def fit(self, dataset: Dict[str, np.ndarray], target_policy) -> "ProxFQE":
        by_t = group_by_t(dataset)
        T = max(by_t.keys())

        next_Q: Optional[KernelRidgeCV] = None

        for t in reversed(range(1, T+1)):
            pack = by_t[t]
            S  = _to32(pack["S"]).to(self.device)
            A  = _to32(pack["A"]).to(self.device)
            R  = _to32(pack["R"]).to(self.device).reshape(-1)
            O  = _to32(pack["O"]).to(self.device).reshape(-1)
            Sp = _to32(pack["Sp"]).to(self.device)
            W  = _to32(pack["W"]).to(self.device)        # shadow: S' (and the reading w, if any)

            mask = (O == 1.0)
            if mask.sum() <= 3:
                raise RuntimeError(f"[FQE] Too few O=1 samples at t={t}.")

            S1, A1, R1, Sp1 = S[mask], A[mask], R[mask], W[mask]
            XH = _stack_cols(Sp1, S1, A1)                 # [S', S, A]
            XF = _stack_cols(R1.reshape(-1,1), S1, A1)    # [R, S, A]
            bridge = self.bridge_cfg.make_estimator().fit(XH, R1.reshape(-1,1), XF)
            self.q[t] = bridge

            XH_all = _stack_cols(W, S, A)
            q_hat_all = bridge.predict(XH_all).reshape(-1)
            beta_t = 0.0
            q_impute = self._penalize(q_hat_all, beta_t)

            R_tilde = torch.where(O == 1.0, R, q_impute)

            if next_Q is None:
                y = R_tilde
            else:
                On = _to32(pack["Onext"]).to(self.device).reshape(-1)
                probs = _policy_probs_target(target_policy, Sp, On)
                v_next = torch.zeros_like(R_tilde)
                for j, a in enumerate(self.action_list):
                    a_col = torch.full((Sp.shape[0], 1), float(a),
                                       dtype=torch.float32, device=self.device)
                    Xa = _stack_cols(Sp, a_col)
                    v_next += probs[:, j] * next_Q.predict(Xa).reshape(-1)
                y = R_tilde + self.gamma * v_next

            Q_t = KernelRidgeCV(**self.krr_kwargs).fit(_stack_cols(S, A), y.reshape(-1,1))
            self.Q[t] = Q_t

            def V_of(S_in: Tensor, Ominus_in: Tensor, Q=Q_t):
                probs = _policy_probs_target(target_policy, S_in, Ominus_in)
                out = torch.zeros(S_in.shape[0], dtype=torch.float32, device=self.device)
                for j, a in enumerate(self.action_list):
                    Xa = _stack_cols(S_in,
                                     torch.full((S_in.shape[0], 1), float(a),
                                                dtype=torch.float32, device=self.device))
                    out += probs[:, j] * Q.predict(Xa).reshape(-1)
                return out

            self.V[t] = V_of
            next_Q = Q_t

        self.V1 = self.V[1]
        return self

    def value(self, S1: np.ndarray, O0: Optional[np.ndarray] = None) -> float:
        S1_t = torch.as_tensor(S1, dtype=torch.float32, device=self.device)
        if O0 is None:
            O0_t = torch.zeros(S1_t.shape[0], dtype=torch.float32, device=self.device)
        else:
            O0_t = torch.as_tensor(O0, dtype=torch.float32, device=self.device).reshape(-1)
        return float(self.V1(S1_t, O0_t).mean().item())



# --------------------------- baseline: Naive FQE (MNAR-blind) ---------------------------

class NaiveFQE:
    r'''
    Naive FQE baseline that drops missing-reward transitions.
    Trains on O_t=1 only:
        y_t = R_t + gamma * V_{t+1}^\pi(S_{t+1}, O_t=1)
    which estimates E[R_t | s,a,O_t=1] and is biased under MNAR.

    Parameters
    ----------
    action_list : iterable of actions (e.g., [-1, +1])
    gamma       : discount factor
    krr_kwargs  : kwargs for KernelRidgeCV (lam_grid, folds, device)
    device      : 'cuda' or 'cpu'
    '''
    def __init__(self,
                 action_list=(-1, +1),
                 gamma: float = 1.0,
                 krr_kwargs=None,
                 device: str = 'cuda'):
        self.action_list = list(action_list)
        self.gamma = float(gamma)
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.krr_kwargs = dict(lam_grid=None, folds=5, device=self.device)
        if krr_kwargs:
            self.krr_kwargs.update(krr_kwargs)

        self.Q = {}
        self.V = {}

    def fit(self, dataset: Dict[str, np.ndarray], target_policy) -> "NaiveFQE":
        '''
        dataset: flat dict from data collection. We convert via group_by_t(.).
        target_policy: object exposing method `prob_a_plus(s, o_prev) -> float`
                       used by _policy_probs_target.
        '''
        by_t = group_by_t(dataset)
        T = max(by_t.keys())
        next_Q = None

        for t in reversed(range(1, T+1)):
            pack = by_t[t]
            S  = _to32(pack["S"]).to(self.device)
            A  = _to32(pack["A"]).to(self.device)
            R  = _to32(pack["R"]).to(self.device).reshape(-1)
            O  = _to32(pack["O"]).to(self.device).reshape(-1)
            Sp = _to32(pack["Sp"]).to(self.device)

            # Keep only O_t = 1 samples
            On_all = _to32(pack["Onext"]).to(self.device).reshape(-1)
            mask = (O == 1.0)
            S1, A1, R1, Sp1 = S[mask], A[mask], R[mask], Sp[mask]
            On1 = On_all[mask]

            if next_Q is None:
                y = R1
            else:
                probs = _policy_probs_target(target_policy, Sp1, On1)
                v_next = torch.zeros_like(R1)
                for j, a in enumerate(self.action_list):
                    a_col = torch.full((Sp1.shape[0], 1), float(a),
                                    dtype=torch.float32, device=self.device)
                    Xa = _stack_cols(Sp1, a_col)
                    v_next += probs[:, j] * next_Q.predict(Xa).reshape(-1)
                y = R1 + self.gamma * v_next

            # Fit Q_t on (S_t, A_t) -> y, using only O=1
            X = _stack_cols(S1, A1)
            Q_t = KernelRidgeCV(**self.krr_kwargs).fit(X, y.reshape(-1, 1))
            self.Q[t] = Q_t

            # Define V_t(S_t, O_{t-1}) via target policy and Q_t
            def V_of(S_in: Tensor, Ominus_in: Tensor, Q=Q_t):
                probs = _policy_probs_target(target_policy, S_in, Ominus_in)
                out = torch.zeros(S_in.shape[0], dtype=torch.float32, device=self.device)
                for j, a in enumerate(self.action_list):
                    Xa = _stack_cols(
                        S_in,
                        torch.full((S_in.shape[0], 1), float(a),
                                   dtype=torch.float32, device=self.device)
                    )
                    out += probs[:, j] * Q.predict(Xa).reshape(-1)
                return out

            self.V[t] = V_of
            next_Q = Q_t

        self.V1 = self.V[1]
        return self

    def value(self, S1: np.ndarray, O0: Optional[np.ndarray] = None) -> float:
        r'''
        Estimate V(pi) = E[ V_1^\pi(S1, O0) ].
        When O0 is None, use zeros.
        '''
        S1_t = torch.as_tensor(S1, dtype=torch.float32, device=self.device)
        if O0 is None:
            O0_t = torch.zeros(S1_t.shape[0], dtype=torch.float32, device=self.device)
        else:
            O0_t = torch.as_tensor(O0, dtype=torch.float32, device=self.device).reshape(-1)
        return float(self.V1(S1_t, O0_t).mean().item())
    


# --------------------------- baseline: Impute-then-FQE ---------------------------

class ImputeFQE:
    r'''
    Impute-then-FQE baseline.

    Step 1: On O_t=1 samples, fit a KRR regression  R_t ~ f(S_t, A_t)
            to learn  E[R_t | S_t, A_t, O_t=1].
    Step 2: Impute missing rewards:
            R_tilde = O_t * R_t + (1 - O_t) * f_hat(S_t, A_t).
    Step 3: Run standard FQE on ALL samples using R_tilde.

    This is biased under MNAR because the regression learns the
    observed-conditional mean E[R|S,A,O=1] instead of E[R|S,A].

    Parameters
    ----------
    action_list : iterable of actions (e.g., [-1, +1])
    gamma       : discount factor
    krr_kwargs  : kwargs for KernelRidgeCV (lam_grid, folds, device)
    device      : 'cuda' or 'cpu'
    '''
    def __init__(self,
                 action_list=(-1, +1),
                 gamma: float = 1.0,
                 krr_kwargs=None,
                 device: str = 'cuda'):
        self.action_list = list(action_list)
        self.gamma = float(gamma)
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.krr_kwargs = dict(lam_grid=None, folds=5, device=self.device)
        if krr_kwargs:
            self.krr_kwargs.update(krr_kwargs)

        self.Q = {}
        self.V = {}
        self.reward_model = {}  # t -> KernelRidgeCV for R_hat

    def fit(self, dataset: Dict[str, np.ndarray], target_policy) -> "ImputeFQE":
        by_t = group_by_t(dataset)
        T = max(by_t.keys())
        next_Q = None

        for t in reversed(range(1, T + 1)):
            pack = by_t[t]
            S  = _to32(pack["S"]).to(self.device)
            A  = _to32(pack["A"]).to(self.device)
            R  = _to32(pack["R"]).to(self.device).reshape(-1)
            O  = _to32(pack["O"]).to(self.device).reshape(-1)
            Sp = _to32(pack["Sp"]).to(self.device)

            mask = (O == 1.0)
            if mask.sum() <= 3:
                raise RuntimeError(f"[ImputeFQE] Too few O=1 samples at t={t}.")

            # ---- Step 1: fit reward model on O=1 ----
            X_obs = _stack_cols(S[mask], A[mask])       # (n_obs, 3)
            R_obs = R[mask]                              # (n_obs,)
            r_model = KernelRidgeCV(**self.krr_kwargs).fit(X_obs, R_obs.reshape(-1, 1))
            self.reward_model[t] = r_model

            # ---- Step 2: impute missing rewards ----
            X_all = _stack_cols(S, A)
            R_hat = r_model.predict(X_all).reshape(-1)
            R_tilde = torch.where(O == 1.0, R, R_hat)

            # ---- Step 3: standard FQE Bellman target ----
            On_all = _to32(pack["Onext"]).to(self.device).reshape(-1)
            if next_Q is None:
                y = R_tilde
            else:
                probs = _policy_probs_target(target_policy, Sp, On_all)
                v_next = torch.zeros_like(R_tilde)
                for j, a in enumerate(self.action_list):
                    a_col = torch.full((Sp.shape[0], 1), float(a),
                                       dtype=torch.float32, device=self.device)
                    Xa = _stack_cols(Sp, a_col)
                    v_next += probs[:, j] * next_Q.predict(Xa).reshape(-1)
                y = R_tilde + self.gamma * v_next

            # Fit Q_t on ALL samples
            Q_t = KernelRidgeCV(**self.krr_kwargs).fit(X_all, y.reshape(-1, 1))
            self.Q[t] = Q_t

            def V_of(S_in: Tensor, Ominus_in: Tensor, Q=Q_t):
                probs = _policy_probs_target(target_policy, S_in, Ominus_in)
                out = torch.zeros(S_in.shape[0], dtype=torch.float32, device=self.device)
                for j, a in enumerate(self.action_list):
                    Xa = _stack_cols(S_in,
                                     torch.full((S_in.shape[0], 1), float(a),
                                                dtype=torch.float32, device=self.device))
                    out += probs[:, j] * Q.predict(Xa).reshape(-1)
                return out

            self.V[t] = V_of
            next_Q = Q_t

        self.V1 = self.V[1]
        return self

    def value(self, S1: np.ndarray, O0: Optional[np.ndarray] = None) -> float:
        S1_t = torch.as_tensor(S1, dtype=torch.float32, device=self.device)
        if O0 is None:
            O0_t = torch.zeros(S1_t.shape[0], dtype=torch.float32, device=self.device)
        else:
            O0_t = torch.as_tensor(O0, dtype=torch.float32, device=self.device).reshape(-1)
        return float(self.V1(S1_t, O0_t).mean().item())


# logistic regression for extended propensity score lambda_t

class _LogitBinary:
    '''
    Tiny logistic regression with L2 penalty implemented in torch.
    Features are scaled outside (use Scaler like KRR).
    '''
    def __init__(self, l2: float = 1e-3, max_iter: int = 200, device: str = 'cpu'):
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.coef_ = None  # (d,)
        self.bias_ = None  # scalar

    def fit(self, X: Tensor, y: Tensor):
        X = _to32(X.to(self.device))           # (n,d)
        y = _to32(y.to(self.device)).reshape(-1)  # (n,)
        n, d = X.shape

        w = torch.zeros(d, dtype=torch.float32, device=self.device, requires_grad=True)
        b = torch.zeros((), dtype=torch.float32, device=self.device, requires_grad=True)

        opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=self.max_iter, line_search_fn='strong_wolfe')
        bce = torch.nn.BCEWithLogitsLoss(reduction='mean')

        l2 = self.l2

        def closure():
            opt.zero_grad()
            logits = X @ w + b
            loss = bce(logits, y) + 0.5 * l2 * (w @ w)   # no penalty on bias
            loss.backward()
            return loss

        opt.step(closure)
        self.coef_ = w.detach().clone()
        self.bias_ = b.detach().clone()
        return self

    def predict_proba(self, X: Tensor) -> Tensor:
        X = _to32(X.to(self.device))
        logits = X @ self.coef_ + self.bias_
        return torch.sigmoid(logits)


# ------------------------- baseline: IPW-FQE (Wang et al. 2025) -------------------------

class WeightedFQE:
    '''
    IPW-FQE baseline:
      - Estimate per-step lambda_t(S_t, A_t, R_t) via a quick logit on [S, A, q_hat],
        where q_hat is the RKHS bridge prediction of R_t using (S_{t+1}, S_t, A_t).
      - Train Q_t by weighted KRR on O_t=1 samples with weights w = 1 / lambda_hat.

      - Adapted from Wang et al. (2025) "Off-Policy Evaluation under Missing Not at Random Data".
    '''
    def __init__(self,
                 action_list=(-1, +1),
                 gamma: float = 1.0,
                 bridge_cv_kwargs=None,       # for RKHS bridge just to get q_hat
                 krr_kwargs=None,             # for weighted KRR
                 logit_l2: float = 1e-3,      # L2 for lambda-logit
                 logit_max_iter: int = 200,
                 pmin: float = 1e-2,          # clip lambda in [pmin, 1-pmin]
                 w_cap: float = 50.0,         # optional cap for 1/lambda
                 device: str = 'cuda'):
        self.action_list = list(action_list)
        self.gamma = float(gamma)
        self.device = device if torch.cuda.is_available() else 'cpu'

        # a small bridge just to compute q_hat; reuse your RKHS config
        self.bridge_cfg = BridgeCVConfig(device=self.device)
        if bridge_cv_kwargs:
            for k, v in bridge_cv_kwargs.items():
                setattr(self.bridge_cfg, k, v)
        # weighted KRR
        self.krr_kwargs = dict(lam_grid=None, folds=5, device=self.device)
        if krr_kwargs:
            self.krr_kwargs.update(krr_kwargs)

        self.logit_l2 = float(logit_l2)
        self.logit_max_iter = int(logit_max_iter)
        self.pmin = float(pmin)
        self.w_cap = float(w_cap)

        self.Q = {}
        self.V = {}
        self._scaler_lambda = {}  # t scaler for lambda features (S,A,qhat)

    def _policy_probs_target(self, pi, S: Tensor, Ominus: Tensor) -> Tensor:
        return _policy_probs_target(pi, S, Ominus)

    def _fit_lambda_and_weights(self, S: Tensor, A: Tensor, Sp: Tensor, O: Tensor) -> Tuple[Tensor, Tensor, object]:
        '''
        Compute q_hat via RKHS bridge (O=1 only to fit), then fit logit on features [S, A, q_hat]
        to estimate lambda_hat; return (lambda_hat_all, weights_for_O1, bridge_obj).
        '''
        O = O.reshape(-1)
        mask = (O == 1.0)
        if mask.sum() <= 3:
            raise RuntimeError("[IPW-FQE] Too few O=1 samples to fit bridge/logit.")

        # 1) bridge q_hat for all points (fit on O=1)
        #    XH: [S', S, A], Y: R (but we don't have R for O=0 → fit uses only O=1)
        #    Here we only need q_hat as a feature to predict lambda.
        #    NOTE: We do NOT impute rewards in IPW baseline.
        #    We still need R1 to fit bridge; grab it from pack outside and pass in as arg if you prefer.
        #    For simplicity, we assume caller already has R and will call this within a 'fit' where R is in scope.
        raise RuntimeError("Internal misuse: _fit_lambda_and_weights should be called with R in scope.")

    def fit(self, dataset: Dict[str, np.ndarray], target_policy) -> "WeightedFQE":
        by_t = group_by_t(dataset)
        T = max(by_t.keys())
        next_Q = None

        for t in reversed(range(1, T+1)):
            pack = by_t[t]
            S  = _to32(pack["S"]).to(self.device)
            A  = _to32(pack["A"]).to(self.device)
            R  = _to32(pack["R"]).to(self.device).reshape(-1)
            O  = _to32(pack["O"]).to(self.device).reshape(-1)
            Sp = _to32(pack["Sp"]).to(self.device)

            mask = (O == 1.0)
            if mask.sum() <= 3:
                raise RuntimeError(f"[IPW-FQE] Too few O=1 samples at t={t}.")

            # ---- (1) bridge q_hat (fit on O=1) ----
            W  = _to32(pack["W"]).to(self.device)              # shadow: S' (and the reading w, if any)
            XH = _stack_cols(W[mask], S[mask], A[mask])           # (S', S, A)
            XF = _stack_cols(R[mask].reshape(-1,1), S[mask], A[mask])  # (R, S, A) for gamma_f heuristic only
            bridge = self.bridge_cfg.make_estimator().fit(XH, R[mask].reshape(-1,1), XF)
            q_hat_all = bridge.predict(_stack_cols(W, S, A)).reshape(-1)

            # ---- (2) fit lambda_t via logit on [S, A, q_hat] ----
            # feature scaling
            Phi = _stack_cols(S, A, q_hat_all.reshape(-1,1))  # (n, 2 + 1 + 1) = (n,4)
            scaler = Scaler().fit(Phi)
            self._scaler_lambda[t] = scaler
            Phi_s = scaler.transform(Phi)

            logit = _LogitBinary(l2=self.logit_l2, max_iter=self.logit_max_iter, device=self.device).fit(Phi_s, O)
            lam_hat = torch.clamp(logit.predict_proba(Phi_s), min=self.pmin, max=1.0 - 1e-6)

            # weights for observed points = 1 / lam_hat
            w_all = torch.zeros_like(lam_hat)
            w_all[mask] = 1.0 / lam_hat[mask]
            if self.w_cap is not None and self.w_cap > 0:
                w_all[mask] = torch.clamp(w_all[mask], max=self.w_cap)
            # stabilize (mean 1 on O=1 subset)
            mean_w = torch.mean(w_all[mask])
            w_all[mask] = w_all[mask] / torch.clamp(mean_w, min=torch.tensor(1e-12, dtype=mean_w.dtype, device=mean_w.device))

            # ---- (3) construct y_t on O=1 and fit weighted KRR ----
            On_all = _to32(pack["Onext"]).to(self.device).reshape(-1)
            if next_Q is None:
                y = R[mask]  # gamma=0 case or last step
            else:
                On1 = On_all[mask]
                probs = self._policy_probs_target(target_policy, Sp[mask], On1)
                v_next = torch.zeros_like(R[mask])
                for j, a in enumerate(self.action_list):
                    a_col = torch.full((Sp[mask].shape[0], 1), float(a),
                                       dtype=torch.float32, device=self.device)
                    Xa = _stack_cols(Sp[mask], a_col)
                    v_next += probs[:, j] * next_Q.predict(Xa).reshape(-1)
                y = R[mask] + self.gamma * v_next

            # weighted KRR on (S,A) with weights for O=1 samples
            X_train = _stack_cols(S[mask], A[mask])
            krr_w = WeightedKernelRidgeCV(**self.krr_kwargs).fit(X_train, y.reshape(-1,1), sample_weight=w_all[mask])
            self.Q[t] = krr_w

            # define V_t for later recursion
            def V_of(S_in: Tensor, Ominus_in: Tensor, Q=krr_w):
                probs = self._policy_probs_target(target_policy, S_in, Ominus_in)
                out = torch.zeros(S_in.shape[0], dtype=torch.float32, device=self.device)
                for j, a in enumerate(self.action_list):
                    Xa = _stack_cols(S_in,
                                     torch.full((S_in.shape[0], 1), float(a),
                                                dtype=torch.float32, device=self.device))
                    out += probs[:, j] * Q.predict(Xa).reshape(-1)
                return out

            self.V[t] = V_of
            next_Q = krr_w

        self.V1 = self.V[1]
        return self

    def value(self, S1: np.ndarray, O0: Optional[np.ndarray] = None) -> float:
        S1_t = torch.as_tensor(S1, dtype=torch.float32, device=self.device)
        if O0 is None:
            O0_t = torch.zeros(S1_t.shape[0], dtype=torch.float32, device=self.device)
        else:
            O0_t = torch.as_tensor(O0, dtype=torch.float32, device=self.device).reshape(-1)
        return float(self.V1(S1_t, O0_t).mean().item())


# --------------------------- baseline: SCOPE (Parbhoo et al. 2020) ---------------------------

class SCOPE:
    r'''
    SCOPE: Shaping Control variates for Off-Policy Evaluation.

    Per-step IS estimator with potential-based reward shaping (PBRS).
    Uses R_obs (= O_t * R_t, i.e. 0 when missing) as the base reward,
    then adds shaping term  gamma * phi(S_{t+1}) - phi(S_t)  to densify
    the reward signal.

    This baseline is biased under MNAR because it treats unobserved
    rewards as 0.  The shaping reduces IS variance but cannot correct
    the reward bias.

    Algorithm
    ---------
    1.  Split data into D_shape (frac_shape) and D_eval (rest).
    2.  On D_shape, learn potential phi(s) by fitting a KRR on
        cumulative observed return -> initial state.
    3.  On D_eval, compute per-step IS estimate with shaped rewards:
        V_hat = (1/n) sum_i sum_t gamma^t * w_{0:t}^(i)
                * [ R_obs_t + gamma*phi(S_{t+1}) - phi(S_t) ]

    Parameters
    ----------
    gamma       : discount factor (must match env)
    frac_shape  : fraction of episodes used to learn phi (default 0.3)
    krr_kwargs  : kwargs for KernelRidgeCV used to learn phi
    w_cap       : cap on cumulative IS weight to limit variance
    device      : 'cuda' or 'cpu'
    '''
    def __init__(self,
                 gamma: float = 1.0,
                 frac_shape: float = 0.3,
                 krr_kwargs=None,
                 w_cap: float = 50.0,
                 device: str = 'cuda'):
        self.gamma = float(gamma)
        self.frac_shape = float(frac_shape)
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.krr_kwargs = dict(lam_grid=None, folds=5, device=self.device)
        if krr_kwargs:
            self.krr_kwargs.update(krr_kwargs)
        self.w_cap = float(w_cap)
        self.phi_model = None
        self._value = None

    def fit(self, dataset: Dict[str, np.ndarray],
            target_policy, behavior_policy) -> "SCOPE":
        '''
        Parameters
        ----------
        dataset : flat dict from collect_episodes
        target_policy  : object with prob_a_plus(s, o_prev)
        behavior_policy: object with prob_a_plus(s)
        '''
        obs   = np.asarray(dataset["obs"],   dtype=np.float32)  # (N, 3)
        a     = np.asarray(dataset["a"],     dtype=np.float32)  # (N,)
        r_obs = np.asarray(dataset["r_obs"], dtype=np.float32)  # (N,)
        obs_n = np.asarray(dataset["obs_n"], dtype=np.float32)  # (N, 3)
        t_idx = np.asarray(dataset["t"],     dtype=np.int16)    # (N,)
        ep    = np.asarray(dataset["ep"],    dtype=np.int32)    # (N,)

        T = int(t_idx.max())
        episodes = np.unique(ep)
        n_ep = len(episodes)

        # ---- split episodes into shape / eval ----
        rng = np.random.default_rng(42)
        perm = rng.permutation(n_ep)
        n_shape = max(1, int(n_ep * self.frac_shape))
        ep_shape_set = set(episodes[perm[:n_shape]].tolist())
        ep_eval_set  = set(episodes[perm[n_shape:]].tolist())

        # ---- Step 1: learn phi(s) on shape episodes ----
        # For each shape episode, compute cumulative observed return
        shape_s1_list = []
        shape_ret_list = []
        for e in ep_shape_set:
            mask_e = (ep == e)
            r_e = r_obs[mask_e]
            obs_e = obs[mask_e]
            # cumulative return
            ret = 0.0
            disc = 1.0
            for tt in range(len(r_e)):
                ret += disc * r_e[tt]
                disc *= self.gamma
            shape_s1_list.append(obs_e[0, :2])   # initial state (s1, s2)
            shape_ret_list.append(ret)

        S_shape = torch.as_tensor(np.stack(shape_s1_list),
                                  dtype=torch.float32, device=self.device)
        Y_shape = torch.as_tensor(np.array(shape_ret_list),
                                  dtype=torch.float32, device=self.device).reshape(-1, 1)

        if S_shape.shape[0] >= 5:
            self.phi_model = KernelRidgeCV(**self.krr_kwargs).fit(S_shape, Y_shape)
        else:
            # too few episodes for CV; use zero potential
            self.phi_model = None

        # ---- Step 2: per-step IS with shaping on eval episodes ----
        estimates = []
        for e in ep_eval_set:
            mask_e = (ep == e)
            obs_e   = obs[mask_e]      # (T_e, 3)
            a_e     = a[mask_e]        # (T_e,)
            r_obs_e = r_obs[mask_e]    # (T_e,)
            obs_n_e = obs_n[mask_e]    # (T_e, 3)

            T_e = obs_e.shape[0]
            cum_w = 1.0
            traj_val = 0.0
            disc = 1.0

            for tt in range(T_e):
                s_t = obs_e[tt, :2]
                o_prev = int(round(obs_e[tt, 2]))
                a_t = float(a_e[tt])          # in {-1, +1}
                s_next = obs_n_e[tt, :2]

                # IS weight for this step
                p_target = target_policy.prob_a_plus(s_t, o_prev)
                p_behav  = behavior_policy.prob_a_plus(s_t)
                if a_t > 0:  # a = +1
                    w_t = p_target / max(p_behav, 1e-8)
                else:        # a = -1
                    w_t = (1.0 - p_target) / max(1.0 - p_behav, 1e-8)
                cum_w *= w_t
                # cap cumulative weight
                cum_w = min(cum_w, self.w_cap)

                # shaped reward
                phi_s = self._phi(s_t)
                phi_sp = self._phi(s_next)
                r_shaped = float(r_obs_e[tt]) + self.gamma * phi_sp - phi_s

                traj_val += disc * cum_w * r_shaped
                disc *= self.gamma

            estimates.append(traj_val)

        self._value = float(np.mean(estimates)) if len(estimates) > 0 else np.nan
        return self

    def _phi(self, s: np.ndarray) -> float:
        '''Evaluate potential function phi(s).'''
        if self.phi_model is None:
            return 0.0
        s_t = torch.as_tensor(s.reshape(1, -1)[:, :2],
                              dtype=torch.float32, device=self.device)
        return float(self.phi_model.predict(s_t).item())

    def value(self, S1: np.ndarray = None, O0: np.ndarray = None) -> float:
        '''Return the pre-computed SCOPE estimate (S1, O0 ignored).'''
        if self._value is None:
            raise RuntimeError("Call .fit() first.")
        return self._value