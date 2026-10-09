"""Learners for the corrected estimator that ShadOPE does not provide.

Both nuisances are fitted by whitened generalized method of moments (GMM) with fixed, finite
feature maps, as in the paper's fixed-feature instance: minimize ||S^{-1/2} m(theta)||^2, where
m(theta) is the empirical moment vector against fixed instruments g and S = mean(g g^T). Neural
min-max learners were tried first; with a few hundred rows per fit, the adversary memorizes which
rows are recorded and the game degenerates (for the recording model, to h = 1), so they are not
used.

- RecordingInverse: h(r, x, a) = 1 / e(r, x, a) with logistic e, i.e. h = 1 + exp(-theta' phi(r, x, a)),
  capped at `cap`, solving E[g(z, x, a) (M h(R, x, a) - 1)] = 0 on all rows. With shadow
  exclusion the true inverse solves these moments.
- RatioModel: marginal ratio w_t(x, a) = rho_t(x) * pi_t(a|x) / mu(a|x). The logging policy mu(a|s)
  is ShadOPE's softmax network; the state ratio rho_t (rho_1 = 1, shared initial law) is
  log-linear in x and solves the balance equations
  E[g(X_t) rho_t(X_t)] = E[g(X_t) rho_{t-1}(X_{t-1}) pi_{t-1}(A_{t-1}|X_{t-1}) / mu(A_{t-1}|S_{t-1})].
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from improxope.nn_fqe import _NNSoftmaxPolicy


# --------------------------------------------------------------------------- feature maps

class Standardizer:
    """Column standardization fitted on training rows; constant columns map to 0."""

    def __init__(self, X: np.ndarray):
        self.mean = X.mean(0)
        std = X.std(0)
        self.std = np.where(std > 1e-8, std, 1.0)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


def dummies(A: np.ndarray, k: int) -> np.ndarray:
    """Action indicators without the reference category 0 (an intercept is added separately)."""
    out = np.zeros((len(A), k - 1), dtype=np.float64)
    rows = np.nonzero(A > 0)[0]
    out[rows, A[rows] - 1] = 1.0
    return out


def _with_intercept(X: np.ndarray) -> np.ndarray:
    return np.concatenate([np.ones((len(X), 1)), X], axis=1)


# --------------------------------------------------------------------------- GMM solver

def _whitener(G: torch.Tensor, ridge: float) -> torch.Tensor:
    S = G.T @ G / G.shape[0]
    S = S + ridge * (torch.trace(S) / S.shape[0]) * torch.eye(S.shape[0], dtype=S.dtype, device=S.device)
    vals, vecs = torch.linalg.eigh(S)
    return vecs @ torch.diag(vals.clamp(min=1e-12) ** -0.5) @ vecs.T


def gmm_fit(theta0: torch.Tensor, moments, W_half: torch.Tensor, max_iter: int) -> Dict:
    """Minimize ||W_half @ moments(theta)||^2 over theta with L-BFGS (full batch)."""
    theta = theta0.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([theta], lr=1.0, max_iter=max_iter, line_search_fn='strong_wolfe',
                            tolerance_grad=1e-10, tolerance_change=1e-14, history_size=50)

    def closure():
        opt.zero_grad()
        loss = (W_half @ moments(theta)).pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        loss = float((W_half @ moments(theta)).pow(2).sum())
        initial = float((W_half @ moments(theta0)).pow(2).sum())
    return {'theta': theta.detach(), 'objective': loss, 'objective_initial': initial}


# --------------------------------------------------------------------------- recording inverse

class RecordingInverse:
    """h(r, x, a) = 1 + exp(-theta' phi), phi = (1, r, x, action dummies); instruments
    g = (1, z, x, action dummies). Fitted on all rows (observed and missing)."""

    def __init__(self, n_actions: int, cap: float = 20.0, max_iter: int = 500, ridge: float = 1e-6,
                 device: str = 'cpu'):
        if cap <= 1:
            raise ValueError('cap must exceed 1')
        self.k, self.cap, self.max_iter, self.ridge, self.device = n_actions, cap, max_iter, ridge, device

    def _phi(self, R, X, A):
        return _with_intercept(np.concatenate([self.std_rx(np.column_stack([R, X])), dummies(A, self.k)], axis=1))

    def _g(self, Z, X, A):
        return _with_intercept(np.concatenate([self.std_zx(np.column_stack([Z, X])), dummies(A, self.k)], axis=1))

    def _h(self, theta, phi):
        eta = (phi @ theta).clamp(min=-float(np.log(self.cap - 1.0)))
        return 1.0 + torch.exp(-eta)

    def fit(self, R, X, A, Z, M):
        R, X, Z, M = (np.asarray(v, dtype=np.float64) for v in (R, X, Z, M))
        obs = M == 1
        if obs.sum() < 2:
            raise RuntimeError('too few observed rows to fit the recording model')
        self.std_rx = Standardizer(np.column_stack([R, X])[obs])  # R is only meaningful where observed
        self.std_zx = Standardizer(np.column_stack([Z, X]))
        dev = torch.device(self.device)
        phi = torch.as_tensor(self._phi(R, X, A), device=dev)
        G = torch.as_tensor(self._g(Z, X, A), device=dev)
        Mt = torch.as_tensor(M, device=dev)
        p = M.mean()
        theta0 = torch.zeros(phi.shape[1], dtype=torch.float64, device=dev)
        theta0[0] = np.log(p / (1 - p))  # missing-at-random start: h = 1 / P(M = 1)
        res = gmm_fit(theta0, lambda th: (G * (Mt * self._h(th, phi) - 1.0)[:, None]).mean(0),
                      _whitener(G, self.ridge), self.max_iter)
        self.theta_ = res['theta']
        h = self._h(self.theta_, phi).cpu().numpy()
        self.info_ = {'objective': res['objective'], 'objective_initial': res['objective_initial'],
                      'mean_Mh': float((M * h).mean()), 'mean_h_observed': float(h[obs].mean()),
                      'cap_fraction': float(np.mean(h[obs] >= self.cap - 1e-6))}
        return self

    def predict(self, R, X, A) -> np.ndarray:
        phi = torch.as_tensor(self._phi(np.asarray(R, np.float64), np.asarray(X, np.float64), A),
                              device=torch.device(self.device))
        with torch.no_grad():
            return self._h(self.theta_, phi).cpu().numpy()


# --------------------------------------------------------------------------- marginal ratios

class RatioModel:
    """w_t(x, a) = min(rho_t(x) * pi_t(a|x) / mu(a|s), cap) for t = 1..T.

    Fit on a panel's states S (N, T, d), decision states X (N, T, dx), actions A (N, T) and target
    probabilities pi (N, T, K). mu is fitted on all (s, a) rows pooled over time."""

    def __init__(self, n_actions: int, cap: float = 100.0, max_iter: int = 500, ridge: float = 1e-6,
                 behavior: Optional[Dict] = None, device: str = 'cpu'):
        self.k, self.cap, self.max_iter, self.ridge, self.device = n_actions, cap, max_iter, ridge, device
        self.behavior = dict(behavior or {})

    def _mu(self, S2d: np.ndarray) -> np.ndarray:
        return self.mu_.predict_proba(torch.as_tensor(S2d, dtype=torch.float32)).cpu().numpy().astype(np.float64)

    def action_ratio(self, S, A, pi) -> np.ndarray:
        """pi(A|X) / mu(A|S), shape (N, T)."""
        N, T, d = S.shape
        mu = self._mu(S.reshape(N * T, d)).reshape(N, T, self.k)
        idx = np.asarray(A)[..., None]
        num = np.take_along_axis(pi, idx, -1)[..., 0]
        den = np.take_along_axis(mu, idx, -1)[..., 0]
        return num / np.maximum(den, 1e-12)

    def fit(self, S, X, A, pi):
        N, T, d = S.shape
        dev = torch.device(self.device)
        self.mu_ = _NNSoftmaxPolicy(n_actions=self.k, device=self.device, **self.behavior).fit(
            torch.as_tensor(S.reshape(N * T, d), dtype=torch.float32, device=dev),
            torch.as_tensor(np.asarray(A).reshape(-1), dtype=torch.long, device=dev))
        r = self.action_ratio(S, A, pi)
        self.std_x = Standardizer(np.asarray(X, np.float64).reshape(N * T, -1))
        self.betas_: List[torch.Tensor] = []
        self.info_ = {'objective': [], 'mean_w': [], 'cap_fraction': []}
        prev = np.minimum(r[:, 0], self.cap)                   # rho_1 = 1, so w_1 = pi / mu
        self.info_['mean_w'].append(float(prev.mean()))
        self.info_['cap_fraction'].append(float(np.mean(r[:, 0] >= self.cap)))
        for t in range(1, T):
            phi = torch.as_tensor(_with_intercept(self.std_x(np.asarray(X[:, t], np.float64))), device=dev)
            target = torch.as_tensor(prev, device=dev)
            res = gmm_fit(torch.zeros(phi.shape[1], dtype=torch.float64, device=dev),
                          lambda b: (phi * (self._rho(b, phi) - target)[:, None]).mean(0),
                          _whitener(phi, self.ridge), self.max_iter)
            self.betas_.append(res['theta'])
            with torch.no_grad():
                rho = self._rho(res['theta'], phi).cpu().numpy()
            w = np.minimum(rho * r[:, t], self.cap)
            self.info_['objective'].append(res['objective'])
            self.info_['mean_w'].append(float(w.mean()))
            self.info_['cap_fraction'].append(float(np.mean(rho * r[:, t] >= self.cap)))
            prev = w
        return self

    def _rho(self, beta, phi):
        return torch.exp((phi @ beta).clamp(max=float(np.log(self.cap))))

    def predict(self, S, X, A, pi) -> np.ndarray:
        N, T, _ = S.shape
        r = self.action_ratio(S, A, pi)
        out = np.empty((N, T))
        out[:, 0] = np.minimum(r[:, 0], self.cap)
        dev = torch.device(self.device)
        for t in range(1, T):
            phi = torch.as_tensor(_with_intercept(self.std_x(np.asarray(X[:, t], np.float64))), device=dev)
            with torch.no_grad():
                rho = self._rho(self.betas_[t - 1], phi).cpu().numpy()
            out[:, t] = np.minimum(rho * r[:, t], self.cap)
        return out.astype(np.float32)
