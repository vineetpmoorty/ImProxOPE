"""
Copied from ShadOPE (Wei, Qu, Miao, ICML 2026), https://github.com/NAIVlab/ShadOPE @ 4231ba5, file sepsis/nn_fqe.py.
Changes from the original: NNBridge import path; removed the sys.path insertion; optional
'shadow' data key (a reduced shadow variable) used by the ProxFQE and IPW-FQE bridges instead of
'next_states' (Q-function targets still use 'next_states'); without it, behaviour is unchanged.
Optional stochastic target: with 'pi_target' / 'pi_target_next' probability arrays in the data,
values and FQE targets use sum_a pi(a|x) Q(x, a) and SCOPE's importance weight uses pi(a_t|x_t)
instead of the deterministic DQN action (`_target_value`); without them, behaviour is unchanged.

Neural-network-based FQE for the sepsis setting.

Supports:
  - High-dimensional state (48 dims, paper's features)
  - 25 discrete actions (5 vaso x 5 iv)
  - NN bridge for reward imputation (ProxFQE)
  - NN Q-function for Bellman regression

Classes:
  - NNQFunction:   neural net Q(s,a) -> R for all 25 actions
  - NNProxFQE:     our method — NN bridge to impute MNAR rewards, then FQE
  - NNNaiveFQE:    baseline — drops missing rewards, FQE on observed only
  - NNOracleFQE:   oracle — uses r_true (ground truth)
  - NNImputeFQE:   baseline — NN regression imputation (biased under MNAR)
  - NNIPWFQE:      baseline — inverse propensity weighted FQE
  - NNSCOPE:       baseline — reward shaping + per-step IS (Parbhoo et al. 2020)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional
from improxope.nn_bridge import NNBridge


# ===================== Q-function via neural net =====================

class NNQFunction:
    """
    Neural net Q(s) -> R^{n_actions}.
    Maps state to Q-values for all actions simultaneously.
    """
    def __init__(self, state_dim, n_actions=25, hidden=(128, 128),
                 n_steps=3000, lr=1e-3, batch_size=512, device='cpu'):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.hidden = hidden
        self.n_steps = n_steps
        self.lr = lr
        self.batch_size = batch_size
        self.device = device

    def fit(self, S, A, y):
        dev = torch.device(self.device)
        S = S.to(dev, dtype=torch.float32)
        A = A.to(dev, dtype=torch.long)
        y = y.to(dev, dtype=torch.float32)
        n = S.shape[0]

        self.s_mean_ = S.mean(0)
        self.s_std_ = S.std(0).clamp(min=1e-6)
        Sn = (S - self.s_mean_) / self.s_std_

        layers = []
        d_in = self.state_dim
        for d_h in self.hidden:
            layers.append(nn.Linear(d_in, d_h))
            layers.append(nn.ReLU())
            d_in = d_h
        layers.append(nn.Linear(d_in, self.n_actions))
        self.net_ = nn.Sequential(*layers).to(dev)

        optimizer = optim.Adam(self.net_.parameters(), lr=self.lr)
        bs = min(self.batch_size, n)

        self.net_.train()
        for step in range(self.n_steps):
            idx = torch.randint(n, (bs,), device=dev)
            q_all = self.net_(Sn[idx])
            q_sa = q_all.gather(1, A[idx].unsqueeze(1)).squeeze(1)
            loss = nn.functional.mse_loss(q_sa, y[idx])

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net_.parameters(), 10.0)
            optimizer.step()

        self.net_.eval()
        return self

    def predict_all(self, S):
        dev = torch.device(self.device)
        S = S.to(dev, dtype=torch.float32)
        Sn = (S - self.s_mean_) / self.s_std_
        with torch.no_grad():
            return self.net_(Sn)

    def predict(self, S, A):
        q_all = self.predict_all(S)
        A = A.to(q_all.device, dtype=torch.long)
        return q_all.gather(1, A.unsqueeze(1)).squeeze(1)


# ===================== Simple NN regressor =====================

class _NNRegressor:
    """Simple NN regression f(X) -> y, used for reward imputation and potential."""
    def __init__(self, hidden=(128, 128), n_steps=2000, lr=1e-3,
                 batch_size=512, device='cpu'):
        self.hidden = hidden
        self.n_steps = n_steps
        self.lr = lr
        self.batch_size = batch_size
        self.device = device

    def fit(self, X, y):
        dev = torch.device(self.device)
        X = X.to(dev, dtype=torch.float32)
        y = y.to(dev, dtype=torch.float32).reshape(-1)
        n = X.shape[0]

        self.x_mean_ = X.mean(0)
        self.x_std_ = X.std(0).clamp(min=1e-6)
        Xn = (X - self.x_mean_) / self.x_std_
        self.y_mean_ = y.mean()
        self.y_std_ = y.std().clamp(min=1e-6)
        yn = (y - self.y_mean_) / self.y_std_

        layers = []
        d_in = X.shape[1]
        for d_h in self.hidden:
            layers.append(nn.Linear(d_in, d_h))
            layers.append(nn.ReLU())
            d_in = d_h
        layers.append(nn.Linear(d_in, 1))
        self.net_ = nn.Sequential(*layers).to(dev)

        optimizer = optim.Adam(self.net_.parameters(), lr=self.lr)
        bs = min(self.batch_size, n)

        self.net_.train()
        for step in range(self.n_steps):
            idx = torch.randint(n, (bs,), device=dev)
            pred = self.net_(Xn[idx]).squeeze(1)
            loss = nn.functional.mse_loss(pred, yn[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.net_.eval()
        return self

    def predict(self, X):
        dev = torch.device(self.device)
        X = X.to(dev, dtype=torch.float32)
        Xn = (X - self.x_mean_) / self.x_std_
        with torch.no_grad():
            pred_n = self.net_(Xn).squeeze(1)
        return pred_n * self.y_std_ + self.y_mean_


# ===================== NN classifier for P(O=1|features) =====================

class _NNClassifier:
    """Binary classifier via NN logistic regression."""
    def __init__(self, hidden=(64, 64), n_steps=1000, lr=1e-3,
                 batch_size=512, device='cpu'):
        self.hidden = hidden
        self.n_steps = n_steps
        self.lr = lr
        self.batch_size = batch_size
        self.device = device

    def fit(self, X, y):
        dev = torch.device(self.device)
        X = X.to(dev, dtype=torch.float32)
        y = y.to(dev, dtype=torch.float32).reshape(-1)
        n = X.shape[0]

        self.x_mean_ = X.mean(0)
        self.x_std_ = X.std(0).clamp(min=1e-6)
        Xn = (X - self.x_mean_) / self.x_std_

        layers = []
        d_in = X.shape[1]
        for d_h in self.hidden:
            layers.append(nn.Linear(d_in, d_h))
            layers.append(nn.ReLU())
            d_in = d_h
        layers.append(nn.Linear(d_in, 1))
        self.net_ = nn.Sequential(*layers).to(dev)

        optimizer = optim.Adam(self.net_.parameters(), lr=self.lr)
        bs = min(self.batch_size, n)

        self.net_.train()
        for step in range(self.n_steps):
            idx = torch.randint(n, (bs,), device=dev)
            logits = self.net_(Xn[idx]).squeeze(1)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.net_.eval()
        return self

    def predict_proba(self, X):
        dev = torch.device(self.device)
        X = X.to(dev, dtype=torch.float32)
        Xn = (X - self.x_mean_) / self.x_std_
        with torch.no_grad():
            return torch.sigmoid(self.net_(Xn).squeeze(1))


# ===================== Softmax policy model for behavior =====================

class _NNSoftmaxPolicy:
    """Estimate behavior policy P(A|S) via softmax NN."""
    def __init__(self, n_actions=25, hidden=(128, 128), n_steps=3000,
                 lr=1e-3, batch_size=512, device='cpu'):
        self.n_actions = n_actions
        self.hidden = hidden
        self.n_steps = n_steps
        self.lr = lr
        self.batch_size = batch_size
        self.device = device

    def fit(self, S, A):
        dev = torch.device(self.device)
        S = S.to(dev, dtype=torch.float32)
        A = A.to(dev, dtype=torch.long)
        n = S.shape[0]

        self.s_mean_ = S.mean(0)
        self.s_std_ = S.std(0).clamp(min=1e-6)
        Sn = (S - self.s_mean_) / self.s_std_

        layers = []
        d_in = S.shape[1]
        for d_h in self.hidden:
            layers.append(nn.Linear(d_in, d_h))
            layers.append(nn.ReLU())
            d_in = d_h
        layers.append(nn.Linear(d_in, self.n_actions))
        self.net_ = nn.Sequential(*layers).to(dev)

        optimizer = optim.Adam(self.net_.parameters(), lr=self.lr)
        bs = min(self.batch_size, n)

        self.net_.train()
        for step in range(self.n_steps):
            idx = torch.randint(n, (bs,), device=dev)
            logits = self.net_(Sn[idx])
            loss = nn.functional.cross_entropy(logits, A[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.net_.eval()
        return self

    def predict_proba(self, S):
        """Return P(A|S) for all actions. Shape (n, n_actions)."""
        dev = torch.device(self.device)
        S = S.to(dev, dtype=torch.float32)
        Sn = (S - self.s_mean_) / self.s_std_
        with torch.no_grad():
            return torch.softmax(self.net_(Sn), dim=1)


# ===================== Data grouping =====================

def _target_value(Q, S, A, P=None):
    """Value of S under the target: Q(S, A) for the deterministic target action A, or
    sum_a P(a) Q(S, a) when target probabilities P are given."""
    if P is None:
        return Q.predict(S, A)
    q_all = Q.predict_all(S)
    return (q_all * P.to(q_all.device)).sum(1)


def _start_probs(data, mask, dev):
    if 'pi_target' not in data:
        return None
    return torch.tensor(data['pi_target'][mask], device=dev, dtype=torch.float32)


def group_by_t(data: dict) -> Dict[int, dict]:
    """
    Group flat data arrays by time step.

    data keys: states, next_states, a_joint, at_joint_next,
               r_true, r_obs, o_t, dones, bloc
    """
    bloc = data['bloc']
    T = int(bloc.max())
    out = {}
    for t in range(1, T + 1):
        m = (bloc == t)
        out[t] = {
            'S': torch.tensor(data['states'][m], dtype=torch.float32),
            'Sp': torch.tensor(data['next_states'][m], dtype=torch.float32),
            'W': torch.tensor(data.get('shadow', data['next_states'])[m], dtype=torch.float32),
            'Pt_next': (torch.tensor(data['pi_target_next'][m], dtype=torch.float32)
                        if 'pi_target_next' in data else None),
            'A_joint': torch.tensor(data['a_joint'][m], dtype=torch.long),
            'At_joint_next': torch.tensor(data['at_joint_next'][m], dtype=torch.long),
            'R_true': torch.tensor(data['r_true'][m], dtype=torch.float32),
            'R_obs': torch.tensor(data['r_obs'][m], dtype=torch.float32),
            'O': torch.tensor(data['o_t'][m], dtype=torch.float32),
            'done': torch.tensor(data['dones'][m], dtype=torch.float32),
        }
    return out


# ===================== ProxFQE with NN bridge (our method) =====================

class NNProxFQE:
    """
    Proximal FQE with NN bridge for MNAR reward imputation.

    Steps per time t (backward from T to 1):
      1. Fit bridge b_t via AGMM on O_t=1 data:
         E[R_t - b_t(S_{t+1}, S_t, A_t) | R_t, S_t, A_t] = 0
      2. Impute: R_tilde = O_t * R_obs + (1-O_t) * b_t(S_{t+1}, S_t, A_t)
      3. FQE target: y_t = R_tilde + gamma * Q_{t+1}(S_{t+1}, a_target_{t+1})
      4. Fit Q_t(S_t, A_t) -> y_t via NN regression
    """
    def __init__(self, state_dim, n_actions=25, gamma=0.99,
                 bridge_kwargs=None, q_kwargs=None, device='cpu'):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = device

        self.bridge_kwargs = dict(
            h_hidden=(128, 128), f_hidden=(128, 128),
            n_steps=2000, lr_h=1e-3, lr_f=1e-3,
            lambda_f=0.1, mu_h=1e-4, n_critic=5,
            batch_size=512, device=device)
        if bridge_kwargs:
            self.bridge_kwargs.update(bridge_kwargs)

        self.q_kwargs = dict(
            state_dim=state_dim, n_actions=n_actions,
            hidden=(128, 128), n_steps=3000, lr=1e-3,
            batch_size=512, device=device)
        if q_kwargs:
            self.q_kwargs.update(q_kwargs)

        self.Q = {}
        self.bridges = {}

    def fit(self, data: dict):
        by_t = group_by_t(data)
        T = max(by_t.keys())
        dev = torch.device(self.device)
        Q_next = None

        for t in reversed(range(1, T + 1)):
            pack = by_t[t]
            S = pack['S'].to(dev)
            A_joint = pack['A_joint'].to(dev)
            R_obs = pack['R_obs'].to(dev)
            O = pack['O'].to(dev)
            Sp = pack['Sp'].to(dev)
            done = pack['done'].to(dev)

            mask = (O == 1.0)
            n_obs = mask.sum().item()
            if n_obs <= 10:
                raise RuntimeError(f"Too few observed samples at t={t}: {n_obs}")

            # Step 1: Fit bridge on O=1 data
            W = pack['W'].to(dev)   # shadow variable (= Sp unless reduced)
            S1, R1, Sp1 = S[mask], R_obs[mask], W[mask]
            A1_float = A_joint[mask].float().unsqueeze(1)
            XH = torch.cat([Sp1, S1, A1_float], dim=1)
            XF = torch.cat([R1.unsqueeze(1), S1, A1_float], dim=1)

            bridge = NNBridge(**self.bridge_kwargs)
            bridge.fit(XH, R1, XF)
            self.bridges[t] = bridge

            # Step 2: Impute rewards
            A_float = A_joint.float().unsqueeze(1)
            XH_all = torch.cat([W, S, A_float], dim=1)
            q_hat = bridge.predict(XH_all)
            R_tilde = torch.where(O == 1.0, R_obs, q_hat)

            # Step 3: FQE target
            if Q_next is None:
                y = R_tilde
            else:
                At_next = pack['At_joint_next'].to(dev)
                v_next = _target_value(Q_next, Sp, At_next, pack['Pt_next'])
                y = R_tilde + self.gamma * v_next * (1 - done)

            # Step 4: Fit Q_t
            Q_t = NNQFunction(**self.q_kwargs).fit(S, A_joint, y)
            self.Q[t] = Q_t
            Q_next = Q_t

            print(f"  t={t}: n_obs={n_obs}/{len(S)}, "
                  f"R_tilde mean={R_tilde.mean():.3f}, y mean={y.mean():.3f}")

        return self

    def value(self, data: dict) -> float:
        dev = torch.device(self.device)
        mask = data['bloc'] == 1
        S0 = torch.tensor(data['states'][mask], device=dev, dtype=torch.float32)
        At0 = torch.tensor(data['at_joint'][mask], device=dev, dtype=torch.long)
        Q1 = self.Q[1]
        v = _target_value(Q1, S0, At0, _start_probs(data, mask, dev))
        return v.mean().item(), v.std().item() / np.sqrt(len(v))


# ===================== Naive FQE (drop missing) =====================

class NNNaiveFQE:
    """
    Naive FQE: only uses O_t=1 data, ignores missingness.
    Biased under MNAR.
    """
    def __init__(self, state_dim, n_actions=25, gamma=0.99,
                 q_kwargs=None, device='cpu'):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = device
        self.q_kwargs = dict(
            state_dim=state_dim, n_actions=n_actions,
            hidden=(128, 128), n_steps=3000, lr=1e-3,
            batch_size=512, device=device)
        if q_kwargs:
            self.q_kwargs.update(q_kwargs)
        self.Q = {}

    def fit(self, data: dict):
        by_t = group_by_t(data)
        T = max(by_t.keys())
        dev = torch.device(self.device)
        Q_next = None

        for t in reversed(range(1, T + 1)):
            pack = by_t[t]
            S = pack['S'].to(dev)
            A_joint = pack['A_joint'].to(dev)
            R_obs = pack['R_obs'].to(dev)
            O = pack['O'].to(dev)
            Sp = pack['Sp'].to(dev)
            done = pack['done'].to(dev)

            mask = (O == 1.0)
            S1, A1 = S[mask], A_joint[mask]
            R1, Sp1 = R_obs[mask], Sp[mask]
            done1 = done[mask]

            if Q_next is None:
                y = R1
            else:
                At_next1 = pack['At_joint_next'].to(dev)[mask]
                Pt1 = None if pack['Pt_next'] is None else pack['Pt_next'].to(dev)[mask]
                v_next = _target_value(Q_next, Sp1, At_next1, Pt1)
                y = R1 + self.gamma * v_next * (1 - done1)

            Q_t = NNQFunction(**self.q_kwargs).fit(S1, A1, y)
            self.Q[t] = Q_t
            Q_next = Q_t

            print(f"  t={t}: n_obs={mask.sum().item()}/{len(S)}, y mean={y.mean():.3f}")

        return self

    def value(self, data: dict) -> float:
        dev = torch.device(self.device)
        mask = data['bloc'] == 1
        S0 = torch.tensor(data['states'][mask], device=dev, dtype=torch.float32)
        At0 = torch.tensor(data['at_joint'][mask], device=dev, dtype=torch.long)
        Q1 = self.Q[1]
        v = _target_value(Q1, S0, At0, _start_probs(data, mask, dev))
        return v.mean().item(), v.std().item() / np.sqrt(len(v))


# ===================== Oracle FQE (uses r_true) =====================

class NNOracleFQE:
    """
    Oracle FQE: uses r_true (fully observed). Ground truth baseline.
    """
    def __init__(self, state_dim, n_actions=25, gamma=0.99,
                 q_kwargs=None, device='cpu'):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = device
        self.q_kwargs = dict(
            state_dim=state_dim, n_actions=n_actions,
            hidden=(128, 128), n_steps=3000, lr=1e-3,
            batch_size=512, device=device)
        if q_kwargs:
            self.q_kwargs.update(q_kwargs)
        self.Q = {}

    def fit(self, data: dict):
        by_t = group_by_t(data)
        T = max(by_t.keys())
        dev = torch.device(self.device)
        Q_next = None

        for t in reversed(range(1, T + 1)):
            pack = by_t[t]
            S = pack['S'].to(dev)
            A_joint = pack['A_joint'].to(dev)
            R = pack['R_true'].to(dev)
            Sp = pack['Sp'].to(dev)
            done = pack['done'].to(dev)

            if Q_next is None:
                y = R
            else:
                At_next = pack['At_joint_next'].to(dev)
                v_next = _target_value(Q_next, Sp, At_next, pack['Pt_next'])
                y = R + self.gamma * v_next * (1 - done)

            Q_t = NNQFunction(**self.q_kwargs).fit(S, A_joint, y)
            self.Q[t] = Q_t
            Q_next = Q_t

            print(f"  t={t}: n={len(S)}, y mean={y.mean():.3f}")

        return self

    def value(self, data: dict) -> float:
        dev = torch.device(self.device)
        mask = data['bloc'] == 1
        S0 = torch.tensor(data['states'][mask], device=dev, dtype=torch.float32)
        At0 = torch.tensor(data['at_joint'][mask], device=dev, dtype=torch.long)
        Q1 = self.Q[1]
        v = _target_value(Q1, S0, At0, _start_probs(data, mask, dev))
        return v.mean().item(), v.std().item() / np.sqrt(len(v))


# ===================== Impute-then-FQE (biased under MNAR) =====================

class NNImputeFQE:
    """
    Impute-then-FQE baseline.

    Step 1: On O_t=1, fit NN regression R ~ f(S_t, A_t) to learn E[R|S,A,O=1].
    Step 2: Impute: R_tilde = O*R_obs + (1-O)*f(S,A).
    Step 3: Standard FQE on all samples with R_tilde.

    Biased under MNAR because f learns E[R|S,A,O=1] != E[R|S,A].
    """
    def __init__(self, state_dim, n_actions=25, gamma=0.99,
                 q_kwargs=None, impute_kwargs=None, device='cpu'):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = device
        self.q_kwargs = dict(
            state_dim=state_dim, n_actions=n_actions,
            hidden=(128, 128), n_steps=3000, lr=1e-3,
            batch_size=512, device=device)
        if q_kwargs:
            self.q_kwargs.update(q_kwargs)
        self.impute_kwargs = dict(
            hidden=(128, 128), n_steps=2000, lr=1e-3,
            batch_size=512, device=device)
        if impute_kwargs:
            self.impute_kwargs.update(impute_kwargs)
        self.Q = {}
        self.reward_models = {}

    def fit(self, data: dict):
        by_t = group_by_t(data)
        T = max(by_t.keys())
        dev = torch.device(self.device)
        Q_next = None

        for t in reversed(range(1, T + 1)):
            pack = by_t[t]
            S = pack['S'].to(dev)
            A_joint = pack['A_joint'].to(dev)
            R_obs = pack['R_obs'].to(dev)
            O = pack['O'].to(dev)
            Sp = pack['Sp'].to(dev)
            done = pack['done'].to(dev)

            mask = (O == 1.0)
            n_obs = mask.sum().item()

            # Step 1: Fit reward model on O=1
            A_float = A_joint.float().unsqueeze(1)
            X_obs = torch.cat([S[mask], A_float[mask]], dim=1)
            r_model = _NNRegressor(**self.impute_kwargs).fit(X_obs, R_obs[mask])
            self.reward_models[t] = r_model

            # Step 2: Impute
            X_all = torch.cat([S, A_float], dim=1)
            R_hat = r_model.predict(X_all)
            R_tilde = torch.where(O == 1.0, R_obs, R_hat)

            # Step 3: FQE target
            if Q_next is None:
                y = R_tilde
            else:
                At_next = pack['At_joint_next'].to(dev)
                v_next = _target_value(Q_next, Sp, At_next, pack['Pt_next'])
                y = R_tilde + self.gamma * v_next * (1 - done)

            Q_t = NNQFunction(**self.q_kwargs).fit(S, A_joint, y)
            self.Q[t] = Q_t
            Q_next = Q_t

            print(f"  t={t}: n_obs={n_obs}/{len(S)}, "
                  f"R_tilde mean={R_tilde.mean():.3f}, y mean={y.mean():.3f}")

        return self

    def value(self, data: dict) -> float:
        dev = torch.device(self.device)
        mask = data['bloc'] == 1
        S0 = torch.tensor(data['states'][mask], device=dev, dtype=torch.float32)
        At0 = torch.tensor(data['at_joint'][mask], device=dev, dtype=torch.long)
        Q1 = self.Q[1]
        v = _target_value(Q1, S0, At0, _start_probs(data, mask, dev))
        return v.mean().item(), v.std().item() / np.sqrt(len(v))


# ===================== IPW-FQE =====================

class NNIPWFQE:
    """
    Inverse Propensity Weighted FQE.

    Step 1: Fit bridge b_t on O=1 data (same as ProxFQE) to get q_hat.
    Step 2: Estimate P(O=1|S,A,q_hat) via NN classifier.
    Step 3: Weight O=1 samples by 1/P_hat, fit weighted FQE.

    Uses the bridge prediction as a feature for the propensity model,
    following Wang et al. (2025).
    """
    def __init__(self, state_dim, n_actions=25, gamma=0.99,
                 bridge_kwargs=None, q_kwargs=None,
                 prop_kwargs=None, p_min=0.05, w_cap=20.0, device='cpu'):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = device
        self.p_min = p_min
        self.w_cap = w_cap

        self.bridge_kwargs = dict(
            h_hidden=(128, 128), f_hidden=(128, 128),
            n_steps=2000, lr_h=1e-3, lr_f=1e-3,
            lambda_f=0.1, mu_h=1e-4, n_critic=5,
            batch_size=512, device=device)
        if bridge_kwargs:
            self.bridge_kwargs.update(bridge_kwargs)

        self.q_kwargs = dict(
            state_dim=state_dim, n_actions=n_actions,
            hidden=(128, 128), n_steps=3000, lr=1e-3,
            batch_size=512, device=device)
        if q_kwargs:
            self.q_kwargs.update(q_kwargs)

        self.prop_kwargs = dict(
            hidden=(64, 64), n_steps=1000, lr=1e-3,
            batch_size=512, device=device)
        if prop_kwargs:
            self.prop_kwargs.update(prop_kwargs)

        self.Q = {}

    def fit(self, data: dict):
        by_t = group_by_t(data)
        T = max(by_t.keys())
        dev = torch.device(self.device)
        Q_next = None

        for t in reversed(range(1, T + 1)):
            pack = by_t[t]
            S = pack['S'].to(dev)
            A_joint = pack['A_joint'].to(dev)
            R_obs = pack['R_obs'].to(dev)
            O = pack['O'].to(dev)
            Sp = pack['Sp'].to(dev)
            done = pack['done'].to(dev)

            mask = (O == 1.0)
            n_obs = mask.sum().item()

            # Step 1: Bridge for q_hat feature
            W = pack['W'].to(dev)   # shadow variable (= Sp unless reduced)
            S1, R1, Sp1 = S[mask], R_obs[mask], W[mask]
            A1_float = A_joint[mask].float().unsqueeze(1)
            XH = torch.cat([Sp1, S1, A1_float], dim=1)
            XF = torch.cat([R1.unsqueeze(1), S1, A1_float], dim=1)
            bridge = NNBridge(**self.bridge_kwargs)
            bridge.fit(XH, R1, XF)

            A_float = A_joint.float().unsqueeze(1)
            XH_all = torch.cat([W, S, A_float], dim=1)
            q_hat = bridge.predict(XH_all).detach()

            # Step 2: Estimate P(O=1|S,A,q_hat)
            Phi = torch.cat([S, A_float, q_hat.unsqueeze(1)], dim=1)
            prop_model = _NNClassifier(**self.prop_kwargs).fit(Phi, O)
            p_hat = prop_model.predict_proba(Phi)
            p_hat = p_hat.clamp(min=self.p_min, max=1.0)

            # Step 3: Weights for O=1 samples
            w = 1.0 / p_hat[mask]
            if self.w_cap > 0:
                w = w.clamp(max=self.w_cap)
            w = w / w.mean()  # normalize

            # Step 4: Weighted FQE target on O=1
            if Q_next is None:
                y = R_obs[mask]
            else:
                At_next1 = pack['At_joint_next'].to(dev)[mask]
                Pt1 = None if pack['Pt_next'] is None else pack['Pt_next'].to(dev)[mask]
                v_next = _target_value(Q_next, Sp[mask], At_next1, Pt1)
                y = R_obs[mask] + self.gamma * v_next * (1 - done[mask])

            # Weighted Q-fit: use weights in the loss
            Q_t = self._fit_weighted_q(S[mask], A_joint[mask], y, w)
            self.Q[t] = Q_t
            Q_next = Q_t

            print(f"  t={t}: n_obs={n_obs}/{len(S)}, "
                  f"w mean={w.mean():.3f}, y mean={y.mean():.3f}")

        return self

    def _fit_weighted_q(self, S, A, y, w):
        """Fit Q-function with sample weights via weighted MSE."""
        dev = torch.device(self.device)
        S = S.to(dev, dtype=torch.float32)
        A = A.to(dev, dtype=torch.long)
        y = y.to(dev, dtype=torch.float32)
        w = w.to(dev, dtype=torch.float32)
        n = S.shape[0]

        q = NNQFunction(**self.q_kwargs)
        q.s_mean_ = S.mean(0)
        q.s_std_ = S.std(0).clamp(min=1e-6)
        Sn = (S - q.s_mean_) / q.s_std_

        layers = []
        d_in = q.state_dim
        for d_h in q.hidden:
            layers.append(nn.Linear(d_in, d_h))
            layers.append(nn.ReLU())
            d_in = d_h
        layers.append(nn.Linear(d_in, q.n_actions))
        q.net_ = nn.Sequential(*layers).to(dev)

        optimizer = optim.Adam(q.net_.parameters(), lr=q.lr)
        bs = min(q.batch_size, n)

        q.net_.train()
        for step in range(q.n_steps):
            idx = torch.randint(n, (bs,), device=dev)
            q_all = q.net_(Sn[idx])
            q_sa = q_all.gather(1, A[idx].unsqueeze(1)).squeeze(1)
            # Weighted MSE
            loss = (w[idx] * (q_sa - y[idx]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q.net_.parameters(), 10.0)
            optimizer.step()

        q.net_.eval()
        return q

    def value(self, data: dict) -> float:
        dev = torch.device(self.device)
        mask = data['bloc'] == 1
        S0 = torch.tensor(data['states'][mask], device=dev, dtype=torch.float32)
        At0 = torch.tensor(data['at_joint'][mask], device=dev, dtype=torch.long)
        Q1 = self.Q[1]
        v = _target_value(Q1, S0, At0, _start_probs(data, mask, dev))
        return v.mean().item(), v.std().item() / np.sqrt(len(v))


# ===================== SCOPE (Parbhoo et al. 2020) =====================

class NNSCOPE:
    """
    SCOPE: Shaping Control variates for Off-Policy Evaluation.

    Per-step IS with potential-based reward shaping:
      V_hat = (1/n) sum_i sum_t gamma^t * w_{0:t} * (R_obs + gamma*phi(S') - phi(S))

    Steps:
      1. Estimate behavior policy P(A|S) via softmax NN.
      2. Split eval data: frac_shape for learning phi, rest for IS.
      3. Learn potential phi(s) via NN regression on cumulative return.
      4. Compute per-step IS with shaped rewards.

    Note: With deterministic target policy and 25 actions, IS weights
    are often 0 (when behavior != target action), causing high variance.
    Uses r_obs (0 when missing) — biased under MNAR.
    """
    def __init__(self, state_dim, n_actions=25, gamma=0.99,
                 frac_shape=0.3, phi_kwargs=None, behavior_kwargs=None,
                 w_cap=50.0, device='cpu'):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.frac_shape = frac_shape
        self.w_cap = w_cap
        self.device = device

        self.phi_kwargs = dict(
            hidden=(128, 128), n_steps=2000, lr=1e-3,
            batch_size=512, device=device)
        if phi_kwargs:
            self.phi_kwargs.update(phi_kwargs)

        self.behavior_kwargs = dict(
            n_actions=n_actions, hidden=(128, 128), n_steps=3000,
            lr=1e-3, batch_size=512, device=device)
        if behavior_kwargs:
            self.behavior_kwargs.update(behavior_kwargs)

        self._value = None

    def fit(self, data: dict):
        dev = torch.device(self.device)

        # Step 1: Estimate behavior policy from ALL data
        S_all = torch.tensor(data['states'], device=dev, dtype=torch.float32)
        A_all = torch.tensor(data['a_joint'], device=dev, dtype=torch.long)
        print("  Fitting behavior policy...")
        self.pi_b = _NNSoftmaxPolicy(**self.behavior_kwargs).fit(S_all, A_all)

        # Organize data into trajectories
        bloc = data['bloc']
        T = int(bloc.max())

        # Group rows into trajectories by finding bloc==1 starts
        starts = np.where(bloc == 1)[0]
        n_traj = len(starts)
        trajs = []
        for i in range(n_traj):
            begin = starts[i]
            end = starts[i + 1] if i + 1 < n_traj else len(bloc)
            trajs.append({
                'S': data['states'][begin:end],
                'Sp': data['next_states'][begin:end],
                'A': data['a_joint'][begin:end],
                'At': data['at_joint'][begin:end],
                'Pt': data['pi_target'][begin:end] if 'pi_target' in data else None,
                'R_obs': data['r_obs'][begin:end],
                'O': data['o_t'][begin:end],
            })

        # Step 2: Split trajectories
        rng = np.random.RandomState(42)
        perm = rng.permutation(n_traj)
        n_shape = max(1, int(n_traj * self.frac_shape))
        shape_idx = perm[:n_shape]
        eval_idx = perm[n_shape:]

        # Step 3: Learn phi(s) on shape trajectories
        print("  Learning potential phi(s)...")
        s1_list, ret_list = [], []
        for i in shape_idx:
            traj = trajs[i]
            ret = 0.0
            disc = 1.0
            for tt in range(len(traj['R_obs'])):
                ret += disc * traj['R_obs'][tt]
                disc *= self.gamma
            s1_list.append(traj['S'][0])
            ret_list.append(ret)

        S_shape = torch.tensor(np.stack(s1_list), device=dev, dtype=torch.float32)
        Y_shape = torch.tensor(np.array(ret_list), device=dev, dtype=torch.float32)

        if len(S_shape) >= 10:
            self.phi_model = _NNRegressor(**self.phi_kwargs).fit(S_shape, Y_shape)
        else:
            self.phi_model = None

        # Step 4: Per-step IS with shaping on eval trajectories
        print("  Computing IS estimates...")
        estimates = []
        for i in eval_idx:
            traj = trajs[i]
            T_i = len(traj['A'])
            cum_w = 1.0
            traj_val = 0.0
            disc = 1.0

            for tt in range(T_i):
                s_t = torch.tensor(traj['S'][tt:tt+1], device=dev, dtype=torch.float32)
                a_t = int(traj['A'][tt])
                at_t = int(traj['At'][tt])
                s_next = torch.tensor(traj['Sp'][tt:tt+1], device=dev, dtype=torch.float32)

                # IS weight: pi_target(a|s) / pi_behavior(a|s)
                # Deterministic target: 1 if a == a_target, 0 otherwise
                p_t = 1.0 if traj['Pt'] is None else float(traj['Pt'][tt, a_t])
                if (traj['Pt'] is None and a_t != at_t) or p_t <= 0.0:
                    cum_w = 0.0
                    break  # rest of trajectory has 0 weight

                p_b = self.pi_b.predict_proba(s_t)[0, a_t].item()
                p_b = max(p_b, 1e-8)
                w_t = p_t / p_b
                cum_w *= w_t
                cum_w = min(cum_w, self.w_cap)

                # Shaped reward
                phi_s = self._phi(s_t)
                phi_sp = self._phi(s_next)
                r_shaped = float(traj['R_obs'][tt]) + self.gamma * phi_sp - phi_s

                traj_val += disc * cum_w * r_shaped
                disc *= self.gamma

            estimates.append(traj_val)

        self._value = float(np.mean(estimates)) if estimates else np.nan
        self._se = float(np.std(estimates) / np.sqrt(len(estimates))) if estimates else np.nan
        n_nonzero = sum(1 for e in estimates if abs(e) > 1e-10)
        print(f"  SCOPE: {n_nonzero}/{len(estimates)} trajectories with nonzero weight")

        return self

    def _phi(self, s_tensor):
        if self.phi_model is None:
            return 0.0
        return float(self.phi_model.predict(s_tensor).item())

    def value(self, data: dict = None) -> float:
        if self._value is None:
            raise RuntimeError("Call .fit() first.")
        return self._value, self._se
