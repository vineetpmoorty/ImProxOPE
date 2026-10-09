"""
Copied from ShadOPE (Wei, Qu, Miao, ICML 2026), https://github.com/NAIVlab/ShadOPE @ 4231ba5, file src/OPE/nn_bridge.py.
Changes from the original: none.

Neural network bridge estimator via Adversarial GMM (AGMM).

Replaces the RKHS minimax bridge with neural networks, following:
  Dikkala et al. (2020) "Minimax Estimation of Conditional Moment Models"
  Section 6: Neural Networks / AGMM.

The conditional moment restriction for the bridge function:
  E[R_t - b(S_{t+1}, S_t, A_t) | R_t, S_t, A_t] = 0

is solved via the minimax objective:
  min_theta max_w  (1/n) sum_i [R_i - h_theta(X_i)] f_w(Z_i)
                   - lambda * ||f_w||^2_F
                   + mu * ||h_theta||^2_H

where:
  X = (S_{t+1}, S_t, A_t)  -- hypothesis input (what bridge depends on)
  Z = (R_t, S_t, A_t)      -- instrument/condition input
  y = R_t                   -- outcome

Interface matches RKHSIV: .fit(X, y, condition) and .predict(X).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class _MLP(nn.Module):
    """Simple MLP with configurable hidden layers."""
    def __init__(self, input_dim, hidden_dims=(128, 128), output_dim=1):
        super().__init__()
        layers = []
        d_in = input_dim
        for d_h in hidden_dims:
            layers.append(nn.Linear(d_in, d_h))
            layers.append(nn.ReLU())
            d_in = d_h
        layers.append(nn.Linear(d_in, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NNBridge:
    """
    Neural network bridge estimator via AGMM.

    Parameters
    ----------
    h_hidden : tuple of int
        Hidden layer sizes for the hypothesis (bridge) network h_theta.
    f_hidden : tuple of int
        Hidden layer sizes for the adversary (test function) network f_w.
    n_steps : int
        Number of adversarial training steps.
    lr_h : float
        Learning rate for the bridge network.
    lr_f : float
        Learning rate for the adversary network.
    lambda_f : float
        Penalty on adversary's second moment (variance regularization).
    mu_h : float
        Weight decay / L2 penalty on bridge network.
    n_critic : int
        Number of adversary updates per bridge update.
    batch_size : int
        Mini-batch size. Use 0 or None for full batch.
    device : str
        'cpu' or 'cuda'.
    """
    def __init__(self,
                 h_hidden=(128, 128),
                 f_hidden=(128, 128),
                 n_steps=2000,
                 lr_h=1e-3,
                 lr_f=1e-3,
                 lambda_f=0.1,
                 mu_h=1e-4,
                 n_critic=5,
                 batch_size=512,
                 device='cpu'):
        self.h_hidden = h_hidden
        self.f_hidden = f_hidden
        self.n_steps = n_steps
        self.lr_h = lr_h
        self.lr_f = lr_f
        self.lambda_f = lambda_f
        self.mu_h = mu_h
        self.n_critic = n_critic
        self.batch_size = batch_size
        self.device = device if torch.cuda.is_available() or device == 'cpu' else 'cpu'

    def fit(self, X, y, condition):
        """
        Fit the bridge function h_theta via AGMM.

        Parameters
        ----------
        X : array-like, shape (n, d_x)
            Hypothesis input: (S_{t+1}, S_t, A_t).
        y : array-like, shape (n,) or (n, 1)
            Outcome: R_t.
        condition : array-like, shape (n, d_z)
            Instrument/condition input: (R_t, S_t, A_t).

        Returns
        -------
        self
        """
        dev = torch.device(self.device)

        X = self._to_tensor(X, dev)
        y = self._to_tensor(y, dev).reshape(-1, 1)
        Z = self._to_tensor(condition, dev)
        n = X.shape[0]

        # Standardize inputs
        self.x_mean_ = X.mean(0)
        self.x_std_ = X.std(0).clamp(min=1e-6)
        self.z_mean_ = Z.mean(0)
        self.z_std_ = Z.std(0).clamp(min=1e-6)
        self.y_mean_ = y.mean()
        self.y_std_ = y.std().clamp(min=1e-6)

        Xn = (X - self.x_mean_) / self.x_std_
        Zn = (Z - self.z_mean_) / self.z_std_
        yn = (y - self.y_mean_) / self.y_std_

        d_x = Xn.shape[1]
        d_z = Zn.shape[1]

        # Build networks
        self.h_net_ = _MLP(d_x, self.h_hidden, 1).to(dev)
        f_net = _MLP(d_z, self.f_hidden, 1).to(dev)

        opt_h = optim.Adam(self.h_net_.parameters(), lr=self.lr_h, weight_decay=self.mu_h)
        opt_f = optim.Adam(f_net.parameters(), lr=self.lr_f)

        bs = self.batch_size if self.batch_size and self.batch_size < n else n

        for step in range(self.n_steps):
            # --- Adversary update (maximize) ---
            for _ in range(self.n_critic):
                idx = torch.randint(n, (bs,), device=dev)
                xb, zb, yb = Xn[idx], Zn[idx], yn[idx]

                with torch.no_grad():
                    h_val = self.h_net_(xb)
                residual = yb - h_val  # (bs, 1)
                f_val = f_net(zb)      # (bs, 1)

                # Adversary objective: maximize E[residual * f] - lambda * E[f^2]
                moment = (residual * f_val).mean()
                penalty = self.lambda_f * (f_val ** 2).mean()
                loss_f = -(moment - penalty)  # negate for minimization

                opt_f.zero_grad()
                loss_f.backward()
                opt_f.step()

            # --- Bridge update (minimize) ---
            idx = torch.randint(n, (bs,), device=dev)
            xb, zb, yb = Xn[idx], Zn[idx], yn[idx]

            h_val = self.h_net_(xb)
            residual = yb - h_val
            f_val = f_net(zb).detach()

            moment = (residual * f_val).mean()
            loss_h = moment  # minimize the moment violation

            opt_h.zero_grad()
            loss_h.backward()
            opt_h.step()

        self.h_net_.eval()
        return self

    def predict(self, X):
        """
        Predict bridge values h_theta(X).

        Parameters
        ----------
        X : array-like, shape (m, d_x)

        Returns
        -------
        predictions : torch.Tensor, shape (m,)
        """
        dev = torch.device(self.device)
        X = self._to_tensor(X, dev)
        Xn = (X - self.x_mean_) / self.x_std_

        with torch.no_grad():
            pred_n = self.h_net_(Xn)

        # Undo y standardization
        pred = pred_n * self.y_std_ + self.y_mean_
        return pred.reshape(-1)

    @staticmethod
    def _to_tensor(x, dev):
        if isinstance(x, torch.Tensor):
            return x.to(device=dev, dtype=torch.float32)
        return torch.tensor(np.asarray(x), device=dev, dtype=torch.float32)


class NNBridgeCV(NNBridge):
    """
    NNBridge with cross-validation over lambda_f and mu_h.

    Tries a grid of (lambda_f, mu_h) pairs, picks the one with lowest
    held-out moment violation.
    """
    def __init__(self,
                 h_hidden=(128, 128),
                 f_hidden=(128, 128),
                 n_steps=2000,
                 lr_h=1e-3,
                 lr_f=1e-3,
                 lambda_f_grid=(0.01, 0.1, 1.0),
                 mu_h_grid=(1e-5, 1e-4, 1e-3),
                 n_critic=5,
                 batch_size=512,
                 cv=3,
                 device='cpu'):
        super().__init__(h_hidden=h_hidden, f_hidden=f_hidden,
                         n_steps=n_steps, lr_h=lr_h, lr_f=lr_f,
                         lambda_f=0.1, mu_h=1e-4,
                         n_critic=n_critic, batch_size=batch_size,
                         device=device)
        self.lambda_f_grid = lambda_f_grid
        self.mu_h_grid = mu_h_grid
        self.cv = cv

    def fit(self, X, y, condition):
        """CV over hyperparams, then refit on full data with best params."""
        dev = torch.device(self.device)
        X = self._to_tensor(X, dev)
        y = self._to_tensor(y, dev).reshape(-1, 1)
        Z = self._to_tensor(condition, dev)
        n = X.shape[0]

        # CV split indices
        idx = torch.randperm(n, device=dev)
        fold_size = n // self.cv
        folds = [idx[i * fold_size:(i + 1) * fold_size] for i in range(self.cv)]

        best_score = float('inf')
        best_lf, best_mu = self.lambda_f, self.mu_h

        for lf in self.lambda_f_grid:
            for mu in self.mu_h_grid:
                scores = []
                for k in range(self.cv):
                    val_idx = folds[k]
                    tr_idx = torch.cat([folds[j] for j in range(self.cv) if j != k])

                    est = NNBridge(
                        h_hidden=self.h_hidden, f_hidden=self.f_hidden,
                        n_steps=self.n_steps // 2,  # fewer steps for CV
                        lr_h=self.lr_h, lr_f=self.lr_f,
                        lambda_f=lf, mu_h=mu,
                        n_critic=self.n_critic,
                        batch_size=self.batch_size,
                        device=self.device)
                    est.fit(X[tr_idx], y[tr_idx], Z[tr_idx])

                    # Evaluate: moment violation on validation set
                    pred = est.predict(X[val_idx])
                    residual = y[val_idx].reshape(-1) - pred
                    # Use squared moment as score (lower = better)
                    score = residual.pow(2).mean().item()
                    scores.append(score)

                avg = np.mean(scores)
                if avg < best_score:
                    best_score = avg
                    best_lf, best_mu = lf, mu

        # Refit on full data with best params
        self.lambda_f = best_lf
        self.mu_h = best_mu
        return super().fit(X, y, condition)
