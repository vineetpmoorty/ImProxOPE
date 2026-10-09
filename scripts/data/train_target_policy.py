"""Copied from ShadOPE (Wei, Qu, Miao, ICML 2026), https://github.com/NAIVlab/ShadOPE @ 4231ba5, file sepsis/train_dqn_sepsis.py."""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os


# ---- Q-Network ----
class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ---- Replay Buffer ----
class ReplayBuffer:
    def __init__(self, capacity=200000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buf.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s_n, d = zip(*batch)
        return (np.array(s, dtype=np.float32),
                np.array(a, dtype=np.int64),
                np.array(r, dtype=np.float32),
                np.array(s_n, dtype=np.float32),
                np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


def get_state_cols(df):
    """State columns: paper's 48 features."""
    drop = ['icustayid', 'vaso_input', 'iv_input', 'reward']
    return [c for c in df.columns if c not in drop]


def load_transitions(csv_path):
    """Load clean CSV and build (s, a, r, s_next, done) transitions."""
    df = pd.read_csv(csv_path)

    state_cols = get_state_cols(df)
    print(f"State dim: {len(state_cols)}")

    # Standardize state features
    means = df[state_cols].mean()
    stds = df[state_cols].std().replace(0, 1)
    df[state_cols] = (df[state_cols] - means) / stds

    # Joint action: vaso * 5 + iv
    df['action'] = (df['vaso_input'] * 5 + df['iv_input']).astype(int)

    T = 10
    states, actions, rewards, next_states, dones = [], [], [], [], []

    for _, traj in df.groupby('icustayid'):
        traj = traj.sort_values('bloc')
        s = traj[state_cols].values
        a = traj['action'].values
        r = traj['reward'].values

        for t in range(T - 1):
            states.append(s[t])
            actions.append(a[t])
            rewards.append(r[t])
            next_states.append(s[t + 1])
            dones.append(0.0)

        # Last step: terminal
        states.append(s[T - 1])
        actions.append(a[T - 1])
        rewards.append(r[T - 1])
        next_states.append(s[T - 1])  # absorbing
        dones.append(1.0)

    return (np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            state_cols, means.values, stds.values)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)

    # Load clean data
    states, actions, rewards, next_states, dones, state_cols, means, stds = \
        load_transitions(args.input)
    n_transitions = len(states)
    state_dim = states.shape[1]  # 48
    n_actions = 25
    print(f"Transitions: {n_transitions}, State dim: {state_dim}, Actions: {n_actions}")

    # Fill replay buffer
    buf = ReplayBuffer(capacity=n_transitions)
    for i in range(n_transitions):
        buf.push(states[i], actions[i], rewards[i], next_states[i], dones[i])

    # Networks
    q_net = QNetwork(state_dim, n_actions, args.hidden).to(device)
    q_target = QNetwork(state_dim, n_actions, args.hidden).to(device)
    q_target.load_state_dict(q_net.state_dict())
    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)

    # Training
    gamma = args.gamma
    losses = []

    for step in range(1, args.n_steps + 1):
        s, a, r, s_n, d = buf.sample(args.batch_size)
        s_t = torch.tensor(s, device=device)
        a_t = torch.tensor(a, device=device).unsqueeze(1)
        r_t = torch.tensor(r, device=device)
        sn_t = torch.tensor(s_n, device=device)
        d_t = torch.tensor(d, device=device)

        # Current Q
        q_vals = q_net(s_t).gather(1, a_t).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            best_a = q_net(sn_t).argmax(dim=1, keepdim=True)
            q_next = q_target(sn_t).gather(1, best_a).squeeze(1)
            target = r_t + gamma * q_next * (1 - d_t)

        loss = nn.functional.mse_loss(q_vals, target)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
        optimizer.step()

        losses.append(loss.item())

        # Update target network
        if step % args.target_update == 0:
            q_target.load_state_dict(q_net.state_dict())

        if step % args.log_every == 0:
            avg_loss = np.mean(losses[-args.log_every:])
            print(f"Step {step}/{args.n_steps}  loss={avg_loss:.4f}")

    # Save model
    os.makedirs(args.outdir, exist_ok=True)
    model_path = os.path.join(args.outdir, 'dqn_sepsis.pt')
    torch.save({
        'state_dict': q_net.state_dict(),
        'state_dim': state_dim,
        'n_actions': n_actions,
        'hidden': args.hidden,
        'state_cols': state_cols,
        'means': means,
        'stds': stds,
        'gamma': gamma,
    }, model_path)
    print(f"\nSaved model to {model_path}")

    # Policy stats on train data
    q_net.eval()
    with torch.no_grad():
        all_s = torch.tensor(states, device=device)
        greedy_actions = q_net(all_s).argmax(dim=1).cpu().numpy()
        vaso_pred = greedy_actions // 5
        iv_pred = greedy_actions % 5

    print(f"\nLearned policy action distribution (train):")
    print(f"  vaso: {dict(zip(*np.unique(vaso_pred, return_counts=True)))}")
    print(f"  iv:   {dict(zip(*np.unique(iv_pred, return_counts=True)))}")

    # Apply DQN target actions to the original data and save
    apply_policy(q_net, args.input, state_cols, means, stds, device, args.outdir)


def apply_policy(q_net, csv_path, state_cols, means, stds, device, outdir):
    """Apply learned DQN policy to all data, save vaso_target / iv_target."""
    print(f"\nApplying policy to: {csv_path}")
    df = pd.read_csv(csv_path)

    # Standardize with train stats
    df_norm = df.copy()
    for i, col in enumerate(state_cols):
        df_norm[col] = (df[col] - means[i]) / stds[i]

    s = df_norm[state_cols].values.astype(np.float32)

    q_net.eval()
    with torch.no_grad():
        s_t = torch.tensor(s, device=device)
        greedy = q_net(s_t).argmax(dim=1).cpu().numpy()

    df['vaso_target'] = greedy // 5
    df['iv_target'] = greedy % 5

    out_path = os.path.join(outdir, 'sepsis_T10_with_targets.csv')
    df.to_csv(out_path, index=False)

    print(f"  Target policy applied to {len(df)} rows")
    print(f"  vaso_target: {dict(zip(*np.unique(df['vaso_target'].values, return_counts=True)))}")
    print(f"  iv_target:   {dict(zip(*np.unique(df['iv_target'].values, return_counts=True)))}")
    print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='realdata/sepsis_T10.csv')
    parser.add_argument('--outdir', type=str, default='realdata')
    parser.add_argument('--n_steps', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--target_update', type=int, default=500)
    parser.add_argument('--log_every', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
