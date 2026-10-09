"""Clinician-policy clone and DQN Q-values per row, for the epsilon-mixture target policies.

The clone is ShadOPE's softmax behaviour network (`_NNSoftmaxPolicy`, the model its SCOPE baseline
uses), trained once on all stays like the DQN target. With ShadOPE's sepsis settings (512-512-256,
8000 steps) it memorizes the training stays (held-out log-loss far above the marginal-frequency
baseline), so the network size and number of steps are chosen by held-out log-loss on 20% of stays;
the saved clone is then refitted on all stays with the chosen settings. The DQN's
Q-values (mimic_processed/dqn_sepsis.pt) are evaluated on every row as well, so target policies
can be built without the networks. Output (mimic_processed/clinician_policy.npz):
icustayid, bloc, clin (N, 25) clone probabilities, q_dqn (N, 25) DQN Q-values.

    python scripts/data/train_clinician_policy.py --device cuda
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from improxope.nn_fqe import _NNSoftmaxPolicy

N_ACTIONS = 25


CANDIDATES = [((512, 512, 256), n) for n in (8000, 2000, 1000, 500, 250)] + \
             [((256, 256), n) for n in (2000, 1000, 500, 250)]


def fit_clone(S, A, hidden, n_steps, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    return _NNSoftmaxPolicy(n_actions=N_ACTIONS, hidden=hidden, n_steps=n_steps, lr=1e-3,
                            batch_size=4096, device=device).fit(
        torch.as_tensor(S, device=dev), torch.as_tensor(A, device=dev))


def nll(p, A):
    return float(-np.log(np.maximum(p[np.arange(len(A)), A], 1e-12)).mean())


def probs(model, S):
    return model.predict_proba(torch.as_tensor(S)).cpu().numpy().astype(np.float32)


def dqn_q_values(path, df, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    h = ck['hidden']
    net = nn.Sequential(nn.Linear(ck['state_dim'], h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                        nn.Linear(h, ck['n_actions'])).to(device)
    net.load_state_dict({k.replace('net.', '', 1): v for k, v in ck['state_dict'].items()})
    net.eval()
    S = ((df[ck['state_cols']].values - np.asarray(ck['means'])) / np.asarray(ck['stds'])).astype(np.float32)
    with torch.no_grad():
        return net(torch.as_tensor(S, device=device)).cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='mimic_processed')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(args.data_dir, 'sepsis_T10_with_targets.csv'))
    df = df.sort_values(['icustayid', 'bloc']).reset_index(drop=True)
    state_cols = [c for c in df.columns if c not in
                  ('icustayid', 'vaso_input', 'iv_input', 'reward', 'vaso_target', 'iv_target')]
    assert len(state_cols) == 48, len(state_cols)
    S = df[state_cols].values.astype(np.float32)
    A = (df['vaso_input'] * 5 + df['iv_input']).values.astype(np.int64)

    # Model selection by held-out log-loss: fit on 80% of stays, score on the rest.
    ids = np.sort(df['icustayid'].unique())
    rng = np.random.RandomState(args.seed)
    rng.shuffle(ids)
    test = df['icustayid'].isin(ids[int(0.8 * len(ids)):]).values
    base = np.bincount(A[~test], minlength=N_ACTIONS) / (~test).sum()
    print(f'marginal-frequency baseline: held-out NLL {-np.log(base[A[test]]).mean():.3f}')
    scores = []
    for hidden, n_steps in CANDIDATES:
        m = fit_clone(S[~test], A[~test], hidden, n_steps, args.device, args.seed)
        p_tr, p_te = probs(m, S[~test]), probs(m, S[test])
        scores.append((nll(p_te, A[test]), hidden, n_steps))
        print(f'clone {str(hidden):16s} {n_steps:5d} steps: NLL train {nll(p_tr, A[~test]):.3f} '
              f'held-out {scores[-1][0]:.3f}, held-out accuracy {(p_te.argmax(1) == A[test]).mean():.3f}', flush=True)
    best_nll, hidden, n_steps = min(scores)
    print(f'chosen: {hidden}, {n_steps} steps (held-out NLL {best_nll:.3f})')

    # Final clone on all stays, and DQN Q-values on every row.
    clin = probs(fit_clone(S, A, hidden, n_steps, args.device, args.seed), S)
    q = dqn_q_values(os.path.join(args.data_dir, 'dqn_sepsis.pt'), df, args.device)
    a_dqn = (df['vaso_target'] * 5 + df['iv_target']).values.astype(np.int64)
    assert (q.argmax(1) == a_dqn).mean() > 0.99, 'DQN Q-values do not reproduce the saved target actions'

    p_dqn = clin[np.arange(len(df)), a_dqn]
    print('\nclone probability of the DQN action (before the O_{t-1} = 0 dose rule):')
    print('  quantiles 10/25/50/75/90%:', np.round(np.quantile(p_dqn, [.1, .25, .5, .75, .9]), 4))
    for tau in (0.01, 0.02, 0.05, 0.1, 0.2):
        print(f'  share of rows with p >= {tau:<4}: {(p_dqn >= tau).mean():.3f}')

    out = os.path.join(args.data_dir, 'clinician_policy.npz')
    np.savez_compressed(out, icustayid=df['icustayid'].values, bloc=df['bloc'].values, clin=clin, q_dqn=q,
                        hidden=np.array(hidden), n_steps=n_steps, heldout_nll=best_nll)
    print(f'\nsaved {out}: {len(df)} rows')


if __name__ == '__main__':
    main()
