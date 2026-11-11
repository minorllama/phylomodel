# Simulate a CTMC on a tree and train a GNN to reconstruct ancestral states.

import numpy as np
import torch

from collections import namedtuple
from scipy.linalg import expm
from scipy.special import logsumexp
from torch import nn
from torch.nn import functional as F

def simulate_sites(adjacency, rates, root, root_probs, n_sites, *, branch_lengths=None, seed=None):
    """Return states (sites × nodes), a tip mask, and edge transition matrices.
    adjacency is symmetric and binary. Key rates and branch_lengths by
    (parent, child) after orienting the tree away from root. Rate matrices
    use the row-generator convention. Branch lengths default to one.
    """
    A = np.asarray(adjacency)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] == 0:
        raise ValueError("adjacency must be a nonempty square matrix")
    assert not (
        not np.all((A == 0) | (A == 1))
        or not np.array_equal(A, A.T)
        or np.any(np.diag(A)))
        

    n = len(A)
    assert not (not 0 <= root < n or n_sites < 1)
    assert not (A.sum() != 2 * (n - 1))

    order = [root]
    seen = {root}
    edges = []
    is_tip = np.ones(n, dtype=bool)
    for u in order:
        for v in np.flatnonzero(A[u]):
            if v in seen:
                continue
            seen.add(v)
            order.append(v)
            edges.append((u, v))
            is_tip[u] = False

    
    assert len(seen) == n
    assert set(rates) == set(edges)
    assert not (branch_lengths is not None and set(branch_lengths) != set(edges))

    pi = np.asarray(root_probs, dtype=float)
    if (
        pi.ndim != 1
        or pi.size < 2
        or not np.all(np.isfinite(pi))
        or np.any(pi < 0)
        or not np.isclose(pi.sum(), 1)
    ):
        raise ValueError("root_probs must be a probability vector")
    pi = pi / pi.sum()
    k = len(pi)

    rng = np.random.default_rng(seed)
    states = np.empty((n_sites, n), dtype=np.int64)
    states[:, root] = rng.choice(k, size=n_sites, p=pi)
    transitions = {}

    for u, v in edges:
        Q = np.asarray(rates[u, v], dtype=float)
        if Q.shape != (k, k) or not np.all(np.isfinite(Q)):
            raise ValueError(f"invalid matrix shape or entries on edge {(u, v)}")
        off_diagonal = Q.copy()
        np.fill_diagonal(off_diagonal, 0.0)
        if np.any(off_diagonal < 0) or not np.allclose(Q.sum(axis=1), 0):
            raise ValueError(f"invalid CTMC generator on edge {(u, v)}")

        t = 1.0 if branch_lengths is None else float(branch_lengths[u, v])
        if not np.isfinite(t) or t < 0:
            raise ValueError(f"invalid branch length on edge {(u, v)}")
        P = np.maximum(expm(t * Q), 0.0)  # Remove roundoff below zero.
        P /= P.sum(axis=1, keepdims=True)
        transitions[u, v] = P

        cdf = P.cumsum(axis=1)
        cdf[:, -1] = 1.0
        draws = rng.random((n_sites, 1))
        states[:, v] = (draws >= cdf[states[:, u]]).sum(axis=1)

    return states, is_tip, transitions


def graph_inputs(transitions):
    """Build two directed messages per branch, with a direction flag."""
    src, dst, features = [], [], []
    for (u, v), P in transitions.items():
        src.extend([u, v])
        dst.extend([v, u])
        features.extend([np.r_[P.ravel(), 1.0], np.r_[P.ravel(), -1.0]])
    return (
        torch.tensor(src, dtype=torch.long),
        torch.tensor(dst, dtype=torch.long),
        torch.tensor(np.stack(features), dtype=torch.float32),
    )


def node_inputs(states, is_tip, root, pi):
    """One-hot tip observations and root prior, with zeros elsewhere."""
    x = np.zeros((*states.shape, 2 * len(pi)), dtype=np.float32)
    x[:, is_tip, :len(pi)] = np.eye(len(pi))[states[:, is_tip]]
    x[:, root, len(pi):] = pi
    return torch.from_numpy(x)


class TreeGNN(nn.Module):
    def __init__(self, k, width=48, rounds=4):
        super().__init__()
        self.rounds = rounds
        self.encode = nn.Linear(2 * k, width)
        self.message = nn.Sequential(
            nn.Linear(width + k * k + 1, width),
            nn.ReLU(),
            nn.Linear(width, width),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * width, width),
            nn.ReLU(),
            nn.Linear(width, width),
        )
        self.readout = nn.Linear(width, k)

    def forward(self, x, src, dst, edge_features):
        h = self.encode(x)
        edge_features = edge_features.unsqueeze(0).expand(x.size(0), -1, -1)

        for _ in range(self.rounds):
            messages = self.message(torch.cat((h[:, src], edge_features), dim=-1))
            incoming = torch.zeros_like(h)
            incoming.index_add_(1, dst, messages)
            h = h + self.update(torch.cat((h, incoming), dim=-1))

        return self.readout(h)  # [sites, nodes, states]


def exact_root_posterior(tip_states, is_tip, transitions, root, pi):
    """Compute the posterior with Felsenstein pruning for simulated tip patterns."""
    n_sites = len(tip_states)
    k = len(pi)
    likelihood = np.ones((n_sites, len(is_tip), k))
    likelihood[:, is_tip] = np.eye(k)[tip_states]

    # The simulator returns edges in parent-before-child order.
    for (u, v), P_uv in reversed(list(transitions.items())):
        likelihood[:, u] *= likelihood[:, v] @ P_uv.T
        likelihood[:, u] /= likelihood[:, u].sum(axis=1, keepdims=True)

    posterior = likelihood[:, root] * pi
    return posterior / posterior.sum(axis=1, keepdims=True)


def exact_node_posteriors(tip_states, is_tip, transitions, root, pi):
    """Return exact state probabilities with shape [sites, nodes, states].

    Pass transitions in parent-before-child order, as returned by simulate_sites.
    Columns of tip_states follow the True entries in is_tip.
    """
    n_sites = len(tip_states)
    k = len(pi)
    children = [[] for _ in is_tip]
    order = [root]
    with np.errstate(divide="ignore"):
        log_pi = np.log(pi)
        log_transitions = {}
        for (u, v), P_uv in transitions.items():
            log_transitions[u, v] = np.log(P_uv)
            children[u].append(v)
            order.append(v)

    # Upward pass. log_inside at a node describes evidence from its subtree.
    log_inside = np.zeros((n_sites, len(is_tip), k))
    log_inside[:, is_tip] = np.where(
        tip_states[..., None] == np.arange(k), 0.0, -np.inf
    )
    upward = {}
    for u, v in reversed(log_transitions):
        upward[u, v] = logsumexp(
            log_transitions[u, v] + log_inside[:, v, None, :], axis=2
        )
        log_inside[:, u] += upward[u, v]

    # Downward pass. Include the parent's outside evidence and the child's
    # siblings. Prefix and suffix sums avoid subtracting log(0).
    log_outside = np.empty_like(log_inside)
    log_outside[:, root] = log_pi
    for u in order:
        prefix = [log_outside[:, u]]
        for v in children[u]:
            prefix.append(prefix[-1] + upward[u, v])

        suffix = np.zeros((n_sites, k))
        for i in range(len(children[u]) - 1, -1, -1):
            v = children[u][i]
            log_context = prefix[i] + suffix
            log_outside[:, v] = logsumexp(
                log_context[:, :, None] + log_transitions[u, v], axis=1
            )
            suffix += upward[u, v]

    log_joint = log_inside + log_outside
    log_evidence = logsumexp(log_joint, axis=2, keepdims=True)
    if np.any(~np.isfinite(log_evidence)):
        raise ValueError("tip pattern has zero probability under the tree model")
    return np.exp(log_joint - log_evidence)

def exact_ancestral_diagnostics(states, is_tip, transitions, root, pi):
    """score exact inference against the simulated states at internal nodes.
    cross-entropy scores the realized labels. Conditional entropy averages
    over the exact posterior at each site, estimating the population minimum
    cross-entropy using the supplied tip patterns.
    """
    internal = ~is_tip
    if not np.any(internal):
        raise ValueError("ancestral diagnostics require at least one internal node")
    posteriors = exact_node_posteriors(
        states[:, is_tip], is_tip, transitions, root, pi
    )
    probabilities = posteriors[:, internal]
    labels = states[:, internal]
    assigned = np.take_along_axis(probabilities, labels[..., None], axis=2)
    root_assigned = posteriors[np.arange(len(states)), root, states[:, root]]
    with np.errstate(divide="ignore"):
        cross_entropy = -np.log(assigned).mean()
        root_cross_entropy = -np.log(root_assigned).mean()
        positive = probabilities[probabilities > 0]
        conditional_entropy = -(positive * np.log(positive)).sum() / labels.size
    return {
        "cross_entropy": float(cross_entropy),
        "conditional_entropy": float(conditional_entropy),
        "accuracy": float((probabilities.argmax(axis=2) == labels).mean()),
        "root_cross_entropy": float(root_cross_entropy),
    }


def main():
    Config = namedtuple("Config", ["epochs", "train_sites", "test_sites"])
    args  = Config(50, 12000, 3000)

    # Seven-node example, with root 0 and tips 3, 4, 5, and 6.
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
    adjacency = np.zeros((7, 7), dtype=int)
    rng = np.random.default_rng(7)
    k = 4
    root = 0
    pi = np.full(k, 1 / k)
    rates = {}
    lengths = {}

    for u, v in edges:
        adjacency[u, v] = adjacency[v, u] = 1
        Q = rng.uniform(0.1, 1.0, size=(k, k))
        np.fill_diagonal(Q, 0.0)
        np.fill_diagonal(Q, -Q.sum(axis=1))
        rates[u, v] = Q
        lengths[u, v] = rng.uniform(0.1, 0.5)

    train, is_tip, transitions = simulate_sites(
        adjacency, rates, root, pi, args.train_sites,
        branch_lengths=lengths, seed=10,
    )
    test, _, _ = simulate_sites(
        adjacency, rates, root, pi, args.test_sites,
        branch_lengths=lengths, seed=11,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    graph = tuple(t.to(device) for t in graph_inputs(transitions))
    internal = torch.from_numpy(~is_tip).to(device)
    train_x = node_inputs(train, is_tip, root, pi).to(device)
    train_y = torch.from_numpy(train).to(device)[:, internal]

    torch.manual_seed(12)
    model = TreeGNN(k).to(device)

    #graph = graph_inputs(transitions)
    #internal = torch.from_numpy(~is_tip)
    #train_x = node_inputs(train, is_tip, root, pi)
    #train_y = torch.from_numpy(train)[:, internal]

    #torch.set_num_threads(1)  # The batches in this CPU example are small.
    #torch.manual_seed(12)
    #model = TreeGNN(k)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in torch.randperm(len(train), device=device).split(256):
        #for batch in torch.randperm(len(train)).split(256):
            logits = model(train_x[batch], *graph)
            loss = F.cross_entropy(
                logits[:, internal].reshape(-1, k), train_y[batch].reshape(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)

        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            print(f"Epoch {epoch + 1:>2} | train loss {total_loss / len(train):.3f}")

    model.eval()
    with torch.no_grad():
        test_x = node_inputs(test, is_tip, root, pi).to(device)
        logits = model(test_x, *graph)
        labels = torch.from_numpy(test).to(device)

        #logits = model(node_inputs(test, is_tip, root, pi), *graph)
        #labels = torch.from_numpy(test)
        predictions = logits.argmax(dim=-1)
        accuracy = (
            predictions[:, internal] == labels[:, internal]
        ).float().mean().item()
        gnn_nll = F.cross_entropy(logits[:, root], labels[:, root]).item()

    posterior = exact_root_posterior(
        test[:, is_tip], is_tip, transitions, root, pi
    )
    exact_nll = -np.log(posterior[np.arange(len(test)), test[:, root]]).mean()
    print(f"held-out internal-node accuracy: {accuracy:.3f}")
    print(f"held-out root negative log-likelihood | GNN {gnn_nll:.3f} | exact {exact_nll:.3f}")

    stats = exact_ancestral_diagnostics(test, is_tip, transitions, root, pi)
    print(stats)

if __name__ == "__main__":
    main()

"""
Using cuda
Epoch  1 | train loss 1.031
Epoch  5 | train loss 0.817
Epoch 10 | train loss 0.789
Epoch 15 | train loss 0.779
Epoch 20 | train loss 0.785
Epoch 25 | train loss 0.778
Epoch 30 | train loss 0.776
Epoch 35 | train loss 0.776
Epoch 40 | train loss 0.775
Epoch 45 | train loss 0.770
Epoch 50 | train loss 0.772
held-out internal-node accuracy: 0.701
held-out root negative log-likelihood | GNN 1.028 | exact 1.010
{'cross_entropy': 0.7574514214613214, 'conditional_entropy': 0.7660633761471697, 'accuracy': 0.7075555555555556, 'root_cross_entropy': 1.0102124171920617}
"""
