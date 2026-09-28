"""Selection-only audit for the APEX revision (no model training, CPU, ~1 minute).

Reproduces three findings cited in docs/APEX_revision_plan.md, section 1:

  1. Lock-in: Oort, FedCS and TiFL (and to a lesser degree PoC) as implemented
     select a fixed subset of clients for the whole run. Their Gini values match
     the submitted Table II (0.800 / 0.800 / 0.796).
  2. Thompson signal: at the reward scale the simulator produces (~1e-3 per
     round), APEX v2's posterior means stay ~1e-4 while its sampling noise is
     ~2e-2, so planted high-value clients are not preferred.
  3. Selection time: APEX v2's greedy diversity step scales as O(N * K^2 * L).
  4. Heterogeneity scalar: the submitted estimate (sampled sqrt-JSD / 0.6)
     saturates near 1 for alpha <= 0.3; compared with I(Z;Y)/H(Y).

Client losses and rewards are synthetic stand-ins with realistic magnitudes; the
point is the selectors' behaviour, not accuracy.

Usage:  python scripts/apex_selection_audit.py
"""
import importlib
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from csfl_simulator.core.client import ClientInfo  # noqa: E402
from csfl_simulator.core.system import init_system_state, simulate_round_env  # noqa: E402


def make_clients(N, L=10, alpha=0.3, seed=0):
    rs = np.random.RandomState(seed)
    clients = []
    for i in range(N):
        p = np.ones(L) / L if alpha is None else rs.dirichlet([alpha] * L)
        n = 50000 // N
        h = np.bincount(rs.choice(L, n, p=p), minlength=L)
        clients.append(ClientInfo(id=i, data_size=n,
                                  label_histogram={k: float(v) for k, v in enumerate(h) if v > 0}))
    random.seed(seed)
    init_system_state(clients, {})
    return clients


def mi_index(clients, L=10):
    """H = I(Z;Y)/H(Y): mutual information between client index and label, normalised."""
    P = np.array([[c.label_histogram.get(k, 0.0) for k in range(L)] for c in clients])
    n = P.sum(1)
    p = P / n[:, None]
    pbar = P.sum(0) / P.sum()
    kl = np.array([np.sum(pi[pi > 0] * np.log(pi[pi > 0] / pbar[pi > 0])) for pi in p])
    h_y = -np.sum(pbar[pbar > 0] * np.log(pbar[pbar > 0]))
    return float((n / n.sum() * kl).sum() / h_y)


def gini(x):
    x = np.asarray(x, float)
    s = x.sum()
    return float(np.abs(x[:, None] - x[None, :]).sum() / (2 * len(x) * s)) if s > 0 else 0.0


def run(method, N=50, K=10, T=200, seed=0, good=None, signal=True):
    mod = importlib.import_module(f"csfl_simulator.selection.{method}")
    clients = make_clients(N, seed=seed)
    rs = np.random.RandomState(seed + 1)
    random.seed(seed + 1)
    history = {"state": {}, "selected": []}
    good = good if good is not None else set()
    for t in range(T):
        simulate_round_env(clients, {}, t)
        ids, _, state = mod.select_clients(t, K, clients, history, random, None, "cpu")
        history["selected"].append(ids)
        if state:
            history["state"].update(state)
        for cid in ids:
            clients[cid].last_selected_round = t
            clients[cid].participation_count += 1
            clients[cid].last_loss = 2.3 * math.exp(-t / 80) + 0.3 + 0.2 * rs.rand()
            clients[cid].grad_norm = 1.0 + rs.rand()
        # Same form as core/simulator.py: reward = change of 0.6*acc + ..., with
        # per-round accuracy changes of order 1e-3 and noise of order 5e-3.
        frac_good = len(set(ids) & good) / K
        dacc = (0.003 * math.exp(-t / 80)
                + (0.004 * (frac_good - 0.2) if signal else 0.0)
                + rs.normal(0, 0.006))
        history["state"]["last_reward"] = 0.6 * dacc
    return [c.participation_count for c in clients], history


def main():
    print("== Lock-in check (N=50, K=10, T=200) ==")
    for m in ["baseline.fedavg", "system_aware.oort", "system_aware.fedcs",
              "system_aware.tifl", "system_aware.poc", "ml.apex_v2"]:
        part, _ = run(m)
        print(f"{m:24s} unique clients ever selected={sum(p > 0 for p in part):3d}/50  Gini={gini(part):.3f}")

    print("\n== APEX v2 Thompson signal check: 10 planted 'good' clients raise cohort reward ==")
    good = set(range(0, 50, 5))
    for sig in (True, False):
        shares = []
        for s in range(5):
            _, history = run("ml.apex_v2", seed=s, good=good, signal=sig)
            late = [c for ids in history["selected"][100:] for c in ids]
            shares.append(np.mean([c in good for c in late]))
        st = history["state"]["apex_v2_state"]
        mu = np.array([st["ts_mu"].get(i, 0.0) for i in range(50)])
        print(f"reward signal={'ON ' if sig else 'OFF'}: share of good clients in rounds 100-200 = "
              f"{np.mean(shares):.3f} (random = 0.200); posterior mean |mu| ~ {np.abs(mu).mean():.2e}; "
              f"mean mu good={mu[list(good)].mean():.2e} vs rest={np.delete(mu, list(good)).mean():.2e}")
    n = 40
    print(f"TS sample std at n={n} with variance floor 0.1/sqrt(n): "
          f"{math.sqrt(0.1 / math.sqrt(n) / n):.3f}  (contextual score range is [0,1])")

    print("\n== APEX v2 selection time vs N (K = N/10, this machine's CPU) ==")
    mod = importlib.import_module("csfl_simulator.selection.ml.apex_v2")
    for N in (50, 200, 500, 1000):
        clients = make_clients(N)
        history = {"state": {}, "selected": []}
        random.seed(0)
        for c in clients:
            c.last_loss = 1.0 + random.random()
            c.grad_norm = random.random()
        times = []
        for t in range(6):
            simulate_round_env(clients, {}, t)
            t0 = time.perf_counter()
            ids, _, state = mod.select_clients(t, N // 10, clients, history, random, None, "cpu")
            times.append(time.perf_counter() - t0)
            history["selected"].append(ids)
            history["state"].update(state)
            history["state"]["last_reward"] = 0.001
        print(f"N={N:5d} K={N // 10:4d}: first round {times[0] * 1e3:8.1f} ms (incl. JSD), "
              f"steady {np.median(times[1:]) * 1e3:8.1f} ms")

    print("\n== Heterogeneity scalar: submitted estimate vs I(Z;Y)/H(Y) (N=50, 10 classes, 3 draws) ==")
    from csfl_simulator.selection.ml.apex_v2 import _estimate_heterogeneity
    for alpha in (0.1, 0.3, 0.6, 1.0, None):
        vals = np.array([(_estimate_heterogeneity(cl, 10), mi_index(cl))
                         for cl in (make_clients(50, alpha=alpha, seed=s) for s in range(3))]).mean(0)
        print(f"alpha={'IID' if alpha is None else alpha}: submitted (sqrtJSD/0.6, capped) = {vals[0]:.3f}   "
              f"I(Z;Y)/H(Y) = {vals[1]:.3f}")


if __name__ == "__main__":
    main()
