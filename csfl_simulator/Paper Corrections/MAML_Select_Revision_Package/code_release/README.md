# MAML-Select

Reference implementation for **"MAML-Select: An Online Adaptive Client Selection
Method for Federated Learning via Meta-Learning"**.

MAML-Select treats each federated round as a meta-learning task. A small ranking
policy is adapted from recent feedback, used to score the available clients, and
then meta-updated once the next round's feedback arrives. The aim is to keep
accuracy competitive while cutting the compute and energy spent on client
training, without ever permanently excluding a client.

This repository is a dependency-light extraction of the selector that produced
the reported results. It is organised so that every equation in the paper can be
read directly against the code, and the test suite checks the paper's claims
rather than merely exercising the functions.

---

## Install and run

```bash
pip install -r requirements.txt

python3 examples/run_demo.py --rounds 200 --clients 100 --cohort 10
python3 -m pytest tests/ -q
```

The demo needs no dataset and no GPU. It builds a synthetic pool with the three
device tiers of the paper and lets each client's utility decay as it is used, so
the selection target is genuinely non-stationary. A representative run:

```
 round         phase   cov%    Jain    T1    T2    T3
-----------------------------------------------------
     1    cold start     10   0.100  0.00  0.60  0.40
    20        policy    100   0.588  0.11  0.48  0.41
   200        policy    100   0.778  0.06  0.46  0.48

final coverage            100%
final Jain index          0.778
Lemma 1, inner descent    198/198 rounds non-increasing
drift increments          mean +0.0007, sum V_T +0.14 over 198 rounds
```

The selector shifts toward the fast tier, as a latency-aware objective should,
but keeps the slow tier represented and reaches full coverage. Both properties
match the behaviour reported in the paper.

---

## Where each part of the paper lives

| Paper | Code |
|---|---|
| Eq. (4), per-round state standardization | `maml_select/policy.py`, `standardize` |
| Eq. (5), support and query sets | `maml_select/selector.py`, `MAMLSelect.observe` |
| Eq. (6), normalized latency penalty | `maml_select/cost.py`, `normalized_latency_penalty` |
| Eq. (8), per-client utility | `maml_select/cost.py`, `client_cost` |
| Eq. (9), inner-loop adaptation | `maml_select/selector.py`, `MAMLSelect._adapt` |
| Eq. (10), first-order meta-update | `maml_select/selector.py`, `MAMLSelect._meta_update` |
| Eq. (11), Top-K selection | `maml_select/selector.py`, `top_k_by_cost` |
| Proposition 1, Top-K is exact | `tests/test_maml_select.py` |
| Proposition 2, coverage guarantee | `maml_select/selector.py`, `MAMLSelect._explorers` |
| Lemma 1, inner-step descent | `Diagnostics.inner_descent` |
| Corollary 1, drift term `V_T` | `Diagnostics.drift_increment` |
| Algorithm 1 | `MAMLSelect.select` and `MAMLSelect.observe` |

---

## Three details that are easy to get wrong

**The two sets are lagged by one round.** Cost feedback exists only after a
cohort has trained, so the support set is the feedback of round `t-2` and the
query set that of round `t-1`. Hence `D_sup(t) = D_query(t-1)`. The inner step
adapts on the older round and the meta-gradient is measured on the newer one, so
the outer update rewards initializations from which one adaptation step
generalises *forward in time*. That lag is what makes this meta-learning rather
than online regression, and `test_support_set_is_the_previous_query_set` pins it
down.

**The latency penalty is normalized by the deadline.** The cost is

```
c_i = lambda * max(0, T_i - T_target) / T_target  -  Delta_i
```

Without the division by `T_target` the penalty carries units of seconds while
the loss reduction is dimensionless, so a single weight `lambda` would depend on
the units of the latency model and would not transfer between datasets with
different round times. `test_normalized_cost_is_scale_free_but_unnormalized_cost_is_not`
demonstrates the difference. `T_target` is the mean expected round latency of the
medium tier, recomputed every round.

**Full coverage comes from the exploration slot, not the cold start.** One of the
`K` places each round is reserved for the stalest, then least-used, client. The
slot is deterministic. Because a client that takes it has its staleness reset,
each of the other `N-1` clients can block a given client at most once, so every
client is selected within any window of `N` rounds regardless of what the policy
learns. `test_exploration_slot_alone_guarantees_full_coverage` verifies this with
the cold start switched off, and the companion test shows coverage is *not*
guaranteed once the slot is removed.

---

## Defaults

These are the values behind the reported results.

| Symbol | Meaning | Value |
|---|---|---|
| `beta` | inner learning rate, plain gradient descent | 0.01 |
| `eta` | outer learning rate, Adam | 0.001 |
| inner steps | adaptation steps per round | 1 |
| `lambda` | latency and loss trade-off | 0.5 |
| `epsilon` | exploration slots per round | 1 |
| cold start | rounds of forced coverage | `ceil(N/K)` |
| hidden width | policy is `6-64-64-1`, 4,673 parameters | 64 |
| selector seed | fixed so reported variance is training and partition only | 2026 |

The outer update applies the first-order MAML direction through Adam. The
analysis in the paper is written for a plain descent step, and the manuscript
states that gap rather than assuming the two are equivalent.

---

## Privacy

No raw client data reaches the server. The server observes only the six-scalar
state vector and, after a round, the scalar cost of Eq. (8).

## Citation

Please cite the paper if you use this code. The BibTeX entry will be added once
the article is assigned its final reference.
