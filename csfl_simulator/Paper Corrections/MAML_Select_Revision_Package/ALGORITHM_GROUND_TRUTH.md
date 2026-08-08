# MAML-Select — what the code actually does

Read directly from `csfl_simulator/experiments/maml_select/selector.py` (registered
as `research.maml_select`, the selector that produced every reported number).
This file is the reference the revised manuscript is written against. Where the
submitted manuscript or the supplementary disagreed with this, the manuscript was
corrected — not the other way round.

---

## 1. State vector

`_client_features`, lines 90–110.

```
s_i,t = [ last_loss, grad_norm, expected_duration, battery_ratio,
          participation_count, recency ]                       (6 entries)
```

Then, and this was **never stated in the submitted paper**:

```python
matrix = _zscore(np.asarray(rows, dtype=np.float32))   # line 109
```

Every feature is **z-scored across the currently available client pool, every
round**. Two consequences that matter for the reviewer's dimensional-consistency
point:

* the policy input is dimensionless, so mixing seconds, watt-hours and counts in
  one vector is harmless;
* the score $h_\phi(s)$ is a *relative* ranking signal within a round, not an
  absolute cost prediction. Only the ordering is used (Top-$K$).

Feature ablation is a multiplicative mask applied after z-scoring (line 110).

## 2. Order of operations inside round $t$

`select_clients`, lines 303–420. The meta-update happens **at the start of the
round, before any client is scored or selected** — not after FL execution as the
submitted Algorithm 1 implied.

```
1.  ingest feedback that arrived from round t-1     _ingest_previous_feedback
2.  one inner GD step on the support set            _adapt
3.  first-order meta-gradient on the query set,     _outer_step
    applied to phi through Adam
4.  score all available clients with phi'_t
5.  cold start  (t < ceil(N/K))  or  Top-K + 1 exploration slot
6.  train, aggregate, stash this round's feedback as "pending"
```

## 3. Support and query sets — the important correction

At line 413 the current round stashes

```python
state["pending"] = { ..., "adaptation_support": query }
```

and at line 347 the next round unpacks

```python
query, previous_adaptation_support = _ingest_previous_feedback(...)
```

`query` is rebuilt from `pending["client_ids"] / durations`, i.e. the cohort of
the **immediately preceding** round. `previous_adaptation_support` is whatever
was `query` one round earlier. Therefore

| set | contents | cohort |
|---|---|---|
| support $\mathcal{D}_{sup}(t)$ | $(s_{i,t-2}, c_{i,t-2})$ | $\mathcal{S}_{t-2}$ |
| query $\mathcal{D}_{query}(t)$ | $(s_{i,t-1}, c_{i,t-1})$ | $\mathcal{S}_{t-1}$ |

**Both sets are past rounds, and the support set is exactly one round staler
than the query set**, so $\mathcal{D}_{sup}(t) = \mathcal{D}_{query}(t-1)$.

This is neither what the submitted manuscript said (support $t-1$, query
*current* round $t$) nor what the supplementary said (both updates on
$\mathcal{D}_{sup}$). It is, however, the *right* design and should be presented
as such: the inner loop adapts on an older round and the meta-gradient is
measured on a newer one, so the meta-update is explicitly rewarded for
adaptation that **generalises forward in time**. That is the meta-learning
formulation the paper claims to be making, and it is only visible once the lag
structure is stated correctly.

It also removes an inconsistency in the old theory. Writing
$\mathcal{D}_{sup}(t) = \mathcal{D}_{query}(t-1)$ makes $g_t \equiv q_{t-1}$, so
assuming a stationary query objective $q_t \equiv q$ forces $g_t \equiv q$ too —
i.e. it assumes client utility never changes, the negation of the paper's own
motivation. See `THEORY_NOTES.md`.

## 4. Cost / utility

`_ingest_previous_feedback`, lines 248–251, with the default at line 318.

```python
latency_penalty = max(0.0, float(duration) - target)
if normalize_latency_penalty:          # default True
    latency_penalty /= target
cost = float(lambda_latency) * latency_penalty - local_reduction
```

So the cost that produced every reported result is the **normalised** form

$$c_{i,t} = \lambda\,\frac{\max(0,\;T_{i,t}-T_{\text{target}})}{T_{\text{target}}} \;-\; \Delta_{i,t}$$

Submitted Eq. (4) omitted the $/T_{\text{target}}$. The supplementary algorithm
had it. **The supplementary was right and the main paper was wrong**, which is
precisely the inconsistency the reviewer identified. Normalisation is also what
makes the two terms dimensionally comparable — a fractional deadline overrun
against a dimensionless loss reduction — so it is not a cosmetic difference.

$T_{\text{target}}$ (lines 397–406): the **mean expected round latency of the
Tier-2 clients**, recomputed each round from the live pool; if no Tier-2 client
is present it falls back to the median latency over the pool
(`target_latency_quantile = 0.5`).

## 5. Cold start and exploration

```python
warmup_rounds = ceil(N / K)                         # line 371-375, default
if round_idx < warmup_rounds:  _cold_start_selection(...)
else:                          _policy_selection(..., exploration_clients=1)
```

* **Cold start**: for the first $\lceil N/K \rceil$ rounds (10 rounds at
  $N=100$, $K=10$) the cohort is taken in a fixed, seed-shuffled coverage order,
  so every client is visited exactly once before the policy takes over. Its
  purpose is to seed the replay buffer with one observation per client; the
  fairness consequence is a side effect, and Sec. `FAIRNESS_REEVALUATION.md`
  shows the coverage result does not depend on it.
* **Exploration**: `exploration_clients = 1`. The submitted supplementary called
  this "$\epsilon$ random exploratory clients" — **it is not random**. Lines
  289–301 sort by `(-recency, participation_count, coverage_rank)`, i.e. the one
  slot is given deterministically to the *stalest, least-used* client. One of
  the $K$ slots is reserved for it; the other $K-1$ go to the Top-$(K-1)$ of the
  learned ranking.

## 6. Optimisers

* inner loop (`_adapt`, lines 125–141): plain gradient descent,
  `params - inner_lr * grad`, `inner_steps = 1`, $\beta = 0.01$. Matches Lemma 1
  exactly.
* outer loop (line 338): `torch.optim.Adam(model.parameters(), lr=outer_lr)`,
  $\eta = 0.001$. The meta-gradient is
  `torch.autograd.grad(loss(adapted), adapted.values())` — differentiated at the
  adapted weights, applied to the base weights, second derivatives dropped.
  That is first-order MAML and matches Eq. (7).
  **Lemma 2 analyses a plain-GD outer step and therefore does not literally
  cover the Adam implementation.** The revision states this rather than hiding
  it; see `THEORY_NOTES.md`.

## 7. Defaults that belong in the paper

| symbol | value | source |
|---|---|---|
| $\beta$ inner LR | 0.01 | line 313 |
| $\eta$ outer LR | 0.001 (Adam) | line 314, 338 |
| inner steps | 1 | line 315 |
| $\lambda$ | 0.5 | line 316 |
| normalise latency | **True** | line 318 |
| cold-start rounds | $\lceil N/K\rceil = 10$ | line 371–375 |
| exploration slot $\epsilon$ | 1 | line 320 |
| hidden width | 64 (6-64-64-1, 4673 params) | line 322 |
| selector seed | 2026 | line 321 |

## 8. `selector_v2.py` and `selection/ml/maml_select.py`

Neither produced a headline number.

* `selector_v2.py` is a different algorithm (ranking losses, quota buckets
  6/2/1/1) used only in the design ablation.
* `csfl_simulator/selection/ml/maml_select.py` is a **legacy** selector with no
  support/query split at all. `docs/MAML_select_method_reference.md` line 97
  ("Unlike full MAML, there is no outer-loop or task distribution") describes
  *that* file, not the paper's method. The doc should not be cited as a
  description of the published algorithm.
