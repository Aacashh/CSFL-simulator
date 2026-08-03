# SCOPE-FD — claim audit and hardening plan

Every claim the paper makes, what the evidence actually says after 278 runs, and
what is still missing. Written to decide where the remaining GPU time goes.

---

## 1. Where we stand

| # | Claim | Status | Evidence |
|---|---|---|---|
| C1 | **Proposition 1**: participation counts differ by at most 1; Gini $\le N/(4KR)$ | **Proved and confirmed** | Counts span exactly 1 at N=30,K=10. Gini falls 3.33→1.33% over R=25→100, below the bound (6.00→1.50%) at every horizon |
| C2 | **Gini invariance**: fairness independent of data, channel, privacy, public set | **Confirmed, and stronger than claimed** | Exactly 1.33% in 176 of 278 runs across 12 of 16 families. Moves only where N, K or realised participation change |
| C3 | Fairness holds when $K \nmid N$ | **Confirmed** | 1.40% at N=47/K=6, 1.25% at N=53/K=7, vs 15.70% and 13.81% for random |
| C4 | Fairness holds off cycle boundaries | **Confirmed** | Rolling-window Gini 4.52–18.58% vs 43.85–52.22% for random |
| C5 | Beats published FD-ported selectors | **Confirmed** | +1.33 pp over DivFL, +1.81 pp over SubTrunc, both with ~30 pp lower Gini |
| C6 | FL selectors do not transfer to FD | **Confirmed but thin** | Oort reaches 21.14% vs 71.21%. **One data point only** |
| C7 | Insensitive to $\alpha_u, \alpha_d$ | **Confirmed inside the tested box** | 30-cell grid spans 1.33 pp; Gini identical in every cell. **The theorem's precondition $\alpha_u+\alpha_d<1$ was never tested at or beyond the boundary** |
| C8 | Histogram privacy is affordable | **Confirmed** | Laplace at every $\varepsilon$ and the surrogate all within one s.d.; Gini unchanged |
| C9 | Survives dropout and staleness | **Confirmed** | Advantage retained; Gini 0.80–1.88% under dropout, exactly 1.33% under staleness |
| C10 | Inherits FedTSKD's $\mathcal{O}(1/t)$ bound | **NOT established** | No full-participation reference exists. Log-log fit of server loss gives slope −0.67 (R²=0.42), not −1 |
| C11 | Three-term score is justified | **Contradicted at the tested config** | The four variants are statistically tied at $\alpha=0.5$. Debt alone carries the whole fairness result |

### Claims from the original submission that the data refuted

| Original claim | Measured | Action taken |
|---|---|---|
| "converges $3\times$ faster at $K/N=10\%$" | ≈1.6× | Removed; replaced with rounds-to-absolute-accuracy |
| "$9.4$ pp gain at $K=1$" | **+1.94 pp** (60.61 vs 58.67) | Corrected throughout |
| "participation Gini identically zero" | 1.33% residual at the headline | Restated as the $\mathcal{O}(1/R)$ bound, which is what Proposition 1 actually gives |
| "2.7× speedup on EMNIST" | single seed, unverified | Cross-reference removed |

---

## 2. The three real gaps

### G1. The theorem's precondition has never been tested
Proposition 1 holds when $\alpha_u + \alpha_d < 1$, because the normalised debt
gap is $1/(1+\varepsilon)$ and the two information terms are bounded by
$\alpha_u+\alpha_d$. Every configuration run so far sits at $0.4$, far inside
the safe region, so the condition has never been exercised. A theorem whose
precondition is never approached is a theorem whose sharpness is unknown.

This is the cheapest and most valuable experiment available: sweep
$\alpha_u+\alpha_d$ across $1$ and check whether the guarantee fails exactly
where the proof says it must. Either outcome is publishable. Confirmation makes
Proposition 1 sharp rather than merely sufficient; a failure to break would show
the bound is conservative and invite a tighter statement.

### G2. There is no full-participation reference
The convergence claim is specifically about replacing full participation with
partial participation, yet the highest ratio ever run is $K/N=33\%$. Without a
$K=N$ run there is nothing to compare against, which is why C10 cannot be
supported. At $K=N$ every selector is identical, so a single run per seed
suffices as the shared reference.

### G3. The information terms were only ever tested at one heterogeneity level
The four-way ablation exists only at $\alpha=0.5$. That is the one setting where
the coverage term is least likely to matter, because a Dirichlet draw at
$\alpha=0.5$ already gives each client several classes, so any five clients
cover most of the label space and there is little for a coverage penalty to fix.

The existing data already hints at where it does matter. Sorting every matched
comparison of the complete score against debt-only:

* wins cluster at **K=3** (+0.60, +0.47, +0.45 pp) and under **dropout** (+0.58 pp)
* losses cluster at **K=1** (−0.43, −0.46) and at **large K** (−1.01 at K=20, −1.90 at K=40)

That is a mechanism, not noise. At $K=1$ there is no within-round composition to
diversify. At $K \gg C$ the cohort covers the label space anyway and the penalty
only distorts selection away from debt. The predicted useful regime is
$1 < K \lesssim C$, and it should widen as heterogeneity increases, because at
low $\alpha$ each client holds fewer classes and the choice of which clients to
combine matters more.

If that prediction holds, the three-term design is rescued and correctly scoped
rather than abandoned. If it fails, the honest paper is a debt-only selector with
a proved guarantee, which is still a contribution.

---

## 3. Experiments

| ID | Experiment | Claim | Prediction | Cost |
|---|---|---|---|---|
| **X1** | $\alpha_u+\alpha_d \in \{0.4, 0.8, 1.0, 1.3, 1.8, 3.0\}$ at the headline | C1, C7 | Gini stays 1.33% below 1 and degrades above it | ~3.7 h |
| **X2** | $K \in \{20, 30\}$ at $N=30$, including full participation | C10 | SCOPE tracks the full-participation trajectory; gap closes as $K \to N$ | ~6.7 h |
| **X3** | Four-way ablation at $\alpha \in \{0.05, 0.1, 0.3\}$, $K=3$ | C11 | Coverage term helps as $\alpha$ falls; separation emerges below $\alpha=0.3$ | ~5.1 h |
| **X4** | Headline extended to $R=300$ | C1 | Gini continues to fall as $\mathcal{O}(1/R)$ towards ~0.44% | ~3.7 h |
| **X5** | FedCS and TiFL ported into FD alongside Oort | C6 | Both collapse like Oort, turning one data point into three | ~1.2 h |

**Total ≈ 20 GPU-hours.**

X1 and X3 are the ones that change what the paper can claim. X2 is what the
convergence section needs to say anything at all. X4 and X5 are cheap
reinforcement of claims already made.

### What needs no GPU time

* The $\mathcal{O}(1/t)$ log-log fit is computable from the per-round history
  already on disk. It currently gives slope −0.67 with R²=0.42, which does **not**
  support a claim of achieving the bound. It does support the modest statement
  Reviewer 2 suggested, namely compatibility with the FedTSKD pipeline under the
  assumptions of the original work.
* Round-to-round server-accuracy variance is also already recorded: 0.414 pp for
  debt-only, 0.481 for the complete score, 0.528 for random. The ordering is in
  the expected direction but the margins are small, so this supports a remark
  rather than a claim.

---

## 4. Position after these runs

The paper's contribution is a **deterministic participation guarantee that is
invariant to every operational condition tested**, proved in Proposition 1 and
confirmed across 278 runs, together with an honest account of the two places it
breaks: a hard resource constraint, and the boundary of its own precondition.

That is a stronger and more useful contribution than a small accuracy win, and
it is one the accuracy-focused literature does not provide. The remaining
experiments are aimed at making the guarantee sharp rather than at chasing
accuracy the data does not support.
