# SCOPE-FD — claim audit and hardening plan

Every claim the paper makes, what the evidence actually says after 278 runs, and
what is still missing. Written to decide where the remaining GPU time goes.

**Sections 1 to 4 are the audit as written at 278 runs. Section 5 records what
changed at 284 runs and supersedes anything above that conflicts with it.**

Regenerate every number with

```
python3 "csfl_simulator/Paper Corrections/Scope_FD_Revision_Package/campaign_numbers.py" > campaign_numbers.txt
```

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

---

## 5. Post-campaign update (284 runs)

Six runs landed after the audit above was written. Five are the reworked audio
cells and one is the first CIFAR-10 seed. Their effect on the paper is larger
than their number suggests, and re-reading the whole campaign at once produced
two results that needed no GPU time at all.

### 5.1 The participation Gini coefficient has a closed form

The proof of Proposition 1 already derives the exact numerator `2a(N-a)` and
then throws it away by bounding it with `N^2/2`. Keeping it gives

```
G(R) = m(N-m) / (N K R),      m = KR mod N
```

Checked against every selector run in the campaign that was not subject to
dropout, this is correct in **457 of 457 cases**, covering 275 runs of the
complete score and 182 of the debt-only variant, largest deviation below
1e-4 pp. Seed-to-seed standard deviation is exactly zero everywhere.

Three consequences.

* The bound `N/(4KR)` is **tight**, not merely sufficient. It is the maximum of
  the closed form at `m = N/2`.
* The coefficient is zero **iff `N` divides `KR`**. This corrects a statement
  the paper made carelessly. `K | N` is neither necessary nor sufficient, and
  the campaign contains counterexamples in both directions. At `N=30, K=5,
  R=100` we have `K | N` yet `G = 1.33%`. At `N=50, K=3` we have `K ∤ N` yet
  `G = 0`. The caption of Fig. 8 asserted the wrong rule and has been fixed.
  So had the caption's description of the figure layout, which referred to an
  upper and a lower panel that the figure does not have.
* The decay in `R` is not smooth. The envelope is `O(1/R)` and the value
  oscillates with `m`, touching zero at every complete rotation.

This turns C1 and C2 from "confirmed across many runs" into "predicted in
advance and correct every time", which is a materially stronger claim, and it
answers the Attached Review's request for different `N`, different `R` and
`K ∤ N` with one expression instead of three stress tests.

### 5.2 The accuracy gap is a monotone function of K/N, and the debt term is the safe part

Fifteen configurations pair the selector against uniform random on matched
seeds, spanning `N` from 30 to 200 and `K/N` from 0.033 to 0.333.

| | mean gap over random | configurations positive | Spearman vs K/N |
|---|---|---|---|
| complete score | −1.42 to +2.43 pp | 12 of 15 | −0.815, p = 0.0002 |
| debt only | +0.05 to +2.39 pp | **15 of 15** | −0.764, p = 0.0009 |

The three configurations where the complete score trails are `N=100/K=20`,
`N=200/K=20` and `N=200/K=40`. All three use a cohort of at least 20, and in
all three debt-only leads the complete score, by 0.67, 1.34 and 1.90 pp.

This confirms the G3 prediction of Section 2 and resolves C11. The rotation
never costs accuracy at any ratio tested. The two information terms are what
cost accuracy once the cohort is large enough to cover the label space anyway.
The paper now scopes the accuracy claim to the sparse regime explicitly and
says that a dense deployment should use the debt-only variant.

Caveat worth keeping. Several of the debt-only gaps are a small fraction of the
seed spread, and the fifteen configurations share a seed set rather than being
independent. The claim is about the **direction** of the effect, not its size,
and the manuscript says so.

### 5.3 The manuscript described the public dataset incorrectly

Every run uses `--public-dataset same`, which
`core/datasets.py:get_public_dataset` resolves to 2000 unlabeled samples drawn
from the **held-out split of the private dataset**. The manuscript said MNIST
was the public set for the FMNIST experiments. That is wrong and it contradicted
the paper's own public-set sensitivity section, which reports what happens when
the public set *is replaced by* MNIST. Table I and Section VI-A now state what
was run.

The correction raises an overlap question, so it is answered in place. Public
labels are never read, which the label-noise null proves structurally, so no
label information can leak. Two configurations use a public corpus disjoint from
the evaluation data, namely MNIST with FMNIST private and STL-10 with CIFAR-10
private, and the selector ordering is unchanged in both. Absolute accuracy
depends on the public set. The comparison between selectors does not.

### 5.4 Audio is now real evidence rather than a weak signal

The earlier 3-seed audio run at one local epoch was still climbing at round 100
and its selector ordering was not separable. The reworked cell uses 5 seeds,
3 local epochs and a 150-sample public set, both changes forced by FSDD holding
~90 train samples per client and a 300-sample held-out split.

| | old, 3 seeds | new, 5 seeds |
|---|---|---|
| SCOPE-FD | 44.08 ± 2.58 | **46.16 ± 1.69** |
| debt only | 43.40 ± 1.46 | 45.54 ± 1.17 |
| uniform random | 43.38 ± 1.44 | 44.99 ± 1.59 |
| SubTrunc | 40.28 ± 1.98 | 40.66 ± 1.48 |
| DivFL | 41.12 ± 2.23 | 40.53 ± 1.49 |

SCOPE-FD now beats random on **all five seeds**, by 0.51 to 1.70 pp, and beats
both submodular methods on all five by more than 5 pp. Gini is 1.33% against
12.49% and ~45%. The complete score and debt-only remain tied, +0.61 pp at
p = 0.188, which reproduces the image finding.

Note on p-values at n=5: the smallest attainable two-sided Wilcoxon p is 0.0625,
so a reported 0.062 means a clean sweep and nothing stronger. Do not describe it
as significant at the 5% level.

### 5.5 CIFAR-10 and EMNIST are out of the paper

Both were single-seed and both are now removed, on the reasoning that the round-2
reviewers criticised single-seed results and volunteering two more invites the
criticism again.

* **CIFAR-10** was never asked for. R1.2 asks about domains *other than images*,
  which FSDD answers at five seeds. Seed 11 gave SCOPE-FD 29.24, debt-only 29.29,
  random 28.66, DivFL 27.97, SubTrunc 27.95, Gini 1.33 / 1.33 / 11.33 / 47.36 /
  42.64, so the ordering reproduces FMNIST. Kept here in case it is wanted later.
  Seeds 22 and 33 land around 10 August. If they are added, write it as a
  three-seed result, not as one seed plus a caveat.
* **EMNIST** was a single-seed replication carried over from round 1. The
  multi-seed MNIST and FSDD studies supersede it. Its removal is stated in one
  sentence in reply R1.2 rather than done silently.

The disjoint-public-corpus control that CIFAR-10 with STL-10 used to provide is
still in the paper. MNIST-as-public on FMNIST-private supplies it at three seeds.

`factcheck.py` now asserts that the words "single seed" and "one seed" do not
appear anywhere in the experimental sections.

### 5.6 What is still open

* **The convergence theorem.** Three replies remain `\PENDING` in the letter
  (R1.1, R2.3, Attached-1). No experiment closes these. C10 is unchanged and
  still not established.
* **CIFAR-10 seeds 22 and 33.** ~45 h of accelerator time each. Running, but
  CIFAR-10 is out of the paper, so nothing depends on them.
* **pub-CIFAR10 sensitivity cells**, 3 jobs, and the **tier-3 audio K sweep**,
  6 jobs. Roughly 6 h total, both behind CIFAR-10 in the queue.
* **EMNIST** never ran as a multi-seed study. It failed on an SSL trust-store
  error and was then dropped from the job list. The old single-seed EMNIST
  replication has been removed from the manuscript, so this is closed.
* **N=500** dropped from the scale sweep by design.

### 5.7 Two claims the fact-check found to be wrong

`factcheck.py` traces every quoted number back to a run and asserts the
structural claims. It caught three things.

**The rotation claim was false.** The paper said the per-round selection matrix
selects clients `{0..9}`, `{10..19}`, `{20..29}` by `r mod 3` at N=30, K=10. It
does not. The cohorts are score-ordered, not index blocks, and the complete
score uses 35 to 54 distinct cohorts across a run. What *is* true, and is now
what the paper says, is that every aligned 3-round window partitions the pool
exactly, in 33 of 33 windows on all three seeds, for both the complete score and
debt-only, against 0 of 33 for uniform random. Debt-only uses exactly 3 cohorts
and repeats them. That contrast is direct evidence for the sentence at the end
of Section V-A about the information terms acting inside the debt-equal cohort,
which previously had none.

**The participation-count numbers were stale.** The paper reported uniform
random spanning `{22..44}` with sd 4.96 and called it a "more than 10x"
reduction. No run on this machine produces that. The three available seeds give
23 to 42 with sd 4.53, so the reduction is about 9.6x, not more than 10x.
Corrected.

**The audio margin was overstated.** "Ahead of both submodular methods on every
seed by more than five" is false for SubTrunc on seed 11, which gives 4.02.
Changed to "at least four".

Everything else traced: 62 of 62 mean-and-sd pairs and 39 of 39 bare
percentages, plus 46 structural assertions. Run `python3 factcheck.py` after any
change to the numbers.

---

## 6. Presentation round

### 6.1 The marked and clean builds would have rendered differently

The source only ever differed by one preamble line, so a text diff looked fine.
The *rendered* output would not have been. A single 25 KB `\rev{}` block wrapped
**11 subsections, 7 figure floats, 2 table floats, 9 captions and 20 labels**
inside `\textcolor{blue}{...}`. LaTeX defers floats to shipout, by which point
the colour group has closed, so float colour and placement stop matching the
clean build, and `\textcolor` around a sectioning command is unsafe in its own
right.

Fixed by splitting at structural boundaries. `\rev{}` now wraps prose paragraphs
only. Floats and headings pass through unwrapped and a marked caption is marked
from inside, as `\caption{\rev{...}}`. 32 blocks became 85, none containing a
float or a heading.

The refactor is safe because `\rev` is the identity in the clean build, so the
clean output must be unchanged by it. That invariant was asserted during the
transformation and holds.

### 6.2 Length

| | before | after | limit |
|---|---|---|---|
| Abstract | 323 | **246** | 250 |
| Impact statement | 153 | **146** | 150 |
| Conclusion + future work | 627 | **362** | shorter |
| Whole paper | ~14.1 pp | **~14.0 pp** | 15 |

The abstract now leads with the closed form rather than the older and vaguer
"drives the Gini to zero when `K | N`" claim, which was the weaker statement.

The introduction was left alone. Its motivation chain already runs FL to FD to
FedTSKD to the full-participation assumption to why FL selectors fail
structurally in FD, which is the argument the paper needs.

### 6.3 Three contribution bullets were stale

* Bullet 2 claimed only an `O(1/R)` bound. It now states the exact law.
* Bullet 3 claimed the FedTSKD bound "continues to hold without modification".
  Three reviewers challenged exactly that, and the reply is still PENDING, so
  the paper was contradicting its own response letter. Reworded to the
  compatibility statement Reviewer 2 proposed. **This does not resolve the
  PENDING replies, it only stops the manuscript from overclaiming while they
  are open.**
* Bullet 4 said the study was Fashion-MNIST only.

### 6.4 Bibliography

Removed two entries.

* `li2020federated` was in the file but never cited.
* `tamboli2026p` was cited to support "the privacy-preserving nature of FL has
  been investigated". It is a paper on privacy-preserving *face-centric
  generative image editing*, which is not that topic. Off-topic citations are
  the kind of thing a reviewer notices.

Added five, all checked against DBLP before use, all IEEE, four of them IEEE TAI.

| key | venue | why |
|---|---|---|
| `arisdakessian2026vox` | IEEE TAI 2026, 7(4):1997-2011 | fairness *as* the selection criterion, the closest prior work to this paper's objective |
| `kazemi2026robust` | IEEE TAI 2026, 7(1):262-271 | RL-based client selection; the contribution list contrasts against RL selectors and previously cited nothing |
| `rao2025privacy` | IEEE TAI 2025, 6(2):333-353 | replaces the off-topic privacy citation |
| `thakur2026grace` | IEEE TAI 2026, 7(6):3221-3236 | resource and energy-aware FL, supports the energy-budget motivation |
| `castillo2024subtrunc` | IEEE CDC 2024, pp. 5496-5502 | the **published** version of SubTrunc, which was cited only as an arXiv preprint despite being a headline baseline |

The arXiv entry is retained for UnionFL, which appears only in that version.

Nothing old was dropped. Every pre-2020 entry is load-bearing: `hinton2015`
(distillation), `mcmahan2017` (FedAvg), `jeong2018` (FD origin), `hsu2019`
(the Dirichlet partitioning protocol the experiments use), `nishio2019` (FedCS),
`ahn2019` (wireless FD). Removing any of them would leave a claim unsupported.

45 entries, 45 cited, no orphans and no dangling keys.

---

## 7. Compliance round

### 7.1 The marked copy was losing text, and the cause was not the one assumed

`\newcommand` makes `\rev` itself `\long`, so a paragraph break inside a `\rev{}`
block is legal in the clean build. `\textcolor` is **not** `\long`. In the marked
build `\rev{A \par B}` expanded to `\textcolor{blue}{A \par B}`, which raises
"Paragraph ended before \textcolor was complete" and drops text during recovery.

Three blocks spanned a paragraph break. One of them was the conclusion's headline
paragraph, so **the marked copy had no statement of the closed-form result at
all**, and another was the coefficient-choice paragraph in Section IV-D. This is
invisible to a text diff, which is why the earlier check passed.

Fixed twice over. The marked build now defines `\rev` as `{\color{blue}#1}`,
a switch that is legal across paragraphs, and the three blocks were split so the
source is safe under either mechanism. `build_versions.sh` now reports the count
of paragraph-spanning and float-wrapping blocks on every run and both must be
zero.

### 7.2 Abstract and impact statement rewritten to the journal's brief

The abstract had formulas and a paragraph of experimental detail, which the
journal explicitly rules out. It now follows the six-part structure requested,
namely problem, state of the art, gap, contribution, result, implications, with
no mathematics and no per-configuration numbers. 249 words against the 250 limit.

The impact statement was close to a paraphrase of the abstract. It now addresses
the legal and social dimensions the journal asks about, in language a
non-specialist can follow, and avoids repeating the abstract's content. 150 words
against the 100 to 150 range.

### 7.3 Length

16.0 pages measured, 15 allowed. Now about 14.6 estimated, leaving roughly 350
words of headroom.

Where it came from:

| cut | saving |
|---|---|
| Fig. 2 and Fig. 3 removed, both reproduced Tables II and III exactly | ~0.4 pp |
| Section VI-L, four paragraphs condensed | ~0.5 pp |
| Section VI-N, the census re-listed conditions already reported | ~0.2 pp |
| Section VI-K, negative result told at length | ~0.15 pp |
| Contributions block, four verbose bullets | ~0.15 pp |
| Theory, three consequences and the divisibility discussion merged | ~0.1 pp |
| Introduction opening and Section II-B, overlap with each other removed | ~0.1 pp |
| remaining prose across seven subsections | ~0.3 pp |

No claim, number or citation was removed. `factcheck.py` still traces every
quoted value and passes all structural assertions. The percentage count fell from
39 to 33 because six of them were duplicate statements of the same value.

Removing two figures renumbered the rest. Every `Fig.~n` in the manuscript and in
the response letter was remapped, and the response letter now states the removal
rather than leaving reviewers to notice it.

### 7.4 Punctuation

29 colons and semicolons removed from prose. What remains is the paper title,
which is fixed from the submission. Run-in headings moved from `\textit{Label:}`
to `\textit{Label.}`, which is equally standard in IEEE style. The 14 remaining
double hyphens are all numeric ranges inside math, where they are correct.

### 7.5 Package layout

Rebuilt to the standard IEEE Overleaf layout, so `main.tex` is auto-detected as
the main document and figures live in `figures/`. No `.cls` or `.bst` is bundled,
because Overleaf maintains IEEEtran and shipping an unverified copy is worse.
