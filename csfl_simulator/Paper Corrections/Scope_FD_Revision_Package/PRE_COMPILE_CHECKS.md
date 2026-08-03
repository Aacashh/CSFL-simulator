# Pre-compile sanity checks — SCOPE-FD revision

Open items that must be resolved before the revised manuscript is compiled and
submitted. Both are decisions only the authors can make, so they were not
resolved silently during drafting.

---

## 1. The revision runs use a different protocol from the original submission

The original submission and the revision campaign do not share the same
experimental configuration. Three parameters differ.

| Parameter | Original submission | Revision campaign |
|---|---|---|
| Client pool `N` | 50 for the sweeps, 30 for the participation study | 30 (swept to 200 separately) |
| Public dataset | cross-pair, MNIST when FMNIST is private | `same`, the FMNIST test split |
| Channel noise | on, DL SNR −20 dB | off for the headline families, with a dedicated SNR sweep alongside |
| Seeds | 1 (seed 42) | 5 headline, 3 sweeps |
| Coefficients | `(0.3, 0.1)` | `(0.3, 0.1)` — unchanged |

**What was done.** Table I in the revised manuscript was updated to describe the
protocol actually used, and every reported number comes from that protocol.

**What to confirm.** That reporting the revision protocol as the paper's primary
setup is intended. The alternative is to rerun the headline families with
`--public-dataset MNIST` and `--channel-noise --dl-snr-db -20` at `N=50` so the
revision matches the original submission exactly.

**Note.** The public-set identity is no longer a fixed choice. The
`public_dataset_sensitivity` family sweeps MNIST, EMNIST, and CIFAR-10 against
the `same` baseline, so the original cross-pair configuration becomes one point
in that sensitivity study once it completes.

---

## 2. Verify the SubTrunc and UnionFL citation venue

`references.bib` currently cites both methods as

```
@article{castillo2025subtrunc,
  title  = {Submodular Maximization Approaches for Equitable Client Selection
            in Federated Learning},
  author = {Castillo Jim{\'e}nez, Andr{\'e}s Catalino and Kaya, Ege C. and
            Ye, Lintao and Hashemi, Abolfazl},
  journal = {arXiv preprint arXiv:2408.13683},
  year    = {2024}
}
```

This arXiv record was verified directly. A separate IEEE ICASSP 2025 paper
titled *Equitable Client Selection in Federated Learning via Truncated
Submodular Maximization* by the same authors was seen referenced but **could not
be confirmed on IEEE Xplore**.

**Action.** Search IEEE Xplore for the ICASSP version. If it exists, replace the
arXiv entry with the published one, since an IEEE-indexed citation is stronger
for an IEEE TAI submission. If it does not, keep the arXiv entry as is.

The same check applies to **FedSTS** (IEEE Xplore document 10689614) if it is
ever cited. Its exact journal, volume, and issue were never confirmed. It is
not currently cited anywhere in the manuscript.

---

## 3. Other items to re-check at compile time

- **Figure and table numbering.** The reply letter cites specific numbers
  (Table I to III, Fig. 1 to 9, Sections IV-D, V-A, VI-A to VI-I). These were
  reconciled against the manuscript on 2026-07-31. If sections or floats are
  added or reordered, re-run the numbering check before submitting.
- **Nine pending replies.** The reply letter has nine `\PENDING` blocks in red,
  one per experiment family still running. None may remain in the submitted
  version.
- **The EMNIST cross-dataset subsection (VI-I) is still single seed.** It states
  this explicitly. It should be replaced when the `dataset_generality` family
  completes, at which point the statement about a single seed is removed.
- **`\PENDING` and `\rev` macros.** The clean manuscript build must have
  `\rev{}` as a no-op. `build_versions.sh` handles this. Do not edit
  `main_scope_marked.tex` or `main_scope_clean.tex` directly, since both are
  generated from `main_scope_revised.tex`.
