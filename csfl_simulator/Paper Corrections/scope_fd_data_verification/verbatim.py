"""Diff every reviewer comment in the letter against the source text.

The source is the editor's mail for the Associate Editor and Reviewers 1 to 3,
and the attached review PDF for Reviewer 4. Both are held verbatim below. The
letter is normalized out of LaTeX before comparison, so quoting marks, maths
delimiters and escaped percent signs do not register as differences, while a
changed word, a changed punctuation mark or a missing space does.

This exists because three transcription defects had survived a read-through,
including a semicolon that a punctuation pass had silently rewritten inside a
Reviewer's own sentence.
"""

import difflib
import io
import re

P = ("c:/Users/drash/OneDrive/Desktop/CSFL-simulator/csfl_simulator/"
     "Paper Corrections/SCOPE_FD_13PAGE/response_to_reviewers.tex")

SOURCE = {
    ("Associate Editor", "Comment:"):
        "A revision stage is needed",

    ("Reviewer 1", "Comment 1:"):
        "The paper aims to demonstrate convergence compatibility with FedTSKD and client "
        "selection. However, the convergence proof is based on FedTSKD, and the effect of "
        "client selection via a client subset with bounded variance can easily be derived. "
        "Therefore, the analytical contribution of Section V-B is minimal.",
    ("Reviewer 1", "Comment 2:"):
        "Only two datasets, Fashion-MNIST and MNIST, which have very similar characteristics, "
        "are tested. The dataset is compatible with the given model. However, as mentioned in "
        "the conclusion and future work, the applicability of the proposed method to domains "
        "other than images is questionable, as the proposed SCOPE-FD was tested under a narrow "
        "scenario.",
    ("Reviewer 1", "Comment 3:"):
        "As the experimental results were conducted in limited settings, it would be useful to "
        "show confidence intervals on the graphs.",
    ("Reviewer 1", "Comment 4:"):
        "The coefficients (e.g., alphau and alphad) are not ablated. The selection process for "
        "these coefficients needs to be described.",
    ("Reviewer 1", "Comment 5:"):
        "The baseline is too simple when considering uniform randomization. There are other "
        "client selection mechanisms in FL (e.g., DevFL) that apply submodular diversity "
        "maximization to the client histogram, which is similar to SCOPE-FD's coverage penalty. "
        "Therefore, the authors need to consider more realistic and fair comparison baselines.",
    ("Reviewer 1", "Comment 6:"):
        "Only a single value of Dirichlet alpha (0.5) is tested, producing moderate heterogeneity.",
    ("Reviewer 1", "Comment 7:"):
        "Experiment scale is too small to represent Massive-MIMO systems.",

    ("Reviewer 2", "Comment 1:"):
        "The most important missing comparison is a debt-only selector with alpha_u = alpha_d = 0. "
        "Section IV-A states that this setting reduces SCOPE-FD to deterministic round-robin, and "
        "Section V-A shows that the debt term dominates the fairness guarantee. Without this "
        "baseline, it is unclear whether the gains in Fig. 3 come from the under-prediction and "
        "coverage terms or simply from replacing random selection with balanced rotation. Please "
        "also report ablations for debt plus under-prediction only and debt plus coverage only.",
    ("Reviewer 2", "Comment 2:"):
        "Equations (12) and (14) require each client to reveal its label histogram h_i. This is "
        "not raw-data sharing, but it still discloses the local class distribution. Since FD is "
        "partly motivated by privacy, this assumption should be treated as a central limitation. "
        "Please either evaluate the Laplace-noise or server-side surrogate variants mentioned in "
        "Section IV-D, or narrow the privacy-related claims.",
    ("Reviewer 2", "Comment 3:"):
        "The participation-fairness proof is convincing, but the convergence claim needs more "
        "care. Section V-B says that the FedTSKD O(1/t) bound holds \"without modification,\" but "
        "SCOPE-FD changes full participation into partial participation. Please provide a clearer "
        "theorem for the partial-participation setting, or state more modestly that SCOPE-FD is "
        "compatible with the FedTSKD pipeline under the assumptions of [13].",
    ("Reviewer 2", "Comment 4:"):
        "The experiments use a single reported seed and do not show confidence intervals or "
        "standard deviations. It remains unclear what the actual performance is under all possible "
        "scenarios. Claims such as \"statistically tied\" and \"3x faster\" should be supported by "
        "multi-seed results and error bars. The rounds-to-80%-of-final-accuracy metric is useful, "
        "but it should be complemented with final accuracy tables, rounds to a fixed absolute "
        "accuracy, and sensitivity to alpha_u, alpha_d, Dirichlet alpha, and K/N.",
    ("Reviewer 2", "Comment 5:"):
        "The motivation emphasizes massive-MIMO, energy budgets, and contention-limited uplinks, "
        "but the selection score in Eq. (8) does not include any channel or energy information. "
        "The SNR sweep mainly shows that FedTSKD remains robust under noise; it does not show that "
        "SCOPE-FD exploits MIMO properties. Please either add channel/energy-aware experiments or "
        "clearly position SCOPE-FD as a fairness/data-coverage selector running on top of an mMIMO "
        "communication substrate.",
    ("Reviewer 2", "Comment 6:"):
        "Please define \"K | N\" clearly as a divisibility condition in the abstract. In Section "
        "VI-A, fix missing spaces such as \"[13].FMNIST\" and \"sweeps.In every configuration.\" "
        "The title should likely use lowercase \"for\" if title case is applied consistently.",
    ("Reviewer 2", "Comment 7:"):
        "Overall, the paper has a clear idea and a readable presentation, but it currently "
        "overstates the evidence. A stronger revision should separate the effect of debt-based "
        "fairness from the added value of the under-prediction and coverage terms, and should make "
        "the privacy and wireless-system assumptions explicit.",

    ("Reviewer 3", "Comment 1:"):
        "The authors clearly explain why traditional federated learning client selection methods "
        "cannot be directly applied to federated distillation.",
    ("Reviewer 3", "Comment 2:"):
        "How sensitive is the proposed algorithm to different values of alpha_u and alpha_d?",
    ("Reviewer 3", "Comment 3:"):
        "Improve the resolution of Figure 1.",
    ("Reviewer 3", "Comment 4:"):
        "How does SCOPE-FD perform under more severe non-IID settings?",
    ("Reviewer 3", "Comment 5:"):
        "How would the method perform under asynchronous federated learning or client dropout "
        "scenarios?",
    ("Reviewer 3", "Comment 6:"):
        "Since the proposed method relies on a public dataset, how sensitive is its performance to "
        "the quality or distribution of that public dataset?",

    ("Reviewer 4", "Comment 1 (Convergence analysis):"):
        "The manuscript states that the convergence behavior of FedTSKD is preserved because "
        "SCOPE-FD only changes the composition of the selected client subset while leaving the "
        "remaining learning procedure unchanged. Since deterministic partial participation "
        "introduces a different client-selection policy, it would be helpful to clarify whether "
        "the assumptions required in the original convergence analysis of FedTSKD remain "
        "satisfied. If a complete theoretical extension is beyond the scope of this work, a brief "
        "discussion of the underlying assumptions would improve the presentation.",
    ("Reviewer 4", "Comment 2 (Fairness evaluation):"):
        "The theoretical participation-fairness guarantee is one of the main strengths of the "
        "manuscript. However, the experimental validation could be expanded to demonstrate the "
        "proposed behavior under a broader range of settings. For example, additional experiments "
        "using different values of N and R, configurations where K does not divide N, or "
        "evaluation windows that do not exactly coincide with complete participation cycles would "
        "provide stronger empirical evidence for the generality of the proposed fairness behavior.",
    ("Reviewer 4", "Comment 3 (Experimental reproducibility):"):
        "The reported results appear to be obtained using a single random seed. Since federated "
        "learning performance can vary due to random initialization and data partitioning, "
        "reporting results over multiple independent runs together with statistics such as mean "
        "+- standard deviation would substantially improve the reproducibility and reliability of "
        "the reported performance, particularly for convergence-speed comparisons and the sparse "
        "participation setting.",
    ("Reviewer 4", "Comment 4 (Ablation study):"):
        "The proposed scoring function consists of three complementary components. An ablation "
        "study comparing participation debt only, debt plus under-prediction bonus, debt plus "
        "class-coverage penalty, and the complete SCOPE-FD score would help quantify the "
        "contribution of each component and provide additional insight into the design choices.",
    ("Reviewer 4", "Comment 5 (Privacy discussion):"):
        "The proposed method requires each client to upload its normalized label histogram. "
        "Although the manuscript briefly discusses possible privacy-preserving alternatives, a "
        "slightly more detailed discussion of the practical implications of sharing these "
        "statistics would further improve the completeness of the paper.",
}


def normalize(s):
    """Strip LaTeX down to the words and punctuation the Reviewer wrote."""
    s = s.replace("``", '"').replace("''", '"')
    s = s.replace(r"$\pm$", "+-").replace(r"$|$", "|")
    s = s.replace(r"\alpha_u", "alpha_u").replace(r"\alpha_d", "alpha_d")
    s = s.replace(r"\alpha", "alpha")
    s = re.sub(r"\\%", "%", s)
    s = re.sub(r"\\textit\{|\\textbf\{", "", s)
    s = s.replace("$", "").replace("\\", "")
    s = s.replace("{", "").replace("}", "")
    s = s.rstrip(". ").rstrip()
    return " ".join(s.split())


def letter_comments():
    t = io.open(P, encoding="utf-8").read()
    heads = [(m.start(), " ".join(m.group(1).split()))
             for m in re.finditer(r"Response to Comments of ([^}]*?)\}\}\}", t)]

    def who(pos):
        name = "?"
        for start, h in heads:
            if start < pos:
                name = h
        return name

    out = {}
    for m in re.finditer(r"\\textbf\{(Comment[^}]*?)\}\s*\\textit\{", t):
        start = m.end()
        depth, i = 1, start
        while i < len(t) and depth > 0:
            if t[i] == "{":
                depth += 1
            elif t[i] == "}":
                depth -= 1
            i += 1
        out[(who(m.start()), m.group(1))] = " ".join(t[start:i - 1].split())
    return out


if __name__ == "__main__":
    got = letter_comments()
    bad = missing = 0
    print("%-16s %-38s %s" % ("reviewer", "comment", "verdict"))
    print("-" * 78)
    for key in SOURCE:
        want = normalize(SOURCE[key])
        if key not in got:
            print("%-16s %-38s NOT IN LETTER" % key)
            missing += 1
            continue
        have = normalize(got[key])
        if have == want:
            print("%-16s %-38s verbatim" % key)
        else:
            bad += 1
            print("%-16s %-38s DIFFERS" % key)
            for line in difflib.unified_diff(want.split(), have.split(),
                                             "source", "letter", lineterm="", n=2):
                if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
                    print("        %s" % line)
    extra = [k for k in got if k not in SOURCE]
    print()
    for k in extra:
        print("  in letter but not in the source: %s %s" % k)
    print()
    print("%d comments checked, %d differ, %d missing, %d unexpected"
          % (len(SOURCE), bad, missing, len(extra)))
    raise SystemExit(1 if (bad or missing or extra) else 0)
