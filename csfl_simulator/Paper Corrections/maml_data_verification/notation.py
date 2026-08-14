"""Audit mathematical notation and terminology in the manuscript.

Three questions.

1. Is every symbol that appears in the body given a meaning somewhere?
2. Does any symbol carry two different meanings, or do two symbols carry one?
3. Is the vocabulary for a single object stable, for example whether the
   quantity c_{i,t} is called a cost, a utility or a score.
"""

import io
import re
import sys
from collections import Counter

TEX = sys.argv[1] if len(sys.argv) > 1 else \
    r"c:/Users/drash/OneDrive/Desktop/CSFL-simulator/csfl_simulator/Paper Corrections/MAML_REVISION_2/build/manuscript_r2_clean.tex"

B = chr(92)


def body(path):
    s = io.open(path, encoding="utf-8").read()
    start = s.find(B + "begin{abstract}")
    end = s.find(B + "begin{thebibliography}")
    s = s[start:end if end > 0 else len(s)]
    return re.sub("(?<!" + B + B + ")%[^\n]*", "", s)


def flat(s):
    return re.sub(r"\s+", " ", s)


# The symbols the paper actually leans on, and the words that would define them.
SYMBOLS = {
    r"\lambda": ["trade-off", "balances", "weight", "controls"],
    r"\beta": ["inner", "step size", "learning rate"],
    r"\eta": ["outer", "meta-learning rate", "step size"],
    r"\phi": ["policy", "meta-policy", "parameter"],
    r"\theta": ["global model"],
    r"\varepsilon": ["exploration"],
    r"\epsilon_0": ["standardiz", "division"],
    r"\rho_{i,t}": ["overrun", "fractional"],
    r"\tau_{i,t}": ["staleness", "rounds since"],
    r"\delta_t": ["displacement", "look-ahead"],
    r"\delta_{i,t}": ["delay"],
    r"q_{i,t}": ["selection frequency", "frequency"],
    r"q_t": ["query"],
    r"g_t": ["support"],
    r"c_{i,t}": ["cost", "utility"],
    r"V_T": ["drift", "cumulative"],
    r"L_q": ["smooth"],
    r"T_{target}": ["deadline", "target"],
    r"C_{model}": ["per-sample"],
    r"n_i": ["samples"],
    r"P": ["parameters", "policy"],
    r"K": ["cohort size"],
    r"N": ["clients"],
    r"E": ["epochs"],
    r"M": ["update", "size"],
    r"\Delta_{i,t}": ["loss reduction"],
    r"a_{i,t}": ["selected", "indicate"],
    r"p_i": ["selection counts"],
    r"f_{i,t}": ["processing rate", "device"],
    r"B_{i,t}": ["bandwidth"],
    r"b_{i,t}": ["battery"],
}

TERMS = ["cost", "utility", "score", "penalty", "reward", "objective"]


if __name__ == "__main__":
    s = body(TEX)
    f = flat(s)

    print("=" * 78)
    print("1. SYMBOL DEFINITIONS")
    print("=" * 78)
    print(f"{'symbol':<18}{'uses':>6}   status")
    undefined = []
    for sym, cues in sorted(SYMBOLS.items()):
        uses = len(re.findall(re.escape(sym), f))
        if uses == 0:
            continue
        first = f.find(sym)
        window = f[max(0, first - 320):first + 320].lower()
        hit = [c for c in cues if c in window]
        if hit:
            print(f"{sym:<18}{uses:>6}   defined near first use ({hit[0]})")
        else:
            # search the whole body before declaring it undefined
            anywhere = [c for c in cues if c in f.lower()]
            if anywhere:
                print(f"{sym:<18}{uses:>6}   defined elsewhere ({anywhere[0]})")
            else:
                print(f"{sym:<18}{uses:>6}   NO DEFINITION FOUND")
                undefined.append(sym)

    print()
    print("=" * 78)
    print("2. COLLISIONS, one symbol used for two things")
    print("=" * 78)
    # P is the classic risk here, policy parameter count versus tier power.
    for sym, meanings in ((r"$P$", ["parameters", "power"]),
                          (r"$E$", ["epochs", "energy"]),
                          (r"$q_", ["query", "frequency"]),
                          (r"$\delta", ["displacement", "delay"]),
                          (r"$\varepsilon", ["exploration", "floor"])):
        found = [m for m in meanings if m in f.lower()]
        flag = "CHECK BY EYE" if len(found) > 1 else "ok"
        print(f"  {sym:<14} meanings present {found}  -> {flag}")

    print()
    print("=" * 78)
    print("3. TERMINOLOGY STABILITY")
    print("=" * 78)
    for t in TERMS:
        print(f"  {t:<12}{len(re.findall(r'[bB]?' + t, f, re.I)):>4} uses")
    print()
    print("  every phrase that names c_{i,t}:")
    for m in re.finditer(r".{0,95}c_\{i,t\}.{0,95}", f):
        seg = m.group(0).strip()
        if any(w in seg.lower() for w in ("cost", "utility", "score")):
            print("    ..." + seg + "...")
