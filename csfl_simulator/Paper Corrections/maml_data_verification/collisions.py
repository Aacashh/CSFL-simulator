"""Confirm that no base symbol carries two distinct meanings.

Checks the symbol forms directly rather than the words around them, which is
what the earlier notation pass could not distinguish.
"""

import io
import re
import sys

BS = chr(92)
TEX = sys.argv[1] if len(sys.argv) > 1 else \
    r"c:/Users/drash/OneDrive/Desktop/CSFL-simulator/csfl_simulator/Paper Corrections/MAML_REVISION_2/build/manuscript_r2_clean.tex"


def body(path):
    s = io.open(path, encoding="utf-8").read()
    s = s[s.find(BS + "begin{abstract}"):s.find(BS + "begin{thebibliography}")]
    return re.sub("(?<!" + BS + BS + ")%[^\n]*", "", s)


# Each row is a base symbol and the distinct forms that must no longer share it.
RESOLVED = [
    ("q", "query objective", "q_t",
     "selection frequency", "q_{i,t}", r"\nu_{i,t}"),
    ("P", "meta-policy parameter count", "P=",
     "device-tier power", "P_i", r"\mathcal{P}_i"),
    ("delta", "look-ahead displacement", BS + "delta_t",
     "communication delay", BS + "delta_{i,t}", "d_{i,t}"),
    ("E", "local epochs", "E ",
     "modelled energy", "E_{total}", r"\mathcal{W}_{total}"),
]

if __name__ == "__main__":
    s = body(TEX)
    print(f"{'base':<8}{'kept for':<30}{'old clashing form':<20}{'now':<22}status")
    print("-" * 96)
    bad = 0
    for base, kept, keptform, other, oldform, newform in RESOLVED:
        old_left = s.count(oldform)
        new_present = s.count(newform)
        ok = old_left == 0 and new_present > 0
        bad += not ok
        print(f"{base:<8}{kept:<30}{oldform:<20}{newform:<22}"
              f"{'resolved' if ok else f'STILL PRESENT x{old_left}'}")

    print()
    print("free-symbol check, the replacements must not clash with anything else")
    for sym in (r"\nu", r"\mathcal{P}", "d_{i,t}", r"\mathcal{W}"):
        n = len(re.findall(re.escape(sym), s))
        print(f"  {sym:<16} {n} uses")

    print()
    print("terminology, the object c_{i,t} is minimized so it must be a cost")
    for word in ("per-client utility", "per-round utility", "per-client cost", "per-round cost"):
        print(f"  {word:<22} {s.count(word)}")

    print()
    print("resolved" if bad == 0 else f"{bad} collisions remain")
