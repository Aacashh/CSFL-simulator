"""Recompute every derived claim in the prose from Table II itself.

The audit checks the table against the run logs. This checks the prose against
the table, which is the other half. A percentage quoted in the abstract and a
percentage quoted in Section V-A have to be the same number, and both have to
follow from the rows a reader can see.

Table II is read out of the manuscript source so the values are exact rather
than reconstructed from the rendered page.
"""

import io
import re
import sys

BASE = ("c:/Users/drash/OneDrive/Desktop/CSFL-simulator/csfl_simulator/"
        "Paper Corrections/MAML_REVISION_2/build/")

COLUMNS = ["acc", "f1", "tflops", "energy", "jain", "cov"]


def table_two():
    """dataset -> method -> {column: value}, from the manuscript source."""
    src = io.open(BASE + "manuscript_r2_clean.tex", encoding="utf-8").read()
    block = src[src.index(r"\label{tab:main_summary}"):]
    block = block[:block.index(r"\end{tabular*}")]
    out, dataset = {}, None
    for line in block.split("\n"):
        s = line.strip()
        m = re.match(r"\\multicolumn\{7\}\{@\{\}l\}\{\\textit\{([^}]*)\}\}", s)
        if m:
            dataset = m.group(1)
            out[dataset] = {}
            continue
        if dataset is None or "&" not in s or r"\pm" not in s:
            continue
        cells = [c.strip() for c in s.rstrip(r"\\").split("&")]
        name = re.sub(r"\\textbf\{|\}", "", cells[0]).strip()
        vals = {}
        for key, cell in zip(COLUMNS, cells[1:]):
            g = re.search(r"([-\d.]+)\$\\pm\$", cell.replace(r"\textbf{", ""))
            if g:
                vals[key] = float(g.group(1))
        out[dataset][name] = vals
    return out


def pct_drop(base, new):
    return 100.0 * (base - new) / base


def main():
    t = table_two()
    pdf_src = io.open(BASE + "manuscript_r2_clean.tex", encoding="utf-8").read()
    bad = 0

    print("=" * 72)
    print("DERIVED FROM TABLE II, AGAINST MAML-SELECT AND FEDAVG")
    print("=" * 72)
    print("%-14s %9s %9s %9s %9s" %
          ("dataset", "TFLOPs%", "energy%", "acc pp", "resid%"))
    derived = {}
    for ds in ("Fashion-MNIST", "CIFAR-10", "CIFAR-100"):
        fa = t[ds]["FedAvg"]
        ms = t[ds]["MAML-Select"]
        dt = pct_drop(fa["tflops"], ms["tflops"])
        de = pct_drop(fa["energy"], ms["energy"])
        da = ms["acc"] - fa["acc"]
        # the part of the energy saving not already explained by the compute
        resid = 100.0 * (1.0 - (1.0 - de / 100.0) / (1.0 - dt / 100.0))
        derived[ds] = (dt, de, da, resid)
        print("%-14s %9.2f %9.2f %9.2f %9.2f" % (ds, dt, de, da, resid))

    print()
    print("=" * 72)
    print("THE SAME QUANTITIES AS THE PROSE STATES THEM")
    print("=" * 72)

    checks = [
        ("TFLOPs reduction", [d[0] for d in derived.values()],
         r"lowers cumulative Tera Floating-Point Operations \(TFLOPs\) by "
         r"\$([\d.]+)\\%\$, \$([\d.]+)\\%\$ and \$([\d.]+)\\%\$",
         "abstract"),
        ("energy reduction", [d[1] for d in derived.values()],
         r"modelled energy by \$([\d.]+)\\%\$, \$([\d.]+)\\%\$ and "
         r"\$([\d.]+)\\%\$",
         "abstract"),
        ("accuracy change", [d[2] for d in derived.values()],
         r"for accuracy changes of \$-([\d.]+)\$, \$-([\d.]+)\$ and "
         r"\$-([\d.]+)\$ percentage points",
         "abstract"),
    ]
    for what, want, pattern, where in checks:
        m = re.search(pattern, pdf_src)
        if not m:
            print("  FAIL  %-20s pattern not found in the %s" % (what, where))
            bad += 1
            continue
        got = [float(g) for g in m.groups()]
        for w, g, ds in zip(want, got, ("Fashion", "CIFAR-10", "CIFAR-100")):
            ok = abs(abs(w) - g) < 0.05
            if not ok:
                bad += 1
            print("  %-5s %-20s %-10s table gives %6.2f, %s says %.2f"
                  % ("ok" if ok else "FAIL", what, ds, abs(w), where, g))

    # Section V-A quotes the compute reduction and the residual energy factor
    m = re.search(r"account for the reported reductions, \$([\d.]+)\\%\$ and "
                  r"\$([\d.]+)\\%\$ on Fashion-MNIST, \$([\d.]+)\\%\$ and "
                  r"\$([\d.]+)\\%\$ on CIFAR-10, and \$([\d.]+)\\%\$ and "
                  r"\$([\d.]+)\\%\$ on CIFAR-100", pdf_src)
    print()
    if not m:
        print("  FAIL  Section V-A reduction pairs not found")
        bad += 1
    else:
        g = [float(x) for x in m.groups()]
        pairs = [(g[0], g[1]), (g[2], g[3]), (g[4], g[5])]
        for (dt_s, re_s), ds in zip(pairs, ("Fashion-MNIST", "CIFAR-10",
                                            "CIFAR-100")):
            dt, _de, _da, resid = derived[ds]
            ok1 = abs(dt - dt_s) < 0.05
            ok2 = abs(resid - re_s) < 0.05
            if not (ok1 and ok2):
                bad += 1
            print("  %-5s V-A %-14s compute %5.2f vs %5.2f, residual %4.2f vs %4.2f"
                  % ("ok" if (ok1 and ok2) else "FAIL", ds, dt, dt_s, resid, re_s))

    # the abstract and Section V-A must not give the same quantity two ways
    print()
    print("=" * 72)
    print("THE ABSTRACT AND SECTION V-A ON THE SAME QUANTITY")
    print("=" * 72)
    a = re.search(r"lowers cumulative Tera Floating-Point Operations "
                  r"\(TFLOPs\) by \$([\d.]+)\\%\$", pdf_src)
    b = re.search(r"account for the reported reductions, \$([\d.]+)\\%\$",
                  pdf_src)
    if a and b:
        ok = a.group(1) == b.group(1)
        if not ok:
            bad += 1
        print("  %-5s Fashion-MNIST TFLOPs reduction, abstract %s%%, "
              "Section V-A %s%%" % ("ok" if ok else "FAIL",
                                    a.group(1), b.group(1)))

    print()
    print("%d problems" % bad)
    return 1 if bad else 0


sys.exit(main())
