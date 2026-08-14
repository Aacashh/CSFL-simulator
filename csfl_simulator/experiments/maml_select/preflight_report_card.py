#!/usr/bin/env python3
"""Checks that must pass before the report-card campaign spends any GPU time.

Run directly, from anywhere:

    python csfl_simulator/experiments/maml_select/preflight_report_card.py

Two things are checked.

1.  The package actually imports.  Several directories under ``csfl_simulator``
    used to have no ``__init__.py``, so they resolved only as PEP 420 namespace
    packages.  ``setuptools.find_packages`` skips such directories, which means
    an installed or editable copy of the project never contained them, and
    ``import csfl_simulator.core.client`` failed on any machine that resolved
    the package through the install rather than through the working directory.
    This reports exactly which package is unreachable and why, instead of
    letting a bare ModuleNotFoundError out.

2.  ``inner_steps=0`` really disables the adaptation.  It used to be clamped to
    1, which would have turned the no-adaptation control into a relabelled copy
    of the default and produced a table row that quietly proved nothing.

The repository root is put on ``sys.path`` from this file's own location, so the
result does not depend on the working directory or on whether the project is
installed.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Packages the campaign needs. Every one of these has to be a real package with
# an __init__.py, not a namespace directory.
REQUIRED = [
    "csfl_simulator",
    "csfl_simulator.core",
    "csfl_simulator.core.client",
    "csfl_simulator.core.datasets",
    "csfl_simulator.selection",
    "csfl_simulator.selection.baseline",
    "csfl_simulator.selection.system_aware",
    "csfl_simulator.selection.ml",
    "csfl_simulator.experiments.maml_select.selector",
    "csfl_simulator.experiments.maml_select.simulator",
    "csfl_simulator.experiments.maml_select.run_experiments",
]


def _fail(message: str, code: int = 1) -> None:
    print()
    print("  " + "-" * 68)
    for line in message.strip().splitlines():
        print("  " + line)
    print("  " + "-" * 68)
    sys.exit(code)


def check_imports() -> None:
    import importlib

    broken = []
    for name in REQUIRED:
        try:
            importlib.import_module(name)
        except Exception as exc:
            broken.append((name, exc))

    if not broken:
        print("                 ok: all %d required modules import" % len(REQUIRED))
        return

    lines = ["Some modules the campaign needs could not be imported.", ""]
    for name, exc in broken:
        lines.append("  %-52s %s" % (name, type(exc).__name__ + ": " + str(exc)))
    lines += [
        "",
        "Repository root used: " + REPO_ROOT,
        "",
        "Most likely causes, in order.",
        "",
        "1. The checkout is incomplete. Check that the directory exists and has",
        "   files in it, then `git status` and `git pull`. On OneDrive or",
        "   iCloud, a folder can list files that are not actually downloaded.",
        "",
        "2. A stale install of csfl_simulator is shadowing the checkout. Run",
        "   `pip uninstall csfl-simulator`, or reinstall from this checkout with",
        "   `pip install -e .` so the new package layout is picked up.",
        "",
        "3. The checkout predates the __init__.py fix. `git pull` and retry.",
    ]
    _fail("\n".join(lines), code=5)


def check_directories() -> None:
    """A missing __init__.py is the failure this campaign hit first."""
    needed = [
        os.path.join("csfl_simulator", "core"),
        os.path.join("csfl_simulator", "selection", "baseline"),
        os.path.join("csfl_simulator", "selection", "system_aware"),
        os.path.join("csfl_simulator", "selection", "ml"),
        os.path.join("csfl_simulator", "experiments", "maml_select"),
    ]
    problems = []
    for rel in needed:
        full = os.path.join(REPO_ROOT, rel)
        if not os.path.isdir(full):
            problems.append("missing directory      " + rel)
        elif not os.path.exists(os.path.join(full, "__init__.py")):
            problems.append("no __init__.py in      " + rel)
    if problems:
        _fail("The package layout is not what the campaign expects.\n\n"
              + "\n".join("  " + p for p in problems)
              + "\n\nRun `git pull` in " + REPO_ROOT + " and try again.", code=6)
    print("                 ok: package layout is complete")


def check_control() -> None:
    import numpy as np
    import torch

    from csfl_simulator.experiments.maml_select import selector as S

    model = S._seeded_policy(2026, "cpu", 64)
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.normal(size=(8, 6)).astype("float32"))
    y = torch.tensor(rng.normal(size=(8,)).astype("float32"))

    zero = S._adapt(model, x, y, 0.01, 0)
    one = S._adapt(model, x, y, 0.01, 1)

    if not all(torch.equal(zero[k], before[k]) for k in before):
        _fail("inner_steps=0 changed the policy weights, so the no-adaptation\n"
              "control is not a control. Do not run the campaign.", code=7)
    if not any(not torch.equal(one[k], before[k]) for k in before):
        _fail("inner_steps=1 did not change the policy weights, so the inner\n"
              "step is inert. Do not run the campaign.", code=8)

    n_params = sum(p.numel() for p in model.parameters())
    print("                 ok: 0 steps leaves phi untouched, 1 step moves it")
    print("                 ok: policy has %d parameters" % n_params)


def main() -> None:
    check_directories()
    check_imports()
    check_control()


if __name__ == "__main__":
    main()
