"""The inner-step count means what the ablation says it means.

`inner_steps=0` is the no-adaptation control the R2 report card asks for. It has
to leave the policy weights untouched, so that the outer update is a plain step
on the query loss and the method reduces to online regression with the same
network. It used to be clamped to 1, which silently turned the control into a
copy of the default.

Values of 1 and above must behave exactly as they did before, or every published
ablation number changes.
"""

from __future__ import annotations

import numpy as np
import torch

from csfl_simulator.experiments.maml_select import selector as S
from csfl_simulator.experiments.maml_select.selector import ReplayRecord


def _batch(n: int = 10, seed: int = 7):
    rng = np.random.default_rng(seed)
    x = torch.tensor(rng.normal(size=(n, 6)).astype("float32"))
    y = torch.tensor(rng.normal(size=(n,)).astype("float32"))
    return x, y


def _records(n: int = 10, seed: int = 11):
    rng = np.random.default_rng(seed)
    return [
        ReplayRecord(rng.normal(size=6).astype("float32"), float(rng.normal()))
        for _ in range(n)
    ]


def test_zero_inner_steps_leaves_the_policy_untouched():
    model = S._seeded_policy(2026, "cpu", 64)
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    x, y = _batch()

    adapted = S._adapt(model, x, y, 0.01, 0)

    assert all(torch.equal(adapted[k], before[k]) for k in before), \
        "inner_steps=0 moved the weights, so it is not a no-adaptation control"


def test_one_and_two_inner_steps_still_move_the_policy():
    model = S._seeded_policy(2026, "cpu", 64)
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    x, y = _batch()

    one = S._adapt(model, x, y, 0.01, 1)
    two = S._adapt(model, x, y, 0.01, 2)

    assert any(not torch.equal(one[k], before[k]) for k in before)
    assert any(not torch.equal(two[k], one[k]) for k in before)


def test_negative_inner_steps_behaves_like_zero():
    """Defensive. A bad config should not silently take one step."""
    model = S._seeded_policy(2026, "cpu", 64)
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    x, y = _batch()

    adapted = S._adapt(model, x, y, 0.01, -3)

    assert all(torch.equal(adapted[k], before[k]) for k in before)


def test_outer_step_skips_the_support_set_at_zero():
    """At 0 the support loss is never evaluated, so its diagnostics are nan."""
    support, query = _records(seed=11), _records(seed=13)

    model = S._seeded_policy(2026, "cpu", 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    diag = S._outer_step(model, optimizer, support, query, "cpu", 0.01, 0)

    assert diag is not None
    assert diag["inner_steps"] == 0
    assert diag["l_sup_before"] != diag["l_sup_before"], "support loss should be nan"

    model = S._seeded_policy(2026, "cpu", 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    diag_one = S._outer_step(model, optimizer, support, query, "cpu", 0.01, 1)

    assert diag_one["inner_steps"] == 1
    assert diag_one["l_sup_descent"] <= 0.0, "Lemma 1 says the inner step descends"


def test_zero_and_one_reach_different_query_objectives():
    """The control must actually be a different algorithm, not a relabelling."""
    support, query = _records(seed=11), _records(seed=13)

    losses = {}
    for steps in (0, 1):
        model = S._seeded_policy(2026, "cpu", 64)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        losses[steps] = S._outer_step(
            model, optimizer, support, query, "cpu", 0.01, steps)["l_query"]

    assert losses[0] != losses[1], \
        "the query objective is identical at 0 and 1 steps, so the inner step is inert"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nall inner-step control tests pass")
