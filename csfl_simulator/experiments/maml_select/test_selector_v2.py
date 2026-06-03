"""Lightweight checks for the experimental MAML-Select v2 selector.

Run with:
    python -m csfl_simulator.experiments.maml_select.test_selector_v2
"""
from __future__ import annotations

from copy import deepcopy

from csfl_simulator.core.client import ClientInfo
from csfl_simulator.experiments.maml_select.selector_v2 import select_clients


def _clients(with_histograms: bool = True):
    clients = []
    for cid in range(20):
        hist = None
        if with_histograms:
            hist = {cid % 5: 30 + cid, (cid + 1) % 5: 10 + cid}
        client = ClientInfo(id=cid, data_size=80 + 3 * cid, label_histogram=hist)
        client.compute_speed = 1.0 + (cid % 3)
        client.channel_quality = 0.5 + 0.05 * (cid % 7)
        client.estimated_duration = client.data_size / client.compute_speed
        client.estimated_energy = 0.01 * client.estimated_duration
        client.battery_capacity = 30.0
        clients.append(client)
    return clients


def _apply_feedback(clients, ids, round_idx: int) -> None:
    for cid in ids:
        client = clients[int(cid)]
        client.meta["last_local_loss_reduction"] = 0.1 + 0.01 * (cid % 5)
        client.last_loss = 1.0 / (1.0 + cid)
        client.grad_norm = 0.5 + 0.1 * (cid % 4)
        client.participation_count += 1
        client.last_selected_round = round_idx


def test_unique_k_and_missing_histograms() -> None:
    history = {"state": {}, "selected": []}
    ids, scores, state = select_clients(0, 5, _clients(with_histograms=False), history, None, device="cpu")
    assert len(ids) == 5
    assert len(set(ids)) == 5
    assert scores is not None and len(scores) == 20
    assert "research_maml_select_v2_state" in state


def test_deterministic_two_round_sequence() -> None:
    clients_a = _clients()
    clients_b = deepcopy(clients_a)
    history_a = {"state": {}, "selected": []}
    history_b = {"state": {}, "selected": []}

    for round_idx in range(6):
        ids_a, _, state_a = select_clients(round_idx, 5, clients_a, history_a, None, device="cpu")
        ids_b, _, state_b = select_clients(round_idx, 5, clients_b, history_b, None, device="cpu")
        assert ids_a == ids_b
        assert len(ids_a) == 5
        assert len(set(ids_a)) == 5
        history_a["state"].update(state_a)
        history_b["state"].update(state_b)
        _apply_feedback(clients_a, ids_a, round_idx)
        _apply_feedback(clients_b, ids_b, round_idx)


def main() -> None:
    test_unique_k_and_missing_histograms()
    test_deterministic_two_round_sequence()
    print("MAML-Select v2 selector self-test passed.")


if __name__ == "__main__":
    main()
