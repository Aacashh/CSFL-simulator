from __future__ import annotations

import json
import random
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch

from csfl_simulator.core.client import ClientInfo
from csfl_simulator.core.datasets import FSDDSpectrogram
from csfl_simulator.core.fd_simulator import _apply_public_label_noise
from csfl_simulator.core.dp import laplace_noise_histogram
from csfl_simulator.core.metrics import (
    participation_gini,
    rolling_window_participation_gini,
    rounds_to_absolute_accuracy,
)
from csfl_simulator.experiments.scope_fd.aggregate_results import aggregate
from csfl_simulator.selection.fd_native import divfl_fd, scope_fd, subtrunc_fd


class ScopeRevisionTests(unittest.TestCase):
    def make_clients(self, count=5):
        return [
            ClientInfo(
                id=cid,
                data_size=10,
                label_histogram={cid % 2: 10},
                channel_quality=0.5 + cid * 0.1,
                estimated_energy=float(cid + 1),
            )
            for cid in range(count)
        ]

    def test_histogram_laplace_release_is_reproducible_and_nonnegative(self):
        first = laplace_noise_histogram({0: 5, 1: 2}, 0.5, random.Random(7), num_classes=3)
        second = laplace_noise_histogram({0: 5, 1: 2}, 0.5, random.Random(7), num_classes=3)
        self.assertEqual(first, second)
        self.assertTrue(all(value >= 0 for value in first.values()))
        self.assertEqual(
            laplace_noise_histogram({0: 5}, float("inf"), random.Random(1)),
            {0: 5.0},
        )

    def test_debt_only_balances_a_nondivisible_cycle(self):
        clients = self.make_clients()
        history = {"state": {}, "selected": []}
        for round_idx in range(5):
            selected, _, state = scope_fd.select_clients(
                round_idx,
                2,
                clients,
                history,
                random.Random(round_idx),
                alpha_uncertainty=0,
                alpha_diversity=0,
                disable_server_signal=True,
                disable_diversity_penalty=True,
            )
            history["state"].update(state)
            history["selected"].append(selected)
            for cid in selected:
                clients[cid].participation_count += 1
        self.assertEqual([client.participation_count for client in clients], [2] * 5)
        self.assertEqual(participation_gini([2] * 5), 0.0)

    def test_private_histogram_is_released_once_and_cached(self):
        clients = self.make_clients(3)
        history = {"state": {}, "selected": []}
        _, _, state = scope_fd.select_clients(
            0, 1, clients, history, random.Random(2), histogram_epsilon=0.5
        )
        history["state"].update(state)
        cached = json.dumps(state["scope_private_histograms"], sort_keys=True)
        _, _, next_state = scope_fd.select_clients(
            1, 1, clients, history, random.Random(999), histogram_epsilon=0.5
        )
        self.assertEqual(
            cached,
            json.dumps(next_state["scope_private_histograms"], sort_keys=True),
        )

    def test_channel_aware_scope_enforces_energy_budget(self):
        clients = self.make_clients(3)
        clients[0].estimated_energy = 10.0
        clients[1].estimated_energy = 1.0
        clients[2].estimated_energy = 1.0
        selected, _, _ = scope_fd.select_clients(
            0,
            2,
            clients,
            {"state": {}},
            random.Random(1),
            alpha_uncertainty=0,
            alpha_diversity=0,
            alpha_channel=1,
            channel_energy_mix=0,
            enforce_energy_budget=True,
            energy_budget=2,
        )
        self.assertEqual(set(selected), {1, 2})

    def test_fd_submodular_ports_consume_logit_signal_cache(self):
        clients = self.make_clients(3)
        history = {
            "state": {
                "fd_client_signals": {
                    0: {"logit_representation": torch.tensor([1.0, 0.0]), "public_loss": 0.1},
                    1: {"logit_representation": torch.tensor([0.9, 0.1]), "public_loss": 2.0},
                    2: {"logit_representation": torch.tensor([-1.0, 0.0]), "public_loss": 0.5},
                }
            },
            "selected": [],
        }
        div_selected, _, div_state = divfl_fd.select_clients(
            1, 2, clients, history, random.Random(3)
        )
        sub_selected, _, sub_state = subtrunc_fd.select_clients(
            1, 2, clients, history, random.Random(3)
        )
        self.assertEqual(len(set(div_selected)), 2)
        self.assertEqual(len(set(sub_selected)), 2)
        self.assertFalse(div_state["divfl_fd"]["exploration_round"])
        self.assertEqual(sub_state["subtrunc_fd"]["observed_public_losses"], 3)

    def test_rolling_metrics_and_absolute_thresholds(self):
        history = [[0, 1], [2, 3], [4, 0]]
        value = rolling_window_participation_gini(history, 5, 2)
        self.assertAlmostEqual(value, participation_gini([1, 0, 1, 1, 1]))
        summary = rounds_to_absolute_accuracy(
            [
                {"round": -1, "accuracy": 0.1, "wall_clock": 0},
                {"round": 0, "accuracy": 0.6, "wall_clock": 1},
                {"round": 1, "accuracy": 0.75, "wall_clock": 2},
            ]
        )
        self.assertEqual(summary["rounds_to_abs_60"], 1)
        self.assertEqual(summary["rounds_to_abs_70"], 2)
        self.assertIsNone(summary["rounds_to_abs_80"])

    def test_public_label_noise_is_seeded_and_replaces_labels(self):
        labels = torch.tensor([0, 1, 2, 3, 4])
        first = _apply_public_label_noise(labels, 1.0, 5, 7)
        second = _apply_public_label_noise(labels, 1.0, 5, 7)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.all(first != labels))
        self.assertTrue(torch.equal(_apply_public_label_noise(labels, 0.0, 5, 7), labels))

    def test_fsdd_spectrogram_adapter(self):
        wave = (np.sin(np.linspace(0, 80, 4000)) * 30000).astype(np.int16)
        dataset = object.__new__(FSDDSpectrogram)
        dataset.items = [(Path("synthetic.wav"), 3)]
        with patch("scipy.io.wavfile.read", return_value=(8000, wave)):
            x, y = dataset[0]
        self.assertEqual(tuple(x.shape), (1, 64, 64))
        self.assertEqual(y, 3)

    def test_seed_aggregation(self):
        records = []
        for seed, accuracy in ((11, 0.6), (22, 0.8)):
            payload = {
                "config": {"seed": seed, "dataset": "synthetic"},
                "results": {
                    "fd_native.scope_fd": {
                        "metrics": [
                            {"round": 0, "accuracy": accuracy, "loss": 1 - accuracy,
                             "fairness_gini": 0.1, "wall_clock": 1.0}
                        ],
                        "convergence": {},
                    },
                    "heuristic.random": {
                        "metrics": [
                            {"round": 0, "accuracy": accuracy - 0.1, "loss": 1.1 - accuracy,
                             "fairness_gini": 0.2, "wall_clock": 1.0}
                        ],
                        "convergence": {},
                    },
                },
            }
            records.append(("group", "smoke", {"dataset": "synthetic"}, payload))

        module = "csfl_simulator.experiments.scope_fd.aggregate_results"
        with patch(f"{module}._result_files", return_value=[Path("a"), Path("b")]), \
                patch(f"{module}._load_record", side_effect=records):
            result = aggregate([Path(".")], "fd_native.scope_fd")
        group = next(iter(result["groups"].values()))
        accuracy = group["methods"]["fd_native.scope_fd"]["final"]["accuracy"]
        self.assertAlmostEqual(accuracy["mean"], 0.7)
        self.assertEqual(accuracy["n"], 2)


if __name__ == "__main__":
    unittest.main()
