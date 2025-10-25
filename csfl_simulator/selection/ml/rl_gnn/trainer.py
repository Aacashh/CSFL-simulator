from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
import torch

from csfl_simulator.core.simulator import FLSimulator, SimConfig
from . import policy as pol


def train_policy(sim_cfg: SimConfig, episodes: int = 3, device: str = 'cpu') -> Path:
    """Simple REINFORCE-like training stub.
    - Uses current FLSimulator with method ml.rl_gnn
    - For each episode, enables TRAIN_MODE to sample selections and collect log_probs
    - Reward = final_acc - initial_acc
    - Loss = -sum(log_probs) * reward
    Saves checkpoint under artifacts/checkpoints/ml.rl_gnn/<run_id>/policy.pt
    Returns path to checkpoint
    """
    # Ensure policy exists
    pol.init_policy(in_dim=6, device=device)
    optimizer = torch.optim.Adam(pol.POLICY.parameters(), lr=1e-3)

    ckpt_path = None
    for ep in range(episodes):
        sim = FLSimulator(sim_cfg)
        pol.set_train_mode(True)
        pol.zero_logs()
        res = sim.run(method_key="ml.rl_gnn")
        # Compute reward from metrics
        mets = res["metrics"]
        if not mets:
            continue
        initial = mets[0]["accuracy"]  # round -1 baseline
        final = mets[-1]["accuracy"]
        reward = final - initial
        logs = pol.get_logs_and_clear()
        if not logs:
            continue
        loss = -reward * torch.stack(logs).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Save checkpoint
        out_dir = Path("CSFL-simulator") / "artifacts" / "checkpoints" / "ml.rl_gnn" / res["run_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_dir / "policy.pt"
        pol.save_checkpoint(str(ckpt_path))
    return ckpt_path if ckpt_path else Path()
