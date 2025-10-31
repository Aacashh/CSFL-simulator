from __future__ import annotations
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.nn.functional as F


def eval_model(model, loader, device: str):
    model.eval()
    ys, yh = [], []
    loss_sum = 0.0
    n_total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = model(x)
            # accumulate CE loss (mean over dataset at the end)
            try:
                loss_sum += float(F.cross_entropy(p, y, reduction="sum").item())
                n_total += int(y.size(0))
            except Exception:
                pass
            yh.extend(p.argmax(1).cpu().numpy().tolist())
            ys.extend(y.cpu().numpy().tolist())
    acc = accuracy_score(ys, yh)
    # Avoid UndefinedMetricWarning when some labels have no predicted samples
    f1 = f1_score(ys, yh, average="macro", zero_division=0)
    pre = precision_score(ys, yh, average="macro", zero_division=0)
    rec = recall_score(ys, yh, average="macro", zero_division=0)
    loss = (loss_sum / max(1, n_total)) if n_total > 0 else 0.0
    return {"accuracy": acc, "f1": f1, "precision": pre, "recall": rec, "loss": loss}
