from __future__ import annotations
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch


def eval_model(model, loader, device: str):
    model.eval()
    ys, yh = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = model(x)
            yh.extend(p.argmax(1).cpu().numpy().tolist())
            ys.extend(y.cpu().numpy().tolist())
    acc = accuracy_score(ys, yh)
    # Avoid UndefinedMetricWarning when some labels have no predicted samples
    f1 = f1_score(ys, yh, average="macro", zero_division=0)
    pre = precision_score(ys, yh, average="macro", zero_division=0)
    rec = recall_score(ys, yh, average="macro", zero_division=0)
    return {"accuracy": acc, "f1": f1, "precision": pre, "recall": rec}
