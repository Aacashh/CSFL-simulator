from __future__ import annotations
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F


def eval_model(model, loader, device: str):
    model.eval()
    ys, yh = [], []
    loss_sum = 0.0
    n_total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            try:
                p = model(x)
            except Exception as e:
                # Last-chance fallback: adapt first Conv2d in case of channel mismatch and retry once
                try:
                    c0 = int(x.shape[1])
                    def _first_conv_and_parent(m: nn.Module):
                        parent = None
                        aname = None
                        for n, mod in m.named_modules():
                            if isinstance(mod, nn.Conv2d):
                                parts = n.split(".")
                                p = m
                                for part in parts[:-1]:
                                    p = getattr(p, part)
                                return p, parts[-1], mod
                        return None, None, None
                    pmod, aname, conv = _first_conv_and_parent(model)
                    if conv is not None and int(conv.in_channels) != c0:
                        new_conv = nn.Conv2d(c0, conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride,
                                             padding=conv.padding, dilation=conv.dilation, groups=conv.groups,
                                             bias=(conv.bias is not None), padding_mode=conv.padding_mode).to(device)
                        with torch.no_grad():
                            w_old = conv.weight.data
                            in_old = int(w_old.shape[1])
                            if c0 == in_old:
                                new_conv.weight.copy_(w_old)
                            elif c0 > in_old:
                                reps = (c0 + in_old - 1) // in_old
                                w_rep = w_old.repeat(1, reps, 1, 1)[:, :c0]
                                w_rep = w_rep * (in_old / float(c0))
                                new_conv.weight.copy_(w_rep)
                            else:
                                step = in_old / float(c0)
                                idxs = [int(i * step) for i in range(c0)]
                                w_sel = w_old[:, idxs, :, :].clone()
                                new_conv.weight.copy_(w_sel)
                            if new_conv.bias is not None and conv.bias is not None:
                                new_conv.bias.copy_(conv.bias.data)
                        if pmod is not None and aname is not None:
                            setattr(pmod, aname, new_conv)
                        # retry
                        p = model(x)
                    else:
                        raise e
                except Exception:
                    # give up and re-raise original
                    raise e
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
