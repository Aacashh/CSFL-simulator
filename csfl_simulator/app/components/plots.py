from __future__ import annotations
import numpy as np
import plotly.graph_objects as go


def plot_accuracy(metrics_list, names=None):
    """Plot accuracy over rounds.

    Accepts several input shapes:
    - List[Dict]: a single run, where each dict has keys like {"round", "accuracy", ...}
    - List[List[Dict]]: multiple runs (each inner list is like the above)
    - List[float|int|str]: a single run of y-values
    - Dict[str, List[Dict|float|int|str]]: mapping series names to runs
    - Dict with key "accuracy" -> List[float]
    """
    fig = go.Figure()

    def series_from_sequence(seq, default_name):
        xs, ys = [], []
        for idx, row in enumerate(seq):
            if isinstance(row, dict):
                xs.append(row.get("round", idx))
                ys.append(row.get("accuracy", 0.0))
            elif isinstance(row, (list, tuple)) and row and isinstance(row[0], dict):
                # Nested list of dicts
                sub_xs = [r.get("round", j) for j, r in enumerate(row)]
                sub_ys = [r.get("accuracy", 0.0) for r in row]
                # Flatten by appending end value and index
                xs.append(sub_xs[-1] if sub_xs else idx)
                ys.append(sub_ys[-1] if sub_ys else 0.0)
            else:
                # Try numeric cast; if fails, drop-in 0.0
                try:
                    ys.append(float(row))
                except Exception:
                    ys.append(0.0)
                xs.append(idx)
        return xs, ys, default_name

    series = []  # list of tuples: (xs, ys, name)

    # Dict input
    if isinstance(metrics_list, dict):
        # Case: {"accuracy": [...]} or {name: sequence, ...}
        if "accuracy" in metrics_list and isinstance(metrics_list["accuracy"], (list, tuple)):
            ys = []
            for v in metrics_list["accuracy"]:
                try:
                    ys.append(float(v))
                except Exception:
                    ys.append(0.0)
            xs = list(range(len(ys)))
            series.append((xs, ys, (names[0] if names else "Run 1")))
        else:
            # Treat as mapping of names -> sequences
            for i, (k, v) in enumerate(metrics_list.items()):
                nm = names[i] if names and i < len(names) else str(k)
                if isinstance(v, (list, tuple)):
                    xs, ys, _ = series_from_sequence(v, nm)
                else:
                    # Fallback: single value
                    try:
                        ys = [float(v)]
                    except Exception:
                        ys = [0.0]
                    xs = [0]
                series.append((xs, ys, nm))

    # List input
    elif isinstance(metrics_list, list):
        if not metrics_list:
            series.append(([], [], (names[0] if names else "Run 1")))
        else:
            first = metrics_list[0]
            if isinstance(first, dict):
                xs, ys, nm = series_from_sequence(metrics_list, (names[0] if names else "Run 1"))
                series.append((xs, ys, nm))
            elif isinstance(first, (list, tuple)):
                for i, seq in enumerate(metrics_list):  # type: ignore
                    nm = names[i] if names and i < len(names) else f"Run {i+1}"
                    xs, ys, _ = series_from_sequence(seq, nm)
                    series.append((xs, ys, nm))
            else:
                xs, ys, nm = series_from_sequence(metrics_list, (names[0] if names else "Run 1"))
                series.append((xs, ys, nm))
    else:
        # Fallback: single numeric
        try:
            val = float(metrics_list)  # type: ignore
            series.append(([0], [val], (names[0] if names else "Run 1")))
        except Exception:
            series.append(([], [], (names[0] if names else "Run 1")))

    for xs, ys, nm in series:
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines+markers', name=nm))

    fig.update_layout(title="Accuracy per Round", xaxis_title="Round", yaxis_title="Accuracy", template="plotly_white")
    return fig


def plot_participation(clients):
    x = [c.id for c in clients]
    y = [c.participation_count for c in clients]
    fig = go.Figure(data=[go.Bar(x=x, y=y)])
    fig.update_layout(title="Client Participation Counts", xaxis_title="Client ID", yaxis_title="Times Selected", template="plotly_white")
    return fig


def plot_selection_heatmap(history_selected, total_clients: int):
    rounds = len(history_selected)
    mat = np.zeros((total_clients, rounds), dtype=float)
    for r, ids in enumerate(history_selected):
        for cid in ids:
            if 0 <= cid < total_clients:
                mat[cid, r] = 1.0
    fig = go.Figure(data=go.Heatmap(z=mat, colorscale='Blues'))
    fig.update_layout(title="Selection Heatmap (clients x rounds)", xaxis_title="Round", yaxis_title="Client ID", template="plotly_white")
    return fig


def plot_dp_usage(clients):
    x = [c.id for c in clients]
    y = [getattr(c, 'dp_epsilon_used', 0.0) for c in clients]
    fig = go.Figure(data=[go.Bar(x=x, y=y)])
    fig.update_layout(title="Cumulative DP epsilon used per Client", xaxis_title="Client ID", yaxis_title="Sigma epsilon", template="plotly_white")
    return fig
