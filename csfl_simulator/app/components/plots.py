from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List

try:
    # Optional import; only used when paper-style is selected
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None

try:
    from plotly.subplots import make_subplots
except Exception:  # pragma: no cover
    make_subplots = None


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


def plot_round_time(metrics):
    ys = []
    xs = []
    for i, row in enumerate(metrics):
        if isinstance(row, dict):
            xs.append(row.get("round", i))
            ys.append(float(row.get("round_time", 0.0) or 0.0))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines+markers', name='round_time'))
    fig.update_layout(title="Estimated Round Time", xaxis_title="Round", yaxis_title="Time (a.u.)", template="plotly_white")
    return fig


def plot_fairness(metrics):
    ys = []
    xs = []
    for i, row in enumerate(metrics):
        if isinstance(row, dict):
            xs.append(row.get("round", i))
            ys.append(float(row.get("fairness_var", 0.0) or 0.0))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines+markers', name='fairness_var'))
    fig.update_layout(title="Participation Fairness (variance)", xaxis_title="Round", yaxis_title="Variance", template="plotly_white")
    return fig


def plot_composite(metrics):
    ys = []
    xs = []
    for i, row in enumerate(metrics):
        if isinstance(row, dict):
            xs.append(row.get("round", i))
            ys.append(float(row.get("composite", 0.0) or 0.0))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines+markers', name='composite'))
    fig.update_layout(title="Composite Score per Round", xaxis_title="Round", yaxis_title="Composite", template="plotly_white")
    return fig


# --------- New comparison plotting helpers ---------

def _safe_template(template: str) -> str:
    # Accept known plotly templates; fall back to plotly_white
    allowed = {
        "plotly_white", "plotly", "simple_white", "ggplot2", "seaborn",
        "presentation", "xgridoff", "ygridoff", "gridon", "none"
    }
    return template if template in allowed else "plotly_white"


def plot_metric_compare_plotly(method_to_series: Dict[str, List[float]], metric_name: str, template: str = "plotly_white"):
    fig = go.Figure()
    tmpl = _safe_template(template)
    for name, ys in method_to_series.items():
        xs = list(range(len(ys)))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name=name))
    fig.update_layout(title=f"{metric_name} per Round", xaxis_title="Round", yaxis_title=metric_name, template=tmpl)
    return fig


def plot_multi_panel_plotly(metric_to_series: Dict[str, Dict[str, List[float]]], template: str = "plotly_white"):
    if make_subplots is None:
        # Fallback: single figure by concatenating traces
        flat = {}
        for metric, mseries in metric_to_series.items():
            for name, ys in mseries.items():
                flat[f"{metric} — {name}"] = ys
        return plot_metric_compare_plotly(flat, "Combined", template)
    metrics = list(metric_to_series.keys())
    # Ensure deterministic ordering and up to 4 panels
    wanted = [m for m in ["Accuracy", "F1", "Precision", "Recall"] if m in metric_to_series] or metrics[:4]
    rows, cols = 2, 2
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=wanted)
    tmpl = _safe_template(template)
    for idx, metric in enumerate(wanted):
        r, c = (idx // cols) + 1, (idx % cols) + 1
        for name, ys in metric_to_series[metric].items():
            xs = list(range(len(ys)))
            fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name=name, showlegend=(idx == 0)), row=r, col=c)
        fig.update_xaxes(title_text="Round", row=r, col=c)
        fig.update_yaxes(title_text=metric, row=r, col=c)
    fig.update_layout(template=tmpl)
    return fig


def _resolve_mpl_style(style_name: str) -> str:
    if plt is None:
        return "classic"
    try:
        available = set(plt.style.available)
    except Exception:
        available = {"classic", "default"}
    if style_name in available:
        return style_name
    # Common aliases
    for candidate in ["classic", "default", "ggplot"]:
        if candidate in available:
            return candidate
    return "classic"


def plot_metric_compare_matplotlib(method_to_series: Dict[str, List[float]], metric_name: str, style_name: str = "classic"):
    if plt is None:
        raise RuntimeError("Matplotlib is not available")
    style = _resolve_mpl_style(style_name)
    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, ys in method_to_series.items():
            xs = list(range(len(ys)))
            ax.plot(xs, ys, label=name)
        ax.set_title(f"{metric_name} per Round")
        ax.set_xlabel("Round")
        ax.set_ylabel(metric_name)
        ax.legend()
        fig.tight_layout()
    return fig


def plot_multi_panel_matplotlib(metric_to_series: Dict[str, Dict[str, List[float]]], style_name: str = "classic"):
    if plt is None:
        raise RuntimeError("Matplotlib is not available")
    style = _resolve_mpl_style(style_name)
    with plt.style.context(style):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        wanted = [m for m in ["Accuracy", "F1", "Precision", "Recall"] if m in metric_to_series] or list(metric_to_series.keys())[:4]
        for idx, metric in enumerate(wanted):
            r, c = divmod(idx, 2)
            ax = axes[r][c]
            for name, ys in metric_to_series[metric].items():
                xs = list(range(len(ys)))
                ax.plot(xs, ys, label=name)
            ax.set_title(metric)
            ax.set_xlabel("Round")
            ax.set_ylabel(metric)
            if idx == 0:
                ax.legend()
        fig.tight_layout()
    return fig
