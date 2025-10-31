from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple

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


def _smooth_series(ys: List[float], window: int) -> List[float]:
    if window is None or window <= 1:
        return ys
    try:
        k = int(window)
        k = max(1, k)
        if k == 1 or len(ys) == 0:
            return ys
        cumsum = np.cumsum(np.insert(np.asarray(ys, dtype=float), 0, 0.0))
        out = (cumsum[k:] - cumsum[:-k]) / float(k)
        # pad to original length by repeating last value
        if len(out) < len(ys):
            out = np.concatenate([out, np.full(len(ys) - len(out), out[-1] if len(out) > 0 else 0.0)])
        return out.tolist()
    except Exception:
        return ys


def plot_metric_compare_plotly(
    method_to_series: Dict[str, List[float]],
    metric_name: str,
    template: str = "plotly_white",
    methods_filter: Optional[List[str]] = None,
    smoothing_window: int = 0,
    y_axis_type: str = "linear",
    line_width: float = 2.0,
    legend_position: str = "right",
) -> go.Figure:
    fig = go.Figure()
    tmpl = _safe_template(template)
    names = list(method_to_series.keys())
    if methods_filter:
        names = [n for n in names if n in methods_filter]
    for name in names:
        ys = method_to_series[name]
        ys = _smooth_series(ys, smoothing_window)
        xs = list(range(len(ys)))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name=name, line={"width": line_width}, legendgroup=name))
    legend = {}
    if legend_position == "top":
        legend = {"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0.0, "xanchor": "left"}
    elif legend_position == "right":
        legend = {"orientation": "v", "y": 1.0, "yanchor": "top", "x": 1.02, "xanchor": "left"}
    # Enable linked legend toggling across subplots/figures
    if isinstance(legend, dict):
        legend["groupclick"] = "togglegroup"
    fig.update_layout(title=f"{metric_name} per Round", xaxis_title="Round", yaxis_title=metric_name, template=tmpl, legend=legend, uirevision="viz_static")
    fig.update_yaxes(type=y_axis_type)
    return fig


def plot_loss(metrics):
    ys = []
    xs = []
    for i, row in enumerate(metrics):
        if isinstance(row, dict):
            xs.append(row.get("round", i))
            ys.append(float(row.get("loss", 0.0) or 0.0))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines+markers', name='loss'))
    fig.update_layout(title="Loss per Round", xaxis_title="Round", yaxis_title="Loss", template="plotly_white")
    return fig


def plot_multi_panel_plotly(
    metric_to_series: Dict[str, Dict[str, List[float]]],
    template: str = "plotly_white",
    methods_filter: Optional[List[str]] = None,
    smoothing_window: int = 0,
    y_axis_type: str = "linear",
    line_width: float = 2.0,
    legend_position: str = "right",
):
    if make_subplots is None:
        # Fallback: single figure by concatenating traces
        flat = {}
        for metric, mseries in metric_to_series.items():
            for name, ys in mseries.items():
                if methods_filter and name not in methods_filter:
                    continue
                flat[f"{metric} — {name}"] = ys
        return plot_metric_compare_plotly(flat, "Combined", template, methods_filter=None, smoothing_window=smoothing_window, y_axis_type=y_axis_type, line_width=line_width)
    metrics = list(metric_to_series.keys())
    # Ensure deterministic ordering and up to 4 panels
    wanted = [m for m in ["Accuracy", "F1", "Precision", "Recall"] if m in metric_to_series] or metrics[:4]
    rows, cols = 2, 2
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=wanted)
    tmpl = _safe_template(template)
    for idx, metric in enumerate(wanted):
        r, c = (idx // cols) + 1, (idx % cols) + 1
        for name, ys in metric_to_series[metric].items():
            if methods_filter and name not in methods_filter:
                continue
            ys = _smooth_series(ys, smoothing_window)
            xs = list(range(len(ys)))
            fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name=name, line={"width": line_width}, showlegend=(idx == 0), legendgroup=name), row=r, col=c)
        fig.update_xaxes(title_text="Round", row=r, col=c)
        fig.update_yaxes(title_text=metric, row=r, col=c)
    legend = {}
    if legend_position == "top":
        legend = {"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0.0, "xanchor": "left"}
    elif legend_position == "right":
        legend = {"orientation": "v", "y": 1.0, "yanchor": "top", "x": 1.02, "xanchor": "left"}
    if isinstance(legend, dict):
        legend["groupclick"] = "togglegroup"
    fig.update_layout(template=tmpl, legend=legend, uirevision="viz_static")
    fig.update_yaxes(type=y_axis_type)
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


def plot_metric_compare_matplotlib(
    method_to_series: Dict[str, List[float]],
    metric_name: str,
    style_name: str = "classic",
    methods_filter: Optional[List[str]] = None,
    legend_outside: bool = True,
    legend_position: Optional[str] = None,
    legend_cols: int = 1,
    smoothing_window: int = 0,
    y_axis_type: str = "linear",
    line_width: float = 2.0,
):
    if plt is None:
        raise RuntimeError("Matplotlib is not available")
    style = _resolve_mpl_style(style_name)
    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=(6, 4))
        # Force white backgrounds regardless of style to avoid tinted/yellowish hue
        try:
            fig.patch.set_alpha(1.0)
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")
        except Exception:
            pass
        names = list(method_to_series.keys())
        if methods_filter:
            names = [n for n in names if n in methods_filter]
        for name in names:
            ys = _smooth_series(method_to_series[name], smoothing_window)
            xs = list(range(len(ys)))
            ax.plot(xs, ys, label=name, linewidth=line_width)
        ax.set_title(f"{metric_name} per Round")
        ax.set_xlabel("Round")
        ax.set_ylabel(metric_name)
        try:
            ax.set_yscale(y_axis_type)
        except Exception:
            pass
        pos = legend_position
        if pos is None:
            pos = "right" if legend_outside else "inside"
        if pos == "right":
            lg = ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=max(1, int(legend_cols)))
            try:
                lg.set_frame_on(True)
            except Exception:
                pass
            # leave room on the right for legend
            try:
                fig.tight_layout(rect=[0.0, 0.0, 0.80, 1.0])
            except Exception:
                fig.tight_layout()
        elif pos == "top":
            fig.legend(loc='upper center', ncol=max(1, int(legend_cols)))
            try:
                fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
            except Exception:
                fig.tight_layout()
        else:
            ax.legend(ncol=max(1, int(legend_cols)))
            fig.tight_layout()
    return fig


def plot_multi_panel_matplotlib(
    metric_to_series: Dict[str, Dict[str, List[float]]],
    style_name: str = "classic",
    methods_filter: Optional[List[str]] = None,
    legend_outside: bool = True,
    legend_position: Optional[str] = None,
    legend_cols: int = 1,
    smoothing_window: int = 0,
    y_axis_type: str = "linear",
    line_width: float = 2.0,
):
    if plt is None:
        raise RuntimeError("Matplotlib is not available")
    style = _resolve_mpl_style(style_name)
    with plt.style.context(style):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        # Force white backgrounds for the whole figure and all subplots
        try:
            fig.patch.set_alpha(1.0)
            fig.patch.set_facecolor("white")
            for row in axes:
                for ax in (row if isinstance(row, (list, tuple, np.ndarray)) else [row]):
                    ax.set_facecolor("white")
        except Exception:
            pass
        wanted = [m for m in ["Accuracy", "F1", "Precision", "Recall"] if m in metric_to_series] or list(metric_to_series.keys())[:4]
        for idx, metric in enumerate(wanted):
            r, c = divmod(idx, 2)
            ax = axes[r][c]
            names = list(metric_to_series[metric].keys())
            if methods_filter:
                names = [n for n in names if n in methods_filter]
            for name in names:
                ys = _smooth_series(metric_to_series[metric][name], smoothing_window)
                xs = list(range(len(ys)))
                ax.plot(xs, ys, label=name, linewidth=line_width)
            ax.set_title(metric)
            ax.set_xlabel("Round")
            ax.set_ylabel(metric)
            try:
                ax.set_yscale(y_axis_type)
            except Exception:
                pass
        # Put a single shared legend
        handles, labels = axes[0][0].get_legend_handles_labels()
        pos = legend_position
        if pos is None:
            pos = "right" if legend_outside else "inside"
        if pos == "right":
            fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=max(1, int(legend_cols)))
            try:
                fig.tight_layout(rect=[0.0, 0.0, 0.80, 1.0])
            except Exception:
                fig.tight_layout()
        elif pos == "top":
            fig.legend(handles, labels, loc='upper center', ncol=max(1, int(legend_cols)))
            try:
                fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.90])
            except Exception:
                fig.tight_layout()
        else:
            fig.legend(handles, labels, loc='upper center', ncol=max(1, int(legend_cols)))
            fig.tight_layout()
    return fig


# ---------- Additional plots for loss and selection counts ----------

def plot_selection_counts_compare_plotly(
    method_to_counts: Dict[str, List[int]],
    template: str = "plotly_white",
    methods_filter: Optional[List[str]] = None,
) -> go.Figure:
    fig = go.Figure()
    tmpl = _safe_template(template)
    names = list(method_to_counts.keys())
    if methods_filter:
        names = [n for n in names if n in methods_filter]
    x = None
    for name in names:
        ys = method_to_counts[name]
        if x is None:
            x = list(range(len(ys)))
        fig.add_trace(go.Bar(x=x, y=ys, name=name))
    fig.update_layout(title="Client Selection Counts", xaxis_title="Client ID", yaxis_title="Times Selected", template=tmpl, barmode='group')
    return fig


def plot_selection_counts_compare_matplotlib(
    method_to_counts: Dict[str, List[int]],
    style_name: str = "classic",
    methods_filter: Optional[List[str]] = None,
    legend_outside: bool = True,
    legend_cols: int = 1,
):
    if plt is None:
        raise RuntimeError("Matplotlib is not available")
    style = _resolve_mpl_style(style_name)
    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=(8, 4))
        try:
            fig.patch.set_alpha(1.0)
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")
        except Exception:
            pass
        names = list(method_to_counts.keys())
        if methods_filter:
            names = [n for n in names if n in methods_filter]
        # bar grouping
        num_clients = len(next(iter(method_to_counts.values()))) if method_to_counts else 0
        idx = np.arange(num_clients)
        w = 0.8 / max(1, len(names))
        for i, name in enumerate(names):
            ys = method_to_counts[name]
            ax.bar(idx + i * w, ys, width=w, label=name)
        ax.set_title("Client Selection Counts")
        ax.set_xlabel("Client ID")
        ax.set_ylabel("Times Selected")
        if legend_outside:
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=max(1, int(legend_cols)))
            try:
                fig.tight_layout(rect=[0.0, 0.0, 0.80, 1.0])
            except Exception:
                fig.tight_layout()
        else:
            ax.legend(ncol=max(1, int(legend_cols)))
            fig.tight_layout()
        return fig
