import streamlit as st
from dataclasses import asdict

from csfl_simulator.core.simulator import FLSimulator, SimConfig
from csfl_simulator.core.utils import ROOT

st.set_page_config(page_title="CSFL Simulator", layout="wide")

if "simulator" not in st.session_state:
    st.session_state.simulator = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "cancel_run" not in st.session_state:
    st.session_state.cancel_run = False

st.title("CSFL Simulator (Playground)")

# Create tabs before referencing them
setup_tab, run_tab, compare_tab, export_tab = st.tabs(["Setup", "Run", "Compare", "Export"]) 

with setup_tab:
    st.subheader("Custom Method Editor")
    from pathlib import Path
    from csfl_simulator.app.components.editors import default_template, save_custom_method
    key_name = st.text_input("Custom method key (e.g., my.selector)", value="my.selector")
    code = st.text_area("Method code (select_clients...)", value=default_template(), height=300)
    if st.button("Validate & Save as Preset"):
        try:
            # quick validation
            compiled_code = compile(code, "<custom>", "exec")
            module_file, presets_file = save_custom_method(ROOT, key_name, code)
            st.success(f"Saved custom method to {module_file} and registered in {presets_file}.")
        except Exception as e:
            st.error(f"Failed to save: {e}")

with st.sidebar:
    st.header("Setup")
    dataset = st.selectbox("Dataset", ["MNIST", "Fashion-MNIST", "CIFAR-10", "CIFAR-100"], index=0)
    partition = st.selectbox("Partition", ["iid", "dirichlet", "label-shard"], index=0)
    alpha = st.slider("Dirichlet alpha", 0.05, 2.0, 0.5, 0.05)
    shards = st.number_input("Label shards per client", 1, 10, 2)

    model = st.selectbox("Model", ["CNN-MNIST", "LightCNN", "ResNet18"], index=0)
    total_clients = st.number_input("Total clients", 2, 1000, 10)
    k_clients = st.number_input("Clients per round (K)", 1, 100, 3)
    rounds = st.number_input("Rounds", 1, 200, 3)
    local_epochs = st.number_input("Local epochs", 1, 10, 1)
    batch_size = st.number_input("Batch size", 8, 512, 32)
    lr = st.number_input("Learning rate", 1e-4, 1.0, 0.01, format="%.5f")

    device_choice = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
    seed = st.number_input("Seed", 0, 10_000, 42)
    fast_mode = st.checkbox("Fast mode (few batches)", True)
    pretrained = st.checkbox("Load pretrained (if available)", False)

    with st.expander("Advanced (System & Privacy)"):
        time_budget = st.number_input("Round time budget (seconds, 0=none)", 0.0, 1000000.0, 0.0, format="%.2f")
        dp_sigma = st.number_input("DP Gaussian noise sigma (per-parameter)", 0.0, 10.0, 0.0, format="%.4f")
        dp_eps = st.number_input("DP epsilon consumed per selection", 0.0, 100.0, 0.0, format="%.3f")
        st.caption("Composite reward weights (optimization target)")
        colw1, colw2, colw3, colw4 = st.columns(4)
        w_acc = colw1.slider("w_acc", 0.0, 1.0, 0.6, 0.05)
        w_time = colw2.slider("w_time", 0.0, 1.0, 0.2, 0.05)
        w_fair = colw3.slider("w_fair", 0.0, 1.0, 0.1, 0.05)
        w_dp = colw4.slider("w_dp", 0.0, 1.0, 0.1, 0.05)

    # Load methods dynamically
    from csfl_simulator.selection.registry import MethodRegistry
    reg = MethodRegistry(); reg.load_presets()
    labels_map = reg.labels_map()
    label_list = list(labels_map.keys())
    default_idx = 0
    method_label = st.selectbox("Selection method", label_list, index=default_idx)
    method = reg.key_from_label(method_label)

    init_btn = st.button("Initialize Simulator", use_container_width=True)

if init_btn:
    cfg = SimConfig(
        dataset=dataset,
        partition=partition,
        dirichlet_alpha=alpha,
        shards_per_client=shards,
        total_clients=int(total_clients),
        clients_per_round=int(k_clients),
        rounds=int(rounds),
        local_epochs=int(local_epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        model=model,
        device=device_choice,
        seed=int(seed),
        fast_mode=fast_mode,
        pretrained=pretrained,
        time_budget=(float(time_budget) if 'time_budget' in locals() and time_budget > 0 else None),
        dp_sigma=float(dp_sigma) if 'dp_sigma' in locals() else 0.0,
        dp_epsilon_per_round=float(dp_eps) if 'dp_eps' in locals() else 0.0,
        reward_weights={"acc": float(w_acc) if 'w_acc' in locals() else 0.6,
                        "time": float(w_time) if 'w_time' in locals() else 0.2,
                        "fair": float(w_fair) if 'w_fair' in locals() else 0.1,
                        "dp": float(w_dp) if 'w_dp' in locals() else 0.1},
    )
    sim = FLSimulator(cfg)
    st.session_state.simulator = sim
    st.success("Simulator initialized. Switch to Run tab.")


with run_tab:
    st.subheader("Run Simulation")
    rl_expander = st.expander("Train RL-GNN Policy")
    with rl_expander:
        if st.session_state.simulator is None:
            st.info("Initialize simulator first.")
        else:
            episodes = st.number_input("Episodes", 1, 50, 3)
            if st.button("Train RL Policy"):
                from csfl_simulator.selection.ml.rl_gnn.trainer import train_policy
                with st.spinner("Training RL policy..."):
                    ckpt = train_policy(st.session_state.simulator.cfg, episodes=int(episodes), device=st.session_state.simulator.device)
                if ckpt and ckpt.exists():
                    st.success(f"Policy checkpoint saved: {ckpt}")
                else:
                    st.warning("Training completed, but no checkpoint was created (possibly due to PyG missing).")
    if st.session_state.simulator is None:
        st.info("Use the sidebar to initialize the simulator.")
    else:
        col_run, col_cancel = st.columns([1,1])
        run_clicked = col_run.button("Run", type="primary")
        cancel_clicked = col_cancel.button("Cancel Run", type="secondary")
        if cancel_clicked:
            st.session_state.cancel_run = True
            st.warning("Cancellation requested. Current round will complete.")
        if run_clicked:
            st.session_state.cancel_run = False
            # Placeholders for live updates
            prog = st.progress(0)
            status = st.empty()
            log_box = st.empty()
            log_lines = []
            total_rounds = max(1, int(st.session_state.simulator.cfg.rounds))
            def on_progress(rnd, info):
                pct = int(((rnd+1)/total_rounds)*100)
                prog.progress(min(100, max(0, pct)))
                acc = info.get("accuracy", 0.0)
                reward = info.get("reward", 0.0)
                status.write(f"Round {rnd+1}/{total_rounds} | acc={acc:.4f} | reward={reward:+.4f} | selected={info.get('selected', [])}")
                log_lines.append(f"[Round {rnd+1}] acc={acc:.4f} reward={reward:+.4f} selected={info.get('selected', [])}")
                log_box.code("\n".join(log_lines[-200:]))
            res = st.session_state.simulator.run(method_key=method, on_progress=on_progress, is_cancelled=lambda: st.session_state.cancel_run)
            st.session_state.last_result = res
            if res.get("stopped_early"):
                st.info("Run stopped early by user.")
        if st.session_state.last_result:
            res = st.session_state.last_result
            st.json({"run_id": res["run_id"], "device": res["device"], "config": res["config"]})
            st.write("Metrics (per round):")
            st.dataframe(res["metrics"]) 
            # Plots
            from csfl_simulator.app.components.plots import plot_accuracy, plot_participation, plot_selection_heatmap, plot_dp_usage, plot_round_time, plot_fairness, plot_composite
            st.plotly_chart(plot_accuracy(res["metrics"]), use_container_width=True)
            st.plotly_chart(plot_round_time(res["metrics"]), use_container_width=True)
            st.plotly_chart(plot_fairness(res["metrics"]), use_container_width=True)
            st.plotly_chart(plot_composite(res["metrics"]), use_container_width=True)
            # Build a lightweight client snapshot for plotting
            # Note: in this session, we use the simulator's current clients
            sim = st.session_state.simulator
            st.plotly_chart(plot_participation(sim.clients), use_container_width=True)
            st.plotly_chart(plot_selection_heatmap(sim.history.get("selected", []), sim.cfg.total_clients), use_container_width=True)
            st.plotly_chart(plot_dp_usage(sim.clients), use_container_width=True)

with compare_tab:
    st.subheader("Compare Methods")
    if st.session_state.simulator is None:
        st.info("Use the sidebar to initialize the simulator.")
    else:
        from csfl_simulator.selection.registry import MethodRegistry
        reg = MethodRegistry(); reg.load_presets()
        labels_map = reg.labels_map()
        label_list = list(labels_map.keys())
        default_labels = [next((lbl for lbl,k in labels_map.items() if k==method), label_list[0])] if label_list else []
        chosen_labels = st.multiselect("Methods to compare", label_list, default=default_labels)
        picks = [reg.key_from_label(lbl) for lbl in chosen_labels]
        repeats = st.number_input("Repeats per method", 1, 10, 1)
        style_col1, style_col2, style_col3 = st.columns([1,1,1])
        with style_col1:
            chart_style = st.radio("Chart style", ["Interactive (Plotly)", "Paper (Matplotlib)"], index=0)
        with style_col2:
            if chart_style.startswith("Interactive"):
                plotly_templates = ["plotly_white", "simple_white", "ggplot2", "seaborn", "presentation"]
                template_choice = st.selectbox("Plotly template", plotly_templates, index=0)
            else:
                try:
                    import matplotlib.pyplot as _plt  # type: ignore
                    mpl_styles = list(getattr(_plt.style, 'available', ["classic", "default", "ggplot", "seaborn"]))
                except Exception:
                    mpl_styles = ["classic", "default", "ggplot", "seaborn"]
                style_choice = st.selectbox("Matplotlib style", mpl_styles, index=0)
        with style_col3:
            show_combined = st.checkbox("Show combined 2x2", value=True)
        go = st.button("Run Comparison")
        if go and picks:
            from collections import defaultdict
            base_seed = int(st.session_state.simulator.cfg.seed)

            def extract_series(rows, key):
                ys = []
                for row in rows:
                    try:
                        ys.append(float(row.get(key, 0.0) or 0.0))
                    except Exception:
                        ys.append(0.0)
                return ys

            metric_names = ["accuracy", "f1", "precision", "recall"]
            pretty = {"accuracy": "Accuracy", "f1": "F1", "precision": "Precision", "recall": "Recall"}
            # metric -> {label -> mean_series}
            metric_to_series = {pretty[m]: {} for m in metric_names}

            for mkey in picks:
                per_metric_runs = {m: [] for m in metric_names}
                for r in range(int(repeats)):
                    cfg = SimConfig(**st.session_state.simulator.cfg.__dict__)
                    cfg.seed = base_seed + r
                    sim = FLSimulator(cfg)
                    res = sim.run(mkey)
                    for m in metric_names:
                        per_metric_runs[m].append(extract_series(res["metrics"], m))
                # pad each metric's runs to max len and take mean
                for m in metric_names:
                    runs = per_metric_runs[m]
                    if not runs:
                        continue
                    maxlen = max(len(a) for a in runs)
                    padded = []
                    for a in runs:
                        if len(a) < maxlen and len(a) > 0:
                            a = a + [a[-1]] * (maxlen - len(a))
                        padded.append(a)
                    mean = [sum(x) / len(x) for x in zip(*padded)]
                    label = next((lbl for lbl, k in labels_map.items() if k == mkey), mkey)
                    metric_to_series[pretty[m]][label] = mean

            if chart_style.startswith("Interactive"):
                from csfl_simulator.app.components.plots import (
                    plot_metric_compare_plotly,
                    plot_multi_panel_plotly,
                )
                for metric_display, series_map in metric_to_series.items():
                    fig = plot_metric_compare_plotly(series_map, metric_display, template=template_choice)
                    st.plotly_chart(fig, use_container_width=True)
                if show_combined:
                    figc = plot_multi_panel_plotly(metric_to_series, template=template_choice)
                    st.plotly_chart(figc, use_container_width=True)
            else:
                from csfl_simulator.app.components.plots import (
                    plot_metric_compare_matplotlib,
                    plot_multi_panel_matplotlib,
                )
                for metric_display, series_map in metric_to_series.items():
                    fig = plot_metric_compare_matplotlib(series_map, metric_display, style_name=style_choice)
                    st.pyplot(fig, clear_figure=True)
                if show_combined:
                    figc = plot_multi_panel_matplotlib(metric_to_series, style_name=style_choice)
                    st.pyplot(figc, clear_figure=True)

with export_tab:
    st.subheader("Export to Notebook")
    if st.session_state.simulator is None or st.session_state.last_result is None:
        st.info("Run a simulation first.")
    else:
        from importlib import import_module
        import inspect
        from csfl_simulator.app.export import export_config_to_ipynb
        from csfl_simulator.selection.registry import MethodRegistry
        reg = MethodRegistry(); reg.load_presets()
        labels_map = reg.labels_map()
        label_list = list(labels_map.keys())
        export_label = st.selectbox("Method to export", label_list)
        export_method = reg.key_from_label(export_label)
        # try to locate source
        try:
            module_path = reg.methods.get(export_method, None)
            if module_path:
                mod = import_module(module_path)
                code = inspect.getsource(mod)
            else:
                code = "# Method source unavailable"
        except Exception as e:
            code = f"# Error reading source: {e}"
        if st.button("Generate Notebook"):
            cfg = asdict(st.session_state.simulator.cfg)
            from pathlib import Path
            out_dir = ROOT / "artifacts" / "exports"
            out_path = out_dir / f"export_{st.session_state.simulator.run_id}.ipynb"
            p = export_config_to_ipynb(cfg, code, out_path)
            st.success(f"Exported: {p}")
            st.code(p)
