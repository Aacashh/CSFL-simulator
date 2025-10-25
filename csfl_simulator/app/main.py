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
    method_list = reg.list_methods()
    method = st.selectbox("Selection method", method_list, index=0)

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
        picks = st.multiselect("Methods to compare", reg.list_methods(), default=[method])
        repeats = st.number_input("Repeats per method", 1, 10, 1)
        go = st.button("Run Comparison")
        if go and picks:
            import plotly.graph_objects as gofig
            fig = gofig.Figure()
            fig2 = gofig.Figure()
            for mkey in picks:
                all_acc = []
                all_comp = []
                for r in range(int(repeats)):
                    sim = FLSimulator(SimConfig(**st.session_state.simulator.cfg.__dict__))
                    res = sim.run(mkey)
                    acc = [row["accuracy"] for row in res["metrics"]]
                    comp = [row.get("composite", 0.0) for row in res["metrics"]]
                    all_acc.append(acc)
                    all_comp.append(comp)
                # pad to same length
                maxlen = max(len(a) for a in all_acc)
                for a in all_acc:
                    if len(a) < maxlen:
                        a.extend([a[-1]]*(maxlen-len(a)))
                maxlen2 = max(len(a) for a in all_comp)
                for a in all_comp:
                    if len(a) < maxlen2 and len(a) > 0:
                        a.extend([a[-1]]*(maxlen2-len(a)))
                mean = [sum(x)/len(x) for x in zip(*all_acc)]
                mean2 = [sum(x)/len(x) for x in zip(*all_comp)] if all_comp and all_comp[0] else []
                fig.add_trace(gofig.Scatter(y=mean, mode='lines', name=mkey))
                if mean2:
                    fig2.add_trace(gofig.Scatter(y=mean2, mode='lines', name=mkey))
            st.plotly_chart(fig, use_container_width=True)
            if len(fig2.data) > 0:
                st.plotly_chart(fig2, use_container_width=True)

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
        method_names = reg.list_methods()
        export_method = st.selectbox("Method to export", method_names)
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
