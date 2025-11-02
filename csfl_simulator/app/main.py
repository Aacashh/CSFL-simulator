import streamlit as st
from dataclasses import asdict
import traceback
import inspect
import sys


# Force reload of core modules to avoid stale bytecode cache issues
# Remove this block after confirming cache is clean
_RELOAD_CORE = True
if _RELOAD_CORE:
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith('csfl_simulator.core'):
            try:
                del sys.modules[mod_name]
            except Exception:
                pass

from csfl_simulator.core.simulator import FLSimulator, SimConfig
from csfl_simulator.core.utils import ROOT, cleanup_memory

st.set_page_config(page_title="CSFL Simulator", layout="wide")

# Diagnostic: verify core modules are fresh
try:
    from csfl_simulator.core import models as _models_mod
    import inspect as _insp
    _fwd_src = _insp.getsource(_models_mod.CNNMnist.forward)
    if "_match_channels" not in _fwd_src:
        st.error("⚠️ STALE CACHE DETECTED: core/models.py is using old bytecode. Run `./clean_cache.sh` and restart!")
except Exception:
    pass

# Plot-call compatibility helper: filter kwargs not supported by the target function
def _call_plot_func(func, *args, **kwargs):
    try:
        sig = inspect.signature(func)
        allowed = set(
            p.name
            for p in sig.parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        )
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        return func(*args, **filtered)
    except Exception:
        # Last resort: try positional-only
        return func(*args)

# Safe render helpers
def _safe_plotly(plot_func, *args, **kwargs):
    try:
        fig = _call_plot_func(plot_func, *args, **kwargs)
        # Ensure a unique key per call to avoid duplicate element IDs
        try:
            if not hasattr(_safe_plotly, "_counter"):
                _safe_plotly._counter = 0  # type: ignore[attr-defined]
            _safe_plotly._counter += 1  # type: ignore[attr-defined]
            key = f"plotly_safe_{_safe_plotly._counter}"  # type: ignore[attr-defined]
        except Exception:
            key = None
        if key is not None:
            st.plotly_chart(fig, use_container_width=True, key=key)
        else:
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Plotly rendering failed: {e}")


def _safe_mpl(plot_func, *args, **kwargs):
    try:
        fig = _call_plot_func(plot_func, *args, **kwargs)
        st.pyplot(fig, clear_figure=True)
    except Exception as e:
        st.error(f"Matplotlib rendering failed: {e}")


def _get_state_module():
    """Import snapshot state module robustly across environments."""
    try:
        from csfl_simulator.app import state as _state  # type: ignore
        return _state
    except Exception:
        try:
            import importlib.util  # type: ignore
            from pathlib import Path as _Path  # type: ignore
            _p = _Path(__file__).resolve().parent / "state.py"
            spec = importlib.util.spec_from_file_location("csfl_simulator.app.state", _p)
            module = importlib.util.module_from_spec(spec)  # type: ignore
            assert spec and spec.loader
            spec.loader.exec_module(module)  # type: ignore
            return module
        except Exception:
            return None

if "simulator" not in st.session_state:
    st.session_state.simulator = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "cancel_run" not in st.session_state:
    st.session_state.cancel_run = False
if "compare_results" not in st.session_state:
    st.session_state.compare_results = None
if "run_data" not in st.session_state:
    st.session_state.run_data = None
if "run_ui" not in st.session_state:
    st.session_state.run_ui = {
        "show_accuracy": True,
        "show_loss": True,
        "show_time": True,
        "show_fair": True,
        "show_composite": True,
    }
if "compare_data" not in st.session_state:
    st.session_state.compare_data = None
if "compare_ui" not in st.session_state:
    st.session_state.compare_ui = {
        "chart_style": "Interactive (Plotly)",
        "plotly_template": "plotly_white",
        "mpl_style": "classic",
        "methods_filter": None,
        "metrics_filter": None,
        "smoothing": 0,
        "y_scale": "linear",
        "legend_position": "right",
        "legend_cols": 1,
        "line_width": 2.0,
        "show_combined": True,
    }
if "visualize_data" not in st.session_state:
    st.session_state.visualize_data = None
if "visualize_ui" not in st.session_state:
    st.session_state.visualize_ui = {
        "source": "compare",
        "chart_style": "Interactive (Plotly)",
        "plotly_template": "plotly_white",
        "mpl_style": "classic",
        "methods": None,
        "metrics": ["Accuracy", "F1", "Precision", "Recall"],
        "smoothing": 0,
        "y_scale": "linear",
        "legend_position": "right",
        "line_width": 2.0,
        "show_combined": True,
        "round_start": 0,
        "round_end": None,
        "lock_data": False,
    }

st.title("CSFL Simulator (Playground)")

# Create tabs before referencing them
setup_tab, run_tab, compare_tab, visualize_tab, export_tab = st.tabs(["Setup", "Run", "Compare", "Visualize", "Export"]) 

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
    dataset = st.selectbox("Dataset", ["MNIST", "Fashion-MNIST", "CIFAR-10", "CIFAR-100"], index=0,
                          help="The dataset to use for federated learning. MNIST/Fashion-MNIST are simpler (28x28 grayscale), CIFAR is more complex (32x32 RGB).")
    partition = st.selectbox("Partition", ["iid", "dirichlet", "label-shard"], index=0,
                            help="How data is distributed across clients:\n• IID: Uniformly random (all clients have similar data)\n• Dirichlet: Non-IID controlled by alpha parameter (realistic heterogeneity)\n• Label-shard: Each client gets data from only a few classes")
    alpha = st.slider("Dirichlet alpha", 0.05, 2.0, 0.5, 0.05,
                     help="Controls data heterogeneity in Dirichlet partition. Lower values (0.1) = more non-IID (each client specializes), higher values (2.0) = closer to IID.")
    shards = st.number_input("Label shards per client", 1, 10, 2,
                            help="For label-shard partition: number of different classes each client has data from. Lower = more heterogeneous.")

    model = st.selectbox("Model", ["CNN-MNIST", "CNN-MNIST (FedAvg)", "LightCNN", "ResNet18"], index=0,
                        help="Neural network architecture:\n• CNN-MNIST: Lightweight for MNIST/Fashion-MNIST\n• CNN-MNIST (FedAvg): 2x(5x5) conv (32,64) + 512 FC for MNIST\n• LightCNN: For CIFAR datasets\n• ResNet18: Deeper model for more complex tasks")
    total_clients = st.number_input("Total clients", 2, 1000, 10,
                                   help="Total number of participating clients (devices) in the federated learning system.")
    k_clients = st.number_input("Clients per round (K)", 1, 100, 3,
                               help="Number of clients selected to participate in each training round. Smaller K = faster but less diverse, larger K = slower but more comprehensive.")
    rounds = st.number_input("Rounds", 1, 200, 3,
                            help="Number of federated learning rounds (communication cycles). Each round: clients train locally → server aggregates → repeat.")
    local_epochs = st.number_input("Local epochs", 1, 10, 1,
                                  help="Number of training epochs each selected client performs on their local data before sending updates to the server.")
    batch_size = st.number_input("Batch size", 8, 512, 32,
                                help="Number of samples per training batch on each client. Larger = faster but needs more memory.")
    lr = st.number_input("Learning rate", 1e-4, 1.0, 0.01, format="%.5f",
                        help="Step size for gradient descent optimization. Typical values: 0.001-0.01. Too high = unstable, too low = slow convergence.")

    device_choice = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0,
                                help="Hardware to run training on:\n• auto: Automatically detect GPU if available\n• cpu: Use CPU only\n• cuda: Force GPU (NVIDIA)")
    seed = st.number_input("Seed", 0, 10_000, 42,
                          help="Random seed for reproducibility. Same seed = same results. Useful for comparing different methods fairly.")
    fast_mode = st.checkbox("Fast mode (few batches)", True,
                           help="When enabled, uses fewer batches per epoch for faster testing. Disable for full training runs.")
    pretrained = st.checkbox("Load pretrained (if available)", False,
                            help="Load pre-trained model weights if available. Useful for transfer learning or continuing from a checkpoint.")

    with st.expander("Advanced (System & Privacy)"):
        time_budget = st.number_input("Round time budget (seconds, 0=none)", 0.0, 1000000.0, 0.0, format="%.2f",
                                     help="Maximum time allowed per round in seconds. Set to 0 for unlimited. Simulates real-world time constraints where some clients may not finish in time.")
        energy_budget = st.number_input("Round energy budget (a.u., 0=none)", 0.0, 1e9, 0.0, format="%.2f",
                                       help="Maximum energy allowed per round (abstract units). When set, energy-aware methods will prefer low-energy clients and respect this cap.")
        bytes_budget = st.number_input("Round bytes budget (a.u., 0=none)", 0.0, 1e12, 0.0, format="%.0f",
                                      help="Maximum uplink bytes allowed per round (abstract units). Useful to simulate bandwidth caps.")
        dp_sigma = st.number_input("DP Gaussian noise sigma (per-parameter)", 0.0, 10.0, 0.0, format="%.4f",
                                  help="Standard deviation of Gaussian noise added to model updates for Differential Privacy. Higher = more privacy but less accuracy. 0 = no DP noise.")
        dp_eps = st.number_input("DP epsilon consumed per selection", 0.0, 100.0, 0.0, format="%.3f",
                                help="Privacy budget (epsilon) consumed per client selection. Lower epsilon = stronger privacy guarantee. Used with DP-aware selection methods.")
        dp_clip = st.number_input("DP gradient clip norm (0 to disable)", 0.0, 10.0, 0.0, format="%.3f",
                                 help="Clips gradient norm before adding DP noise to bound sensitivity. Required for formal DP guarantees. 0 = no clipping. Typical values: 1.0-5.0.")
        
        st.caption("⚡ CUDA Parallelization")
        parallel_clients = st.selectbox(
            "Parallel clients",
            options=[0, -1, 2, 3, 4, 6, 8],
            index=0,
            help="Number of clients to train in parallel using CUDA streams:\n• 0: Sequential training (no parallelization)\n• -1: Auto-detect optimal based on GPU memory\n• 2-8: Fixed number of parallel clients\nParallel training can give 2-5x speedup on GPU with minimal memory overhead."
        )
        
        st.caption("Composite reward weights (optimization target)")
        colw1, colw2, colw3, colw4 = st.columns(4)
        w_acc = colw1.slider("w_acc", 0.0, 1.0, 0.6, 0.05,
                            help="Weight for accuracy in composite reward. Higher = prioritize model accuracy.")
        w_time = colw2.slider("w_time", 0.0, 1.0, 0.2, 0.05,
                             help="Weight for training time in composite reward. Higher = prioritize faster training rounds.")
        w_fair = colw3.slider("w_fair", 0.0, 1.0, 0.1, 0.05,
                             help="Weight for fairness in composite reward. Higher = ensure more equal participation across clients.")
        w_dp = colw4.slider("w_dp", 0.0, 1.0, 0.1, 0.05,
                           help="Weight for differential privacy in composite reward. Higher = prioritize privacy-preserving selections.")

    # Load methods dynamically
    from csfl_simulator.selection.registry import MethodRegistry
    reg = MethodRegistry(); reg.load_presets()
    labels_map = reg.labels_map()
    label_list = list(labels_map.keys())
    default_idx = 0
    method_label = st.selectbox("Selection method", label_list, index=default_idx,
                               help="Algorithm for selecting which K clients participate each round. Options include:\n• Random: Baseline random selection\n• Heuristic: Data size, loss, gradient-based\n• System-aware: Consider device capabilities, deadlines\n• ML-based: Neural network, RL, bandit approaches")
    method = reg.key_from_label(method_label)

    # Preselect multiple methods for later comparison runs
    default_compare_labels = [method_label] if method_label in label_list else (label_list[:1] if label_list else [])
    compare_labels = st.multiselect("Methods for comparison (preset)", label_list, default=default_compare_labels,
                                   help="Pre-select multiple methods to compare side-by-side. You can run them all together in the Compare tab.")
    st.session_state.compare_methods = [reg.key_from_label(lbl) for lbl in compare_labels]
    compare_repeats_val = int(st.session_state.get("compare_repeats", 1))
    compare_repeats = st.number_input("Repeats per method (comparison)", 1, 10, compare_repeats_val,
                                     help="Number of times to repeat each method with different random seeds. Higher repeats = more reliable results with error bars. Useful for statistical significance.")
    st.session_state.compare_repeats = int(compare_repeats)

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
        energy_budget=(float(energy_budget) if 'energy_budget' in locals() and energy_budget > 0 else None),
        bytes_budget=(float(bytes_budget) if 'bytes_budget' in locals() and bytes_budget > 0 else None),
        dp_sigma=float(dp_sigma) if 'dp_sigma' in locals() else 0.0,
        dp_epsilon_per_round=float(dp_eps) if 'dp_eps' in locals() else 0.0,
        dp_clip_norm=float(dp_clip) if 'dp_clip' in locals() else 0.0,
        reward_weights={"acc": float(w_acc) if 'w_acc' in locals() else 0.6,
                        "time": float(w_time) if 'w_time' in locals() else 0.2,
                        "fair": float(w_fair) if 'w_fair' in locals() else 0.1,
                        "dp": float(w_dp) if 'w_dp' in locals() else 0.1},
        parallel_clients=int(parallel_clients) if 'parallel_clients' in locals() else 0,
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
            st.session_state.run_data = res
            st.session_state.last_result = res
            # Autosave latest run snapshot
            try:
                _state_mod = _get_state_module()
                if _state_mod:
                    _state_mod.save_run(st.session_state.run_data, st.session_state.run_ui, None)
            except Exception:
                pass
            if res.get("stopped_early"):
                st.info("Run stopped early by user.")
        if st.session_state.last_result:
            res = st.session_state.last_result
            st.json({"run_id": res["run_id"], "device": res["device"], "config": res["config"]})
            st.write("Metrics (per round):")
            st.dataframe(res["metrics"]) 
            try:
                _state_mod = _get_state_module()
                if _state_mod:
                    st.caption(f"Snapshot schema v{_state_mod.SCHEMA_VERSION}. Plot UI is decoupled from data; toggling controls will not reset results.")
            except Exception:
                pass
            # Plots
            from csfl_simulator.app.components.plots import plot_accuracy, plot_participation, plot_selection_heatmap, plot_dp_usage, plot_round_time, plot_fairness, plot_composite, plot_loss, plot_wall_clock, plot_clients_per_hour, plot_energy, plot_bytes
            with st.expander("Plot Controls", expanded=True):
                ui = st.session_state.run_ui
                ui["show_accuracy"] = st.checkbox("Show Accuracy", value=bool(ui.get("show_accuracy", True)))
                ui["show_loss"] = st.checkbox("Show Loss", value=bool(ui.get("show_loss", True)))
                ui["show_time"] = st.checkbox("Show Round Time", value=bool(ui.get("show_time", True)))
                ui["show_wall_clock"] = st.checkbox("Show Wall-Clock", value=bool(ui.get("show_wall_clock", True)))
                ui["show_throughput"] = st.checkbox("Show Clients/hour", value=bool(ui.get("show_throughput", True)))
                ui["show_energy"] = st.checkbox("Show Energy", value=bool(ui.get("show_energy", True)))
                ui["show_bytes"] = st.checkbox("Show Bytes", value=bool(ui.get("show_bytes", True)))
                ui["show_fair"] = st.checkbox("Show Fairness", value=bool(ui.get("show_fair", True)))
                ui["show_composite"] = st.checkbox("Show Composite", value=bool(ui.get("show_composite", True)))
                col_reset, col_clear = st.columns([1,1])
                if col_reset.button("Reset Plot UI"):
                    st.session_state.run_ui = {"show_accuracy": True, "show_loss": True, "show_time": True, "show_wall_clock": True, "show_throughput": True, "show_energy": True, "show_bytes": True, "show_fair": True, "show_composite": True}
                    st.experimental_rerun()
                if col_clear.button("Clear Results"):
                    st.session_state.run_data = None
                    st.session_state.last_result = None
                    st.experimental_rerun()
            if st.session_state.run_ui.get("show_accuracy", True):
                st.plotly_chart(plot_accuracy(res["metrics"]), use_container_width=True, key="run_plot_accuracy")
            if st.session_state.run_ui.get("show_loss", True):
                st.plotly_chart(plot_loss(res["metrics"]), use_container_width=True, key="run_plot_loss")
            if st.session_state.run_ui.get("show_time", True):
                st.plotly_chart(plot_round_time(res["metrics"]), use_container_width=True, key="run_plot_round_time")
            if st.session_state.run_ui.get("show_wall_clock", True):
                st.plotly_chart(plot_wall_clock(res["metrics"]), use_container_width=True, key="run_plot_wall_clock")
            if st.session_state.run_ui.get("show_throughput", True):
                st.plotly_chart(plot_clients_per_hour(res["metrics"]), use_container_width=True, key="run_plot_throughput")
            if st.session_state.run_ui.get("show_energy", True):
                st.plotly_chart(plot_energy(res["metrics"]), use_container_width=True, key="run_plot_energy")
            if st.session_state.run_ui.get("show_bytes", True):
                st.plotly_chart(plot_bytes(res["metrics"]), use_container_width=True, key="run_plot_bytes")
            if st.session_state.run_ui.get("show_fair", True):
                st.plotly_chart(plot_fairness(res["metrics"]), use_container_width=True, key="run_plot_fairness")
            if st.session_state.run_ui.get("show_composite", True):
                st.plotly_chart(plot_composite(res["metrics"]), use_container_width=True, key="run_plot_composite")
            # Build a lightweight client snapshot for plotting
            # Note: in this session, we use the simulator's current clients
            sim = st.session_state.simulator
            st.plotly_chart(plot_participation(sim.clients), use_container_width=True, key="run_plot_participation")
            st.plotly_chart(plot_selection_heatmap(sim.history.get("selected", []), sim.cfg.total_clients), use_container_width=True, key="run_plot_selection_heatmap")
            st.plotly_chart(plot_dp_usage(sim.clients), use_container_width=True, key="run_plot_dp_usage")

            # Snapshot controls (Run)
            with st.expander("Snapshot (Run)"):
                snap_name = st.text_input("Snapshot name", value="")
                col_s1, col_s2 = st.columns([1,1])
                if col_s1.button("Save Snapshot"):
                    try:
                        _state_mod = _get_state_module()
                        if _state_mod:
                            _state_mod.save_run(st.session_state.run_data or st.session_state.last_result, st.session_state.run_ui, snap_name if snap_name else None)
                            st.success("Snapshot saved.")
                        else:
                            st.error("Failed to save snapshot: state module unavailable")
                    except Exception as e:
                        st.error(f"Failed to save snapshot: {e}")
                with col_s2:
                    try:
                        _state_mod = _get_state_module()
                        snaps = _state_mod.list_snapshots(kind='run') if _state_mod else []
                        pick = st.selectbox("Load snapshot", [str(p.name) for p in snaps], index=0 if snaps else None)
                        apply_ui = st.checkbox("Apply UI from snapshot (overwrite current)", value=False)
                        if st.button("Load Selected Snapshot") and snaps:
                            snap = next((p for p in snaps if p.name == pick), None)
                            if snap and _state_mod:
                                data, ui_loaded = _state_mod.load_run(snap)
                                st.session_state.run_data = data
                                st.session_state.last_result = data
                                if apply_ui and ui_loaded:
                                    st.session_state.run_ui = ui_loaded
                                st.success("Run snapshot loaded.")
                    except Exception as e:
                        st.error(f"Failed to load snapshot: {e}")

        # Comparison runner using preselected methods
        with st.expander("Run Selected Methods (Comparison)"):
            picks_keys = list(st.session_state.get("compare_methods", []) or [])
            if not picks_keys:
                st.info("Select methods in the Setup sidebar under 'Methods for comparison'.")
            else:
                # Style controls (same as Compare tab)
                style_col1, style_col2, style_col3 = st.columns([1,1,1])
                with style_col1:
                    chart_style2 = st.radio("Chart style", ["Interactive (Plotly)", "Paper (Matplotlib)"], index=0, key="cmp_style_run")
                with style_col2:
                    if chart_style2.startswith("Interactive"):
                        plotly_templates2 = ["plotly_white", "simple_white", "ggplot2", "seaborn", "presentation"]
                        template_choice2 = st.selectbox("Plotly template", plotly_templates2, index=0, key="cmp_tmpl_run")
                    else:
                        try:
                            import matplotlib.pyplot as _plt  # type: ignore
                            mpl_styles2 = list(getattr(_plt.style, 'available', ["classic", "default", "ggplot", "seaborn"]))
                        except Exception:
                            mpl_styles2 = ["classic", "default", "ggplot", "seaborn"]
                        style_choice2 = st.selectbox("Matplotlib style", mpl_styles2, index=0, key="cmp_mpl_run")
                with style_col3:
                    show_combined2 = st.checkbox("Show combined 2x2", value=True, key="cmp_combined_run")
                repeats2 = st.number_input("Repeats per method", 1, 10, int(st.session_state.get("compare_repeats", 1)), key="cmp_repeats_run")
                go2 = st.button("Run All Selected Methods")
                if go2:
                    # Build label map
                    label_map = labels_map
                    metric_names = ["accuracy", "f1", "precision", "recall", "loss"]
                    pretty = {"accuracy": "Accuracy", "f1": "F1", "precision": "Precision", "recall": "Recall", "loss": "Loss"}
                    metric_to_series = {pretty[m]: {} for m in metric_names}
                    selection_counts = {}
                    failures_run_tab: dict[str, list[str]] = {}

                    base_seed = int(st.session_state.simulator.cfg.seed)

                    # Progress UI
                    prog2 = st.progress(0)
                    status2 = st.empty()
                    prog2_rounds = st.progress(0)
                    status2_round = st.empty()
                    log2_box = st.empty()
                    total2 = max(1, len(picks_keys) * int(repeats2))
                    done2 = 0

                    def extract_series(rows, key):
                        ys = []
                        for row in rows:
                            try:
                                ys.append(float(row.get(key, 0.0) or 0.0))
                            except Exception:
                                ys.append(0.0)
                        return ys

                    for mkey in picks_keys:
                        per_metric_runs = {m: [] for m in metric_names}
                        for r in range(int(repeats2)):
                            cfg = SimConfig(**st.session_state.simulator.cfg.__dict__)
                            cfg.seed = base_seed + r
                            sim2 = FLSimulator(cfg)
                            # Reset round progress
                            try:
                                prog2_rounds.progress(0)
                            except Exception:
                                pass
                            total_rounds2 = max(1, int(cfg.rounds))
                            label2 = next((lbl for lbl, k in label_map.items() if k == mkey), mkey)
                            log2_lines = []
                            def on_prog_round2(rnd, info):
                                try:
                                    pct2r = int(((rnd + 1) / total_rounds2) * 100)
                                    prog2_rounds.progress(min(100, max(0, pct2r)))
                                except Exception:
                                    pass
                                acc = float(info.get("accuracy", 0.0) or 0.0)
                                reward = float(info.get("reward", 0.0) or 0.0)
                                comp = float(info.get("composite", 0.0) or 0.0)
                                chosen = info.get("selected", [])
                                try:
                                    status2_round.write(f"{label2} | repeat {r+1}/{int(repeats2)} | round {rnd+1}/{total_rounds2} | acc={acc:.4f} | comp={comp:.4f} | reward={reward:+.4f} | selected={chosen}")
                                except Exception:
                                    pass
                                try:
                                    log2_lines.append(f"[{label2} rep {r+1}] round {rnd+1}: acc={acc:.4f} comp={comp:.4f} reward={reward:+.4f} selected={chosen}")
                                    log2_box.code("\n".join(log2_lines[-200:]))
                                except Exception:
                                    pass
                            try:
                                res2 = sim2.run(mkey, on_progress=on_prog_round2)
                            except Exception as e:
                                label2 = next((lbl for lbl, k in label_map.items() if k == mkey), mkey)
                                err_msg = f"{type(e).__name__}: {e}\n" + traceback.format_exc()
                                failures_run_tab.setdefault(label2, []).append(err_msg)
                                done2 += 1
                                pct2 = int(done2 / total2 * 100)
                                status2.write(f"[{done2}/{total2}] {label2} — repeat {r+1} (FAILED)")
                                prog2.progress(min(100, pct2))
                                continue
                            for m in metric_names:
                                per_metric_runs[m].append(extract_series(res2["metrics"], m))
                            try:
                                counts = res2.get("participation_counts") or []
                                curr = selection_counts.get(label2)
                                if curr is None:
                                    selection_counts[label2] = counts
                                else:
                                    L = max(len(curr), len(counts))
                                    a = (curr + [0]*(L-len(curr))) if len(curr) < L else curr
                                    b = (counts + [0]*(L-len(counts))) if len(counts) < L else counts
                                    selection_counts[label2] = [int(a[i]) + int(b[i]) for i in range(L)]
                            except Exception:
                                pass
                            done2 += 1
                            pct2 = int(done2 / total2 * 100)
                            label2 = next((lbl for lbl, k in label_map.items() if k == mkey), mkey)
                            status2.write(f"[{done2}/{total2}] {label2} — repeat {r+1}")
                            prog2.progress(min(100, pct2))
                            
                            # Cleanup memory after each run to prevent accumulation
                            try:
                                sim2.cleanup()
                                cleanup_memory(force_cuda_empty=True, verbose=False)
                            except Exception:
                                pass
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
                            label = next((lbl for lbl, k in label_map.items() if k == mkey), mkey)
                            metric_to_series[pretty[m]][label] = mean

                    if chart_style2.startswith("Interactive"):
                        from csfl_simulator.app.components.plots import (
                            plot_metric_compare_plotly,
                            plot_multi_panel_plotly,
                            plot_selection_counts_compare_plotly,
                        )
                        for metric_display, series_map in metric_to_series.items():
                            fig = _call_plot_func(plot_metric_compare_plotly, series_map, metric_display, template=template_choice2)
                            st.plotly_chart(fig, use_container_width=True, key=f"cmp_run_plotly_{metric_display}")
                        if show_combined2:
                            figc = _call_plot_func(plot_multi_panel_plotly, metric_to_series, template=template_choice2)
                            st.plotly_chart(figc, use_container_width=True, key="cmp_run_plotly_combined")
                        if selection_counts:
                            figsc = _call_plot_func(plot_selection_counts_compare_plotly, selection_counts, template=template_choice2)
                            st.plotly_chart(figsc, use_container_width=True, key="cmp_run_plotly_counts")
                    else:
                        try:
                            from csfl_simulator.app.components.plots import (
                                plot_metric_compare_matplotlib,
                                plot_multi_panel_matplotlib,
                                plot_selection_counts_compare_matplotlib as _counts_mpl_run_tab,
                            )
                        except Exception:
                            from csfl_simulator.app.components.plots import (
                                plot_metric_compare_matplotlib,
                                plot_multi_panel_matplotlib,
                            )
                            _counts_mpl_run_tab = None
                        for metric_display, series_map in metric_to_series.items():
                            try:
                                fig = _call_plot_func(plot_metric_compare_matplotlib, series_map, metric_display, style_name=style_choice2)
                                st.pyplot(fig, clear_figure=True)
                            except Exception as e:
                                st.error(f"Matplotlib plotting failed for {metric_display}: {e}")
                        if show_combined2:
                            try:
                                figc = _call_plot_func(plot_multi_panel_matplotlib, metric_to_series, style_name=style_choice2)
                                st.pyplot(figc, clear_figure=True)
                            except Exception as e:
                                st.error(f"Matplotlib multi-panel plotting failed: {e}")
                        if selection_counts:
                            if _counts_mpl_run_tab is not None:
                                try:
                                    figsc = _call_plot_func(_counts_mpl_run_tab, selection_counts, style_name=style_choice2)
                                    st.pyplot(figsc, clear_figure=True)
                                except Exception as e:
                                    st.error(f"Matplotlib selection counts plotting failed: {e}")
                            else:
                                st.info("Selection counts (matplotlib) plot not available on this setup.")

                    # Show failures summary (Run tab expander)
                    if failures_run_tab:
                        st.warning("Some methods failed during comparison. See details below.")
                        for lbl, errs in failures_run_tab.items():
                            with st.expander(f"{lbl} — {len(errs)} failure(s)"):
                                for i, em in enumerate(errs, 1):
                                    st.code(em)

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
            chart_style = st.radio("Chart style", ["Interactive (Plotly)", "Paper (Matplotlib)"], index=0, key="cmp_style_main")
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
            show_combined = st.checkbox("Show combined 2x2", value=True, key="cmp_combined_main")
        go = st.button("Run Comparison")
        if go and picks:
            from collections import defaultdict
            base_seed = int(st.session_state.simulator.cfg.seed)

            # Progress UI
            prog = st.progress(0)
            status = st.empty()
            # Per-run (rounds) progress and live log
            prog_rounds = st.progress(0)
            status_round = st.empty()
            log_box = st.empty()
            total = max(1, len(picks) * int(repeats))
            done = 0

            def extract_series(rows, key):
                ys = []
                for row in rows:
                    try:
                        ys.append(float(row.get(key, 0.0) or 0.0))
                    except Exception:
                        ys.append(0.0)
                return ys

            metric_names = ["accuracy", "f1", "precision", "recall", "loss"]
            pretty = {"accuracy": "Accuracy", "f1": "F1", "precision": "Precision", "recall": "Recall", "loss": "Loss"}
            # metric -> {label -> mean_series}
            metric_to_series = {pretty[m]: {} for m in metric_names}
            selection_counts = {}
            failures_compare_tab: dict[str, list[str]] = {}

            for mkey in picks:
                per_metric_runs = {m: [] for m in metric_names}
                for r in range(int(repeats)):
                    cfg = SimConfig(**st.session_state.simulator.cfg.__dict__)
                    cfg.seed = base_seed + r
                    sim = FLSimulator(cfg)
                    # Reset per-run progress
                    try:
                        prog_rounds.progress(0)
                    except Exception:
                        pass
                    label = next((lbl for lbl, k in labels_map.items() if k == mkey), mkey)
                    total_rounds = max(1, int(cfg.rounds))
                    log_lines = []
                    def on_prog_round(rnd, info):
                        try:
                            pct_r = int(((rnd + 1) / total_rounds) * 100)
                            prog_rounds.progress(min(100, max(0, pct_r)))
                        except Exception:
                            pass
                        acc = float(info.get("accuracy", 0.0) or 0.0)
                        reward = float(info.get("reward", 0.0) or 0.0)
                        comp = float(info.get("composite", 0.0) or 0.0)
                        chosen = info.get("selected", [])
                        try:
                            status_round.write(f"{label} | repeat {r+1}/{int(repeats)} | round {rnd+1}/{total_rounds} | acc={acc:.4f} | comp={comp:.4f} | reward={reward:+.4f} | selected={chosen}")
                        except Exception:
                            pass
                        try:
                            log_lines.append(f"[{label} rep {r+1}] round {rnd+1}: acc={acc:.4f} comp={comp:.4f} reward={reward:+.4f} selected={chosen}")
                            log_box.code("\n".join(log_lines[-200:]))
                        except Exception:
                            pass
                    try:
                        res = sim.run(mkey, on_progress=on_prog_round)
                    except Exception as e:
                        label = next((lbl for lbl, k in labels_map.items() if k == mkey), mkey)
                        err_msg = f"{type(e).__name__}: {e}\n" + traceback.format_exc()
                        failures_compare_tab.setdefault(label, []).append(err_msg)
                        done += 1
                        pct = int(done / total * 100)
                        status.write(f"[{done}/{total}] {label} — repeat {r+1} (FAILED)")
                        prog.progress(min(100, pct))
                        continue
                    for m in metric_names:
                        per_metric_runs[m].append(extract_series(res["metrics"], m))
                    # accumulate selection counts
                    try:
                        counts = res.get("participation_counts") or []
                        curr = selection_counts.get(label)
                        if curr is None:
                            selection_counts[label] = counts
                        else:
                            L = max(len(curr), len(counts))
                            a = (curr + [0]*(L-len(curr))) if len(curr) < L else curr
                            b = (counts + [0]*(L-len(counts))) if len(counts) < L else counts
                            selection_counts[label] = [int(a[i]) + int(b[i]) for i in range(L)]
                    except Exception:
                        pass
                    done += 1
                    pct = int(done / total * 100)
                    status.write(f"[{done}/{total}] {label} — repeat {r+1}")
                    prog.progress(min(100, pct))
                    
                    # Cleanup memory after each run to prevent accumulation
                    try:
                        sim.cleanup()
                        cleanup_memory(force_cuda_empty=True, verbose=False)
                    except Exception:
                        pass
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

            # Store results for post-hoc rendering and autosave snapshot
            st.session_state.compare_data = {
                "metric_to_series": metric_to_series,
                "selection_counts": selection_counts,
                "labels_map": labels_map,
                "methods": list(metric_to_series.get("Accuracy", {}).keys()),
            }
            st.session_state.compare_results = st.session_state.compare_data
            try:
                _state_mod = _get_state_module()
                if _state_mod:
                    _state_mod.save_compare(st.session_state.compare_data, st.session_state.compare_ui, None)
            except Exception:
                pass

            if chart_style.startswith("Interactive"):
                from csfl_simulator.app.components.plots import (
                    plot_metric_compare_plotly,
                    plot_multi_panel_plotly,
                    plot_selection_counts_compare_plotly,
                )
                # Post-hoc controls
                methods_display = st.multiselect("Methods to display", st.session_state.compare_results["methods"], default=st.session_state.compare_results["methods"]) 
                smoothing = st.slider("Smoothing window (rounds)", 0, 20, int(st.session_state.compare_ui.get("smoothing", 0)))
                y_axis = st.radio("Y scale", ["linear", "log"], index=(0 if st.session_state.compare_ui.get("y_scale", "linear") == "linear" else 1), key="cmp_y_scale_plotly")
                lw = st.slider("Line width", 1, 6, int(st.session_state.compare_ui.get("line_width", 2)))
                legend_pos = st.selectbox("Legend position", ["right", "top", "inside"], index=( ["right","top","inside"].index(st.session_state.compare_ui.get("legend_position","right")) ))
                # Persist UI state
                st.session_state.compare_ui.update({
                    "chart_style": "Interactive (Plotly)",
                    "plotly_template": template_choice,
                    "methods_filter": methods_display,
                    "smoothing": int(smoothing),
                    "y_scale": y_axis,
                    "legend_position": legend_pos,
                    "line_width": float(lw),
                    "show_combined": bool(show_combined),
                })
                st.caption("Tip: Click legend items in Plotly to toggle series; use smoothing to reduce noise.")
                for metric_display, series_map in metric_to_series.items():
                    fig = _call_plot_func(
                        plot_metric_compare_plotly,
                        series_map,
                        metric_display,
                        template=template_choice,
                        methods_filter=methods_display,
                        smoothing_window=int(smoothing),
                        y_axis_type=y_axis,
                        line_width=float(lw),
                        legend_position=legend_pos,
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"cmp_plotly_{metric_display}")
                if show_combined:
                    figc = _call_plot_func(
                        plot_multi_panel_plotly,
                        metric_to_series,
                        template=template_choice,
                        methods_filter=methods_display,
                        smoothing_window=int(smoothing),
                        y_axis_type=y_axis,
                        line_width=float(lw),
                        legend_position=legend_pos,
                    )
                    st.plotly_chart(figc, use_container_width=True, key="cmp_plotly_combined")
                if selection_counts:
                    figsc = _call_plot_func(
                        plot_selection_counts_compare_plotly,
                        selection_counts,
                        template=template_choice,
                        methods_filter=methods_display,
                    )
                    st.plotly_chart(figsc, use_container_width=True, key="cmp_plotly_counts")
                # Reset/Clear controls
                col_rc1, col_rc2 = st.columns([1,1])
                if col_rc1.button("Reset Plot UI (Compare)"):
                    st.session_state.compare_ui = {
                        "chart_style": "Interactive (Plotly)",
                        "plotly_template": "plotly_white",
                        "mpl_style": "classic",
                        "methods_filter": st.session_state.compare_results.get("methods", []),
                        "metrics_filter": None,
                        "smoothing": 0,
                        "y_scale": "linear",
                        "legend_position": "right",
                        "legend_cols": 1,
                        "line_width": 2.0,
                        "show_combined": True,
                    }
                    st.experimental_rerun()
                if col_rc2.button("Clear Comparison Results"):
                    st.session_state.compare_data = None
                    st.session_state.compare_results = None
                    st.experimental_rerun()
            else:
                try:
                    from csfl_simulator.app.components.plots import (
                        plot_metric_compare_matplotlib,
                        plot_multi_panel_matplotlib,
                        plot_selection_counts_compare_matplotlib as _counts_mpl_cmp_tab,
                    )
                except Exception:
                    from csfl_simulator.app.components.plots import (
                        plot_metric_compare_matplotlib,
                        plot_multi_panel_matplotlib,
                    )
                    _counts_mpl_cmp_tab = None
                methods_display = st.multiselect("Methods to display", list(metric_to_series.get("Accuracy", {}).keys()), default=list(metric_to_series.get("Accuracy", {}).keys()))
                smoothing = st.slider("Smoothing window (rounds)", 0, 20, int(st.session_state.compare_ui.get("smoothing", 0)))
                y_axis = st.radio("Y scale", ["linear", "log"], index=(0 if st.session_state.compare_ui.get("y_scale", "linear") == "linear" else 1), key="cmp_y_scale_mpl")
                legend_pos = st.selectbox("Legend position", ["right", "top", "inside"], index=( ["right","top","inside"].index(st.session_state.compare_ui.get("legend_position","right")) ))
                lw = st.slider("Line width", 1, 6, int(st.session_state.compare_ui.get("line_width", 2)))
                st.session_state.compare_ui.update({
                    "chart_style": "Paper (Matplotlib)",
                    "mpl_style": style_choice,
                    "methods_filter": methods_display,
                    "smoothing": int(smoothing),
                    "y_scale": y_axis,
                    "legend_position": legend_pos,
                    "line_width": float(lw),
                    "show_combined": bool(show_combined),
                })
                st.caption("Paper-style: legend moved outside to avoid overlap; adjust line width and smoothing as needed.")
                for metric_display, series_map in metric_to_series.items():
                    try:
                        fig = _call_plot_func(
                            plot_metric_compare_matplotlib,
                            series_map,
                            metric_display,
                            style_name=style_choice,
                            methods_filter=methods_display,
                            legend_position=legend_pos,
                            smoothing_window=int(smoothing),
                            y_axis_type=y_axis,
                            line_width=float(lw),
                        )
                        st.pyplot(fig, clear_figure=True)
                    except Exception as e:
                        st.error(f"Matplotlib plotting failed for {metric_display}: {e}")
                if show_combined:
                    try:
                        figc = _call_plot_func(
                            plot_multi_panel_matplotlib,
                            metric_to_series,
                            style_name=style_choice,
                            methods_filter=methods_display,
                            legend_position=legend_pos,
                            smoothing_window=int(smoothing),
                            y_axis_type=y_axis,
                            line_width=float(lw),
                        )
                        st.pyplot(figc, clear_figure=True)
                    except Exception as e:
                        st.error(f"Matplotlib multi-panel plotting failed: {e}")
                if selection_counts:
                    if _counts_mpl_cmp_tab is not None:
                        try:
                            figsc = _call_plot_func(
                                _counts_mpl_cmp_tab,
                                selection_counts,
                                style_name=style_choice,
                                methods_filter=methods_display,
                                legend_position=legend_pos,
                            )
                            st.pyplot(figsc, clear_figure=True)
                        except Exception as e:
                            st.error(f"Matplotlib selection counts plotting failed: {e}")
                    else:
                        st.info("Selection counts (matplotlib) plot not available on this setup.")
                col_rc3, col_rc4 = st.columns([1,1])
                if col_rc3.button("Reset Plot UI (Compare)"):
                    st.session_state.compare_ui = {
                        "chart_style": "Interactive (Plotly)",
                        "plotly_template": "plotly_white",
                        "mpl_style": "classic",
                        "methods_filter": list(metric_to_series.get("Accuracy", {}).keys()),
                        "metrics_filter": None,
                        "smoothing": 0,
                        "y_scale": "linear",
                        "legend_position": "right",
                        "legend_cols": 1,
                        "line_width": 2.0,
                        "show_combined": True,
                    }
                    st.experimental_rerun()
                if col_rc4.button("Clear Comparison Results"):
                    st.session_state.compare_data = None
                    st.session_state.compare_results = None
                    st.experimental_rerun()

            # Failures summary (Compare tab)
            if failures_compare_tab:
                st.warning("Some methods failed during comparison. See details below.")
                for lbl, errs in failures_compare_tab.items():
                    with st.expander(f"{lbl} — {len(errs)} failure(s)"):
                        for i, em in enumerate(errs, 1):
                            st.code(em)

            # Manual snapshot controls (Comparison)
            with st.expander("Snapshot (Comparison)"):
                snap_name2 = st.text_input("Snapshot name", value="", key="cmp_snap_name")
                col_s3, col_s4 = st.columns([1,1])
                if col_s3.button("Save Snapshot", key="cmp_save_btn"):
                    try:
                        _state_mod = _get_state_module()
                        if _state_mod:
                            _state_mod.save_compare(st.session_state.compare_results, st.session_state.compare_ui, snap_name2 if snap_name2 else None)
                            st.success("Snapshot saved.")
                        else:
                            st.error("Failed to save snapshot: state module unavailable")
                    except Exception as e:
                        st.error(f"Failed to save snapshot: {e}")
                with col_s4:
                    try:
                        _state_mod = _get_state_module()
                        snaps2 = _state_mod.list_snapshots(kind='compare') if _state_mod else []
                        pick2 = st.selectbox("Load snapshot", [str(p.name) for p in snaps2], index=0 if snaps2 else None, key="cmp_snap_pick")
                        apply_ui2 = st.checkbox("Apply UI from snapshot (overwrite current)", value=False, key="cmp_apply_ui")
                        if st.button("Load Selected Snapshot", key="cmp_load_btn") and snaps2:
                            snap2 = next((p for p in snaps2 if p.name == pick2), None)
                            if snap2 and _state_mod:
                                data2, ui2 = _state_mod.load_compare(snap2)
                                st.session_state.compare_data = data2
                                st.session_state.compare_results = data2
                                if apply_ui2 and ui2:
                                    st.session_state.compare_ui = ui2
                                st.success("Comparison snapshot loaded. Rerun the controls to render.")
                    except Exception as e:
                        st.error(f"Failed to load snapshot: {e}")

        # Bridge: open current comparison in Visualize
        if st.session_state.compare_results:
            if st.button("Open in Visualize", key="cmp_open_viz"):
                st.session_state.visualize_data = {"kind": "compare", "data": st.session_state.compare_results}
                st.info("Sent current comparison to Visualize tab. Switch to 'Visualize' to customize.")

        # Render last comparison if available (no fresh compute required)
        elif st.session_state.compare_results:
            metric_to_series = st.session_state.compare_results.get("metric_to_series", {})
            selection_counts = st.session_state.compare_results.get("selection_counts", {})
            # Use the same rendering controls as above
            chart_style = st.radio("Chart style", ["Interactive (Plotly)", "Paper (Matplotlib)"], index=0, key="cmp_style_mem")
            if chart_style.startswith("Interactive"):
                from csfl_simulator.app.components.plots import (
                    plot_metric_compare_plotly,
                    plot_multi_panel_plotly,
                    plot_selection_counts_compare_plotly,
                )
                methods_display = st.multiselect("Methods to display", list(metric_to_series.get("Accuracy", {}).keys()), default=(st.session_state.compare_ui.get("methods_filter") or list(metric_to_series.get("Accuracy", {}).keys())))
                smoothing = st.slider("Smoothing window (rounds)", 0, 20, int(st.session_state.compare_ui.get("smoothing", 0)))
                y_axis = st.radio("Y scale", ["linear", "log"], index=(0 if st.session_state.compare_ui.get("y_scale", "linear") == "linear" else 1), key="cmp_y_scale_plotly_mem")
                lw = st.slider("Line width", 1, 6, int(st.session_state.compare_ui.get("line_width", 2)))
                legend_pos = st.selectbox("Legend position", ["right", "top", "inside"], index=( ["right","top","inside"].index(st.session_state.compare_ui.get("legend_position","right")) ))
                st.session_state.compare_ui.update({
                    "chart_style": "Interactive (Plotly)",
                    "plotly_template": "plotly_white",
                    "methods_filter": methods_display,
                    "smoothing": int(smoothing),
                    "y_scale": y_axis,
                    "legend_position": legend_pos,
                    "line_width": float(lw),
                })
                for metric_display, series_map in metric_to_series.items():
                    fig = _call_plot_func(
                        plot_metric_compare_plotly,
                        series_map,
                        metric_display,
                        template="plotly_white",
                        methods_filter=methods_display,
                        smoothing_window=int(smoothing),
                        y_axis_type=y_axis,
                        line_width=float(lw),
                        legend_position=legend_pos,
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"cmp_mem_plotly_{metric_display}")
                if st.checkbox("Show combined 2x2", value=True, key="cmp_combined_mem_plotly"):
                    figc = _call_plot_func(
                        plot_multi_panel_plotly,
                        metric_to_series,
                        template="plotly_white",
                        methods_filter=methods_display,
                        smoothing_window=int(smoothing),
                        y_axis_type=y_axis,
                        line_width=float(lw),
                        legend_position=legend_pos,
                    )
                    st.plotly_chart(figc, use_container_width=True, key="cmp_mem_plotly_combined")
                if selection_counts:
                    figsc = _call_plot_func(
                        plot_selection_counts_compare_plotly,
                        selection_counts,
                        template="plotly_white",
                        methods_filter=methods_display,
                    )
                    st.plotly_chart(figsc, use_container_width=True, key="cmp_mem_plotly_counts")
                col_rc5, col_rc6 = st.columns([1,1])
                if col_rc5.button("Reset Plot UI (Compare)"):
                    st.session_state.compare_ui = {
                        "chart_style": "Interactive (Plotly)",
                        "plotly_template": "plotly_white",
                        "mpl_style": "classic",
                        "methods_filter": list(metric_to_series.get("Accuracy", {}).keys()),
                        "metrics_filter": None,
                        "smoothing": 0,
                        "y_scale": "linear",
                        "legend_position": "right",
                        "legend_cols": 1,
                        "line_width": 2.0,
                        "show_combined": True,
                    }
                    st.experimental_rerun()
                if col_rc6.button("Clear Comparison Results"):
                    st.session_state.compare_data = None
                    st.session_state.compare_results = None
                    st.experimental_rerun()
            else:
                try:
                    from csfl_simulator.app.components.plots import (
                        plot_metric_compare_matplotlib,
                        plot_multi_panel_matplotlib,
                        plot_selection_counts_compare_matplotlib as _counts_mpl_cmp_mem,
                    )
                except Exception:
                    from csfl_simulator.app.components.plots import (
                        plot_metric_compare_matplotlib,
                        plot_multi_panel_matplotlib,
                    )
                    _counts_mpl_cmp_mem = None
                methods_display = st.multiselect("Methods to display", list(metric_to_series.get("Accuracy", {}).keys()), default=(st.session_state.compare_ui.get("methods_filter") or list(metric_to_series.get("Accuracy", {}).keys())))
                smoothing = st.slider("Smoothing window (rounds)", 0, 20, int(st.session_state.compare_ui.get("smoothing", 0)))
                y_axis = st.radio("Y scale", ["linear", "log"], index=(0 if st.session_state.compare_ui.get("y_scale", "linear") == "linear" else 1), key="cmp_y_scale_mpl_mem")
                legend_pos = st.selectbox("Legend position", ["right", "top", "inside"], index=( ["right","top","inside"].index(st.session_state.compare_ui.get("legend_position","right")) ))
                lw = st.slider("Line width", 1, 6, int(st.session_state.compare_ui.get("line_width", 2)))
                st.session_state.compare_ui.update({
                    "chart_style": "Paper (Matplotlib)",
                    "methods_filter": methods_display,
                    "smoothing": int(smoothing),
                    "y_scale": y_axis,
                    "legend_position": legend_pos,
                    "line_width": float(lw),
                })
                for metric_display, series_map in metric_to_series.items():
                    try:
                        fig = _call_plot_func(
                            plot_metric_compare_matplotlib,
                            series_map,
                            metric_display,
                            style_name="classic",
                            methods_filter=methods_display,
                            legend_position=legend_pos,
                            smoothing_window=int(smoothing),
                            y_axis_type=y_axis,
                            line_width=float(lw),
                        )
                        st.pyplot(fig, clear_figure=True)
                    except Exception as e:
                        st.error(f"Matplotlib plotting failed for {metric_display}: {e}")
                if st.checkbox("Show combined 2x2", value=True, key="cmp_combined_mem_mpl"):
                    try:
                        figc = _call_plot_func(
                            plot_multi_panel_matplotlib,
                            metric_to_series,
                            style_name="classic",
                            methods_filter=methods_display,
                            legend_position=legend_pos,
                            smoothing_window=int(smoothing),
                            y_axis_type=y_axis,
                            line_width=float(lw),
                        )
                        st.pyplot(figc, clear_figure=True)
                    except Exception as e:
                        st.error(f"Matplotlib multi-panel plotting failed: {e}")
                if selection_counts and _counts_mpl_cmp_mem is not None:
                    try:
                        figsc = _call_plot_func(
                            _counts_mpl_cmp_mem,
                            selection_counts,
                            style_name="classic",
                            methods_filter=methods_display,
                            legend_position=legend_pos,
                        )
                        st.pyplot(figsc, clear_figure=True)
                    except Exception as e:
                        st.error(f"Matplotlib selection counts plotting failed: {e}")
                col_rc7, col_rc8 = st.columns([1,1])
                if col_rc7.button("Reset Plot UI (Compare)"):
                    st.session_state.compare_ui = {
                        "chart_style": "Interactive (Plotly)",
                        "plotly_template": "plotly_white",
                        "mpl_style": "classic",
                        "methods_filter": list(metric_to_series.get("Accuracy", {}).keys()),
                        "metrics_filter": None,
                        "smoothing": 0,
                        "y_scale": "linear",
                        "legend_position": "right",
                        "legend_cols": 1,
                        "line_width": 2.0,
                        "show_combined": True,
                    }
                    st.experimental_rerun()
                if col_rc8.button("Clear Comparison Results"):
                    st.session_state.compare_data = None
                    st.session_state.compare_results = None
                    st.experimental_rerun()

with visualize_tab:
    st.subheader("Visualize Results")
    viz_ui = st.session_state.visualize_ui
    # Source selector
    src_idx = (0 if viz_ui.get("source", "compare").lower() == "compare" else (1 if viz_ui.get("source") == "run" else 2))
    src = st.radio("Source", ["Compare", "Run", "Snapshot"], index=src_idx, key="viz_src_radio")
    viz_ui["source"] = src.lower()
    viz_ui["lock_data"] = st.checkbox("Lock data source (prevent change)", value=bool(viz_ui.get("lock_data", False)), key="viz_lock")
    data_obj = None
    if viz_ui["source"] == "compare" and st.session_state.compare_results:
        data_obj = {"kind": "compare", "data": st.session_state.compare_results}
    elif viz_ui["source"] == "run" and st.session_state.run_data:
        rd = st.session_state.run_data
        series = {}
        for key, pretty in [("accuracy","Accuracy"),("f1","F1"),("precision","Precision"),("recall","Recall"),("loss","Loss")]:
            vals = []
            for row in rd.get("metrics", []):
                try:
                    vals.append(float(row.get(key, 0.0) or 0.0))
                except Exception:
                    vals.append(0.0)
            series.setdefault(pretty, {})["Run"] = vals
        data_obj = {"kind": "compare", "data": {"metric_to_series": series, "selection_counts": {}, "methods": ["Run"]}}
    elif viz_ui["source"] == "snapshot":
        try:
            _state_mod = _get_state_module()
            kind_choice = st.radio("Snapshot kind", ["compare", "run"], index=0, key="viz_snap_kind")
            snaps = _state_mod.list_snapshots(kind=kind_choice) if _state_mod else []
            pick = st.selectbox("Snapshot file", [str(p.name) for p in snaps], index=0 if snaps else None, key="viz_snap_pick")
            if snaps and st.button("Load Snapshot", key="viz_snap_load"):
                snap = next((p for p in snaps if p.name == pick), None)
                if snap and _state_mod:
                    if kind_choice == "compare":
                        d, _ = _state_mod.load_compare(snap)
                        st.session_state.visualize_data = {"kind": "compare", "data": d}
                    else:
                        d, _ = _state_mod.load_run(snap)
                        series = {}
                        for key, pretty in [("accuracy","Accuracy"),("f1","F1"),("precision","Precision"),("recall","Recall"),("loss","Loss")]:
                            vals = []
                            for row in d.get("metrics", []):
                                try:
                                    vals.append(float(row.get(key, 0.0) or 0.0))
                                except Exception:
                                    vals.append(0.0)
                            series.setdefault(pretty, {})["Run"] = vals
                        st.session_state.visualize_data = {"kind": "compare", "data": {"metric_to_series": series, "selection_counts": {}, "methods": ["Run"]}}
                    st.success("Snapshot loaded into Visualize.")
        except Exception as e:
            st.error(f"Snapshot error: {e}")
    # Effective data (prefer previously loaded)
    data_obj = st.session_state.visualize_data or data_obj
    if not data_obj:
        st.info("No data available. Run a comparison, select Run, or load a snapshot.")
    else:
        mts = data_obj["data"].get("metric_to_series", {})
        methods_all = list(next(iter(mts.values()), {}).keys())
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            viz_ui["chart_style"] = st.radio("Chart style", ["Interactive (Plotly)", "Paper (Matplotlib)"], index=(0 if viz_ui.get("chart_style","Interactive (Plotly)").startswith("Interactive") else 1), key="viz_style")
        with col2:
            if viz_ui["chart_style"].startswith("Interactive"):
                viz_ui["plotly_template"] = st.selectbox("Plotly template", ["plotly_white","simple_white","ggplot2","seaborn","presentation"], index=0, key="viz_tpl")
            else:
                try:
                    import matplotlib.pyplot as _plt  # type: ignore
                    mpl_styles = list(getattr(_plt.style, 'available', ["classic","default","ggplot","seaborn"]))
                except Exception:
                    mpl_styles = ["classic","default","ggplot","seaborn"]
                viz_ui["mpl_style"] = st.selectbox("Matplotlib style", mpl_styles, index=0, key="viz_mpl")
        with col3:
            viz_ui["show_combined"] = st.checkbox("Show combined 2x2", value=bool(viz_ui.get("show_combined", True)), key="viz_combined")
        # Compute safe default for methods multiselect
        methods_default = methods_all
        if "viz_methods" in st.session_state and st.session_state["viz_methods"]:
            # Keep user's previous selection if it's valid
            prev_selection = st.session_state["viz_methods"]
            methods_default = [m for m in prev_selection if m in methods_all]
            if not methods_default:
                methods_default = methods_all
        viz_ui["methods"] = st.multiselect("Methods", options=methods_all, default=methods_default, key="viz_methods")
        metrics_all = [m for m in ["Accuracy","F1","Precision","Recall","Loss"] if m in mts]
        # Compute safe default for metrics multiselect
        metrics_default = metrics_all
        if "viz_metrics" in st.session_state and st.session_state["viz_metrics"]:
            # Keep user's previous selection if it's valid
            prev_selection = st.session_state["viz_metrics"]
            metrics_default = [m for m in prev_selection if m in metrics_all]
            if not metrics_default:
                metrics_default = metrics_all
        viz_ui["metrics"] = st.multiselect("Metrics", options=metrics_all, default=metrics_default, key="viz_metrics")
        col4, col5, col6, col7 = st.columns([1,1,1,1])
        viz_ui["smoothing"] = int(col4.slider("Smoothing", 0, 20, int(viz_ui.get("smoothing", 0)), key="viz_smooth"))
        viz_ui["y_scale"] = col5.radio("Y scale", ["linear","log"], index=(0 if viz_ui.get("y_scale","linear") == "linear" else 1), key="viz_y")
        viz_ui["legend_position"] = col6.selectbox("Legend", ["right","top","inside"], index=( ["right","top","inside"].index(viz_ui.get("legend_position","right")) ), key="viz_legend")
        viz_ui["line_width"] = float(col7.slider("Line width", 1, 6, int(viz_ui.get("line_width", 2)), key="viz_lw"))
        # Rounds
        try:
            first_metric = next(iter(mts.keys()))
            first_method = next(iter(mts[first_metric].keys()))
            max_len = len(mts[first_metric][first_method])
        except Exception:
            max_len = 0
        colr1, colr2 = st.columns([1,1])
        # Ensure minimum range of 100 for better UX even when no data
        max_index = max(100, max_len - 1) if max_len > 0 else 100
        # Clamp any persisted widget state before rendering
        try:
            if isinstance(st.session_state.get("viz_r0"), (int, float)):
                st.session_state["viz_r0"] = int(min(max(0, int(st.session_state.get("viz_r0", 0))), max_index))
            if isinstance(st.session_state.get("viz_r1"), (int, float)):
                st.session_state["viz_r1"] = int(min(max(int(st.session_state.get("viz_r0", 0)), int(st.session_state.get("viz_r1", max_index))), max_index))
        except Exception:
            pass
        _start_default = viz_ui.get("round_start", 0)
        try:
            _start_default = int(_start_default)
        except Exception:
            _start_default = 0
        _start_default = min(max(0, _start_default), max_index)
        viz_ui["round_start"] = int(colr1.number_input("Round start", min_value=0, max_value=max_index, value=_start_default, step=1, key="viz_r0"))
        _end_default = viz_ui.get("round_end")
        if _end_default is None:
            _end_default = max_index if max_len > 0 else max_index
        else:
            try:
                _end_default = int(_end_default)
            except Exception:
                _end_default = max_index
        _end_default = min(max(_end_default, viz_ui["round_start"]), max_index)
        viz_ui["round_end"] = int(colr2.number_input("Round end", min_value=viz_ui["round_start"], max_value=max_index, value=_end_default, step=1, key="viz_r1"))
        # Build filtered data
        filtered_mts = {}
        for metric in viz_ui["metrics"]:
            mm = {}
            for mname, ys in (mts.get(metric, {}) or {}).items():
                if viz_ui["methods"] and mname not in viz_ui["methods"]:
                    continue
                start_i = int(viz_ui["round_start"]) if isinstance(viz_ui.get("round_start"), (int, float)) else 0
                end_i = int(viz_ui["round_end"]) if isinstance(viz_ui.get("round_end"), (int, float)) else start_i
                ys2 = []
                try:
                    if ys is None:
                        ys2 = []
                    else:
                        # Convert numpy array to list if needed
                        try:
                            import numpy as np
                            if isinstance(ys, np.ndarray):
                                ys = ys.tolist()
                        except Exception:
                            pass
                        
                        if max_len > 0 and len(ys) > 0:
                            # Clamp indices to actual data length
                            actual_len = len(ys)
                            safe_start = min(start_i, actual_len - 1) if actual_len > 0 else 0
                            safe_end = min(end_i, actual_len - 1) if actual_len > 0 else 0
                            ys2 = list(ys[safe_start:safe_end + 1])
                        else:
                            ys2 = list(ys) if ys else []
                except Exception as e:
                    try:
                        ys2 = list(ys or [])
                    except Exception:
                        ys2 = []
                mm[mname] = ys2
            filtered_mts[metric] = mm
        
        # Debug: check if we have data
        if not filtered_mts or all(not v for v in filtered_mts.values()):
            st.warning(f"⚠️ No data to display. Metrics selected: {viz_ui['metrics']}, Methods selected: {viz_ui['methods']}, Data available: {list(mts.keys())}")
        
        # Render only if we have data
        if filtered_mts and any(v for v in filtered_mts.values()):
            if viz_ui["chart_style"].startswith("Interactive"):
                from csfl_simulator.app.components.plots import plot_metric_compare_plotly, plot_multi_panel_plotly
                for metric, series_map in filtered_mts.items():
                    if not series_map:
                        continue
                    fig = _call_plot_func(
                        plot_metric_compare_plotly,
                        series_map,
                        metric,
                        template=viz_ui["plotly_template"],
                        methods_filter=viz_ui["methods"],
                        smoothing_window=viz_ui["smoothing"],
                        y_axis_type=viz_ui["y_scale"],
                        line_width=viz_ui["line_width"],
                        legend_position=viz_ui["legend_position"],
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"viz_plotly_{metric}")
                if viz_ui["show_combined"] and filtered_mts:
                    figc = _call_plot_func(
                        plot_multi_panel_plotly,
                        filtered_mts,
                        template=viz_ui["plotly_template"],
                        methods_filter=viz_ui["methods"],
                        smoothing_window=viz_ui["smoothing"],
                        y_axis_type=viz_ui["y_scale"],
                        line_width=viz_ui["line_width"],
                        legend_position=viz_ui["legend_position"],
                    )
                    st.plotly_chart(figc, use_container_width=True, key="viz_plotly_combined")
                # Export
                try:
                    from datetime import datetime as _dt
                    figx = _call_plot_func(
                        plot_multi_panel_plotly,
                        filtered_mts,
                        template=viz_ui["plotly_template"],
                        methods_filter=viz_ui["methods"],
                        smoothing_window=viz_ui["smoothing"],
                        y_axis_type=viz_ui["y_scale"],
                        line_width=viz_ui["line_width"],
                        legend_position=viz_ui["legend_position"],
                    )
                    html = figx.to_html(include_plotlyjs="cdn")
                    st.download_button("Download HTML", data=html, file_name=f"viz_{_dt.now().strftime('%Y%m%d_%H%M%S')}.html", mime="text/html", key="viz_dl_html")
                    try:
                        buf = figx.to_image(format="png")
                        st.download_button("Download PNG", data=buf, file_name=f"viz_{_dt.now().strftime('%Y%m%d_%H%M%S')}.png", mime="image/png", key="viz_dl_png")
                    except Exception:
                        st.caption("PNG export requires kaleido; install to enable.")
                except Exception:
                    pass
            else:
                from csfl_simulator.app.components.plots import plot_metric_compare_matplotlib, plot_multi_panel_matplotlib
                for metric in viz_ui["metrics"]:
                    fig = _call_plot_func(
                        plot_metric_compare_matplotlib,
                        filtered_mts.get(metric, {}),
                        metric,
                        style_name=viz_ui["mpl_style"],
                        methods_filter=viz_ui["methods"],
                        legend_position=viz_ui["legend_position"],
                        smoothing_window=viz_ui["smoothing"],
                        y_axis_type=viz_ui["y_scale"],
                        line_width=viz_ui["line_width"],
                    )
                    st.pyplot(fig, clear_figure=True)
                if viz_ui["show_combined"] and filtered_mts:
                    figc = _call_plot_func(
                        plot_multi_panel_matplotlib,
                        filtered_mts,
                        style_name=viz_ui["mpl_style"],
                        methods_filter=viz_ui["methods"],
                        legend_position=viz_ui["legend_position"],
                        smoothing_window=viz_ui["smoothing"],
                        y_axis_type=viz_ui["y_scale"],
                        line_width=viz_ui["line_width"],
                    )
                    st.pyplot(figc, clear_figure=True)

        # Presets & CSV export
        with st.expander("Presets & Export"):
            from pathlib import Path as _Path
            from csfl_simulator.core.utils import ROOT as _ROOT
            import json as _json
            import io as _io
            import csv as _csv
            preset_dir = (_ROOT / "artifacts" / "checkpoints" / "visualize_presets").resolve()
            preset_dir.mkdir(parents=True, exist_ok=True)
            colp1, colp2 = st.columns([1,1])
            name = colp1.text_input("Preset name", value="default", key="viz_preset_name")
            if colp1.button("Save UI Preset", key="viz_preset_save"):
                try:
                    (_Path(preset_dir) / f"{name}.json").write_text(_json.dumps(viz_ui, indent=2))
                    st.success("Preset saved.")
                except Exception as e:
                    st.error(f"Preset save failed: {e}")
            files = list(preset_dir.glob("*.json"))
            pickp = colp2.selectbox("Load preset", [f.name for f in files], index=0 if files else None, key="viz_preset_pick")
            if colp2.button("Load UI Preset", key="viz_preset_load") and files:
                try:
                    p = next((f for f in files if f.name == pickp), None)
                    if p:
                        st.session_state.visualize_ui = _json.loads(p.read_text())
                        st.success("Preset loaded. Adjust controls above if needed.")
                except Exception as e:
                    st.error(f"Preset load failed: {e}")
            # CSV export of filtered data
            if filtered_mts:
                try:
                    sio = _io.StringIO()
                    writer = _csv.writer(sio)
                    writer.writerow(["metric", "round", "method", "value"])
                    for metric, series_map in filtered_mts.items():
                        for method, ys in (series_map or {}).items():
                            for i, v in enumerate(ys):
                                writer.writerow([metric, viz_ui["round_start"] + i, method, float(v)])
                    st.download_button("Download CSV", data=sio.getvalue(), file_name="viz_series.csv", mime="text/csv", key="viz_dl_csv")
                except Exception as e:
                    st.error(f"CSV export failed: {e}")

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
