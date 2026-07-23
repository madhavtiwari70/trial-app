"""
Divi Demo Console

A single Streamlit app for running Divi demos end-to-end: pick a demo,
edit its data file, click Run, get real results back from real Divi execution.

Run locally with:  streamlit run app.py
"""

import sys
import os
import yaml
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "demos", "spin_dynamics"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "demos", "cluster_maxcut"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "demos", "molecular_ground_state"))

st.set_page_config(page_title="Divi Demo Console", layout="wide")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ─────────────────────────────────────────────────────────────────────
#  Demo registry — add a new demo by adding one entry here.
#  "status": "ready"   -> fully wired, runs real Divi execution
#  "status": "pending" -> not refactored yet, shown greyed out
# ─────────────────────────────────────────────────────────────────────

DEMOS = {
    "Spin Dynamics (TFIM)": {
        "category": "Time Evolution",
        "status": "ready",
        "data_file": "spin_dynamics.yaml",
    },
    "Cluster MaxCut": {
        "category": "QAOA",
        "status": "ready",
        "data_file": "cluster_maxcut.yaml",
    },
    "Molecular Ground State": {
        "category": "Custom VQE",
        "status": "ready",
        "data_file": "molecular_ground_state.yaml",
    },
    "Travelling Salesman": {"category": "QAOA · QUBO", "status": "pending"},
    "Portfolio Optimization": {"category": "QAOA · PCE", "status": "pending"},
    "Economic Load Dispatch": {"category": "Optimization · PCE-VQE", "status": "pending"},
    "Minimum Birkhoff Decomposition": {"category": "Optimization", "status": "pending"},
    "Quantum-Guided Cluster": {"category": "QAOA", "status": "pending"},
}

st.title("Divi Demo Console")
st.caption("Pick a demo, edit its data file, click Run. No notebook, no code changes.")

# ─────────────────────────────────────────────────────────────────────
#  Sidebar: demo picker
# ─────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.subheader("Demos")
    labels = list(DEMOS.keys())

    def fmt(label):
        d = DEMOS[label]
        tag = "✅" if d["status"] == "ready" else "🚧"
        return f"{tag} {label}"

    selected_label = st.radio(
        "Select a demo", labels, format_func=fmt, label_visibility="collapsed"
    )
    st.divider()
    st.caption("✅ ready to run · 🚧 not refactored yet")

demo = DEMOS[selected_label]
st.subheader(f"{selected_label}")
st.caption(demo["category"])

# ─────────────────────────────────────────────────────────────────────
#  Pending demo — nothing to run yet
# ─────────────────────────────────────────────────────────────────────

if demo["status"] == "pending":
    st.warning(
        f"**{selected_label}** hasn't been refactored into the data-driven "
        "framework yet. It still exists as the original script/notebook in "
        "`divi-demos`, but doesn't have a `data.yaml` + config-driven library "
        "wired into this app. Follow the same pattern as Spin Dynamics or "
        "Cluster MaxCut to add it."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────
#  Ready demo — load its data file and run
# ─────────────────────────────────────────────────────────────────────

data_path = os.path.join(DATA_DIR, demo["data_file"])
with open(data_path) as f:
    raw_yaml = f.read()

col_config, col_results = st.columns([1, 1.6])

with col_config:
    st.markdown("**1. Configure** — edit the data file, no code involved.")
    edited_yaml = st.text_area(demo["data_file"], raw_yaml, height=420, key=selected_label)
    run_clicked = st.button("Run demo", type="primary", use_container_width=True)

with col_results:
    st.markdown("**2. Output**")

    if not run_clicked:
        st.info("Edit the config and click **Run demo** to execute the real Divi program.")
    else:
        try:
            cfg = yaml.safe_load(edited_yaml)
        except yaml.YAMLError as e:
            st.error(f"Couldn't parse the data file: {e}")
            st.stop()

        status_box = st.empty()

        def on_progress(*args):
            # supports both progress signatures used across demos:
            #   (message)  or  (i, n, name)
            if len(args) == 1:
                status_box.info(args[0])
            else:
                i, n, name = args
                status_box.info(f"Running: {name} ({i + 1}/{n})")

        with st.spinner("Executing on backend…"):
            if selected_label == "Spin Dynamics (TFIM)":
                from spin_dynamics_lib import run_from_config
                results = run_from_config(cfg, progress_callback=on_progress)
                backend_label = "QoroService (cloud)" if cfg["backend"]["use_cloud"] else "Local simulator"
                st.success(f"Ran {len(results)} experiment(s) on **{backend_label}**")
                for r in results:
                    st.pyplot(r.figure)
                    m1, m2 = st.columns(2)
                    m1.metric("Exact Trotterization runtime", f"{r.runtime_exact_s:.2f} s")
                    m2.metric("QDrift runtime", f"{r.runtime_qdrift_s:.2f} s")
                    st.divider()

            elif selected_label == "Cluster MaxCut":
                from cluster_maxcut_lib import run_from_config
                result = run_from_config(cfg, progress_callback=on_progress)
                backend_label = "QoroService (cloud)" if cfg["backend"]["use_cloud"] else "Local simulator"
                st.success(f"Ran on **{backend_label}** in {result.runtime_s:.1f}s")
                st.pyplot(result.graph_figure)
                m1, m2, m3 = st.columns(3)
                m1.metric("Quantum cut size", result.quantum_cut_size)
                m2.metric("Classical cut size", result.classical_cut_size)
                m3.metric("Quantum / classical ratio", result.ratio)

            elif selected_label == "Molecular Ground State":
                from molecular_ground_state_lib import run_from_config
                result = run_from_config(cfg, progress_callback=on_progress)
                backend_label = "QoroService (cloud)" if cfg["backend"]["use_cloud"] else "Local simulator"
                st.success(f"Ran on **{backend_label}** · {result.n_iterations_run} iterations")
                if result.figure is not None:
                    st.pyplot(result.figure)
                m1, m2 = st.columns(2)
                m1.metric("Ground state energy", f"{result.ground_state_energy:.6f} Ha")
                m2.metric("Optimizer", result.method)

        status_box.empty()
