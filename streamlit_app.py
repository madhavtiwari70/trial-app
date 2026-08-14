"""
Divi Demo Console — all current divi-demos, data-file driven.

Pick a demo, edit its data file, click Run, get real results from real
Divi execution. No code changes needed to change the demo's parameters.

Run locally with:  streamlit run streamlit_app.py
"""

import sys
import os
import yaml
import streamlit as st

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "data")

for demo_dir in [
    "spin_dynamics", "economic_load_dispatch", "quantum_guided_cluster",
    "travelling_salesman", "minimum_birkhoff_decomposition", "portfolio_optimization",
]:
    sys.path.insert(0, os.path.join(HERE, "demos", demo_dir))

st.set_page_config(page_title="Divi Demo Console", layout="wide")

DEMOS = {
    "Spin Dynamics (TFIM)": {
        "category": "Time Evolution",
        "data_file": "spin_dynamics.yaml",
        "module": "spin_dynamics_wrapper",
    },
    "Economic Load Dispatch": {
        "category": "Optimization · PCE-VQE",
        "data_file": "economic_load_dispatch.yaml",
        "module": "eld_wrapper",
    },
    "Quantum-Guided Cluster": {
        "category": "QAOA",
        "data_file": "quantum_guided_cluster.yaml",
        "module": "qgc_wrapper",
    },
    "Travelling Salesman": {
        "category": "QAOA · QUBO",
        "data_file": "travelling_salesman.yaml",
        "module": "tsp_wrapper",
    },
    "Minimum Birkhoff Decomposition": {
        "category": "Optimization",
        "data_file": "minimum_birkhoff_decomposition.yaml",
        "module": "birkhoff_wrapper",
    },
    "Portfolio Optimization": {
        "category": "QAOA",
        "data_file": "portfolio_optimization.yaml",
        "module": "portfolio_wrapper",
    },
}

st.title("Divi Demo Console")
st.caption("Pick a demo, edit its data file, click Run. No notebook, no code changes.")

with st.sidebar:
    st.subheader("Demos")
    selected_label = st.radio("Select a demo", list(DEMOS.keys()), label_visibility="collapsed")
    st.divider()
    st.caption(
        "Some demos here cover a focused part of the original notebook/script "
        "(noted in each data file's comments) rather than every scenario, "
        "to keep each config schema clear."
    )

demo = DEMOS[selected_label]
st.subheader(selected_label)
st.caption(demo["category"])

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
            if len(args) == 1:
                status_box.info(args[0])
            else:
                i, n, name = args
                status_box.info(f"Running: {name} ({i + 1}/{n})")

        import importlib
        module = importlib.import_module(demo["module"])

        with st.spinner("Executing on backend…"):
            result = module.run_from_config(cfg, progress_callback=on_progress)

        status_box.empty()
        st.success("Run complete")

        # ── Render per-demo, based on what each wrapper returns ──────
        if selected_label == "Spin Dynamics (TFIM)":
            for r in result:
                st.markdown(f"**{r['name']}**")
                st.pyplot(r["fig_dynamics"])
                st.pyplot(r["fig_error"])
                m1, m2 = st.columns(2)
                m1.metric("Exact runtime", f"{r['runtime_exact_s']:.2f} s")
                m2.metric("QDrift runtime", f"{r['runtime_qdrift_s']:.2f} s")
                st.divider()

        elif selected_label == "Economic Load Dispatch":
            m1, m2, m3 = st.columns(3)
            m1.metric("Variables", result["n_variables"])
            m2.metric("PCE qubits", result["n_qubits"])
            m3.metric("Runtime", f"{result['runtime_s']:.1f} s")
            st.markdown(f"**Classical optimum:** {result['classical_powers']} MW → ${result['classical_cost']:.1f}")
            if result["quantum_result"]:
                q = result["quantum_result"]
                st.markdown(
                    f"**Quantum (PCE-VQE) solution:** {q['powers']} MW → "
                    f"${q['cost']:.1f}  (seed probability {q['seed_probability']:.2%})"
                )
            else:
                st.warning("No valid quantum solution found — try increasing max_iterations.")

        elif selected_label == "Quantum-Guided Cluster":
            st.metric("Exact ground energy", result["E_ground"])
            for label, r in result["quantum_results"].items():
                st.markdown(
                    f"**{label}** — best energy: {r.best_energy:.2f}, "
                    f"best cut: {r.best_cut}, acceptance rate: {r.acceptance_rate:.1%}"
                )
            for p in result["plot_paths"]:
                st.image(p)

        elif selected_label == "Travelling Salesman":
            st.pyplot(result["figure"])
            m1, m2, m3 = st.columns(3)
            m1.metric("Classical distance", f"{result['classical_distance']:.4f}")
            m2.metric("Quantum distance", f"{result['quantum_distance']:.4f}")
            m3.metric("Ratio", f"{result['ratio']:.3f}" if result["ratio"] else "-")

        elif selected_label == "Minimum Birkhoff Decomposition":
            st.code(result["output_text"], language=None)

        elif selected_label == "Portfolio Optimization":
            m1, m2 = st.columns(2)
            m1.metric("Assets", result["n_assets"])
            m2.metric("Runtime", f"{result['runtime_s']:.1f} s")
            st.markdown(f"**Assets selected:** {result['assets_selected']}")
            st.code(result["evaluation_text"], language=None)
