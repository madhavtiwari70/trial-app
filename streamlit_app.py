"""
Divi Demo Console — Molecular Ground State (VQE)

Edit the config file below and click Run to execute a real Divi VQE program.
No code changes needed to change the molecule, ansatz depth, or backend.

Run locally with:  uv run streamlit run streamlit_app.py
"""

import sys
import os
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "demos", "molecular_ground_state"))
from molecular_ground_state_lib import run_from_config

st.set_page_config(page_title="Divi Demo Console — Molecular Ground State", layout="wide")

st.title("Divi Demo Console")
st.subheader("Molecular Ground State (VQE)")
st.caption("Custom VQE · edit the config file, click Run, no code changes.")

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "data", "molecular_ground_state_config.py")
LIB_PATH = os.path.join(
    os.path.dirname(__file__), "demos", "molecular_ground_state", "molecular_ground_state_lib.py"
)

with open(CONFIG_PATH) as f:
    raw_config = f.read()
with open(LIB_PATH) as f:
    lib_code = f.read()

tab_demo, tab_code = st.tabs(["Run demo", "View Divi code"])

with tab_demo:
    col_config, col_results = st.columns([1, 1.6])

    with col_config:
        st.markdown("**1. Configure** — edit the config file, no other code involved.")
        edited_config = st.text_area(
            "data/molecular_ground_state_config.py", raw_config, height=420
        )
        run_clicked = st.button("Run demo", type="primary", use_container_width=True)

    with col_results:
        st.markdown("**2. Output**")

        if not run_clicked:
            st.info("Edit the config and click **Run demo** to execute the real Divi VQE program.")
        else:
            # The config file is a plain Python dict literal named `config`.
            # We execute it in an isolated namespace and pull that variable out —
            # this is the .py equivalent of yaml.safe_load() for a data file.
            namespace = {}
            try:
                exec(edited_config, namespace)
                cfg = namespace["config"]
            except Exception as e:
                st.error(f"Couldn't parse the config file: {e}")
                st.stop()

            status_box = st.empty()

            def on_progress(message):
                status_box.info(message)

            with st.spinner("Executing on backend…"):
                result = run_from_config(cfg, progress_callback=on_progress)

            status_box.empty()

            backend_label = "QoroService (cloud)" if cfg["backend"]["use_cloud"] else "Local simulator"
            st.success(f"Ran on **{backend_label}** · {result.n_iterations_run} iterations")

            if result.figure is not None:
                st.pyplot(result.figure)

            m1, m2 = st.columns(2)
            m1.metric("Ground state energy", f"{result.ground_state_energy:.6f} Ha")
            m2.metric("Optimizer", result.method)

with tab_code:
    st.markdown(
        "This is the actual Divi code that runs when you click **Run demo** — "
        "read-only here, since only the config file is meant to be edited."
    )
    st.code(lib_code, language="python", line_numbers=True)