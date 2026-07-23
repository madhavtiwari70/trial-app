"""
Molecular Ground State (VQE) — refactored as a config-driven library.

Same Divi logic as the snippet this was built from. The only change:
molecule geometry, ansatz depth, optimizer, and backend are read from a
data file (data/molecular_ground_state.yaml) instead of hardcoded values.

To make a new demo variant (different molecule, different bond length):
edit the YAML file. Never edit this file.
"""

from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qp

from divi.backends import QoroService, QiskitSimulator, JobConfig
from divi.qprog import VQE, HartreeFockAnsatz
from divi.qprog.optimizers import ScipyOptimizer, ScipyMethod


def _resolve_backend(cfg: dict):
    b = cfg["backend"]
    if b["use_cloud"]:
        return QoroService(job_config=JobConfig(shots=b["shots"]))  # reads QORO_API_KEY from env
    return QiskitSimulator(shots=b["shots"])


def _resolve_method(name: str) -> ScipyMethod:
    return ScipyMethod[name.upper().replace("-", "_")]


@dataclass
class MolecularResult:
    ground_state_energy: float
    n_iterations_run: int
    loss_history: list
    symbols: list
    n_layers: int
    method: str
    figure: object


def _plot_convergence(loss_history, symbols):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(loss_history) + 1), loss_history, "o-",
            color="#3b82f6", linewidth=2, markersize=6)
    ax.set_title(f"VQE convergence — {''.join(symbols)}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy (Hartree)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def run_from_config(cfg: dict, progress_callback=None) -> MolecularResult:
    m_cfg = cfg["molecule"]
    v_cfg = cfg["vqe"]

    if progress_callback:
        progress_callback("Building molecule...")

    molecule = qp.qchem.Molecule(
        symbols=m_cfg["symbols"],
        coordinates=np.array([tuple(c) for c in m_cfg["coordinates"]]),
    )

    backend = _resolve_backend(cfg)

    if progress_callback:
        backend_name = "QoroService" if cfg["backend"]["use_cloud"] else "local simulator"
        progress_callback(f"Running VQE on {backend_name}...")

    vqe = VQE(
        molecule=molecule,
        ansatz=HartreeFockAnsatz(),
        n_layers=v_cfg["n_layers"],
        backend=backend,
        optimizer=ScipyOptimizer(method=_resolve_method(v_cfg["optimizer_method"])),
        max_iterations=v_cfg["max_iterations"],
        seed=v_cfg["seed"],
    )
    vqe.run()

    loss_history = list(vqe.losses_history) if vqe.losses_history else []

    return MolecularResult(
        ground_state_energy=vqe.best_loss,
        n_iterations_run=len(loss_history),
        loss_history=loss_history,
        symbols=m_cfg["symbols"],
        n_layers=v_cfg["n_layers"],
        method=v_cfg["optimizer_method"],
        figure=_plot_convergence(loss_history, m_cfg["symbols"]) if loss_history else None,
    )
