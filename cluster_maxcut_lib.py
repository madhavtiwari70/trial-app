"""
Cluster MaxCut — refactored as a config-driven library.

Same Divi logic as the original cluster_maxcut/main.py. The only change:
graph size, QAOA settings, and backend are read from a data file
(data/cluster_maxcut.yaml) instead of hardcoded constants.

To make a new demo variant: edit the YAML file. Never edit this file.
"""

from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from utils import generate_clustered_graph

from divi.qprog.problems import MaxCutProblem
from divi.qprog.problems._graph_partitioning_utils import GraphPartitioningConfig
from divi.qprog.optimizers import MonteCarloOptimizer
from divi.qprog import QAOA
from divi.backends import QoroService, QiskitSimulator, JobConfig


def _resolve_backend(cfg: dict):
    if cfg["backend"]["use_cloud"]:
        return QoroService(job_config=JobConfig(shots=cfg["backend"]["shots"]))
    return QiskitSimulator()


def draw_graph(G, node_to_cluster, title, seed=42):
    """Return a matplotlib Figure showing the graph colored by cluster."""
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = [node_to_cluster[u] for u in G.nodes()]
    n = G.number_of_nodes()
    pos = nx.spring_layout(G, seed=seed, k=1.2 / max(n, 1) ** 0.5)
    nx.draw_networkx(
        G, pos=pos, node_color=colors, node_size=80, with_labels=False,
        edge_color="#999999", width=0.8, ax=ax, cmap="tab10",
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    return fig


@dataclass
class MaxCutResult:
    graph_figure: object
    n_nodes: int
    n_clusters: int
    quantum_cut_size: int
    classical_cut_size: int
    ratio: float
    runtime_s: float


def run_from_config(cfg: dict, progress_callback=None) -> MaxCutResult:
    import time

    g_cfg = cfg["graph"]
    q_cfg = cfg["qaoa"]
    p_cfg = cfg["partitioning"]

    if progress_callback:
        progress_callback("Generating clustered graph...")

    G, node_to_cluster, clusters = generate_clustered_graph(
        n_qubits=g_cfg["n_qubits"],
        n_clusters=g_cfg["n_clusters"],
        inter_edges=g_cfg["inter_edges"],
        p_intra=g_cfg["p_intra"],
        seed=g_cfg["seed"],
        weight=1.0,
    )

    fig = draw_graph(
        G, node_to_cluster,
        title=f"{g_cfg['n_qubits']} nodes, {g_cfg['n_clusters']} clusters, {g_cfg['inter_edges']} inter-edges",
        seed=g_cfg["seed"],
    )

    if progress_callback:
        progress_callback("Computing classical baseline...")
    classical_cut_size, _ = nx.approximation.one_exchange(G, seed=1)

    optim = MonteCarloOptimizer(
        population_size=q_cfg["population_size"],
        n_best_sets=q_cfg["n_best_sets"],
    )
    partition_config = GraphPartitioningConfig(
        minimum_n_clusters=p_cfg["minimum_n_clusters"],
        partitioning_algorithm=p_cfg["algorithm"],
    )
    backend = _resolve_backend(cfg)

    if progress_callback:
        progress_callback(f"Running partitioned QAOA on {'QoroService' if cfg['backend']['use_cloud'] else 'local simulator'}...")

    t0 = time.time()
    qaoa_problem = QAOA(
        problem=MaxCutProblem(graph=G, config=partition_config),
        n_layers=q_cfg["n_layers"],
        optimizer=optim,
        max_iterations=q_cfg["max_iterations"],
        backend=backend,
        grouping_strategy="qwc",
    )
    qaoa_problem.run(blocking=True)
    runtime = time.time() - t0

    quantum_solution = qaoa_problem.solution
    cut_edges = 0
    for u, v in G.edges():
        if (u in quantum_solution) != (v in quantum_solution):
            cut_edges += 1

    return MaxCutResult(
        graph_figure=fig,
        n_nodes=g_cfg["n_qubits"],
        n_clusters=g_cfg["n_clusters"],
        quantum_cut_size=cut_edges,
        classical_cut_size=classical_cut_size,
        ratio=round(cut_edges / classical_cut_size, 2) if classical_cut_size else 0.0,
        runtime_s=runtime,
    )
