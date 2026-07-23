import networkx as nx
from typing import Dict, List, Tuple
import random


def analyze_results(graph, quantum_solution, classical_cut_size, use_index=True):
    cut_edges = 0

    if use_index:
        quantum_solution = [i for i, j in enumerate(quantum_solution) if j == 1]

    for u, v in graph.edges():
        if (u in quantum_solution) != (v in quantum_solution):
            cut_edges += 1

    print(
        f"Quantum Cut Size to Classical Cut Size Ratio = {round(cut_edges / classical_cut_size, 2)}"
    )


def show_graph(G, node_to_cluster, n_qubits, n_clusters, inter_edges, seed=42):
    import matplotlib.pyplot as plt

    # Color nodes by cluster for a quick visual check
    colors = [node_to_cluster[u] for u in G.nodes()]
    pos = nx.spring_layout(G, seed=seed, k=1.2 / (n_qubits**0.5))  # layout hint
    nx.draw_networkx(
        G,
        pos=pos,
        node_color=colors,
        node_size=80,
        with_labels=False,
        edge_color="#999999",
        width=0.8,
    )
    plt.title(
        f"Clustered graph: {n_qubits} nodes, {n_clusters} clusters, {inter_edges} inter edges"
    )
    plt.tight_layout()
    plt.show()


def generate_clustered_graph(
    n_qubits: int,
    n_clusters: int,
    inter_edges: int,
    p_intra: float = 0.6,
    seed: int | None = None,
    weight: float | None = None,
) -> Tuple[nx.Graph, Dict[int, int], List[List[int]]]:
    """
    Create a graph with n_clusters densely connected communities and a fixed number
    of edges between clusters.

    Parameters
    ----------
    n_qubits : int
        Total number of nodes (a.k.a. "qubits").
    n_clusters : int
        Number of clusters/communities to create.
    inter_edges : int
        Total number of edges placed between different clusters (across all pairs).
    p_intra : float, default=0.6
        Probability for adding extra intra-cluster edges on top of a spanning tree
        (keeps clusters connected but tunably dense).
    seed : int | None, default=None
        Random seed for reproducibility.
    weight : float | None, default=None
        If provided, all edges get an attribute {'weight': weight}.

    Returns
    -------
    G : nx.Graph
        The resulting graph.
    node_to_cluster : Dict[int, int]
        Mapping node -> cluster_id (0..n_clusters-1), useful as ground truth labels.
    clusters : List[List[int]]
        List of node lists for each cluster.
    """
    if n_clusters <= 0:
        raise ValueError("n_clusters must be >= 1")
    if n_qubits < n_clusters:
        raise ValueError("n_qubits must be >= n_clusters")
    if inter_edges < 0:
        raise ValueError("inter_edges must be >= 0")
    if not (0.0 <= p_intra <= 1.0):
        raise ValueError("p_intra must be in [0, 1]")

    rng = random.Random(seed)

    # --- Distribute node counts as evenly as possible across clusters ---
    base = n_qubits // n_clusters
    rem = n_qubits % n_clusters
    sizes = [base + (1 if i < rem else 0) for i in range(n_clusters)]

    G = nx.Graph()
    clusters: List[List[int]] = []
    node_to_cluster: Dict[int, int] = {}

    next_node = 0
    for c, sz in enumerate(sizes):
        nodes = list(range(next_node, next_node + sz))
        next_node += sz
        clusters.append(nodes)
        for u in nodes:
            node_to_cluster[u] = c

        # Ensure cluster connectivity: start with a random spanning tree
        T = nx.Graph()
        T.add_nodes_from(range(sz))
        for i in range(1, sz):
            parent = rng.randint(0, i - 1)
            T.add_edge(parent, i)

        mapping = {i: nodes[i] for i in range(sz)}
        T = nx.relabel_nodes(T, mapping)

        # Add extra intra-cluster edges with probability p_intra
        # over all non-tree possible pairs inside the cluster.
        G_cluster = nx.Graph()
        G_cluster.add_nodes_from(nodes)
        G_cluster.add_edges_from(T.edges())

        # All possible intra-cluster pairs
        for i in range(sz):
            for j in range(i + 1, sz):
                u, v = nodes[i], nodes[j]
                if not G_cluster.has_edge(u, v) and rng.random() < p_intra:
                    G_cluster.add_edge(u, v)

        # Merge cluster into global graph
        G.add_nodes_from(G_cluster.nodes())
        G.add_edges_from(G_cluster.edges())

    # --- Add the requested number of inter-cluster edges ---
    # We try random pairs until we reach the target or exhaust attempts.
    # (Exhaustion only happens with very tiny graphs / huge inter_edges.)
    added = 0
    attempts = 0
    max_attempts = inter_edges * 50 + 10_000  # generous cap

    while added < inter_edges and attempts < max_attempts:
        attempts += 1
        # choose two *different* clusters
        c1, c2 = rng.sample(range(n_clusters), 2)
        u = rng.choice(clusters[c1])
        v = rng.choice(clusters[c2])
        if u == v or G.has_edge(u, v):
            continue
        G.add_edge(u, v)
        added += 1

    if added < inter_edges:
        raise RuntimeError(
            f"Could not place all inter-cluster edges: placed {added} of {inter_edges}."
        )

    # Optionally set a uniform weight on all edges
    if weight is not None:
        nx.set_edge_attributes(G, {e: weight for e in G.edges()}, "weight")

    return G, node_to_cluster, clusters
