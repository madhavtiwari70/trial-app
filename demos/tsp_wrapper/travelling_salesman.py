# =============================================================================
#  Travelling Salesman Problem via QUBO + QAOA
# =============================================================================
#
#  Solves the Travelling Salesman Problem (TSP) for small city instances
#  using the Divi quantum SDK.
#
#  The TSP is encoded as a QUBO (Quadratic Unconstrained Binary Optimisation)
#  and solved with the Quantum Approximate Optimisation Algorithm (QAOA).
#
#  Scroll to the bottom to see the main flow — it reads like plain English!
#
# =============================================================================

import itertools

import dimod
import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit.library import CXGate, RYGate, RZGate

from divi.backends import MaestroSimulator, QoroService, JobConfig
from divi.qprog import BeamSearchStrategy, QAOA, EarlyStopping
from divi.qprog.optimizers import MonteCarloOptimizer
from divi.qprog.problems import BinaryOptimizationProblem, CommunityDecomposer
from divi.qprog.workflows import PartitioningProgramEnsemble

# ─────────────────────────────────────────────────────────────────────
#  STEP 1 — Generate a random set of cities
# ─────────────────────────────────────────────────────────────────────


def generate_cities(n_cities: int, seed: int = 42) -> np.ndarray:
    """Generate random 2D city coordinates in the unit square.

    Args:
        n_cities: Number of cities.
        seed:     Random seed for reproducibility.

    Returns:
        Array of shape (n_cities, 2) with (x, y) coordinates.
    """
    rng = np.random.default_rng(seed)
    return rng.random((n_cities, 2))


def compute_distance_matrix(cities: np.ndarray) -> np.ndarray:
    """Compute the Euclidean distance matrix between all city pairs.

    Args:
        cities: Array of shape (n_cities, 2).

    Returns:
        Symmetric distance matrix of shape (n_cities, n_cities).
    """
    diff = cities[:, np.newaxis, :] - cities[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))


# ─────────────────────────────────────────────────────────────────────
#  STEP 2 — Build the QUBO for TSP
# ─────────────────────────────────────────────────────────────────────
#
#  Binary variables x_{i,t} = 1 if city i is visited at step t.
#  The QUBO encodes three things:
#    1. Route cost       — minimise the total travel distance
#    2. One-city-per-step — exactly one city at each time step
#    3. One-step-per-city — each city is visited exactly once
#


def _var(city: int, step: int) -> str:
    """Variable name for 'city visited at step'."""
    return f"x_{city}_{step}"


def build_tsp_qubo(
    dist_matrix: np.ndarray,
    penalty: float | None = None,
) -> tuple[dimod.BinaryQuadraticModel, list[str]]:
    """Build a Binary Quadratic Model encoding the TSP as a QUBO.

    Uses the standard one-hot encoding with n² binary variables:
      x_{i,t} = 1  ⟺  city i is at position t in the tour.

    Args:
        dist_matrix: Distance matrix of shape (n, n).
        penalty:     Constraint penalty weight λ. If None, auto-set
                     to 8× the total distance sum — large enough to
                     make constraint-violating solutions energetically
                     unfavourable for QAOA.

    Returns:
        bqm:       The dimod BinaryQuadraticModel.
        var_names: Ordered list of variable names.
    """
    n = len(dist_matrix)
    if penalty is None:
        penalty = 8.0 * np.sum(dist_matrix) / n

    bqm = dimod.BinaryQuadraticModel(vartype="BINARY")
    var_names = [_var(i, t) for i in range(n) for t in range(n)]

    # ── 1. Route cost: sum of d(i,j) · x_{i,t} · x_{j,t+1} ──
    for t in range(n):
        t_next = (t + 1) % n  # cyclic tour
        for i in range(n):
            for j in range(n):
                if i != j:
                    bqm.add_quadratic(
                        _var(i, t),
                        _var(j, t_next),
                        dist_matrix[i, j],
                    )

    # ── 2. One-city-per-step: λ · (∑_i x_{i,t} − 1)² for each t ──
    for t in range(n):
        # Expand (∑ x_{i,t} - 1)²  =  ∑ x² - 2∑ x + 1 + 2∑_{i<j} x_i x_j
        # Since x² = x for binary:     = -∑ x + 1 + 2∑_{i<j} x_i x_j
        bqm.offset += penalty
        for i in range(n):
            bqm.add_linear(_var(i, t), -penalty)
        for i in range(n):
            for j in range(i + 1, n):
                bqm.add_quadratic(_var(i, t), _var(j, t), 2 * penalty)

    # ── 3. One-step-per-city: λ · (∑_t x_{i,t} − 1)² for each i ──
    for i in range(n):
        bqm.offset += penalty
        for t in range(n):
            bqm.add_linear(_var(i, t), -penalty)
        for t in range(n):
            for s in range(t + 1, n):
                bqm.add_quadratic(_var(i, t), _var(i, s), 2 * penalty)

    return bqm, var_names


# ─────────────────────────────────────────────────────────────────────
#  STEP 3 — Decode a bitstring back to a tour
# ─────────────────────────────────────────────────────────────────────


def decode_tour(sample: dict[str, int], n_cities: int) -> list[int] | None:
    """Decode a binary assignment into a tour.

    Args:
        sample:   Dict mapping variable names to 0/1 values.
        n_cities: Number of cities.

    Returns:
        A list of city indices in visit order, or None if the assignment
        is infeasible (constraint violations).
    """
    tour = []
    for t in range(n_cities):
        cities_at_t = [i for i in range(n_cities) if sample.get(_var(i, t), 0) == 1]
        if len(cities_at_t) != 1:
            return None  # constraint violated
        tour.append(cities_at_t[0])

    if len(set(tour)) != n_cities:
        return None  # city visited twice
    return tour


def assignment_diagnostics(sample: dict[str, int], n_cities: int) -> dict:
    """Describe one-hot assignment violations before applying tour repair."""
    time_counts = [
        sum(sample.get(_var(city, step), 0) for city in range(n_cities))
        for step in range(n_cities)
    ]
    city_counts = [
        sum(sample.get(_var(city, step), 0) for step in range(n_cities))
        for city in range(n_cities)
    ]
    return {
        "time_counts": time_counts,
        "city_counts": city_counts,
        "invalid_time_slots": [i for i, count in enumerate(time_counts) if count != 1],
        "invalid_city_assignments": [
            i for i, count in enumerate(city_counts) if count != 1
        ],
    }


def repair_tour(sample: dict[str, int], n_cities: int) -> list[int]:
    """Greedily repair an infeasible bitstring into a valid tour.

    For each time slot, pick the city with the highest activation that
    hasn't already been assigned. This rescues near-feasible quantum
    samples that have minor constraint violations.

    Args:
        sample:   Dict mapping variable names to 0/1 values.
        n_cities: Number of cities.

    Returns:
        A valid tour (list of city indices in visit order).
    """
    # Build a score matrix: higher = QAOA prefers this assignment
    scores = np.zeros((n_cities, n_cities))  # scores[t][i]
    for t in range(n_cities):
        for i in range(n_cities):
            scores[t][i] = sample.get(_var(i, t), 0)

    tour = []
    used_cities = set()
    for t in range(n_cities):
        # Sort cities by descending score, pick first unused
        order = np.argsort(-scores[t])
        for i in order:
            if i not in used_cities:
                tour.append(int(i))
                used_cities.add(int(i))
                break

    # Safety: if still missing cities (shouldn't happen), fill remaining
    remaining = set(range(n_cities)) - used_cities
    tour.extend(sorted(remaining))

    return tour


def select_best_tour(
    samples, dist_matrix: np.ndarray
) -> tuple[tuple[list[int], float] | None, int]:
    """Decode candidate assignments, repairing invalid tours when needed."""
    n_cities = len(dist_matrix)
    best = None
    n_feasible = 0
    repair_reported = False
    for sample in samples:
        tour = decode_tour(sample, n_cities)
        if tour is not None:
            n_feasible += 1
        else:
            if not repair_reported:
                diagnostics = assignment_diagnostics(sample, n_cities)
                print(
                    "   Repair diagnostics: invalid time slots "
                    f"{diagnostics['invalid_time_slots'] or 'none'}, invalid city "
                    f"assignments {diagnostics['invalid_city_assignments'] or 'none'}"
                )
                repair_reported = True
            tour = repair_tour(sample, n_cities)
        distance = tour_length(tour, dist_matrix)
        if best is None or distance < best[1]:
            best = (tour, distance)
    return best, n_feasible


def tour_length(tour: list[int], dist_matrix: np.ndarray) -> float:
    """Total distance of a cyclic tour.

    Args:
        tour:        Ordered list of city indices.
        dist_matrix: Distance matrix.

    Returns:
        Total tour distance (sum of edges including return to start).
    """
    d = sum(dist_matrix[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour)))
    return d


# ─────────────────────────────────────────────────────────────────────
#  STEP 4 — Classical brute-force solver (for comparison)
# ─────────────────────────────────────────────────────────────────────


def classical_brute_force(dist_matrix: np.ndarray) -> tuple[list[int], float]:
    """Find the shortest tour by exhaustive enumeration.

    Fixes city 0 as the tour start (symmetry reduction) and checks
    all (n−1)! permutations.

    Args:
        dist_matrix: Distance matrix of shape (n, n).

    Returns:
        (best_tour, best_distance)
    """
    n = len(dist_matrix)
    best_tour, best_dist = None, float("inf")

    for perm in itertools.permutations(range(1, n)):
        candidate = [0] + list(perm)
        d = tour_length(candidate, dist_matrix)
        if d < best_dist:
            best_dist = d
            best_tour = candidate

    return best_tour, best_dist


# ─────────────────────────────────────────────────────────────────────
#  STEP 5 — Solve with QAOA via Divi
# ─────────────────────────────────────────────────────────────────────


def solve_with_qaoa(
    bqm: dimod.BinaryQuadraticModel,
    n_layers: int = 2,
    max_iterations: int = 15,
    population_size: int = 50,
    n_best_sets: int = 5,
    shots: int = 10_000,
    backend=None,
) -> QAOA:
    """Run QAOA on the TSP QUBO.

    Args:
        bqm:              The Binary Quadratic Model.
        n_layers:         Number of QAOA layers (circuit depth).
        max_iterations:   Optimizer iterations.
        population_size:  Monte Carlo population size.
        n_best_sets:      Number of elite parameter sets to keep.
        shots:            Measurement samples per circuit evaluation.
        backend:          Divi backend (defaults to MaestroSimulator).

    Returns:
        The solved QAOA instance.
    """
    if backend is None:
        backend = MaestroSimulator(shots=shots)

    optimizer = MonteCarloOptimizer(
        population_size=population_size,
        n_best_sets=n_best_sets,
    )

    qaoa = QAOA(
        problem=BinaryOptimizationProblem(bqm),
        n_layers=n_layers,
        max_iterations=max_iterations,
        optimizer=optimizer,
        backend=backend,
    )

    qaoa.run()
    return qaoa


# ─────────────────────────────────────────────────────────────────────
#  STEP 6 — Extract the best feasible tour from QAOA results
# ─────────────────────────────────────────────────────────────────────


def extract_best_tour(
    qaoa: QAOA,
    bqm: dimod.BinaryQuadraticModel,
    dist_matrix: np.ndarray,
    top_n: int = 100,
) -> tuple[list[int], float] | None:
    """Scan the top QAOA solutions and return the best feasible tour.

    First tries exact decoding (perfectly valid bitstrings), then falls
    back to greedy repair for near-feasible bitstrings.

    Args:
        qaoa:        Solved QAOA instance.
        bqm:         The BQM (for variable ordering).
        dist_matrix: Distance matrix.
        top_n:       How many top bitstrings to inspect.

    Returns:
        (tour, distance) or None if no feasible tour was found.
    """
    var_list = list(bqm.variables)

    top_solutions = qaoa.get_top_solutions(n=top_n)

    print(f"\n   Inspecting top {len(top_solutions)} QAOA bitstrings...")
    samples = [
        {var_list[k]: int(sol.bitstring[k]) for k in range(len(var_list))}
        for sol in top_solutions
    ]
    best, n_feasible = select_best_tour(samples, dist_matrix)

    if n_feasible > 0:
        print(
            f"   Found {n_feasible} exactly feasible bitstrings out of {len(top_solutions)}"
        )
    else:
        print(
            f"   No exactly feasible bitstrings; used greedy repair on best candidates"
        )

    return best


# ─────────────────────────────────────────────────────────────────────
#  STEP 7 — Visualisation
# ─────────────────────────────────────────────────────────────────────


def plot_cities(cities: np.ndarray, title: str = "City Locations"):
    """Plot city locations as a scatter plot.

    Args:
        cities: Array of shape (n, 2).
        title:  Plot title.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        cities[:, 0],
        cities[:, 1],
        s=120,
        c="#4FC3F7",
        edgecolors="#0D47A1",
        zorder=5,
        linewidths=1.5,
    )
    for i, (x, y) in enumerate(cities):
        ax.annotate(
            str(i), (x, y), fontsize=10, ha="center", va="center", fontweight="bold"
        )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


def plot_tour(
    cities: np.ndarray,
    tour: list[int],
    distance: float,
    title: str = "Tour",
    color: str = "#4FC3F7",
    save_path: str | None = None,
):
    """Plot a tour on the city map.

    Draws the cyclic route connecting cities in tour order with arrows
    showing direction.

    Args:
        cities:    Array of shape (n, 2).
        tour:      Ordered list of city indices.
        distance:  Total tour distance (shown in title).
        title:     Plot title prefix.
        color:     Line/arrow colour.
        save_path: If given, save the figure to this path.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw edges with arrows
    for k in range(len(tour)):
        i, j = tour[k], tour[(k + 1) % len(tour)]
        ax.annotate(
            "",
            xy=cities[j],
            xytext=cities[i],
            arrowprops=dict(arrowstyle="->", color=color, lw=2),
        )

    # Draw city nodes
    ax.scatter(
        cities[:, 0],
        cities[:, 1],
        s=180,
        c="#E3F2FD",
        edgecolors="#0D47A1",
        zorder=5,
        linewidths=2,
    )
    for i, (x, y) in enumerate(cities):
        ax.annotate(
            str(i),
            (x, y),
            fontsize=11,
            ha="center",
            va="center",
            fontweight="bold",
            zorder=6,
        )

    ax.set_title(f"{title}  (distance = {distance:.4f})", fontsize=13)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"   Plot saved to {save_path}")
    plt.show()


def plot_comparison(
    cities: np.ndarray,
    classical_tour: list[int],
    classical_dist: float,
    quantum_tour: list[int],
    quantum_dist: float,
    save_path: str | None = None,
):
    """Side-by-side comparison of classical and quantum tours.

    Args:
        cities:         Array of shape (n, 2).
        classical_tour: Best classical tour.
        classical_dist: Classical tour distance.
        quantum_tour:   Best quantum tour.
        quantum_dist:   Quantum tour distance.
        save_path:      If given, save the figure to this path.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    for ax, t, d, label, col in [
        (ax1, classical_tour, classical_dist, "Classical (Brute Force)", "#FF7043"),
        (ax2, quantum_tour, quantum_dist, "Quantum (QAOA)", "#4FC3F7"),
    ]:
        for k in range(len(t)):
            i, j = t[k], t[(k + 1) % len(t)]
            ax.annotate(
                "",
                xy=cities[j],
                xytext=cities[i],
                arrowprops=dict(arrowstyle="->", color=col, lw=2),
            )
        ax.scatter(
            cities[:, 0],
            cities[:, 1],
            s=180,
            c="#E3F2FD",
            edgecolors="#0D47A1",
            zorder=5,
            linewidths=2,
        )
        for i, (x, y) in enumerate(cities):
            ax.annotate(
                str(i),
                (x, y),
                fontsize=11,
                ha="center",
                va="center",
                fontweight="bold",
                zorder=6,
            )
        ax.set_title(f"{label}\ndistance = {d:.4f}", fontsize=12)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

    plt.suptitle(
        "Travelling Salesman — Classical vs. Quantum",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"   Comparison plot saved to {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────
#  STEP 8 — Compare results
# ─────────────────────────────────────────────────────────────────────


def print_comparison(
    classical_tour: list[int],
    classical_dist: float,
    quantum_tour: list[int],
    quantum_dist: float,
):
    """Print a one-line comparison of classical vs quantum solutions."""
    ratio = quantum_dist / classical_dist
    print(
        f"  classical {classical_tour} d={classical_dist:.4f}  |  "
        f"quantum {quantum_tour} d={quantum_dist:.4f}  |  ratio={ratio:.3f}"
    )


# ─────────────────────────────────────────────────────────────────────
#  STEP 9 — Partitioned QUBO solving (Divi's unique capability)
# ─────────────────────────────────────────────────────────────────────
#
#  For larger TSP instances the QUBO has n² variables — too many for a
#  single QAOA run.  Divi's PartitioningProgramEnsemble (driven by a
#  community-based decomposer) automatically:
#    1. Decomposes the QUBO into smaller sub-problems
#    2. Solves each sub-problem with QAOA in parallel
#    3. Reassembles a global solution via beam-search aggregation
#


def solve_partitioned_tsp(
    bqm: dimod.BinaryQuadraticModel,
    dist_matrix: np.ndarray,
    decomposer_size: int = 15,
    n_layers: int = 2,
    max_iterations: int = 10,
    population_size: int = 30,
    n_best_sets: int = 5,
    shots: int = 10_000,
    backend=None,
    patience: int = 3,
) -> tuple[list[int], float]:
    """Solve a large TSP via ``PartitioningProgramEnsemble``.

    The QUBO is decomposed into sub-problems of at most ``decomposer_size``
    variables using Divi's :class:`CommunityDecomposer`, each
    sub-QAOA runs in parallel, and the global tour is reassembled via beam
    search.

    .. note::
        TSP's one-hot constraints span the entire QUBO, so partitioned QAOA
        is fundamentally a stress test of the partitioning machinery here —
        cuts that bisect a city or a time slot's constraint produce
        infeasible sub-problems. Solution quality on TSP via this route is
        expected to be poor; the value of the demo is showing the wiring on
        a hard problem.

    Args:
        bqm: Binary Quadratic Model for the TSP.
        dist_matrix: Distance matrix (n_cities × n_cities).
        decomposer_size: Max variables per sub-problem.
        n_layers: QAOA circuit depth per sub-problem.
        max_iterations: Optimizer iterations per sub-problem.
        population_size: MonteCarloOptimizer population.
        n_best_sets: Top parameter sets retained per iteration.
        shots: Measurement shots.
        backend: Divi backend (defaults to MaestroSimulator).
        patience: EarlyStopping patience per sub-problem.

    Returns:
        ``(tour, distance)`` — tour is decoded (or repaired) from the
        aggregated bitstring.
    """
    if backend is None:
        backend = MaestroSimulator(shots=shots)

    n_cities = len(dist_matrix)
    print(
        f"  QUBO: {len(bqm.variables)} variables → decomposing into "
        f"sub-problems of ≤{decomposer_size}"
    )

    problem = BinaryOptimizationProblem(
        bqm,
        decomposer=CommunityDecomposer(
            max_cluster_size=decomposer_size,
            method="modularity",
            seed=42,
        ),
    )
    ensemble = PartitioningProgramEnsemble(
        problem=problem,
        quantum_routine="qaoa",
        n_layers=n_layers,
        optimizer=MonteCarloOptimizer(
            population_size=population_size,
            n_best_sets=n_best_sets,
        ),
        max_iterations=max_iterations,
        early_stopping=EarlyStopping(patience=patience),
        backend=backend,
    )
    ensemble.create_programs()
    print(f"  Created {len(ensemble.programs)} sub-programs")
    ensemble.run().join()

    solution_array, _ = ensemble.aggregate_results(
        BeamSearchStrategy(beam_width=3, n_partition_candidates=5)
    )

    # The aggregated solution is in BQM variable order. Re-key by name and
    # decode (or repair) into a TSP tour.
    var_list = list(bqm.variables)
    sample = {var_list[i]: int(solution_array[i]) for i in range(len(var_list))}
    tour = decode_tour(sample, n_cities) or repair_tour(sample, n_cities)

    return tour, tour_length(tour, dist_matrix)


# ─────────────────────────────────────────────────────────────────────
#  STEP 10 — Solve with PCE (Pauli Correlation Encoding)
# ─────────────────────────────────────────────────────────────────────
#
#  PCE compresses n² QUBO variables into far fewer qubits using
#  polynomial encoding.  For example, 16 variables → ~6 qubits.
#  This is a fundamentally different approach from QAOA: instead of
#  encoding the QUBO as a cost Hamiltonian, PCE maps variables to
#  parity correlations of qubits, achieving logarithmic compression.
#


def solve_with_pce(
    bqm: dimod.BinaryQuadraticModel,
    dist_matrix: np.ndarray,
    n_layers: int = 3,
    max_iterations: int = 20,
    alpha: float = 3.0,
    population_size: int = 30,
    shots: int = 10_000,
    encoding_type: str = "poly",
    backend=None,
) -> tuple[list[int], float, int]:
    """Solve TSP using PCE (Pauli Correlation Encoding).

    PCE compresses the QUBO variables into fewer qubits using
    polynomial or dense encoding, then solves with a variational
    quantum eigensolver.

    Args:
        bqm:              Binary Quadratic Model.
        dist_matrix:      Distance matrix.
        n_layers:         Ansatz circuit depth.
        max_iterations:   Optimizer iterations.
        alpha:            Binary activation sharpness (higher = harder).
        population_size:  DE optimizer population.
        shots:            Measurement samples.
        encoding_type:    "poly" (fewer qubits) or "dense" (even fewer).
        backend:          Divi backend.

    Returns:
        (tour, distance, n_qubits_used)
    """
    from divi.qprog import PCE, GenericLayerAnsatz
    from divi.qprog.optimizers import PymooMethod, PymooOptimizer
    from divi.hamiltonians import qubo_to_matrix

    if backend is None:
        backend = MaestroSimulator(shots=shots)

    n_cities = len(dist_matrix)
    qubo_mat = qubo_to_matrix(bqm)

    ansatz = GenericLayerAnsatz(
        gate_sequence=[RYGate, RZGate],
        entangler=CXGate,
        entangling_layout="all-to-all",
    )

    pce_solver = PCE(
        BinaryOptimizationProblem(qubo_mat),
        ansatz=ansatz,
        n_layers=n_layers,
        encoding_type=encoding_type,
        optimizer=PymooOptimizer(
            method=PymooMethod.DE, population_size=population_size
        ),
        max_iterations=max_iterations,
        alpha=alpha,
        backend=backend,
    )

    n_qubits = pce_solver.n_qubits
    print(
        f"   PCE qubits: {n_qubits}  "
        f"({encoding_type} encoding of {len(bqm.variables)} variables)"
    )

    pce_solver.run()

    # Decode the top solutions
    var_list = list(bqm.variables)
    top_solutions = pce_solver.get_top_solutions(n=50, include_decoded=True)
    samples = [
        {var_list[k]: int(sol.decoded[k]) for k in range(len(var_list))}
        for sol in top_solutions
        if sol.decoded is not None
    ]
    best, n_feasible = select_best_tour(samples, dist_matrix)

    if n_feasible > 0:
        print(f"   Found {n_feasible} exactly feasible solutions")
    else:
        print(f"   Used greedy repair on best candidates")

    if best is None:
        # Fallback: use raw solution
        sample = {
            var_list[k]: int(pce_solver.solution[k]) for k in range(len(var_list))
        }
        tour = repair_tour(sample, n_cities)
        best = (tour, tour_length(tour, dist_matrix))

    return best[0], best[1], n_qubits


# =====================================================================
#  MAIN — The high-level flow (start reading here!)
# =====================================================================

if __name__ == "__main__":
    USE_CLOUD = False
    SHOTS = 20_000
    SEED = 42
    N_CITIES_SMALL = 4
    N_CITIES_LARGE = 8

    backend = (
        QoroService(job_config=JobConfig(shots=SHOTS))
        if USE_CLOUD
        else MaestroSimulator(shots=SHOTS)
    )
    print(
        f"Backend: {'QoroService' if USE_CLOUD else 'MaestroSimulator'} (shots={SHOTS})"
    )

    # ── Part A: Direct QAOA ────────────────────────────────────────
    print(
        f"\n=== Part A: Direct QAOA ({N_CITIES_SMALL} cities, "
        f"{N_CITIES_SMALL**2} qubits) ==="
    )
    cities_small = generate_cities(N_CITIES_SMALL, seed=SEED)
    dist_small = compute_distance_matrix(cities_small)
    bqm_small, _ = build_tsp_qubo(dist_small)
    classical_tour_small, classical_dist_small = classical_brute_force(dist_small)

    qaoa = solve_with_qaoa(
        bqm_small, n_layers=3, max_iterations=10, shots=SHOTS, backend=backend
    )
    result_small = extract_best_tour(qaoa, bqm_small, dist_small)
    q_tour_small, q_dist_small = result_small
    print_comparison(
        classical_tour_small, classical_dist_small, q_tour_small, q_dist_small
    )
    plot_cities(cities_small, title=f"TSP — {N_CITIES_SMALL} Cities (Direct QAOA)")

    # ── Part B: Partitioned QAOA on a larger instance ──────────────
    # 64 qubits is well beyond a single QAOA — partition + parallelize.
    # NB: TSP one-hot constraints don't decompose cleanly, so this is a
    # stress-test of the partitioning machinery, not a way to set records.
    print(
        f"\n=== Part B: Partitioned QAOA ({N_CITIES_LARGE} cities, "
        f"{N_CITIES_LARGE**2} qubits) ==="
    )
    cities_large = generate_cities(N_CITIES_LARGE, seed=SEED)
    dist_large = compute_distance_matrix(cities_large)
    bqm_large, _ = build_tsp_qubo(dist_large)
    classical_tour_large, classical_dist_large = classical_brute_force(dist_large)

    q_tour_large, q_dist_large = solve_partitioned_tsp(
        bqm_large,
        dist_large,
        decomposer_size=15,
        n_layers=2,
        max_iterations=10,
        shots=SHOTS,
        backend=backend,
    )
    print_comparison(
        classical_tour_large, classical_dist_large, q_tour_large, q_dist_large
    )
    plot_comparison(
        cities_large,
        classical_tour_large,
        classical_dist_large,
        q_tour_large,
        q_dist_large,
        save_path="tsp_partitioned_comparison.png",
    )

    # ── Part C: PCE compression ────────────────────────────────────
    # Same small instance as Part A, but PCE compresses N² variables
    # into far fewer qubits via parity-encoding.
    print(
        f"\n=== Part C: PCE compression (same {N_CITIES_SMALL}-city "
        f"instance, qubit-compressed) ==="
    )
    pce_tour, pce_dist, pce_qubits = solve_with_pce(
        bqm_small,
        dist_small,
        n_layers=3,
        max_iterations=20,
        shots=SHOTS,
        backend=backend,
    )
    print(f"  PCE qubits: {pce_qubits} (vs {N_CITIES_SMALL**2} for direct QAOA)")
    print_comparison(classical_tour_small, classical_dist_small, pce_tour, pce_dist)

    # ── Summary ────────────────────────────────────────────────────
    print("\n=== Summary ===")
    rows = [
        (f"Classical (brute force)", N_CITIES_SMALL, "—", classical_dist_small, 1.000),
        (
            f"A: Direct QAOA",
            N_CITIES_SMALL,
            str(N_CITIES_SMALL**2),
            q_dist_small,
            q_dist_small / classical_dist_small,
        ),
        (
            f"C: PCE (poly encoding)",
            N_CITIES_SMALL,
            str(pce_qubits),
            pce_dist,
            pce_dist / classical_dist_small,
        ),
        (f"Classical (brute force)", N_CITIES_LARGE, "—", classical_dist_large, 1.000),
        (
            f"B: Partitioned QAOA",
            N_CITIES_LARGE,
            "≤15",
            q_dist_large,
            q_dist_large / classical_dist_large,
        ),
    ]
    print(f"  {'Method':<28} {'Cities':>6} {'Qubits':>7} {'Distance':>10} {'Ratio':>7}")
    for name, nc, nq, dist, ratio in rows:
        print(f"  {name:<28} {nc:>6} {nq:>7} {dist:>10.4f} {ratio:>7.3f}")
