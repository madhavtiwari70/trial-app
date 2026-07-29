# Molecular Ground State (VQE) — demo config file
# Edit the values below to change the demo. No changes needed anywhere else.

config = {
    "molecule": {
        "symbols": ["H", "H"],              # atomic symbols
        "coordinates": [                    # one [x, y, z] per atom, in Angstrom
            [0, 0, 0],
            [0, 0, 0.735],
        ],
    },
    "vqe": {
        "n_layers": 2,
        "optimizer_method": "COBYLA",       # COBYLA, NELDER_MEAD, or L_BFGS_B
        "max_iterations": 30,
        "seed": 42,
    },
    "backend": {
        "use_cloud": False,                 # True = QoroService, False = local simulator
        "shots": 5000,
    },
}
