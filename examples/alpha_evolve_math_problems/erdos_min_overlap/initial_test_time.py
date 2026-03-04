# EVOLVE-BLOCK-START

import numpy as np

def run():
    """Function to construct step-function h with low C5 bound."""
    rng = np.random.default_rng()
    n_points = rng.integers(40, 100)
    h_values = np.array([rng.random()] * n_points)
    # Normalize so sum(h) * dx == 1, i.e. sum(h) == n_points / 2
    h_values = h_values * (n_points / 2.0 / np.sum(h_values))
    c5_bound = evaluate_sequence(h_values)

    return h_values


def evaluate_sequence(h_values):
    dx = 2.0 / len(h_values)
    c5_bound = float(np.max(np.correlate(h_values, 1 - h_values, mode="full") * dx))
    return c5_bound

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    h_values = run()
    print(f"h_values: {h_values.tolist()}")