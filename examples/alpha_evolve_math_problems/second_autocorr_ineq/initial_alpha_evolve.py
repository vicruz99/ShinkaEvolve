# EVOLVE-BLOCK-START

import numpy as np

def run() -> list[float]:
    """Function to construct step-function with high C2 value."""
    f_values = [np.random.random()] * np.random.randint(100,10000)
    return f_values

# EVOLVE-BLOCK-END

def evaluate_sequence(heights: np.ndarray) -> float:
    """
    Compute the C₂ lower bound:
        R(f) = ||f*f||_2^2 / ( ||f*f||_1 * ||f*f||_inf )

    Implementation matches the paper's verification logic:
      * full autoconvolution
      * piecewise-linear (Simpson-like) integral with zero padding for ||.||_2^2
      * L1 as dx * sum(conv), dx = 1/(M+1)
      * Linf as max(conv)
    """
    conv = np.convolve(heights, heights, mode="full")
    if conv.size == 0:
        return 0.0

    # L2 norm squared via piecewise-linear rule with endpoint zeros
    M = len(conv)
    dx = 1.0 / (M + 1)
    y = np.empty(M + 2, dtype=conv.dtype)
    y[0] = 0.0
    y[1:-1] = conv
    y[-1] = 0.0
    l2_sq = (dx / 3.0) * np.sum(y[:-1] ** 2 + y[:-1] * y[1:] + y[1:] ** 2)

    # L1 and Linf norms
    l1 = dx * float(np.sum(conv))
    linf = float(np.max(conv))

    if l1 <= 0.0 or linf <= 0.0:
        return 0.0

    return float(l2_sq) / (l1 * linf)

# EVOLVE-BLOCK-END

# Reads file best_sequence_prev.txt and returns the sequence as a list of floats.
def read_best_sequence_prev() -> list[float]:
    with open("best_sequence_prev_alpha_evolve.txt", "r") as f:
        line = f.readline().strip()
        if not line:
            return []
        return [float(x) for x in line.split(",")]


# if __name__ == "__main__":
#     f_values = run()
#     print(f"Generated sequence: {f_values}")
#     c2_value = evaluate_sequence(np.array(f_values))
#     print(f"Evaluated C2 value: {c2_value}")
#     print(f"Best previous sequence: {read_best_sequence_prev()}")