# EVOLVE-BLOCK-START

import numpy as np


def evaluate_sequence(sequence: np.ndarray) -> float:
    """Compute the C3 upper bound."""
    conv = np.convolve(sequence, sequence, mode="full")
    n = len(sequence)
    
    max_conv_abs = float(np.max(conv))
    sum_heights = float(np.sum(sequence))
    sum_squared = sum_heights ** 2

    # Compute C3 upper bound
    c3 = abs(2 * n * max_conv_abs / sum_squared)
    return c3


def run() -> list[float]:
    """Function to construct step-function with low C3 value."""
    f_values = [np.random.random()] * np.random.randint(100,1000)
    return f_values


# EVOLVE-BLOCK-END

# Reads file best_sequence_prev.txt and returns the sequence as a list of floats.
def get_best_sequence_prev() -> list[float]:
    with open("best_sequence_prev_alpha_evolve.txt", "r") as f:
        line = f.readline().strip()
        if not line:
            return []
        return [float(x) for x in line.split(",")]


if __name__ == "__main__":
    f_values = run()
    print(f"Generated sequence: {f_values}")
    c3_value = evaluate_sequence(np.array(f_values))
    print(f"Evaluated C3 value: {c3_value}")
    best_sequence_prev = get_best_sequence_prev()
    print(f"Best previous sequence: {best_sequence_prev}")
    print(f"Evaluated best previous sequence: {evaluate_sequence(best_sequence_prev)}")