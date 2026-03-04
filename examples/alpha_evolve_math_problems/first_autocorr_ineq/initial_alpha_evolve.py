# EVOLVE-BLOCK-START

import numpy as np
import time
from scipy import optimize
linprog = optimize.linprog


def solve_convolution_lp(f_sequence, rhs):
    """Solves the convolution LP for a given sequence and RHS."""
    n = len(f_sequence)
    c = -np.ones(n)
    a_ub = []
    b_ub = []
    for k in range(2 * n - 1):
        row = np.zeros(n)
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                row[j] = f_sequence[i]
        a_ub.append(row)
        b_ub.append(rhs)

    # Non-negativity constraints: b_i >= 0
    a_ub_nonneg = -np.eye(n)  # Negative identity matrix for b_i >= 0
    b_ub_nonneg = np.zeros(n)  # Zero vector

    a_ub = np.vstack([a_ub, a_ub_nonneg])
    b_ub = np.hstack([b_ub, b_ub_nonneg])

    result = linprog(c, A_ub=a_ub, b_ub=b_ub)

    if result.success:
        g_sequence = result.x
        return g_sequence
    else:
        print('LP optimization failed.')
        return None


def get_good_direction_to_move_into(sequence: list[float]) -> list[float] | None:
    """Returns the direction to move into the sequence."""
    n = len(sequence)
    sum_sequence = np.sum(sequence)
    normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]
    rhs = np.max(np.convolve(normalized_sequence, normalized_sequence))
    g_fun = solve_convolution_lp(normalized_sequence, rhs)
    if g_fun is None:
        return None
    sum_sequence = np.sum(g_fun)
    normalized_g_fun = [x * np.sqrt(2 * n) / sum_sequence for x in g_fun]
    t = 0.01
    new_sequence = [(1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)]
    return new_sequence


def run() -> list[float]:
    """Function to search for the best coefficient sequence."""
    best_sequence_prev = get_best_sequence_prev()
    if np.random.rand() < 0.5 and best_sequence_prev:
        best_sequence = best_sequence_prev
    else:
        best_sequence = [np.random.random()] * np.random.randint(100,1000)
    curr_sequence = best_sequence.copy()
    best_score = np.inf
    start_time = time.time()
    while time.time() - start_time < 1000:
        h_function = get_good_direction_to_move_into(curr_sequence)
        if h_function is None:
            curr_sequence[1] = (curr_sequence[1] + np.random.rand()) % 1
        else:
            curr_sequence = h_function

        curr_score = evaluate_sequence(curr_sequence)
        if curr_score < best_score:
            best_score = curr_score
            best_sequence = curr_sequence

    return best_sequence

# EVOLVE-BLOCK-END


def evaluate_sequence(sequence: list[float]) -> float:
  
  if not isinstance(sequence, list):
    return np.inf
  # Reject empty lists
  if not sequence:
    return np.inf

  # Check each element in the list for validity
  for x in sequence:
    # Reject boolean types (as they are a subclass of int) and
    # any other non-integer/non-float types (like strings or complex numbers).
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return np.inf

    # Reject Not-a-Number (NaN) and infinity values.
    if np.isnan(x) or np.isinf(x):
        return np.inf

  # Convert all elements to float for consistency
  sequence = [float(x) for x in sequence]

  # Protect against negative numbers
  sequence = [max(0, x) for x in sequence]

  # Protect against numbers that are too large
  sequence = [min(1000.0, x) for x in sequence]

  n = len(sequence)
  b_sequence = np.convolve(sequence, sequence)
  max_b = max(b_sequence)
  sum_a = np.sum(sequence)

  # Protect against the case where the sum is too close to zero
  if sum_a < 0.01:
    return np.inf

  return float(2 * n * max_b / (sum_a**2))


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
    c2_value = evaluate_sequence(np.array(f_values))
    print(f"Evaluated C2 value: {c2_value}")
    print(f"Best previous sequence: {get_best_sequence_prev()}")