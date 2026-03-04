import argparse
import numpy as np
import os
from typing import Any, Dict, List, Tuple

from shinka.core.wrap_eval import run_shinka_eval

BENCHMARK = 0.8962799441554083                  # found by AlphaEvolve

MAX_CHECK_VALUE = 1e10
MIN_CHECK_VALUE = 1e-10


def validate_sequence(run_output):
    """
    Validates the sequence returned by the evolved program. Must comply with the following criteria:
        - expects list of floats
        - Non-negative: f >= 0 (with small numerical tolerance)
    """
    sequence = run_output

    if not isinstance(run_output, (list, np.ndarray)):
        return False, f"Expected list or np.ndarray, received {type(run_output)}"
    sequence = list(sequence)   # Convert to list if it's a numpy array for easier validation and error messages

    if not sequence:
        return False, "Sequence cannot be None or empty."

    if any(type(x) is bool for x in sequence):
        return False, "Sequence entries must be integers or floats, not booleans."

    try:
        seq_array = np.asarray(sequence, dtype=float)
    except (ValueError, TypeError):
        return False, "Sequence entries must be valid numeric types."

    if not np.isfinite(seq_array).all():
        return False, "Sequence cannot contain nans or infs."

    if (seq_array < -MIN_CHECK_VALUE).any():
        return False, f"Sequence cannot contain values below {MIN_CHECK_VALUE}."

    if (seq_array > MAX_CHECK_VALUE).any():
        return False, f"Sequence cannot contain values above {MAX_CHECK_VALUE}."

    return True, "Sequence is valid."


def get_experiment_kwargs(run_index: int) -> Dict[str, Any]:
    return {}


def compute_c2(sequence: Any) -> float:
    """
    Compute C2 lower bound R(f) (to be maximized):
      1) Full autoconvolution conv = f * f
      2) Piecewise-linear squared integral (zero-padded at endpoints) for ||f*f||_2^2
      3) ||f*f||_1 = sum(|conv|)/(N+1)
      4) ||f*f||_inf = max(|conv|)
      5) R(f) = L2^2 / (L1 * Linf)
    """

    try:
        # 1) Full autoconvolution
        conv = np.convolve(sequence, sequence, mode="full")
        num_points = conv.size

        # Guard against extreme convolution values from numerical instability
        max_conv_abs = float(np.max(np.abs(conv)))
        if max_conv_abs > MAX_CHECK_VALUE:
            raise ValueError(f"Convolution values exceed maximum check value: {MAX_CHECK_VALUE}")

        # 2) L2 norm squared via piecewise-linear integral with zero padding at ends
        x_points = np.linspace(-0.5, 0.5, num_points + 2)
        x_intervals = np.diff(x_points)
        y_points = np.concatenate(([0.0], conv, [0.0]))

        l2_norm_squared = 0.0
        # Integral formula: h/3 * (y1^2 + y1*y2 + y2^2) for piecewise linear
        for i in range(num_points + 1):
            y1 = y_points[i]
            y2 = y_points[i + 1]
            h  = x_intervals[i]
            l2_norm_squared += (h / 3.0) * (y1 * y1 + y1 * y2 + y2 * y2)

        # Guard against non-finite l2 norm
        if not np.isfinite(l2_norm_squared) or l2_norm_squared < MIN_CHECK_VALUE * MIN_CHECK_VALUE:
            raise ValueError(f"Convolution values exceed maximum check value: {MAX_CHECK_VALUE} or are below minimum check value: {MIN_CHECK_VALUE}")

        # 3) L1 norm (averaged definition)
        norm_1 = float(np.sum(np.abs(conv))) / float(num_points + 1)

        # 4) Linf norm
        norm_inf = float(np.max(np.abs(conv)))

        if norm_1 <= MIN_CHECK_VALUE or norm_inf <= MIN_CHECK_VALUE:
            raise ValueError(f"Convolution values exceed maximum check value: {MAX_CHECK_VALUE} or are below minimum check value: {MIN_CHECK_VALUE}")

        # 5) C2 lower bound: R(f)
        c2_upper = float(l2_norm_squared) / (norm_1 * norm_inf)

        # Final check for invalid results
        if not np.isfinite(c2_upper):
            raise ValueError(f"Convolution values exceed maximum check value: {MAX_CHECK_VALUE}")

        return c2_upper

    except Exception as e:
        raise ValueError(f"Error computing C2: {str(e)}") 
    

# Not used, but left here for reference as a simpler, non-robust version of the same calculation (without checks or safeguards)
def compute_c2_simple(sequence: Any) -> float:
    conv = np.convolve(sequence, sequence, mode="full")
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

    c2 = float(l2_sq) / (l1 * linf)
    if not np.isfinite(c2):
            raise ValueError(f"Problem computing c2")
    
    return c2


def aggregate_metrics(results: List[Any]) -> Dict[str, Any]:
    """
    Calculates the score based on the validated sequence.
    """
    if not results:
        return {"combined_score": 0.0, "error": "No results"}

    c2_results = []
    best_c2 = float('-inf')
    for sequence in results:
        sequence = np.asarray(sequence, dtype=float)
        sequence = np.clip(sequence, MIN_CHECK_VALUE, MAX_CHECK_VALUE)
        c2 = compute_c2(sequence)
        c2_results.append(c2)
        if c2 > best_c2:
            best_c2 = c2
            best_sequence = sequence


    metrics = {
        "combined_score": best_c2 / BENCHMARK,
        "best_c2": best_c2,
        "best_sequence_length": len(best_sequence),
        "c2_mean": float(np.mean(c2_results)),
        "c2_std": float(np.std(c2_results)),
        "c2": c2_results,
        "best_sequence": list(best_sequence),
    }
    
    return metrics

def main(program_path: str, results_dir: str):
    print(f"Evaluating program: {program_path}")
    
    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run",
        num_runs=10,                                    # CHANGE THIS TO 1, TO GET ONLY ONE EVALUATION PER PROGRAM VARIANT - IT MAY BE USEFUL TO INCREASE THIS AS SOME PROGRAMS ARE NOT DETERMINISTIC AND INSTEAD SEARCH FOR SOLUTIONS
        get_experiment_kwargs=get_experiment_kwargs,
        validate_fn=validate_sequence,
        aggregate_metrics_fn=aggregate_metrics,
    )

    if correct:
        print("Evaluation completed successfully.")
        print("Metrics:", metrics)
    else:
        print(f"Evaluation failed: {error_msg}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", type=str, default="initial.py")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    main(args.program_path, args.results_dir)