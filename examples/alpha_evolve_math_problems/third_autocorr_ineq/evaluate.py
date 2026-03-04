import argparse
import numpy as np
import os
from typing import Any, Dict, List, Tuple

from shinka.core.wrap_eval import run_shinka_eval

BENCHMARK = 1.4556427953745406                  # found by AlphaEvolve and AlphaEvolve v2

MAX_CHECK_VALUE = 1e10
MIN_CHECK_VALUE = 1e-10


def validate_sequence(run_output):
    """
    Validates the sequence returned by the evolved program
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

    if (seq_array > MAX_CHECK_VALUE).any():
        return False, f"Sequence cannot contain values above {MAX_CHECK_VALUE}."
    
    # Sum cannot be zero (otherwise integral^2 is zero)
    total = float(np.sum(seq_array))
    if abs(total) < MIN_CHECK_VALUE:
        return False, "Sum of heights is zero; invalid for C3 objective"

    # Check that sum squared is not too small
    if total ** 2 < MIN_CHECK_VALUE:
        return False, f"Sum squared is too small (< {MIN_CHECK_VALUE}); invalid for C3 objective"

    return True, "Sequence is valid."


def get_experiment_kwargs(run_index: int) -> Dict[str, Any]:
    return {}


def compute_c3(sequence: Any) -> float:
    """
    Compute C3 upper bound (to be minimized):
    """

    try:
        # 1) Full autoconvolution
        conv = np.convolve(sequence, sequence, mode="full")

        # Guard against extreme convolution values from numerical instability
        max_conv_abs = float(np.max(np.abs(conv)))
        if max_conv_abs > MAX_CHECK_VALUE:
            raise ValueError(f"Convolution values exceed maximum check value: {MAX_CHECK_VALUE}")

        sum_heights = float(np.sum(sequence))
        sum_squared = sum_heights ** 2

        # Compute C3 upper bound
        c3 = abs(2 * len(sequence) * max_conv_abs / sum_squared)

        # Final check for invalid results
        if not np.isfinite(c3):
            raise ValueError(f"Convolution values exceed maximum check value: {MAX_CHECK_VALUE}")

        return c3

    except Exception as e:
        raise ValueError(f"Error computing C3: {str(e)}")
    

# Not used, but left here for reference as a simpler, non-robust version of the same calculation (without checks or safeguards)
def compute_c3_simple(sequence: Any) -> float:
    conv = np.convolve(sequence, sequence, mode="full")
    n = len(sequence)
    
    max_conv_abs = float(np.max(np.abs(conv)))
    sum_heights = float(np.sum(sequence))
    sum_squared = sum_heights ** 2

    # Compute C3 upper bound
    c3 = abs(2 * n * max_conv_abs / sum_squared)
    return c3


def aggregate_metrics(results: List[Any]) -> Dict[str, Any]:
    """
    Calculates the score based on the validated sequence.
    """
    if not results:
        return {"combined_score": 0.0, "error": "No results"}

    c3_results = []
    best_c3 = float('inf')
    for sequence in results:
        sequence = np.asarray(sequence, dtype=float)
        c3 = compute_c3(sequence)
        c3_results.append(c3)
        if c3 < best_c3:
            best_c3 = c3
            best_sequence = sequence


    metrics = {
        "combined_score": BENCHMARK / best_c3 if best_c3 != 0 else 0.0,
        "best_c3": best_c3,
        "best_sequence_length": len(best_sequence),
        "c3_mean": float(np.mean(c3_results)),
        "c3_std": float(np.std(c3_results)),
        "c3": c3_results,
        "best_sequence": best_sequence.tolist(),
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