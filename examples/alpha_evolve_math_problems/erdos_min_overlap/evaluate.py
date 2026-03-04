import argparse
import numpy as np
import os
from typing import Any, Dict, List, Tuple

from shinka.core.wrap_eval import run_shinka_eval

BENCHMARK = 0.38092303510845016                  # found by AlphaEvolve and AlphaEvolve v2


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

    if np.any(seq_array < 0) or np.any(seq_array > 1):
        return False, f"h(x) is not in [0, 1]. Range: [{seq_array.min()}, {seq_array.max()}]"

    dx = 2.0 / len(sequence)
    integral_h = np.sum(sequence) * dx
    if not np.isclose(integral_h, 1.0, atol=1e-5):
        return False, f"Integral of h is not close to 1. Got: {integral_h:.6f}"

    return True, "The sequence generates a valid step function for Erdős' minimum overlap problem."


def get_experiment_kwargs(run_index: int) -> Dict[str, Any]:
    return {}


def compute_upper_bound(sequence: Any) -> float:
    """
    Returns the upper bound for a sequence of coefficients.
    """
    convolution_values = np.correlate(np.array(sequence), 1 - np.array(sequence), mode='full')
    return np.max(convolution_values) / len(sequence) * 2
    

def aggregate_metrics(results: List[Any]) -> Dict[str, Any]:
    """
    Calculates the score based on the validated sequence.
    """
    if not results:
        return {"combined_score": 0.0, "error": "No results"}

    c5_results = []
    best_c5 = float('inf')
    for sequence in results:
        sequence = np.asarray(sequence, dtype=float)
        c5 = compute_upper_bound(sequence)
        c5_results.append(c5)
        if c5 < best_c5:
            best_c5 = c5
            best_sequence = sequence


    metrics = {
        "combined_score": BENCHMARK / best_c5 if best_c5 != 0 else 0.0,
        "best_c5": best_c5,
        "best_sequence_length": len(best_sequence),
        "c5_mean": float(np.mean(c5_results)),
        "c5_std": float(np.std(c5_results)),
        "c5": c5_results,
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