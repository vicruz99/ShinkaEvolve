"""
Standard Template for ShinkaEvolve Task Evaluation.

Usage:
1. Copy this file to examples/<your_task>/evaluate.py
2. Implement validate_fn checks.
3. Implement aggregate_metrics_fn scoring.
4. Update the experiment_fn_name in main() to match your initial.py.
"""

import argparse
import numpy as np
from typing import Any, Dict, List, Tuple

# The core runner from the Shinka framework
from shinka.core import run_shinka_eval

def validate_fn(run_output: Any, **kwargs) -> Tuple[bool, str]:
    """
    Validates the output of a single run of the evolved function.
    
    Args:
        run_output: The return value from the evolved function.
        **kwargs: Additional arguments if needed.

    Returns:
        (is_valid: bool, error_message: str)
        If is_valid is False, the result is discarded.
    """
    # ---------------------------------------------------------
    # TODO: Add your custom validation logic here.
    # ---------------------------------------------------------
    
    # Example Check 1: Ensure it's not None
    if run_output is None:
        return False, "Output cannot be None"
    
    # Example Check 2: Check type (e.g., ensure it's a list)
    # if not isinstance(run_output, list):
    #     return False, f"Expected list, got {type(run_output)}"

    # Example Check 3: Check constraints (e.g., no NaNs)
    # if np.any(np.isnan(run_output)):
    #      return False, "Output contains NaNs"

    return True, "Valid"


def get_experiment_kwargs(run_index: int) -> Dict[str, Any]:
    """
    Generates the arguments to pass INTO the evolved function.
    
    Args:
        run_index: The index of the current run (0 to num_runs-1).
        
    Returns:
        Dict of keyword arguments.
    """
    # ---------------------------------------------------------
    # TODO: Define input arguments for your function here.
    # ---------------------------------------------------------
    
    # If your function is `def solve(seed):`, return {"seed": 42 + run_index}
    # If your function takes no arguments, return {}
    
    return {}


def aggregate_metrics_fn(results: List[Any]) -> Dict[str, Any]:
    """
    Calculates the final score based on results from all VALID runs.
    
    Args:
        results: A list of `run_output`s from successful validations.

    Returns:
        Dict containing metrics. MUST include "combined_score".
    """
    if not results:
        return {"combined_score": 0.0, "error": "No valid results"}

    # ---------------------------------------------------------
    # TODO: Implement your scoring logic here.
    # ---------------------------------------------------------
    
    # Example: If results are numbers, take the mean
    # score = float(np.mean(results))
    
    # Placeholder score
    score = 0.0
    
    # Return the dictionary
    return {
        "combined_score": score,  # CRITICAL: This is what Shinka optimizes!
        "num_valid_runs": len(results),
        # Add any other metrics you want to log
        # "max_val": float(np.max(results))
    }


def main(program_path: str, results_dir: str):
    """
    Main entry point for evaluation.
    """
    print(f"Evaluating program: {program_path}")
    
    # ---------------------------------------------------------
    # TODO: Update these settings for your specific task
    # ---------------------------------------------------------
    FUNCTION_NAME_TO_TEST = "solve"  # Must match the function name in initial.py
    NUMBER_OF_RUNS = 1               # How many times to run the code
    TIMEOUT_SECONDS = 10.0           # Max time per run
    
    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name=FUNCTION_NAME_TO_TEST,
        num_runs=NUMBER_OF_RUNS,
        get_experiment_kwargs=get_experiment_kwargs,
        validate_fn=validate_fn,
        aggregate_metrics_fn=aggregate_metrics_fn,
        timeout=TIMEOUT_SECONDS
    )

    print("Metrics:", metrics)
    
    if correct:
        print("Evaluation completed successfully.")
    else:
        print(f"Evaluation failed: {error_msg}")


if __name__ == "__main__":
    # Standard boilerplate for Hydra/Shinka execution
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", type=str, default="initial.py")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    
    main(args.program_path, args.results_dir)