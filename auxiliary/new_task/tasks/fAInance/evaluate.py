import argparse
import numpy as np
from typing import Any, Dict, List, Tuple


DATA_PATH = "/home/guests2/vic/work/data/"

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

    # 1. Check if it is actually a pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        return False, f"Expected a pandas DataFrame, but got {type(df).__name__}."
    
    # 2. Check if it's completely filled with NaNs (or empty)
    # .notna().any().any() returns True if there is at least ONE non-NaN value anywhere.
    if not df.notna().any().any():
        return False, "The DataFrame cannot be empty or contain only NaNs."
        
    # 3. Check if all values are strictly 1, -1, or NaN
    # .isin([-1, 1]) automatically handles both ints (1) and floats (1.0).
    is_valid_value = df.isin([-1, 1]) | df.isna()
    
    # .all().all() ensures every single cell in the 2D grid returned True
    if not is_valid_value.all().all():
        return False, "The DataFrame contains invalid values. Only 1, -1, and NaN are allowed."

    return True, "Valid"


def get_experiment_kwargs(run_index: int) -> Dict[str, Any]:

    companies_we_have_data_on = "sp500_companies_we_have_data_on.csv"
    sp500_companies_we_have_data_on = pd.read_csv(DATA_PATH + companies_we_have_data_on, header=None)[0].tolist()


    #Select one company randomly to be representative of the market (we will fit this company and use its parameters for all others)
    rand_index = random.randrange(len(missed_companies)) #Getting a random index
    random_company = sp500_companies_we_have_data_on[rand_index]
    sample_company = pd.read_csv("/home/guilherme/Desktop/fAInance/SP500 Daily Stock Values - Alpha Vantage/" + random_company + "_ohlc_full_history.csv") #Loading the data for that company

    return {"df" : df}



def evaluation_function(data_db, signal):
    
    percentual_returns = (data_db['open'].shift(-1) -  data_db['open'])/data_db['open']
    
    # strategy daily return
    strategy_r = signal * percentual_returns

    #Final return over all trading days
    overall_return_sum_n = strategy_r.sum(skipna=True)
    
    return overall_return_sum_n
    

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

    signal = results[0]
    score = evaluation_function(signal)
    
    # Return the dictionary
    return {
        "combined_score": score,
        "num_valid_runs": len(results),
        # Add any other metrics you want to log
        # "max_val": float(np.max(results))
    }


def main(program_path: str, results_dir: str):
    """
    Main entry point for evaluation.
    """
    print(f"Evaluating program: {program_path}")
    
    FUNCTION_NAME_TO_TEST = "strategy_function"                 # Must match the function name in initial.py
    NUMBER_OF_RUNS = 1                                          # How many times to run the code
    
    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name=FUNCTION_NAME_TO_TEST,
        num_runs=NUMBER_OF_RUNS,
        get_experiment_kwargs=get_experiment_kwargs,
        validate_fn=validate_fn,
        aggregate_metrics_fn=aggregate_metrics_fn
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