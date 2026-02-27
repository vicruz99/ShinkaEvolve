import argparse
import numpy as np
from typing import Any, Dict, List, Tuple
import pandas as pd
import random

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

    # 1. Check if it is actually a pandas series
    if not isinstance(run_output, pd.Series):
        return False, f"Expected a pandas Series, but got {type(run_output).__name__}."
    
    # 2. Check if it's completely filled with NaNs (or empty)
    # .notna().any().any() returns True if there is at least ONE non-NaN value anywhere.
    if not run_output.notna().any().any():
        return False, "The Series cannot be empty or contain only NaNs."
        
    # 3. Check if all values are strictly 1, -1, or NaN
    # .isin([-1, 1]) automatically handles both ints (1) and floats (1.0).
    is_valid_value = run_output.isin([-1, 1]) | run_output.isna()
    
    # .all().all() ensures every single cell in the 2D grid returned True
    if not is_valid_value.all().all():
        return False, "The Series contains invalid values. Only 1, -1, and NaN are allowed."

    df = kwargs.get("df", None)  # Get the original DataFrame if needed for further checks
    # Check if the DataFrame is not None
    if df is None:
        return False, "Input dataFrame is missing."

    # Check if the dataframe in run_output has the same number of rows as the input dataframe
    if len(run_output) != len(df):
        return False, f"Output DataFrame has {len(run_output)} rows, but expected {len(df)} rows."

    return True, "Valid"



def get_experiment_kwargs(run_index: int) -> Dict[str, Any]:

    DATA_PATH = "data/"
    sp500_companies_we_have_data_on = pd.read_csv(DATA_PATH + "sp500_companies_we_have_data_on.csv", header=None)[0].tolist()
    # Remove an element from the list
    sp500_companies_we_have_data_on.remove("CPWR")
    sp500_companies_we_have_data_on.remove("PCP")
    sp500_companies_we_have_data_on.remove("SIVB")
    random_company = random.choice(sp500_companies_we_have_data_on)
    sample_company = pd.read_csv(DATA_PATH + "SP 500 Daily Stock Values - Normalized/" + random_company + "_normalized.csv") #Loading the data for that company

    sample_company = sample_company.sort_values("timestamp")
    sample_company['close'] = sample_company['normalized_close'].copy()
    sample_company['open'] = sample_company['normalized_open'].copy()

    return {"df" : sample_company,
            "company_name": random_company}


def evaluation_function(data_db, signal):
    
    percentual_returns = (data_db['open'].shift(-1) -  data_db['open'])/data_db['open']
    
    # strategy daily return
    strategy_r = signal * percentual_returns

    #Final return over all trading days
    overall_return_sum_n = strategy_r.sum(skipna=True)

    return overall_return_sum_n


# def no_model_strategy(data_db):

#     signal = pd.Series(1, index=data_db.index)  # Simulate always buying (signal = +1)
#     return evaluation_function(data_db, signal)

#     # #Calculating percentual change at t from t-1: (Price_t - Price_t-1) / Price_t-1
#     # data_db["pct_change"] = data_db["open"].pct_change()

#     # #Calculating no_model (simulating always buy)
#     # no_model = data_db['pct_change'].cumsum().copy()

#     return no_model.iloc[-1]
    

def aggregate_metrics_fn(results: List[Any], all_run_kwargs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculates the final score based on results from all VALID runs.
    
    Args:
        results: A list of `run_output`s from successful validations.
        all_run_kwargs: A list of kwargs used for each run.

    Returns:
        Dict containing metrics. MUST include "combined_score".
    """
    if not results:
        return {"combined_score": 0.0, "error": "No valid results"}

    companies_score = {}
    for signal, kwargs in zip(results, all_run_kwargs):
        company_name = kwargs.get("company_name")
        data_db = kwargs.get("df")
        score = evaluation_function(data_db, signal)
        companies_score[company_name] = score
    
    # Return the dictionary
    return {
        "combined_score": np.mean(list(companies_score.values())) if companies_score else np.NaN,
        "individual_companie_scores": companies_score
    }


def main(program_path: str, results_dir: str):
    """
    Main entry point for evaluation.
    """
    print(f"Evaluating program: {program_path}")
    
    FUNCTION_NAME_TO_TEST = "strategy_function"                     # Must match the function name in initial.py
    NUMBER_OF_RUNS = 200                                            # How many times to run the code  - IT WOULD BE COOL FOR THIS TO BE CONFIGURABLE FROM THE OUTSIDE, BUT FOR NOW LET'S KEEP IT FIXED TO A HIGH NUMBER TO ENCOURAGE ROBUSTNESS AND AVOID OVERFITTING TO THE TEST SET.
    
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