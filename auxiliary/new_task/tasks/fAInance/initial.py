# You can import other standard libraries here.
# Everything between the markers <EVOLVE-BLOCK-START> and <EVOLVE-BLOCK-END> is what the LLM will rewrite and optimize.

# EVOLVE-BLOCK-START

import sys, datetime as dt
import pandas as pd
import math
import numpy as np
import scipy as scp
import itertools

def strategy_function(
    df,
    price_col="open",
    time_col="timestamp",
    short_window=10,
    long_window=30,
    default=1,
):
    """
    action[t] in {+1, -1} computed using prices up to time t (including price[t]),
    intended to be applied to the next move: (price[t+1] - price[t]) or % version.
    """
    
    x = df[[time_col, price_col]].copy()
    x[time_col] = pd.to_datetime(x[time_col])
    x = x.sort_values(time_col)
    
    sma_s = x[price_col].rolling(short_window, min_periods=short_window).mean()
    sma_l = x[price_col].rolling(long_window,  min_periods=long_window).mean()
    
    
    # Signal/action at time t (uses info through t)
    action = np.where(sma_s > sma_l, 1, -1)
    action = pd.Series(action, index=x.index, name="action").astype("float")

    # Fill early NaNs (before long_window exists)
    action = action.where(~(sma_l.isna() | sma_s.isna()))  # keep NaN early

    # Return aligned to original df index order (optional)
    return action.reindex(df.index)



# EVOLVE-BLOCK-END


if __name__ == "__main__":
    # This block allows you to run the script manually to test it.
    # It is NOT used during the evolution process.
    print("Running initial program manually...")
    
    try:
        test_company = pd.read_csv("/home/guests2/vic/work/SP 500 Daily Stock Values - Normalized/AOS_normalized.csv")
        test_company = test_company.sort_values("timestamp")
        test_company['close'] = test_company['normalized_close'].copy()
        test_company['open'] = test_company['normalized_open'].copy()

        output = strategy_function(test_company)
         
        print(f"Function returned: {output}")
    except Exception as e:
        print(f"Error running function: {e}")