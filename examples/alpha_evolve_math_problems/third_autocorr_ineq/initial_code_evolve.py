# EVOLVE-BLOCK-START

import numpy as np

def run() -> list[float]:
    """Function to construct step-function with low C3 value."""
    f_values = [np.random.random()] * np.random.randint(100,1000)
    return f_values

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = run()
    print(f"Function: {f_values}")