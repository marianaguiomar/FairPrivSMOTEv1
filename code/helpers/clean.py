import pandas as pd
import os
import glob

def unpack_value(val):
    if isinstance(val, str) and val.startswith("[") and val.endswith("]"):  
        try:
            val = eval(val)  # Convert string to actual list
            if isinstance(val, list) and len(val) == 1:
                return val[0]  # Extract the first element
        except:
            pass  # If eval fails, keep the value unchanged
    return val  

# Function to standardize binary values (convert all 1-like values to 1 and 0-like values to 0)
def standardize_binary(val):
    try:
        val = float(val)  # Convert to float
        if val in {1, 1.0, 1., 0, 0.0, 0.}:
            return int(val)  # Convert to integer (1 or 0)
    except:
        pass  # If conversion fails, keep original value
    return val 