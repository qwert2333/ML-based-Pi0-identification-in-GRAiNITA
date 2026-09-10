###Shared helper functions for splitting and mixing datasets###
import numpy as np
import awkward as ak

def safe_to_numpy(
    ak_array: ak.Array, dtype: type = np.float64, default_val: float = 0.0
) -> np.ndarray:
    """Safely converts a 1D event-level Awkward Array to a NumPy array.

    Resolves UnionTypes, OptionTypes (Nones), and mixed types by extracting values
    via Python list conversion.
    """
    # 1. Fill missing/None values
    filled = ak.fill_none(ak_array, default_val)
    # 2. Convert to Python list to resolve internal Awkward UnionType layouts
    py_list = ak.to_list(filled)
    # 3. Convert to contiguous 1D NumPy array with desired dtype
    return np.array(py_list, dtype=dtype)

