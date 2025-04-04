import pandas as pd
import os
from typing import List, Optional

def load_dataset(file_path: str, expected_cols: Optional[List[str]] = None) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    data = pd.read_csv(file_path)

    if expected_cols:
        missing = set(expected_cols) - set(data.columns)
        if missing:
            raise ValueError(f"Missing expected columns: {missing}")

    return data