import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Optional, Tuple

class Preprocess:
    def __init__(self, target_col: str, numeric_cols: List[str], invert_col_scale: Optional[List[str]]=None):
        """
        Parameters:
        - target_col: str, name of the target column
        - numeric_cols: list of numeric columns to robust scale
        - invert_col_scale: list of ordinal columns to invert (e.g., [‘GenHlth’])
        """
        self.target_col = target_col
        self.numeric_cols = numeric_cols
        self.invert_col_scale = invert_col_scale if invert_col_scale else []
        self.scaling_params = {}

    def robust_scale(self, train_col: pd.Series) -> Tuple[float, float]:
        median = np.median(train_col)
        q1 = np.percentile(train_col, 25)
        q3 = np.percentile(train_col, 75)
        iqr = q3 - q1 + 1e-9
        return median, iqr

    def apply_scaling(self, col: pd.Series, median: float, iqr: float):
        return (col - median) / iqr

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        X = df.drop(columns=[self.target_col]).copy()
        y = df[self.target_col].copy()

        for col in self.invert_col_scale:
            X[col] = 6 - X[col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=17, stratify=y
        )

        for col in self.numeric_cols:
            median, iqr = self.robust_scale(self.X_train[col])
            self.scaling_params[col] = (median, iqr)
            self.X_train[col] = self.apply_scaling(self.X_train[col], median, iqr)
            self.X_test[col] = self.apply_scaling(self.X_test[col], median, iqr)

        self.X_train[self.numeric_cols] = self.X_train[self.numeric_cols].round(0).astype(int)
        self.X_test[self.numeric_cols] = self.X_test[self.numeric_cols].round(0).astype(int)

        return (
            self.X_train.astype(int),
            self.X_test.astype(int),
            self.y_train.values,
            self.y_test.values
        )
