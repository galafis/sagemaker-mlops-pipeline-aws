"""Data preprocessing for SageMaker Processing jobs.

This module handles feature engineering, data splitting, and validation
before model training. Designed to run both locally and as a
SageMaker Processing job.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PreprocessingReport:
    """Report from a preprocessing execution."""

    input_rows: int
    output_rows: int
    features_count: int
    train_samples: int
    validation_samples: int
    test_samples: int
    columns_dropped: List[str]
    columns_encoded: List[str]
    missing_values_handled: int


class SageMakerPreprocessor:
    """Preprocessor designed for SageMaker Processing jobs.

    Handles data cleaning, feature engineering, encoding, and
    train/val/test splitting with reproducible results.
    """

    def __init__(
        self,
        target_column: str = "target",
        test_size: float = 0.15,
        validation_size: float = 0.15,
        random_state: int = 42,
        max_missing_ratio: float = 0.4,
        numeric_fill_strategy: str = "median",
    ):
        self.target_column = target_column
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        self.max_missing_ratio = max_missing_ratio
        self.numeric_fill_strategy = numeric_fill_strategy
        self._scaler = StandardScaler()
        self._encoders: Dict[str, LabelEncoder] = {}

    def process(
        self, input_path: str, output_dir: str
    ) -> PreprocessingReport:
        """Run the full preprocessing pipeline.

        Args:
            input_path: Path to raw input data (CSV or Parquet).
            output_dir: Directory for processed output splits.

        Returns:
            PreprocessingReport with execution details.
        """
        logger.info(f"Starting preprocessing: {input_path}")

        df = self._load_data(input_path)
        input_rows = len(df)

        df, dropped = self._drop_high_missing(df)
        missing_handled = int(df.isnull().sum().sum())
        df, encoded_cols = self._encode_categoricals(df)
        df = self._fill_missing(df)
        df = self._engineer_features(df)

        train_df, val_df, test_df = self._split_data(df)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(output_path / "train.csv", index=False, header=False)
        val_df.to_csv(output_path / "validation.csv", index=False, header=False)
        test_df.to_csv(output_path / "test.csv", index=False, header=False)

        feature_cols = [c for c in df.columns if c != self.target_column]
        manifest = {
            "features": feature_cols,
            "target": self.target_column,
            "train_samples": len(train_df),
            "validation_samples": len(val_df),
            "test_samples": len(test_df),
        }
        with open(output_path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        report = PreprocessingReport(
            input_rows=input_rows,
            output_rows=len(df),
            features_count=len(feature_cols),
            train_samples=len(train_df),
            validation_samples=len(val_df),
            test_samples=len(test_df),
            columns_dropped=dropped,
            columns_encoded=encoded_cols,
            missing_values_handled=missing_handled,
        )

        logger.info(
            f"Preprocessing complete: {report.input_rows} -> "
            f"{report.output_rows} rows, {report.features_count} features"
        )
        return report

    def _load_data(self, path: str) -> pd.DataFrame:
        p = Path(path)
        if p.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)

    def _drop_high_missing(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        missing_ratios = df.isnull().mean()
        to_drop = missing_ratios[missing_ratios > self.max_missing_ratio].index.tolist()
        if to_drop:
            df = df.drop(columns=to_drop)
            logger.info(f"Dropped {len(to_drop)} high-missing columns: {to_drop}")
        return df, to_drop

    def _encode_categoricals(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if self.target_column in cat_cols:
            cat_cols.remove(self.target_column)

        encoded = []
        for col in cat_cols:
            if df[col].nunique() > 50:
                df = df.drop(columns=[col])
                logger.info(f"Dropped high-cardinality column: {col}")
                continue
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))
            self._encoders[col] = encoder
            encoded.append(col)

        return df, encoded

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if self.numeric_fill_strategy == "median":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif self.numeric_fill_strategy == "mean":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        else:
            df[numeric_cols] = df[numeric_cols].fillna(0)
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_column in numeric_cols:
            numeric_cols.remove(self.target_column)

        if len(numeric_cols) >= 2:
            col_a, col_b = numeric_cols[0], numeric_cols[1]
            df[f"{col_a}_x_{col_b}_interaction"] = df[col_a] * df[col_b]

        for col in numeric_cols[:3]:
            df[f"{col}_squared"] = df[col] ** 2
            df[f"{col}_log1p"] = np.log1p(np.abs(df[col]))

        return df

    def _split_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split into train/validation/test with target as first column."""
        cols = [self.target_column] + [
            c for c in df.columns if c != self.target_column
        ]
        df = df[cols]

        train_val, test = train_test_split(
            df, test_size=self.test_size, random_state=self.random_state
        )
        relative_val = self.validation_size / (1 - self.test_size)
        train, val = train_test_split(
            train_val, test_size=relative_val, random_state=self.random_state
        )

        logger.info(
            f"Split: train={len(train)}, val={len(val)}, test={len(test)}"
        )
        return train, val, test


if __name__ == "__main__":
    """Entry point for SageMaker Processing job."""
    input_dir = os.environ.get(
        "SM_INPUT_DIR", "/opt/ml/processing/input"
    )
    output_dir = os.environ.get(
        "SM_OUTPUT_DIR", "/opt/ml/processing/output"
    )

    input_files = list(Path(input_dir).glob("*.csv"))
    if not input_files:
        input_files = list(Path(input_dir).glob("*.parquet"))

    if input_files:
        preprocessor = SageMakerPreprocessor()
        preprocessor.process(str(input_files[0]), output_dir)
