import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from utils import read_json


class BaseCounterValidator(ABC):
    def __call__(self, annotations_file_path: Path, count_file_path: Path, output_file: Path) -> float:
        validation_data = self._create_validation_data(
            annotations_file_path, count_file_path
        )
        validation_data.to_parquet(output_file)
        result = self._validate(validation_data)
        return result

    @abstractmethod
    def _validate(self, validation_data: pd.DataFrame):
        pass

    def _create_validation_data(
        self, annotations_file_path: Path, count_file_path: Path
    ) -> pd.DataFrame:
        count_dict = read_json(count_file_path)
        annotations_df = pd.read_parquet(annotations_file_path)
        annotations_df["found"] = annotations_df["image"].apply(
            lambda image: int(count_dict[image])
        )
        return annotations_df


class SimpleCounterValidator(BaseCounterValidator):
    def _validate(self, validation_data: pd.DataFrame):
        columns = validation_data.columns
        assert "count" in columns and "found" in columns
        y = validation_data["count"]
        y_pred = validation_data["found"]
        return mean_squared_error(np.array(y), np.array(y_pred))
