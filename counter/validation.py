from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class BaseValidator(ABC):
    @abstractmethod
    def validate_count(self):
        pass

    def _create_ground_truth_dict(self, annotations_path: Path):
        annotations_df = pd.read_parquet(annotations_path)


class SimpleValidator(BaseValidator):
    pass
