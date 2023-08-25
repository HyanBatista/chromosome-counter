import cv2 as cv
import os
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from tqdm import tqdm
from utils import get_image_paths


class BaseCounter(ABC):
    def __call__(
        self, source_dir: Path, output_file: Path, save=True
    ) -> dict[str, int]:
        image_paths: list[Path] = self._get_image_paths(source_dir)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        path_count_dict = self._count_from_images(image_paths)
        if save:
            self._save_result(path_count_dict, output_file)
        return path_count_dict

    def _count_from_images(self, image_paths: list[Path]) -> dict[str, int]:
        path_count_dict = {}
        for image_path in tqdm(image_paths):
            if image_path is None:
                continue
            count = self._count_from_image(image_path)
            path_count_dict[str(image_path)] = count
        return path_count_dict

    @abstractmethod
    def _count_from_image(self, image_path: Path) -> int:
        pass

    def _save_result(self, path_count_dict: dict[str, int], output_file: Path):
        with open(output_file, "w") as json_file:
            json.dump(path_count_dict, json_file, indent=4)

    def _get_image_paths(self, source_dir: Path) -> list[Path]:
        return get_image_paths(source_dir)


class ConnectedComponentsCounter(BaseCounter):
    def _count_from_image(self, image_path: Path) -> int:
        binary_image = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
        (num_labels, _, stats, _) = cv.connectedComponentsWithStats(
            binary_image, -1, 8
        )
        chromosome_count = 0
        for label in range(num_labels):
            area = stats[label, cv.CC_STAT_AREA]
            if area <= 5000 and area >= 30:
                chromosome_count += 1
        return chromosome_count
