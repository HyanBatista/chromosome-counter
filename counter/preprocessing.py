import os
from pathlib import Path

import cv2 as cv
import numpy as np
from numpy import ndarray
from tqdm import tqdm
from counter.utils import get_image_paths


def binarize_image(image: ndarray) -> ndarray:
    blurred_image = cv.GaussianBlur(image, (5, 5), 0)
    _, binary_image = cv.threshold(
        blurred_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU
    )
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    opening = cv.morphologyEx(binary_image, cv.MORPH_OPEN, kernel)
    binary_image = cv.cvtColor(opening, cv.COLOR_GRAY2BGR)
    return binary_image


def erode_image(image: ndarray, iterations:np.uint8 = 1) -> ndarray:
    eroded_image = image

    for _ in range(iterations):
        kernel1 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)
        eroded_image = cv.erode(eroded_image, kernel1, iterations=1)
        kernel2 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
        eroded_image = cv.erode(eroded_image, kernel2, iterations=5)

    return eroded_image


class Preprocessor:
    def __call__(self, source: Path, target: Path) -> list[Path]:
        image_paths: list[Path] = self._get_image_paths(source)
        os.makedirs(target, exist_ok=True)
        target_image_paths = self._preprocess(image_paths, target)
        return target_image_paths

    def _preprocess(self, image_paths: list[Path], target: Path) -> list[Path]:
        target_image_paths = []
        for image_path in tqdm(image_paths, desc="Preprocessing", unit="iteration"):
            image = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
            if image is None:
                continue
            binary_image = binarize_image(image)
            eroded_image = erode_image(binary_image, 2)
            processed_image_filename = os.path.basename(image_path)
            processed_image_path = os.path.join(target, processed_image_filename)
            target_image_paths.append(processed_image_path)
            cv.imwrite(processed_image_path, eroded_image)
        return target_image_paths

    def _get_image_paths(self, source: Path) -> list[Path]:
        return get_image_paths(source)
