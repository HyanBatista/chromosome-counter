import os
from abc import ABC, abstractmethod
import skfuzzy as fuzz
from pathlib import Path

import cv2 as cv
import numpy as np
from tqdm import tqdm
from counter.utils import get_image_paths, write_images


class BaseImageSegmenter(ABC):
    def __call__(self, source_dir: Path, target_dir: Path) -> list[Path]:
        image_paths: list[Path] = self._get_image_paths(source_dir)
        os.makedirs(target_dir, exist_ok=True)
        target_image_paths = self._segment_images(image_paths, target_dir)
        return target_image_paths

    def _segment_images(
        self, image_paths: list[Path], target_dir: Path
    ) -> list[Path]:
        target_image_paths = []
        for image_path in tqdm(image_paths):
            if image_path is None:
                continue
            segmented_image = self._segment_image(image_path)
            target_image_path = os.path.join(target_dir, image_path.name)
            target_image_paths.append(target_image_path)
            write_images(
                [{"path": target_image_path, "image": segmented_image}]
            )
        return target_image_paths

    @abstractmethod
    def _segment_image(self, image_path: Path) -> np.ndarray:
        pass

    def _get_image_paths(self, source_dir: Path) -> list[Path]:
        return get_image_paths(source_dir)


class KMeansImageSegmenter(BaseImageSegmenter):
    def __init__(self, k=2):
        self.k = k

    def _segment_image(self, image_path: Path) -> np.ndarray:
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.85)
        image = cv.cvtColor(cv.imread(str(image_path)), cv.COLOR_BGR2RGB)
        reshaped = np.float32(image.reshape((-1, 3)))
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.85)
        _, labels, centers = cv.kmeans(
            reshaped, self.k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS
        )
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        segmented_image = segmented_data.reshape((image.shape))
        return segmented_image


class MeanShiftImageSegmenter(BaseImageSegmenter):
    def __init__(self, sp=30, sr=20):
        self.sp = sp
        self.sr = sr

    def _segment_image(self, image_path: Path) -> np.ndarray:
        image: np.ndarray = cv.imread(str(image_path))
        shifted_image = cv.pyrMeanShiftFiltering(
            src=image, sp=self.sp, sr=self.sr
        )
        return shifted_image


class FuzzyCMeansImageSegmenter(BaseImageSegmenter):
    def __init__(self, c=2, m=2, max_iter=100, error=0.005):
        self.c = c
        self.m = m
        self.max_iter = max_iter
        self.error = error

    def _segment_image(self, image_path: Path) -> np.ndarray:
        image: np.ndarray = cv.imread(str(image_path))
        data = image.reshape((-1, 3))
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data.T, self.c, self.m, error=self.error, maxiter=self.max_iter, init=None)
        cluster_membership = np.argmax(u, axis=0)
        segmented_image = cluster_membership.reshape((image.shape[0], image.shape[1]))
        return segmented_image
