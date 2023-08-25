import os
import cv2 as cv
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from tqdm import tqdm
from utils import get_image_paths, write_images


class ImageSegmenter(ABC):
    def __call__(self, source_dir: Path, target_dir: Path) -> list[Path]:
        image_paths: list[Path] = self._get_image_paths(source_dir)
        os.makedirs(target_dir, exist_ok=True)
        target_image_paths = self._segment_images(image_paths, target_dir)
        return target_image_paths

    def _segment_images(self, image_paths: list[Path], target_dir: Path) -> list[Path]:
        target_image_paths = []
        for image_path in tqdm(image_paths):
            if image_path is None:
                continue
            segmented_image = self._segment_image(image_path)
            target_image_path = os.path.join(target_dir, image_path.name)
            target_image_paths.append(target_image_path)
            write_images([{"path": target_image_path, "image": segmented_image}])
        return target_image_paths

    @abstractmethod
    def _segment_image(self, image_path: Path) -> np.ndarray:
        pass

    def _get_image_paths(self, source_dir: Path) -> list[Path]:
        return get_image_paths(source_dir)


class KMeansImageSegmenter(ImageSegmenter):    
    def _segment_image(self, image_path: Path, k=3) -> np.ndarray:
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.85)
        image = cv.cvtColor(cv.imread(str(image_path)), cv.COLOR_BGR2RGB)
        reshaped = np.float32(image.reshape((-1, 3)))
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.85)
        _, labels, centers = cv.kmeans(
            reshaped, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS
        )
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        segmented_image = segmented_data.reshape((image.shape))
        return segmented_image
