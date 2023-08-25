import cv2 as cv
import os
import numpy as np
from pathlib import Path


def get_image_paths(source: Path) -> list[Path]:
    image_paths = []
    for filename in os.listdir(source):
        if filename.endswith(".tif"):
            image_paths.append(Path(os.path.join(source, filename)))
    return image_paths


def write_images(path_image_dicts: list[dict[np.ndarray, Path]]):
    for path_image_dict in path_image_dicts:
        cv.imwrite(path_image_dict['path'], path_image_dict['image'])
