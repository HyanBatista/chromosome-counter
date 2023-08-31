import json
import os
from pathlib import Path

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def get_image_paths(source: Path) -> list[Path]:
    image_paths = []
    for filename in os.listdir(source):
        if filename.endswith(".tif") or filename.endswith(".png"):
            image_paths.append(Path(os.path.join(source, filename)))
    return image_paths


def write_images(path_image_dicts: list[dict[np.ndarray, Path]]):
    for path_image_dict in path_image_dicts:
        path = path_image_dict["path"].replace('.tif', '.png')
        plt.imsave(path, path_image_dict["image"], cmap='viridis')


def read_json(path: Path) -> dict:
    with open(path, "r") as f:
        data = json.load(f)
    return data
