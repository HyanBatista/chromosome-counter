from numpy import ndarray
import cv2 as cv
import os
from matplotlib import pyplot as plt
import numpy as np

def binarize_image(image: ndarray) -> ndarray:
    blurred_image = cv.GaussianBlur(image, (5, 5), 0)
    _, binary_image = cv.threshold(blurred_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    opening = cv.morphologyEx(binary_image, cv.MORPH_OPEN, kernel)
    inverted_image = 255 - binary_image

    return inverted_image

def erode_image(image: ndarray) -> ndarray:
    kernel1 = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ], np.uint8)

    eroded_image = cv.erode(image, kernel1, iterations=1)

    kernel2 = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], np.uint8)

    eroded_image = cv.erode(eroded_image, kernel2, iterations=5)

    return eroded_image

original_images_path = './data/original_images'
processed_images_path = './data/processed_images'

image_paths = []

for filename in os.listdir(original_images_path):
    if filename.endswith('.tif'):
        image_paths.append(os.path.join(original_images_path, filename))

os.makedirs(processed_images_path, exist_ok=True)

for image_path in image_paths:
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if image is not None:
        binarized_image = binarize_image(image)
        eroded_image = erode_image(binarized_image)
        processed_image_filename = os.path.basename(image_path)
        processed_image_path = os.path.join(processed_images_path, processed_image_filename)
        cv.imwrite(processed_image_path, eroded_image)
