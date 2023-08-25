import os
import click
import cv2 as cv
from pathlib import Path
from preprocessing import binarize_image, erode_image, Preprocessor
from segmentation import (
    KMeansImageSegmenter,
    MeanShiftSegmenter,
    FuzzyCMeansSegmenter
)


@click.group()
def cli():
    pass

@click.command()
@click.option("--source_dir", prompt="Souce path", help="Original images source path.")
@click.option("--target_dir", prompt="Target path", help="Preprocessed images target path.")
def preprocess(source_dir: Path, target_dir: Path):
    preprocessor = Preprocessor()
    preprocessor(source_dir, target_dir)


@click.command()
@click.option("--source_dir", prompt="Souce path", help="Original images source path.")
@click.option("--target_dir", prompt="Target path", help="Preprocessed images target path.")
@click.option("--method", default="kmeans", prompt="Segmentation method", help="Segmentation algorithm to be used.")
def segment(source_dir: Path, target_dir: Path, method: str):
    if method == 'kmeans':
        segmenter = KMeansImageSegmenter()
    elif method == 'shift':
        segmenter = MeanShiftSegmenter()
    elif method == 'fuzzy':
        segmenter = FuzzyCMeansSegmenter()
    else:
        raise Exception("Algorithm not supported.")
    segmenter(source_dir, target_dir)


cli.add_command(preprocess)
cli.add_command(segment)


if __name__ == "__main__":
    cli()
