import os
import click
import cv2 as cv
from pathlib import Path
from preprocessing import binarize_image, erode_image, Preprocessor
from segmentation import KMeansImageSegmenter


@click.group()
def cli():
    pass

@click.command()
@click.option("--source", prompt="Souce path", help="Original images source path.")
@click.option("--target", prompt="Target path", help="Preprocessed images target path.")
def preprocess(source: Path, target: Path):
    preprocessor = Preprocessor()
    preprocessor(source, target)


@click.command()
@click.option("--source", prompt="Souce path", help="Original images source path.")
@click.option("--target", prompt="Target path", help="Preprocessed images target path.")
def segment(source: Path, target: Path):
    segmenter = KMeansImageSegmenter()
    segmenter(source, target)


cli.add_command(preprocess)
cli.add_command(segment)


if __name__ == "__main__":
    cli()
