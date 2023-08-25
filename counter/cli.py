from pathlib import Path

import click
from counters import ConnectedComponentsCounter
from preprocessing import Preprocessor
from segmentation import (
    FuzzyCMeansImageSegmenter,
    KMeansImageSegmenter,
    MeanShiftImageSegmenter,
)


@click.group()
def cli():
    pass


@click.command()
@click.option("--source_dir", prompt="Souce path", help="Original images source path.")
@click.option(
    "--target_dir", prompt="Target path", help="Preprocessed images target path."
)
def preprocess(source_dir: Path, target_dir: Path):
    preprocessor = Preprocessor()
    preprocessor(source_dir, target_dir)


@click.command()
@click.option("--source_dir", prompt="Souce path", help="Original images source path.")
@click.option(
    "--target_dir", prompt="Target path", help="Preprocessed images target path."
)
@click.option(
    "--method",
    default="kmeans",
    prompt="Segmentation method",
    help="Segmentation algorithm to be used.",
)
def segment(source_dir: Path, target_dir: Path, method: str):
    if method == "kmeans":
        segmenter = KMeansImageSegmenter()
    elif method == "shift":
        segmenter = MeanShiftImageSegmenter()
    elif method == "fuzzy":
        segmenter = FuzzyCMeansImageSegmenter()
    else:
        raise Exception("Algorithm not supported.")
    segmenter(source_dir, target_dir)


@click.command()
@click.option(
    "--source_dir",
    prompt="Source path",
    help="Path for the preprocessed/segmented image folder.",
)
@click.option(
    "--output_file", prompt="Output file path", help="Path for the output file."
)
def count(source_dir: Path, output_file: Path):
    counter = ConnectedComponentsCounter()
    counter(source_dir, output_file)


cli.add_command(preprocess)
cli.add_command(segment)
cli.add_command(count)


if __name__ == "__main__":
    cli()
