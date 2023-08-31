from pathlib import Path

import click
import pandas as pd
from counter.counters import ConnectedComponentsCounter
from counter.preprocessing import Preprocessor
from counter.segmentation import (
    FuzzyCMeansImageSegmenter,
    KMeansImageSegmenter,
    MeanShiftImageSegmenter,
)
from counter.validation import SimpleCounterValidator


@click.group()
def cli():
    pass


@click.command()
@click.option(
    "--source_dir", prompt="Souce path", help="Original images source path."
)
@click.option(
    "--target_dir",
    prompt="Target path",
    help="Preprocessed images target path.",
)
def preprocess(source_dir: Path, target_dir: Path):
    preprocessor = Preprocessor()
    preprocessor(source_dir, target_dir)


@click.command()
@click.option(
    "--source_dir", prompt="Souce path", help="Original images source path."
)
@click.option(
    "--target_dir",
    prompt="Target path",
    help="Preprocessed images target path.",
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


@click.command()
@click.option(
    "--annotations",
    prompt="Annotations file",
    help="Path for the annotations file.",
)
@click.option(
    "--counts",
    prompt="Count file",
    help="File holding the count of chromosomes for each image.",
)
@click.option(
    "--output_file",
    prompt="Output file path.",
    help="Output file path for the validation data.",
)
def validate(annotations: Path, counts: Path, output_file: Path):
    validator = SimpleCounterValidator()
    mse = validator(annotations, counts, output_file)
    print(f"[INFO] {mse}")


@click.command()
@click.option(
    "--path",
    prompt="Validation data path.",
    help="Preview some of the validation data.",
)
def show(path: Path):
    validation_data = pd.read_parquet(path)
    print(validation_data.head())


cli.add_command(preprocess)
cli.add_command(segment)
cli.add_command(count)
cli.add_command(validate)
cli.add_command(show)


if __name__ == "__main__":
    cli()
