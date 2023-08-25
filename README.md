# Chromosome Counter
Sistema de processamento de imagens para contagem de cromossomos.

# Visão Geral do Projeto

# Pré-requisitos

- Python 3.10

# Instalação

```
pip install -r requirements.txt
```

# Uso

## Preprocessing images
```
python counter/cli.py preprocess --source_dir=./data/ori
ginal_images --target_dir=./data/preprocessed
```

## Apply segmentation to a set of images
Opções de algoritmo de segmentação:
- **Mean-Shift:** shift
- **KMeans:** kmeans
- **Fuzzy C-Means:** fuzzy

```
python counter/cli.py preprocess --source_dir=./data/original_images --target_dir=./data/preprocessed
```

## Validate results
```
python counter/cli.py validate --annotations=./data/annotations.parquet --counts=./data/count_result.json --output_file=./data/validation_data.parquet
```

## Preview some of the validation data
```
python counter/cli.py show --path=./data/validation_data.parquet
```

## Count the chromosomes
```
python counter/cli.py count --source_dir=./data/processed_images --output_file=./data/count_result.json
```

# References
- [OpenCV: Image Thresholding](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)
- [K-means image segmentation using OpenCV - medium](https://medium.com/towardssingularity/k-means-clustering-for-image-segmentation-using-opencv-in-python-17178ce3d6f3)
- [K-means image segmentation using OpenCV - kdnuggets](https://www.kdnuggets.com/2019/08/introduction-image-segmentation-k-means-clustering.html)
- [Image segmentation with Mean-Shift](https://stackoverflow.com/questions/62575894/how-to-find-clusters-in-image-using-mean-shift-in-python-opencv)
- [OpenCV's Mean-Shift filtering method](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga9fabdce9543bd602445f5db3827e4cc0)
- [FCM Implementations](https://github.com/jeongHwarr/various_FCM_segmentation)
