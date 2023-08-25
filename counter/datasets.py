import torchvision.transforms.functional as TF
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CRCNNEDataset(Dataset):
    def __init__(self, image_directory: Path, annotations_file: Path, transform=None, transform_params=None):
        self.image_directory = image_directory
        self.annotations = pd.read_parquet(annotations_file)
        self.transform = transform
        self.transform_params = transform_params
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_directory, self.annotations.iloc[index]['image'])
        image = Image.open(image_path).convert('RGB')
        labels = self.annotations.iloc[index]['labels']
        return image, labels, image_path


if __name__ == '__main__':
    dataset = CRCNNEDataset('data/original_images/', 'data/annotations.parquet')
    image, labels, image_path = dataset[0]
    print('dataset (length):', len(dataset))
    print('image (path) (size) (type):', image_path, image.size, type(image))
    print('labels (shape):', labels.shape)
