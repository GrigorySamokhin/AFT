"""Module with functions for dataset manipulations"""
import math
import os
import glob
from typing import Tuple, Callable, Dict, Optional, List

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from .dataset import ImageDataset


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def load_data(data_dir, batch_size, train_transforms, valid_transforms):

    validation_size = 0.15
    test_size = 0.15
    train_size = 0.70

    full_dataset = torchvision.datasets.ImageFolder(data_dir)
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset=full_dataset,
        lengths=[math.floor(len(full_dataset) * train_size),
                 math.ceil(len(full_dataset) * validation_size),
                 math.floor(len(full_dataset) * test_size)],
        generator=torch.Generator().manual_seed(42))

    train_set.dataset.transform = train_transforms
    val_set.dataset.transform = valid_transforms
    test_set.dataset.transform = valid_transforms


    # weights = make_weights_for_balanced_classes(train_set.dataset.imgs, len(train_set.dataset.classes))
    # weights = torch.DoubleTensor(weights)
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    # train_classes = [label for _, label in train_set]
    # if True:
    #     # Need to get weight for every image in the dataset
    #     class_count = Counter(train_classes)
    #     class_weights = torch.Tensor([len(train_classes)/c for c in pd.Series(class_count).sort_index().values])
    #     # Can't iterate over class_count because dictionary is unordered
    #
    #     sample_weights = [0] * len(train_set)
    #     for idx, (image, label) in enumerate(train_set):
    #         class_weight = class_weights[label]
    #         sample_weights[idx] = class_weight
    #
    #     sampler = WeightedRandomSampler(weights=sample_weights,
    #                                     num_samples=len(train_set),
    #                                     replacement=True)
    #     train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler)

    train_loader = DataLoader(train_set, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, val_loader, test_loader

def create_dataloader(dataset: ImageDataset, batch_size: int = 32) -> DataLoader:
    """
    Creates dataloader from dataset

    :param dataset: dataset with images
    :param batch_size: size of batches
    :return: resulted dataloader
    """

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=True)

    return dataloader


def get_image_paths_labels(
        images_root: str,
        short_hair_folder: str,
        long_hair_folder: str,
        ponitail_hair_folder: str,
        class_labels: Dict[str, int],
        file_paths_to_exclude: Optional[List[str]] = None
) -> List[Tuple[str, int]]:
    """
    Gets image paths and labels from folder with images

    :param images_root: root folder with image folders for each class
    :param short_hair_folder: short hair folder name
    :param long_hair_folder: long hair folder name
    :param class_labels: classes of labels
    :param file_paths_to_exclude: paths to exclude from list with paths
    :return: list with paths and labels
    """

    short_hair_images_pattern = os.path.join(images_root, short_hair_folder, '*.png')
    short_hair_image_paths = set(glob.glob(short_hair_images_pattern))

    long_hair_images_pattern = os.path.join(images_root, long_hair_folder, '*.png')
    long_hair_image_paths = set(glob.glob(long_hair_images_pattern))

    ponitail_hair_images_pattern = os.path.join(images_root, ponitail_hair_folder, '*.png')
    ponitail_hair_image_paths = set(glob.glob(ponitail_hair_images_pattern))

    short_hair_image_paths_labels = list(zip(
        short_hair_image_paths,
        [class_labels[short_hair_folder]] * len(short_hair_image_paths)
    ))

    long_hair_image_paths_labels = list(zip(
        long_hair_image_paths,
        [class_labels[long_hair_folder]] * len(long_hair_image_paths)
    ))

    ponitail_hair_image_paths_labels = list(zip(
        ponitail_hair_image_paths,
        [class_labels[ponitail_hair_folder]] * len(ponitail_hair_image_paths)
    ))

    image_paths_labels = short_hair_image_paths_labels + long_hair_image_paths_labels \
                         + ponitail_hair_image_paths_labels

    if file_paths_to_exclude:
        image_paths_labels = [curr_image_path_label
                              for curr_image_path_label in image_paths_labels
                              if curr_image_path_label[0] not in file_paths_to_exclude]

    return image_paths_labels


def get_split_datasets(
        images_root: str,
        short_hair_folder: str,
        long_hair_folder: str,
        ponytail_hair_folder: str,
        train_augmentations: Optional[Callable] = None,
        valid_augmentations: Optional[Callable] = None,
        file_paths_to_exclude: Optional[List[str]] = None,
        image_size: Tuple[int, int] = (256, 256), valid_size: float = 0.3
) -> Tuple[ImageDataset,
           ImageDataset]:
    """
    Shuffles dataframe and creates train and valid datasets with it

    :param images_root: root folder with image folders for each class
    :param short_hair_folder: short hair folder name
    :param long_hair_folder: long hair folder name
    :param train_augmentations: training augs
    :param valid_augmentations: validation augs
    :param file_paths_to_exclude: paths to exclude from list with paths
    :param image_size: size of image
    :param valid_size: size of validation part
    :return: train_dataset, valid_dataset
    """

    class_labels = {
        short_hair_folder: 0,
        long_hair_folder: 1,
        ponytail_hair_folder: 2
    }

    image_paths_labels = get_image_paths_labels(
        images_root=images_root,
        short_hair_folder=short_hair_folder,
        long_hair_folder=long_hair_folder,
        ponitail_hair_folder=ponytail_hair_folder,
        class_labels=class_labels,
        file_paths_to_exclude=file_paths_to_exclude
    )

    np.random.shuffle(image_paths_labels)

    split_idx = round(valid_size * len(image_paths_labels))

    image_paths_labels_train = image_paths_labels[:-split_idx]
    image_paths_labels_valid = image_paths_labels[-split_idx:]

    train_dataset = ImageDataset(image_paths_labels=image_paths_labels_train,
                                 augmentations=train_augmentations,
                                 image_size=image_size)
    valid_dataset = ImageDataset(image_paths_labels=image_paths_labels_valid,
                                 augmentations=valid_augmentations,
                                 image_size=image_size)

    return train_dataset, valid_dataset
