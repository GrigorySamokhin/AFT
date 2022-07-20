"""Module with augmentations"""
import PIL.Image
import numpy as np
import torchvision
from albumentations import (RandomRotate90, GlassBlur,
                            ShiftScaleRotate,
                            GaussNoise, Compose,
                            Flip,
                            Normalize, HorizontalFlip, RandomBrightnessContrast)
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.transforms import RandomRotation, AutoAugment, AutoAugmentPolicy


# TODO rewrite to imgaug
def get_data_transforms(input_size: int):
    return {
        'train': torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(input_size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': torchvision.transforms.Compose([
            torchvision.transforms.Resize(input_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': torchvision.transforms.Compose([
            torchvision.transforms.Resize(input_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }