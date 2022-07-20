"""
Module with training of networks
"""
import argparse
import os
import torch
import torchvision
from torch.utils.data import random_split

from utils.augs import get_data_transforms
from utils.dataloader import make_weights_for_balanced_classes
from utils.training import train_model, eval_model
from models.hairstylenet import FaceNet

torch.manual_seed(42)


def parse_args():
    """
    Parse python script parameters.

    Returns
    -------
    argparse.Namespace
        Resulted args.
    """

    parser = argparse.ArgumentParser(
        description="Main train script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '-i',
        '--input_dataset',
        type=str,
        required=False,
        default='data/hairstyle_dataset',
        help="Path to input dataset. Note: need to be split on train/test/val sets")
    parser.add_argument(
        '-bs',
        '--batch_size',
        type=int,
        required=False,
        default=32,
        help="Batch size for training networks")
    parser.add_argument(
        '-is',
        '--input_size',
        type=int,
        required=False,
        default=224,
        help="Input image size"
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        raise Warning("Train processing on CPU mode")

    data_transforms = get_data_transforms(args.input_size)

    image_datasets = {
        x: torchvision.datasets.ImageFolder(
            os.path.join(args.input_dataset, x),
            data_transforms[x]
        ) for x in ['train', 'val', 'test']
    }

    output_nodes = 3
    model = FaceNet(num_of_output_nodes=output_nodes)

    weights = make_weights_for_balanced_classes(
        image_datasets['train'].imgs, len(image_datasets['train'].classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    dataloaders_dict = {
        'train': torch.utils.data.DataLoader(
            image_datasets['train'],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True),
        'val': torch.utils.data.DataLoader(
            image_datasets['val'],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4),
        'test': torch.utils.data.DataLoader(
            image_datasets['test'],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4)
    }

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.0001,
        momentum=0.9,
        weight_decay=1e-8,
        nesterov=True
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=7,
        gamma=0.1
    )

    loss_func = torch.nn.CrossEntropyLoss()

    out_model_name_base = ''
    model, val_acc_history = train_model(
        model=model,
        dataloaders=dataloaders_dict,
        criterion=loss_func,
        optimizer=optimizer,
        device=device,
        sheduler=None,
        num_epochs=29,
        output_nodes=output_nodes,
        output_model_name=out_model_name_base)

    eval_model(
        model=model,
        dataloaders=dataloaders_dict,
        criterion=loss_func,
        device=device,
        num_epochs=1)
