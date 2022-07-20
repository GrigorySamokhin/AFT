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
from utils.logger import initialize_logging
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

    parser.add_argument('-i', '--input_dataset', type=str, required=False, default='data/hairstyle_dataset',
                        help="Path to input dataset. Note: need to be split on train/test/val sets")

    # Train
    parser.add_argument('-bs', '--batch_size', type=int, required=False, default=64,
                        help="Batch size for training networks")
    parser.add_argument('-is', '--input_size', type=int, required=False, default=256,
                        help="Input image size")
    parser.add_argument('-nm', '--num_epoch', type=int, required=False, default=100,
                        help="Number of epochs")
    # Data
    parser.add_argument('-wc', '--weight_classes', type=bool, required=False, default=True,
                        help="Enable use weights for unbalanced data")
    # Optimizer
    parser.add_argument('-lr', '--learning_rate', type=float, required=False, default=0.0005,
                        help="Learning rate")
    parser.add_argument('-lr_m', '--momentum', type=float, required=False, default=0.9,
                        help="Learning rate momentum")
    parser.add_argument('-wd', '--weight_decay', type=float, required=False, default=1e-3,
                        help="Weight decay")
    # Schedulers
    parser.add_argument('-uss', '--use_scheduler_step', type=bool, required=False, default=True,
                        help="Enable use sampler step_lr")
    parser.add_argument('-uss_ss', '--scheduler_step_size', type=int, required=False, default=7,
                        help="StepLR sampler step size")
    parser.add_argument('-uss_g', '--scheduler_step_gamma', type=float, required=False, default=0.1,
                        help="StepLR sampler gamma")
    # Log
    parser.add_argument("--log", type=str, default="process.log",
                        help="file name for processing log (relative to the root)")


    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    initialize_logging(
        logging_dir_path='./logs',
        logging_file_name=args.log,
        main_script_path=__file__,
        script_args=args,
        check_ffmpeg=True)

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

    sampler = None
    if args.weight_classes:
        weights = make_weights_for_balanced_classes(
            image_datasets['train'].imgs, len(image_datasets['train'].classes))
        weights = torch.DoubleTensor(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    dataloaders_dict = {
        'train': torch.utils.data.DataLoader(
            image_datasets['train'],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            sampler=sampler),
        'val': torch.utils.data.DataLoader(
            image_datasets['val'],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4),
        'test': torch.utils.data.DataLoader(
            image_datasets['test'],
            batch_size=args.batch_size // 2,
            shuffle=False,
            num_workers=4)
    }

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
    )

    scheduler = None
    if args.use_scheduler_step:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.scheduler_step_size,
            gamma=args.scheduler_step_gamma
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
        num_epochs=args.num_epoch,
        output_nodes=output_nodes,
        output_model_name=out_model_name_base)

    eval_model(
        model=model,
        dataloaders=dataloaders_dict,
        criterion=loss_func,
        device=device,
        num_epochs=1)
