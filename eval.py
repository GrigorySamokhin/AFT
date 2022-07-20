"""Module with training of networks"""

import argparse
import os

import torch
import torchvision

from utils.logger import initialize_logging
from utils.training import eval_model
from models.hairstylenet import FaceNet


def parse_args():
    """Parses arguments for CLI"""

    parser = argparse.ArgumentParser(description=f'Starts training script')

    parser.add_argument('--config-path', default='config.ini', type=str,
                        help='Path to config file')
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

    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_name = 'test_selfie_wo_seg'

    input_size = 256
    data_transforms = {
        set_name: torchvision.transforms.Compose([
            torchvision.transforms.Resize(input_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'data/hairstyle_dataset'
    image_datasets = {
        x: torchvision.datasets.ImageFolder(
            os.path.join(data_dir, x),
            data_transforms[x]
        ) for x in [set_name]
    }

    dataloaders_dict = {
        set_name:
            torch.utils.data.DataLoader(image_datasets[set_name],
                                        batch_size=32,
                                        shuffle=False,
                                        num_workers=4)  # for Kaggle
    }

    model = FaceNet(num_of_output_nodes=len(image_datasets[set_name].classes))

    load_torch = torch.load('weights/best_val_acc_weights.h5')
    load_torch_new = load_torch.copy()

    for key in load_torch:
        new_key = key[9:]
        load_torch_new[new_key] = load_torch[key]
        del load_torch_new[key]

    model.load_state_dict(load_torch)

    # optimizer_ft = torch.optim.RMSprop(
    #     model.parameters(),
    #     lr=1e-4
    # )

    optimizer_ft = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    loss_func = torch.nn.CrossEntropyLoss()

    eval_model(
        model=model,
        dataloaders=dataloaders_dict,
        criterion=loss_func,
        device=device,
        num_epochs=1,
        set_name=set_name
    )


