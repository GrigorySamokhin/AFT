"""Module with network code"""

import torch
import torchsummary
from torchvision import models

first_run = True


class FaceNet(torch.nn.Module):
    """Neural network class"""

    def __init__(
            self,
            num_of_output_nodes: int = 2,
            model='rn',
            use_head=True,
            freeze_n_block=0
    ):
        """
        Init class method

        :param model_type: type of model to use as backbone
        :param in_channels: number of input network channels
        :param num_of_output_nodes: number of output network nodes
        """
        super().__init__()

        if model == 'rn':

            self.model_ft = models.resnet50(pretrained=True)
            self.num_filters = self.model_ft.fc.in_features
            self.model_ft.fc = torch.nn.Linear(self.num_filters, num_of_output_nodes)
            if use_head:
                self.model_ft.fc = torch.nn.Sequential(
                    torch.nn.Linear(self.num_filters, 1024),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(p=0.3),
                    torch.nn.Linear(1024, num_of_output_nodes))

        if model == 'mb':
            self.model_ft = models.mobilenet_v2(pretrained=True)
            self.model_ft.classifier = torch.nn.Linear(1280, num_of_output_nodes)

        if freeze_n_block > 0:
            for i, param in enumerate(self.model_ft.parameters()):
                if i == freeze_n_block:
                    break
                param.requires_grad = False

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Performs forward pass threw network

        :param x: input tensor
        :return: network result
        """

        x = self.model_ft(x)

        return x


