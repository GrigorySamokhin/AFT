"""Module with network training"""
import logging
import os.path
import sys
from datetime import datetime
from typing import List, Dict, Optional

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from .metrics import get_accuracy, get_f1_score
from .utils import print_report, save_weights, save_report
import time
import copy
from models.hairstylenet import FaceNet


def train_model(model,
                   dataloaders,
                   criterion,
                   optimizer,
                   num_epochs=25,
                   device='cpu',
                   sheduler=None,
                   output_nodes=3,
                   output_model_name="model",
                   is_inception=False):
    since = time.time()

    val_acc_history = []

    model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        logging.info('train epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train' and sheduler is not None:
                sheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            logging.info('{} loss: {:.4f}, acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                model_save = FaceNet(
                    num_of_output_nodes=output_nodes,
                )
                model_save.load_state_dict(best_model_wts)
                if not os.path.exists('weights'):
                    os.mkdir('weights')
                torch.save(model.state_dict(), 'weights/best_val_acc_weights.h5')
                logging.info('debug: save weights.')
            if phase == 'val':
                val_acc_history.append(epoch_acc)


    time_elapsed = time.time() - since
    logging.info('training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logging.info('best validation accuracy: {:4f}\n'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def eval_model(model, dataloaders, criterion, num_epochs=25, device='cpu', set_name='test'):
    since = time.time()

    model.to(device)
    model.eval()

    running_corrects = 0
    for epoch in range(num_epochs):
        logging.info('{} epoch {}/{}'.format(set_name, epoch, num_epochs - 1))
        logging.info('-' * 10)

        for inputs, labels in dataloaders[set_name]:

            inputs = inputs.to(device)
            labels = labels.to(device)


            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)

    epoch_acc = running_corrects.double() / len(dataloaders[set_name].dataset)

    logging.info('{} evaluation accuracy: {}'.format(set_name, epoch_acc))


