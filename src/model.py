import os
import gc
import json
import pickle
import signal
from datetime import datetime
import warnings
from tqdm import tqdm
from uuid import uuid4
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms



import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from torch import nn
from torch import nn, optim
from torch.nn import (Sequential, Conv2d, MaxPool2d, ReLU, 
                      BatchNorm2d, Dropout, CrossEntropyLoss, 
                      AdaptiveAvgPool2d, Flatten, Linear)



if torch.cuda.is_available():
  device = "cuda"
else:
  device = "cpu"

print(f"Device: {device}")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"


# Restnet block(Residual network block)
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, identity_downsample = None, stride=1):
        super(Block, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, padding=0, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)

        self.gelu = nn.GELU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.gelu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.gelu(x)
        return x
    

class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, out_channels, expansion, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = image_channels
        self.conv1 = nn.Conv2d(self.in_channels, self.in_channels, kernel_size = 7, stride = 1, padding = 3) # output shape 224 -> When input shape is 224
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.gelu = nn.GELU()
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=3) # output shape 73x73

        # ResNet Layers
        self.layer1 = self._make_layer(block, layers[0], out_channels=out_channels, stride=1, expansion=expansion) # output 73
        self.layer2 = self._make_layer(block, layers[1], out_channels=out_channels * expansion//2, stride=2, expansion=expansion) # output 33
        self.layer3 = self._make_layer(block, layers[2], out_channels=out_channels * expansion, stride=2, expansion=expansion) # output 33
        self.layer4 = self._make_layer(block, layers[3], out_channels=out_channels * expansion * 2, stride=2, expansion=expansion) # output 33

        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        self.fc = nn.Linear(out_channels * expansion * 2 * expansion * 5 * 5, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1) # Reshaping before sending to the fully connected layer

        x = self.fc(x)
        return x

    def _make_layer(self, block, num_residual_blocks, out_channels, stride, expansion):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * expansion:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * expansion)
            )

        layers.append(block(self.in_channels, out_channels, expansion, identity_downsample, stride))
        self.in_channels = out_channels * expansion

        for i in range(1, num_residual_blocks):
            layers.append(block(self.in_channels, out_channels, expansion))

        return nn.Sequential(*layers)
    
# Creating ResNet 50(fifty layers)
class Resnet50(ResNet):
    def __init__(self, block, image_channels, out_channels, expansion, num_classes):
        super(Resnet50, self).__init__(block, [3, 4, 6, 3], image_channels, out_channels, expansion, num_classes)

    def train_model(self, optimizer, dataloader, criterion, scaler, device='cpu'):
        self.train() # put the model into train mode
        total_batches = len(dataloader)
        num_correct = 0
        train_acc = 0
        total_loss = 0
        total_images = 0 # images model has already seen
        batch_bar = tqdm(total=total_batches, dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=5)

        for index, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device) # moving data to same device with the model
            total_images += len(images)
            optimizer.zero_grad() # zeroing the previous gradients

            with torch.cuda.amp.autocast(): # allowing mixed precission during forward propagation
                logits = self(images) # forward pass
                loss = criterion(logits, labels) # calculating the loss/divergence

            # Find number of correct predictions and add them to previous correct total of predictions
            num_correct += int((torch.argmax(logits, axis=1) == labels).sum()) # make summation and cast it to integer
            train_acc = num_correct * 100 / total_images
            total_loss += float(loss.item()) # find the loss

            # Adding monitoring data to tqdm bar
            batch_bar.set_postfix(
                train_acc="{:.04f}%".format(train_acc),
                train_loss="{:.04f}".format(total_loss / (index + 1)),
                correct_preds="{}".format(num_correct),
                lr="{:.04f}".format(optimizer.param_groups[0]['lr'])
            )

            batch_bar.update()

            # Backward pass
            scaler.scale(loss).backward()
            # Gradient descent to update parameter
            scaler.step(optimizer)
            scaler.update()

            # Release some memory
            del images, labels, logits

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        batch_bar.close()
        total_loss /= total_batches

        return train_acc, total_loss

    def validate_model(self, dataloader, criterion, device='cpu'):
        self.eval() # put the model into evaluation mode
        total_batches = len(dataloader)
        num_correct = 0
        val_acc = 0
        val_loss = 0
        total_images = 0 # images model has already seen
        batch_bar = tqdm(total=total_batches, dynamic_ncols=True, leave=False, position=0, desc='Validation', ncols=5)

        for index, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            total_images += len(images)

            with torch.inference_mode(): # inferring
                logits = self(images) # forward pass
                loss = criterion(logits, labels) # Loss calculation

            num_correct += int((torch.argmax(logits, axis=1) == labels).sum())
            val_acc = num_correct * 100 / total_images
            val_loss = loss.item()

            # Adding monitoring data to tqdm bar
            batch_bar.set_postfix(
                train_acc="{:.04f}%".format(val_acc),
                train_loss="{:.04f}".format(val_loss / (index + 1)),
                correct_preds="{}".format(num_correct)
            )

            batch_bar.update()
            del images, labels, logits
            if torch.cuda.is_available():

                torch.cuda.empty_cache()

        batch_bar.close()
        val_loss /= total_batches

        return val_acc, val_loss


    def predict(self, dataloader, device='cpu', labels_names=None):
        self.eval() # put the model into evaluation mode
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Classifier Predict', ncols=5)
        prediction_results = []
        for _, images in enumerate(dataloader):
            if type(images) in [tuple, list]:
              images = images[0]
            images = images.to(device)
            with torch.inference_mode():
                logits = self(images)
            # detaching the from the computational graph and convert result to list

            logits = torch.argmax(logits, axis=1).detach().cpu().numpy().tolist()
            prediction_results.extend(logits)

            batch_bar.update()
            del logits

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        batch_bar.close()

        if labels_names: # return predicted class names
            predicted_classes = []
            for prediction in prediction_results:
              predicted_classes.append(labels_names[prediction])

            return prediction_results, predicted_classes

        return prediction_results, []