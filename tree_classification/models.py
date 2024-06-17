# Host Tree Classification CNN EfficientNet Model Module

# Imports
# Imports for Pytorch
import torch # version 2.1.2
import torchvision # version 0.16.2
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision.utils import make_grid
from torchvision.transforms import v2
import torch.nn as nn # contains base class for all neural network modules
import torch.nn.functional as F #https://pytorch.org/docs/stable/nn.functional.html contains common functions for training NNs (convolutions, losses, etc..)
from sklearn.metrics import classification_report
import torch.cuda.amp as amp

# Image processing and display
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.image import imread

# Other Imports
import os
import shutil
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import time
from imageio import imread

# Setup EfficientNetV2 Model to Run Experiments

class ImageClassificationBase(nn.Module): # https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module
    # Define a base class with functionality for model training, validation, and evaluation per epoch
    
    def training_step(self, batch):
        images, labels = batch
        out = self(images) # Generate predictions
        loss = F.cross_entropy(out, labels, label_smoothing = 0.05) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images) # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        acc = accuracy(out, labels) # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# Functions to define a CNN Model using EfficientNetV2-S

class EfficientNetImageClassification(ImageClassificationBase):
    def __init__(self, num_classes):
        super().__init__()
        # Load the pre-trained EfficientNetV2-L Model
        self.network = torchvision.models.efficientnet_v2_s(pretrained=True)
        # Modify the final fully connected layer to match the number of classes in your dataset
        in_features = self.network.classifier[1].in_features
        self.network.classifier = nn.Linear(in_features, num_classes)

    def forward(self, xb):
        return self.network(xb)

# Functions  to visualize training data
def display_img(img,label):
    print(f"Label : {train_dataset.classes[label]}")
    plt.imshow(img.permute(1,2,0)) #reshape image from (3, H, W) to (H, W, 3)

def show_batch(dl):
    """Plot images grid of single batch"""
    for images, labels in dl:
        fig,ax = plt.subplots(figsize = (16,12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images,nrow=16).permute(1,2,0))
        break

# Functions to define GPU device and load data to GPU
def get_default_device():
    """ Set Device to GPU or CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    "Move data to the device"
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking = True)

class DeviceDataLoader():
    """ Wrap a dataloader to move data to a device """
    
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    
    def __iter__(self):
        """ Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b,self.device)
            
    def __len__(self):
        """ Number of batches """
        return len(self.dl)

def save_checkpoint(model, epoch, path="model_checkpoints"):
    if not os.path.exists(path):
        os.makedirs(path)
    filename = f"checkpoint_epoch_{epoch+1}.pth"
    filepath = os.path.join(path, filename)
    torch.save(model.state_dict(), filepath)
    print(f"Checkpoint saved: {filepath}")

# Do not compute new gradients when evaluating a model
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

# Fit model with FP16 mixed precision
def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam, 
        outpath="model_checkpoint", lr_patience=5, es_patience=10):

    history = []
    optimizer = opt_func(model.parameters(), lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=lr_patience, verbose=True)
    scaler = amp.GradScaler()

    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        for batch in train_loader:
            # Forward pass: prediction & calculate loss. Run with autoscaler for fp16.
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss = model.training_step(batch)
                train_losses.append(loss)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
        scheduler.step(result['val_loss'])
        
        # Print the learning rate
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            print(f"Learning Rate: {current_lr}")
            
        # Early stopping
        val_loss = result['val_loss']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            save_checkpoint(model, epoch, outpath)
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= es_patience:
                print(f"Early stopping triggered after {es_patience} consecutive epochs of no improvement.")
                save_checkpoint(model, epoch, outpath)
                break
        
    return history


def load_classifier_model(classifier_path: str, selected_genera: list) -> torch.nn.Module:
    """Load the tree classification model."""
    device = get_default_device()
    num_classes = len(selected_genera)
    model = to_device(EfficientNetImageClassification(num_classes), device)
    model.load_state_dict(torch.load(classifier_path))
    model.eval()
    return model


# Functions to visualize the results
def plot_accuracies(history, outpath):
    """ Plot the history of accuracies"""
    outpath = os.path.join(outpath, 'model_accuracy.png')
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x');
    plt.xlabel('epoch');
    plt.ylabel('validation accuracy');
    plt.title('Accuracy vs. No. of epochs');
    # Reduce plot margins
    plt.autoscale();
    plt.margins(0.2);
    plt.savefig(outpath, dpi=300);
    plt.close();

def plot_losses(history, outpath):
    """ Plot the losses in each epoch"""
    outpath = os.path.join(outpath, 'model_losses.png')
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx');
    plt.plot(val_losses, '-rx');
    plt.xlabel('epoch');
    plt.ylabel('loss');
    plt.legend(['Training', 'Validation']);
    plt.title('Loss vs. No. of epochs');
    # Reduce plot margins
    plt.autoscale();
    plt.margins(0.2);
    plt.savefig(outpath, dpi=300);
    plt.close();