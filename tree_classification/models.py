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
        device = images.device
        out = self(images) # Generate predictions
        # Weights inversely proportional to class frequency
        class_weights_inv_freq = torch.tensor([16.3308, 19.5188, 21.4612, 22.6739, 24.8179, 24.4910, 35.0551, 37.3372, 33.2994, 39.6037, 
        43.3180, 51.9065, 48.3771,  50.1605,  52.3941,  67.4315,  60.1786,  57.4440, 73.4803, 58.2222, 64.7716, 83.9775, 66.5532, 73.7845,
        74.5828,  81.2869,  90.6673, 105.0896,  90.5999, 114.4741, 144.0915, 135.4631, 139.3379, 119.6280, 166.3679, 153.0237,
        112.6041, 156.9347, 217.6610, 115.1778, 138.6768, 123.6777, 134.1697, 186.0677, 141.5513, 214.4038, 129.2567, 121.2358,
        157.8842, 202.1826, 203.8181, 176.3669, 149.6698, 197.5366, 279.1016, 269.9254, 255.3068, 144.1483, 214.9715, 192.7937,
        198.5026, 200.6833, 310.7988, 208.7084, 387.4274, 366.0761, 271.8333, 164.0889, 280.9258, 336.7226, 372.9903, 315.3595,
        377.8118, 238.8650, 263.0266, 244.7046, 327.9569, 353.6728, 286.2076, 289.4960, 496.7288, 393.9019, 430.5763, 481.3492,
        330.1798, 598.4341, 334.2580, 521.5475, 363.5264, 431.8487, 471.1077, 583.1508, 1043.8400, 495.0461, 458.1116, 928.4473,
        688.0302, 477.5739, 900.9716, 578.9921], device=device)
        # Weights inversely proportion to square root of class frequency
        class_weights_sqrt_inv_freq = torch.tensor([4.0411, 4.4180, 4.6326, 4.7617, 4.9818, 4.9488, 5.9207, 6.1104,
        5.7706, 6.2931, 6.5816, 7.2046, 6.9554, 7.0824, 7.2384, 8.2117, 7.7575, 7.5792, 8.5721, 7.6303, 8.0481, 9.1639, 8.1580, 8.5898,
        8.6361, 9.0159, 9.5219, 10.2513, 9.5184, 10.6993, 12.0038, 11.6389, 11.8041, 10.9375, 12.8984, 12.3703, 10.6115, 12.5274, 14.7533, 10.7321,
        11.7761, 11.1210, 11.5832, 13.6407, 11.8975, 14.6425, 11.3691, 11.0107, 12.5652, 14.2191, 14.2765, 13.2803, 12.2340, 14.0548, 16.7063, 16.4294,
        15.9783, 12.0062, 14.6619, 13.8850, 14.0891, 14.1663, 17.6295, 14.4467, 19.6832, 19.1331, 16.4874, 12.8097, 16.7608, 18.3500, 19.3130, 17.7584,
        19.4374, 15.4553, 16.2181, 15.6430, 18.1096, 18.8062, 16.9177, 17.0146, 22.2874, 19.8470, 20.7503, 21.9397, 18.1709, 24.4629, 18.2827, 22.8374,
        19.0664, 20.7810, 21.7050, 24.1485, 32.3085, 22.2496, 21.4035, 30.4704, 26.2303, 21.8535, 30.0162, 24.0623], device=device)
        loss = F.cross_entropy(out, labels, label_smoothing = 0.05, weight=class_weights_sqrt_inv_freq) # Calculate loss
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

def save_checkpoint(model, epoch, optimizer, path="model_checkpoints"):
    if not os.path.exists(path):
        os.makedirs(path)
    filename = f"checkpoint_epoch_{epoch+1}.tar"
    filepath = os.path.join(path, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        }, filepath)
    print(f"Checkpoint saved: {filepath}")

# Do not compute new gradients when evaluating a model
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

# Fit model with FP16 mixed precision
def fit(epochs, lr, model, train_loader, val_loader, optimizer=None, 
        outpath="model_checkpoint", lr_patience=5, es_patience=10):

    history = []
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr)
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
            save_checkpoint(model, epoch, optimizer, outpath)
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
    optimizer = torch.optim.Adam(model.parameters())
    checkpoint = torch.load(classifier_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    return model, optimizer


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
