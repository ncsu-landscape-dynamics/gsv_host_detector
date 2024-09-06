# Host Tree Classification Training Module

# Thomas Lake June 2024

# Imports

# Imports for Pytorch
import torch # version 2.1.2
import torch.nn as nn
import torch.nn.functional as F
import torchvision # version 0.16.2
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision.utils import make_grid
from torchvision.transforms import v2
import torch.cuda.amp as amp # mixed precision training

# Image processing and display
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.image import imread

# Other Imports
import os
import shutil
import random
import numpy as np
import pandas as pd
import time
import argparse
import logging
import json
from imageio import imread
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Imports for Host Tree Classification
from tree_classification import models, preprocess, utilities
from tree_classification.models import *
from tree_classification.utilities import *

# Set up logging
def setup_logging(log_path):
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

# Define main function to run classification model
def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train a tree classification model with the specified datasets.")

    # Paths for input datasets
    parser.add_argument('--config', type=str, required=True, 
                        help="Path to the configuration file")

    # Parse arguments
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    # File Path Args
    experiment = config['experiment']
    train_data_dir = config['train_data_dir']
    test_data_dir = config['test_data_dir']
    output_path = config['output_path']

    # Model Parameters
    bs = config['batch_size']
    epochs = config['epochs']
    lr = config['learning_rate']
    lr_patience = config.get('lr_patience', 5)
    es_patience = config.get('es_patience', 10)

    # Select Genera for Classification
    selected_genera = config['selected_genera']

    # Outputs
    model_weights_path = os.path.join(output_path, '{}_checkpoint_epoch_last.pth'.format(experiment))
    model_history_path = os.path.join(output_path, '{}_model_history.csv'.format(experiment))
    confusion_matrix_path = os.path.join(output_path, '{}_confusion_matrix.png'.format(experiment))
    classification_report_path = os.path.join(output_path, '{}_classification_report.csv'.format(experiment))

    # Create output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Setup Logging
    log_path = os.path.join(output_path, f'log-{experiment}.log')
    setup_logging(log_path)
    logging.info(f"Configuration File: {config}")
    logging.info("Starting Model Training")
    
    # Data Augmentations

    train_transforms = v2.Compose([
        v2.ToImage(),
        v2.CenterCrop(size=(768, 768)),
        v2.RandomResizedCrop(size=(512, 512), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness = (0.1), contrast = (0.1), saturation = (0.1), hue = (0.01)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.5712, 0.6121, 0.5827), std=(0.2224, 0.2136, 0.2586)),
        v2.RandomAffine(degrees=(-5,5), shear = (0, 0, -5, 5), 
                        interpolation = v2.InterpolationMode.BILINEAR, fill=0),
    ])

    # Log training transformations
    for transform in train_transforms.transforms:
            logging.info(f"  - {transform}\n")

    # Ensure images are resized during testing as same dimension for training
    test_transforms = v2.Compose([
        v2.ToImage(),
        v2.CenterCrop(size=(768, 768)),
        v2.Resize(size=(512, 512), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.5712, 0.6121, 0.5827), std=(0.2224, 0.2136, 0.2586))
    ])

    # Create Image Folder Datasets

    # Use Pytorch ImageFolder class to prepare training and testing datasets
    train_dataset = ImageFolder(train_data_dir, transform = train_transforms)
    test_dataset = ImageFolder(test_data_dir, transform = test_transforms)

    # How many classes are in the training and testing datasets?
    logging.info(f"Classes in the Training Dataset: {len(train_dataset.classes)}")
    logging.info(f"Classes in the Testing Dataset: {len(test_dataset.classes)}")
    logging.info(f"Training Genera: {selected_genera}")
    num_classes = len(train_dataset.classes)

    # Define number of images for validation (10% of the training set)
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size

    # Randomly split training data into train_data and val_data sets
    train_data, val_data = random_split(train_dataset, [train_size, val_size])

    logging.info(f"Length of Train Data: {len(train_data)}")
    logging.info(f"Length of Validation Data: {len(val_data)}")

    # Create DataLoaders
    train_dl = DataLoader(dataset = train_data, batch_size = bs, shuffle = True, num_workers = 4, pin_memory = True)
    val_dl = DataLoader(dataset = val_data, batch_size = bs, num_workers = 4, pin_memory = True)
    test_dl = DataLoader(dataset = test_dataset, batch_size = 1, num_workers = 4, pin_memory = True)

    # Get GPU Device
    device = get_default_device()
    print(device)

    # Load data to GPU
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)

    # Load the model to the GPU
    model = to_device(EfficientNetImageClassification(num_classes), device)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Number of Model Parameters: ', params)

    # Print the experiment details
    print(f"Starting Model Training: Experiment '{experiment}'")
    print(f"Train Data Directory: {train_data_dir}")
    print(f"Test Data Directory: {test_data_dir}")
    print(f"Batch Size: {bs}")
    print(f"Number of Epochs: {epochs}")
    print(f"Learning Rate: {lr}")
    print(f"LR Decay Patience: {lr_patience}")
    print(f"Early Stopping Patience: {es_patience}")
    print(f"Validation Dataset Size: {val_size}")
    print(f"Output Path: {output_path}")
    print("Starting Model:")
    
    start_time = time.time()
    
    # Fit model and record result after epoch
    history = fit(epochs, lr, model, train_dl, val_dl, opt_func = torch.optim.Adam, outpath = output_path, lr_patience = lr_patience, es_patience = es_patience)
    
    elapsed_time = time.time() - start_time
    
    logging.info(f"Total Model Training Time: {elapsed_time:.2f} seconds")
    
    # Export the model history to a csv file
    history_df = pd.DataFrame(history)
    history_df['epoch'] = history_df.index + 1
    history_df.to_csv(os.path.join(output_path, f'{experiment}_history.csv'), index=False)

    # Export plots of accuracy and loss
    plot_accuracies(history, output_path)
    plot_losses(history, output_path)

    # Model Evaluation
    model.eval()
    y_pred = []
    y_true = []

    with torch.no_grad():
        for inputs, labels in test_dl:
            inputs, labels = inputs.cuda(), labels.cuda()
            output = model(inputs)
            output = torch.argmax(output, dim=1).cpu().numpy()  # Extract predicted labels directly
            y_pred.extend(output)
            labels = labels.cpu().numpy()
            y_true.extend(labels)

    # Confusion Matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    logging.info(cf_matrix)
    print(cf_matrix)

    # Plot Confusion Matrix
    disp = ConfusionMatrixDisplay(cf_matrix, display_labels=selected_genera)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical', ax=ax)
    plt.savefig(confusion_matrix_path, dpi=300)
    plt.close()

    # Export classification report to csv
    report_data = classification_report(y_true, y_pred, target_names = selected_genera, output_dict=True)
    report_df = pd.DataFrame(report_data).transpose()
    logging.info(report_df)
    print(report_df)
    report_df.to_csv(classification_report_path)

if __name__ == '__main__':
    main()
