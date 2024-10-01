# Host Tree Classification CNN Evaluation Module

# Imports
# Imports for Pytorch
import torch # version 2.1.2
import torchvision # version 0.16.2
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader

# Other Imports
import argparse
import os
import json
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import time
from imageio import imread
from sklearn.metrics import classification_report

# Imports for Host Tree Classification
from tree_classification import models, preprocess, utilities
from tree_classification.models import *
from tree_classification.utilities import *


def evaluate_model(model, data_loader, selected_genera, output_dir, dataset_name):
    y_pred = []
    y_true = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            output = model(inputs)
            output = torch.argmax(output, dim=1).cpu().numpy()  # Extract predicted labels directly
            y_pred.extend(output)
            labels = labels.cpu().numpy()
            y_true.extend(labels)

    # Export classification report to CSV
    report_data = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_data).transpose()
    report_df = report_df[report_df['support'] != 0] # remove genera with 0 support (images)
    print(f"Classification report for {dataset_name}:")
    print(report_df)
    report_df.to_csv(os.path.join(output_dir, f'{dataset_name}_test_city_classification_report.csv'))
    

# Define main function to run classification model
def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Evaluate a tree classification model with the specified datasets.")

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
    test_data_dir = config['test_data_dir']
    output_path = config['output_path']
    model_checkpoint = config['checkpoint']

    # Select Genera for Classification
    selected_genera = config['selected_genera']
    
    # Sort genera alphabetically
    selected_genera = sorted(selected_genera)

    # Outputs
    model_path = os.path.join(output_path, model_checkpoint)
    
    print("Loading Trained CNN Model")
    # Load Pre-trained image classification model (and optimizer) and use model.eval for evaluation
    model, _ = load_classifier_model(model_path, selected_genera)
    
    # Ensure images are resized during testing as same dimension for training
    test_transforms = v2.Compose([
        v2.ToImage(),
        v2.CenterCrop(size=(768, 768)),
        v2.Resize(size=(512, 512), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.5708, 0.6118, 0.5824), std=(0.2223, 0.2135, 0.2583))
    ])
    
    # Load Testing Datasets and use allow_empty = True to allow empty (placeholder) folders. Necessary for prediction and label indices to align.
    test_dataset = ImageFolder(test_data_dir, transform = test_transforms, allow_empty = True)
    
    # Create DataLoaders
    test_dl = DataLoader(dataset = test_dataset, batch_size = 1, num_workers = 4, pin_memory = True)
    
     # Evaluate model on each test dataset
    print("Evaluating Models")
    evaluate_model(model, test_dl, selected_genera, output_path, experiment)


if __name__ == '__main__':
    main()
