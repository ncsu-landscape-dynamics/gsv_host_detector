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

    # Confusion Matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    print(f"Confusion matrix for {dataset_name}:")
    print(cf_matrix)

    # Plot Confusion Matrix
    disp = ConfusionMatrixDisplay(cf_matrix, display_labels=selected_genera)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical', ax=ax)
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_confusion_matrix.png'), dpi=300)
    plt.close()

    # Export classification report to CSV
    report_data = classification_report(y_true, y_pred, target_names=selected_genera, output_dict=True)
    report_df = pd.DataFrame(report_data).transpose()
    print(f"Classification report for {dataset_name}:")
    print(report_df)
    report_df.to_csv(os.path.join(output_dir, f'{dataset_name}_classification_report.csv'))


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
    train_data_dir = config['train_data_dir']
    test_data_dir = config['test_data_dir']
    test_data_dir_aa = config['test_data_dir_aa']
    test_data_dir_inat = config['test_data_dir_inat']
    output_path = config['output_path']
    model_checkpoint = config['checkpoint']

    # Select Genera for Classification
    selected_genera = config['selected_genera']

    # Outputs
    model_path = os.path.join(output_path, model_checkpoint)
    
    print("Loading Trained CNN Model")
    # Load Pre-trained image classification model and use model.eval for evaluation
    model = load_classifier_model(model_path, selected_genera)
    
    # Ensure images are resized during testing as same dimension for training
    test_transforms = v2.Compose([
        v2.ToImage(),
        v2.CenterCrop(size=(768, 768)),
        v2.Resize(size=(512, 512), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.5046, 0.5396, 0.4885), std=(0.2176, 0.2147, 0.2471))
    ])
    
    # Load Testing Datasets
    test_dataset = ImageFolder(test_data_dir, transform = test_transforms)
    test_dataset_aa = ImageFolder(test_data_dir_aa, transform = test_transforms)
    test_dataset_inat = ImageFolder(test_data_dir_inat, transform = test_transforms)
    
    # Create DataLoaders
    test_dl = DataLoader(dataset = test_dataset, batch_size = 1, num_workers = 4, pin_memory = True)
    test_dl_aa = DataLoader(dataset = test_dataset_aa, batch_size = 1, num_workers = 4, pin_memory = True)
    test_dl_inat = DataLoader(dataset = test_dataset_inat, batch_size = 1, num_workers = 4, pin_memory = True)
    
     # Evaluate model on each test dataset
    print("Evaluating Models")
    evaluate_model(model, test_dl, selected_genera, output_path, 'aa_inat_test')
    evaluate_model(model, test_dl_aa, selected_genera, output_path, 'aa_test')
    evaluate_model(model, test_dl_inat, selected_genera, output_path, 'inat_test')


if __name__ == '__main__':
    main()