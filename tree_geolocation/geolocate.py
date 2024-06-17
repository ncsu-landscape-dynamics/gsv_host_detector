# Host Tree Geolocation Module

# Thomas Lake June 2024

# Imports

# PyTorch
import torch
from torchvision import transforms

# Other Imports
import sys
import os
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import KDTree, cKDTree
import random
import json
import argparse
import logging
from math import asin, atan2, cos, degrees, radians, sin
from geopy.distance import geodesic
from shapely.geometry import Point, LineString

# Plotting
import matplotlib.pyplot as plt
from matplotlib.image import imread

# Imports for Host Tree Geolocation
from tree_geolocation import geoutilities
from tree_geolocation.geoutilities import *

# Imports for Host Tree Classification
from tree_classification import models, preprocess, utilities
from tree_classification.models import *
from tree_classification.utilities import *


def setup_logging(log_path: str) -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )



def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run Tree Detection, Classification, and Geolocation Model")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # File Path Args
    experiment = config['experiment']
    gsv_img_folder = config['gsv_img_folder']
    output_path = config['output_path']
    
    # Model Paths and Parameters
    yolo_model_path = config['yolo_model_path']
    classifier_path = config['classifier_path']
    selected_genera = config['selected_genera']

    # Create output directory
    create_output_directory(output_path)
    
    # Setup Logging
    log_path = os.path.join(output_path, f'log-{experiment}.log')
    setup_logging(log_path)
    logging.info(f"Configuration File: {config}")

    # Load Models
    tree_model = load_yolo_model(yolo_model_path)
    zoe = load_zoedepth_model()
    classifier = load_classifier_model(classifier_path, selected_genera)

    # Run Initial Tree Geolocation
    all_tree_geolocation_results_df = pd.DataFrame()

    for img_name in os.listdir(gsv_img_folder):
        if img_name.endswith('.jpg'):
            tree_geolocation_results_df = geoutilities.process_image(gsv_img_folder, img_name, tree_model, zoe, classifier, selected_genera)
            all_tree_geolocation_results_df = pd.concat([all_tree_geolocation_results_df, tree_geolocation_results_df], ignore_index=True)

    print(all_tree_geolocation_results_df)
    
     # Write results to CSV
    csv_output_file = os.path.join(output_path, f'tree_geolocation_results_{experiment}.csv')
    all_tree_geolocation_results_df.to_csv(csv_output_file, index=False)


if __name__ == '__main__':
    main()
