# Host Tree Geolocation Module

# Thomas Lake June 2024

# Imports

import os
import sys
import json
import logging
import argparse
from math import asin, atan2, cos, degrees, radians, sin

import torch
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import KDTree
from sklearn.cluster import AgglomerativeClustering
from geopy.distance import geodesic
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
from matplotlib.image import imread

from tree_geolocation import geoutilities
from tree_geolocation.geoutilities import *
from tree_classification import models, preprocess, utilities
from tree_classification.models import *
from tree_classification.utilities import *

# Constants
LINE_DISTANCE = 30 # Distance (meters) to cast lines from panoramic image to detected trees
TOLERANCE = 1e-6 # Tolerance (decimal degrees) for minimum distance between intersections during triangulation

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

    # Create output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Setup Logging
    log_path = os.path.join(output_path, f'log-{experiment}.log')
    setup_logging(log_path)
    logging.info(f"Configuration File: {config}")

    # Load Detection, Depth Estimation, and Classification Models
    tree_model = load_yolo_model(yolo_model_path)
    zoe = load_zoedepth_model()
    classifier = load_classifier_model(classifier_path, selected_genera)

    # Run Initial Tree Geolocation
    all_tree_geolocation_results_df = pd.DataFrame()
    for img_name in os.listdir(gsv_img_folder):
        if img_name.endswith('.jpg'):
            tree_geolocation_results_df = geoutilities.process_image(gsv_img_folder, img_name, tree_model, zoe, classifier, selected_genera)
            all_tree_geolocation_results_df = pd.concat([all_tree_geolocation_results_df, tree_geolocation_results_df], ignore_index=True)

     # Export initial geolocation results
    csv_output_file = os.path.join(output_path, f'tree_geolocation_initial_results_{experiment}.csv')
    all_tree_geolocation_results_df.to_csv(csv_output_file, index=False)
    logging.info(all_tree_geolocation_results_df)

    # Perform Triangulation
    # Group the inital tree geolocation results by Pano_ID and cast lines from each pano origin to tree based on bearing for traingulation
    panos_grouped = all_tree_geolocation_results_df.groupby('Pano_ID').apply(lambda x:[extend_lines(row, LINE_DISTANCE) for idx, row in x.iterrows()])
    
    # Combine lines with initial tree detection and classification
    lines = [line for sublist in panos_grouped for line in sublist]
    print(f'Created', len(lines), "lines between panoramic images and detected trees")
    lines_gdf = gpd.GeoDataFrame(geometry=lines, columns=['geometry'], crs="EPSG:4326")
    combined_geolocation_lines = pd.concat([all_tree_geolocation_results_df, lines_gdf], axis=1)
    
    print('Triangulating tree locations and averaging triangulated tree genus predictions')
    
    # Find intersections between lines and average class probabilites at intersections
    triangulated_tree_detections_gdf = find_intersections(combined_geolocation_lines)
    print(triangulated_tree_detections_gdf)

    # Create a KDTree based on initial tree geolocations to filter against nearby intersections
    tree_points = all_tree_geolocation_results_df[['Est_Tree_Lon', 'Est_Tree_Lat']].values
    tree_points_kdtree = KDTree(tree_points)

    # Filter triangulated tree intersections based on distance from KDTree to two nearby initial tree detections
    filtered_triangulated_tree_detections_gdf = filter_intersections(triangulated_tree_detections_gdf, tree_points_kdtree, max_distance=5)
    logging.info(filtered_triangulated_tree_detections_gdf)
    print(filtered_triangulated_tree_detections_gdf)

     # Write filtered tree geolocation results to CSV
    csv_output_file = os.path.join(output_path, f'tree_geolocation_filtered_results_{experiment}.csv')
    filtered_triangulated_tree_detections_gdf.to_csv(csv_output_file, index=False)

    logging.info(filtered_triangulated_tree_detections_gdf)

if __name__ == '__main__':
    main()
