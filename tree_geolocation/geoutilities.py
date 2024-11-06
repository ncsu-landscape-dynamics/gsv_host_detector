
# Host Tree Classification and Geolocation Utilities

# Imports

import torch

import torch
from torchvision import transforms
import os
import sys
import json
import pandas as pd
import numpy as np
import geopandas as gpd
from math import radians, degrees, sin, cos, asin, atan2
from shapely.geometry import Point, LineString
from skimage.io import imread
from PIL import UnidentifiedImageError
import cv2
import logging

# Imports for Host Tree Classification
from tree_classification import models, preprocess, utilities
from tree_classification.models import *
from tree_classification.utilities import *


def load_config(config_path: str) -> dict:
    """Load configuration from a JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def load_yolo_model(yolo_model_path: str) -> torch.nn.Module:
    """Load the YOLO model."""
    model = torch.hub.load(
        r'C:/Users/talake2/Desktop/auto_arborist_cvpr2022_v015/yolov5', 
        'custom', 
        path=yolo_model_path, 
        source='local'
    )
    model.conf = 0.25
    return model

def load_zoedepth_model() -> torch.nn.Module:
    """Load the ZoeDepth model."""
    sys.path.insert(1, 'C:/users/talake2/Desktop/ZoeDepth')
    from zoedepth.models.builder import build_model
    from zoedepth.utils.config import get_config
    import timm  # Ensure timm version 0.6.7 is installed
    conf = get_config("zoedepth", "infer", config_version="kitti", config_mode="eval")
    model_zoe_k = build_model(conf)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    zoe = model_zoe_k.to(DEVICE)
    return zoe


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


def calculate_distance(p1, p2):
    """Calculate the Euclidean distance between two (lat/lon) shapely Points in Meters."""
    return p1.distance(p2) * 111139 #multiply by 111139 to convert lat/long dist to meters (approx)

def read_image_metadata(img_folder, img_name):
    """
    Read metadata from panoramic images.
    Return metadata information: pano_id, image width, height, pano_lat, pano_lon, year, month, elevation.
    """
    img_path = os.path.join(img_folder, img_name)
    json_path = img_path.replace('.jpg', '.metadata.json')
    
    try:
        # Read metadata from panoramic image
        with open(json_path, 'r') as json_file:
            img_metadata = json.load(json_file)
    except FileNotFoundError:
        # Handle the case where metadata file is not found
        print(f"Metadata file not found for image: {img_name}")
        return None
    
    # Get metadata for panorama
    pano_id = img_metadata['panoId']
    resolution = img_metadata['resolution']
    pano_lat = img_metadata['lat']
    pano_lon = img_metadata['lng']
    date = img_metadata['date']
    elevation = img_metadata['elevation']
    
    return pano_id, resolution, pano_lat, pano_lon, date, elevation
    

def read_panoramic_image_and_metadata(img_folder: str, img_name: str):
    """
    Read panoramic image and its metadata from the specified folder.
    Returns the image, height, width, channels, pano_id, pano_rotation_value, pano_lat, pano_lon.
    """
    img_path = os.path.join(img_folder, img_name)
    json_path = img_path.replace('.jpg', '.metadata.json')
    
    try:
        # Read panoramic image
        img = imread(img_path)
        height, width, channels = img.shape
        # Check if image size is correct. Images should be shape (8192,16384,3).
        if width != 16384:
            print("Resizing GSV Image to Dimensions (16384, 8192)")
            img = cv2.resize(img, dsize = (16384, 8192), interpolation = cv2.INTER_AREA)
        height, width, channels = img.shape
    except (FileNotFoundError, UnidentifiedImageError):
        # Handle the case where image file is not found or is not a valid image
        print(f"Image file not found or is not a valid image: {img_name}")
        return None
    
    try:
        # Read panoramic image metadata
        with open(json_path, 'r') as json_file:
            img_metadata = json.load(json_file)
    except FileNotFoundError:
        # Handle the case where metadata file is not found
        print(f"Metadata file not found for image: {img_name}")
        return None

    # Get metadata for panorama
    pano_id = img_metadata['panoId']
    pano_rotation_value = img_metadata['rotation']  # North
    pano_lat = img_metadata['lat']
    pano_lon = img_metadata['lng']

    return img, width, pano_id, pano_rotation_value, pano_lat, pano_lon


def calculate_pano_tree_angle(row, img_width, pano_rotation_north):
    """
    Calculate the angle of rotation of a tree in the panoramic image, based on the 
    vertical center of the bounding box given from a YOLO model.
    """
    # Extract vertical center of the tree bounding box
    tree_center_x = (row['xmin'] + row['xmax']) / 2
    # Calculate the angle relative to the north
    px_per_degree = (img_width/360) # eq. 45.5111
    tree_angle = tree_center_x / px_per_degree
    # Adjust the angle based on the north rotation value
    adjusted_angle = (tree_angle - pano_rotation_north) % 360
    return adjusted_angle

def get_point_at_distance(lat1, lon1, d, bearing, R=6371):
    """
    lat: initial latitude, in degrees
    lon: initial longitude, in degrees
    d: target distance from initial
    bearing: (true) heading in degrees
    R: optional radius of sphere, defaults to mean radius of earth

    Returns new lat/lon coordinate {d}km from initial, in degrees
    Source: https://stackoverflow.com/questions/7222382/get-lat-long-given-current-point-distance-and-bearing
    """
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    a = radians(bearing)
    d = d/1000 # Convert from Km to meters
    lat2 = asin(sin(lat1) * cos(d/R) + cos(lat1) * sin(d/R) * cos(a))
    lon2 = lon1 + atan2(
        sin(a) * sin(d/R) * cos(lat1),
        cos(d/R) - sin(lat1) * sin(lat2)
    )
    return (degrees(lat2), degrees(lon2),)

def extend_lines(row, distance):
    '''
    Create a shapely.LineString (line) between a panoramic image and detected tree
    Given the panoramic image origin (lat, long), and the bearing (angle of tree),
    create a line with a distance.
    '''
    extended_point = get_point_at_distance(row['Pano_Origin_Lat'], row['Pano_Origin_Lon'], distance, row['Pano_Tree_Angle'])
    return LineString([(row['Pano_Origin_Lon'], row['Pano_Origin_Lat']), (extended_point[1], extended_point[0])])


def process_yolo_results(model_results_df, num_detections):
    '''
    Process results of a YOLO model that detects trees in a panoramic image.
    For geolocation: select trees with large bounding boxes that are nearer to the image orign
    Calculate area and aspect ratio of each detected tree by bounding box coordinates
    Filter by the largest trees with high detection confidence and small aspect ratio
    
    Parameter: num_detections. Integer specifying how many detected trees are selected by bounding box area (N-largest trees).
    '''
    # Check if the dataframe is empty
    if model_results_df.empty:
        print("No trees detected in the image.")
        return None

    # Calculate bounding box area and aspect ratio
    model_results_df['area'] = (model_results_df['xmax'] - model_results_df['xmin']) * (model_results_df['ymax'] - model_results_df['ymin'])
    model_results_df['aspect_ratio'] = (model_results_df['xmax'] - model_results_df['xmin']) / (model_results_df['ymax'] - model_results_df['ymin'])

    # Select detection by bounding box area. Specify how many detected objects to return with num_detections.
    topN_by_area = model_results_df.nlargest(num_detections, 'area', 'all')

    # Exclude values with aspect ratio above 1.5
    topN_filtered = topN_by_area[(topN_by_area['aspect_ratio'] <= 2.0) & (topN_by_area['confidence'] > 0.6)]

    return topN_filtered


def predict_genus_cnn(img, model, genera, xmin, ymin, xmax, ymax):
    '''
    Function to apply a trained CNN model to classify trees detected in street view imagery.
    Classifies without TenCrop.
    Takes in a panoramic street view image, applies transformations (resize, normalize) and runs prediction.
    Returns the most likely class prediction (argmax) and a dictionary of the predicted class probabilities.
    '''
    # Genera to detect in google street view images
    selected_genera = genera

    # Crop image to detected tree bounding box
    crop_img = img[ymin:ymax, xmin:xmax, ...]

    # Check if image is at least 512 x 512 to run predictions
    # Create larger bounding box around detected tree if dimensions are too small
    if crop_img.shape[0] < 512:
        print("Re-cropping image to meet required dimensions: 512 x 512")
        ymin = max(0, ymin - (512 - crop_img.shape[0]) // 2)
        ymax = ymin + 512
        crop_img = img[ymin:ymax, xmin:xmax, ...]

    if crop_img.shape[1] < 512:
        print("Re-cropping image to meet required dimensions: 512 x 512")
        xmin = max(0, xmin - (512 - crop_img.shape[1]) // 2)
        xmax = xmin + 512
        crop_img = img[ymin:ymax, xmin:xmax, ...]    

    # Define transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert to PIL Image
        transforms.Resize((512, 512)),  # Resize image to 512x512
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=(0.5708, 0.6118, 0.5824), std=(0.2223, 0.2135, 0.2583))  # Normalize the image
    ])

    # Apply transformations
    transformed_img = transform(crop_img)

    # Run CNN model prediction on the transformed image
    transformed_img_tensor = transformed_img.unsqueeze(0).cuda()  # add first dimension of batch size and push to GPU
    output = model(transformed_img_tensor)

    # Get the softmax class probability for each genus
    class_probs = torch.softmax(output, dim=1)

    # Get the argmax predicted class
    class_argmax = torch.argmax(output).item()

    # From predicted class index, get genus name
    predicted_tree_genus = selected_genera[class_argmax]

    # Initialise a dictionary to hold the output genus probabilities
    genus_softmax_dict = {}
    for i, genus in enumerate(selected_genera):
        genus_softmax_dict[f'{genus}_prob'] = class_probs[0][i].item()

    return predicted_tree_genus, genus_softmax_dict


def process_image(img_folder: str, img_name: str, tree_model: torch.nn.Module, zoe: torch.nn.Module, classifier: torch.nn.Module, selected_genera: list) -> pd.DataFrame:
    """Process a single image for tree detection, classification, and geolocation."""
    result = read_panoramic_image_and_metadata(img_folder, img_name)
    if result is None:
        logging.warning(f"Failed to read image or metadata for {img_name}")
        return pd.DataFrame()
        
    img, width, pano_id, pano_rotation_value, pano_lat, pano_lon = result
    logging.info(f"Running Tree Detection and Initial Geolocation on: {pano_id}")
    
    # Apply YOLO model to detect trees
    model_results = tree_model(img) 
    model_results_df = model_results.pandas().xyxy[0] 
    
    # Subset trees from YOLO model by detection heuristics: bounding box size, aspect ratio, confidence
    top_detection_results = process_yolo_results(model_results_df, 3) # Select top N detections by area, aspect ratio, and confidence
    logging.info(top_detection_results)

    if top_detection_results is None:
        logging.warning("No trees detected in the image.")
        return pd.DataFrame()

    # Depth Estimation with ZoeDepth
    metric_depth_resized_zoe = zoe.infer_pil(img)
    DEPTH_SCALING_FACTOR = 3.0  # Scale factor for panoramic images

    tree_geolocation_results = []

    for index, row in top_detection_results.iterrows():
        # Get bounding box for detected tree in the image
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        
        # Run Ten-Crop CNN Model Classification on Detected Tree
        predicted_genus, genus_softmax_dict = predict_genus_cnn(img, classifier, selected_genera, xmin, ymin, xmax, ymax)

        # Save predicted image
        crop_img = img[ymin:ymax, xmin:xmax, ...]
        # Define the directory and file path
        save_dir = f"C:/users/talake2/Desktop/tree-geolocation-exp/siouxfalls_predictions_genus/predicted_{predicted_genus}/"
        filename = f"{save_dir}{pano_id}-{index}.jpg"
        
        # Check if the directory exists, if not, create it
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Save the cropped image
        plt.imshow(crop_img)
        plt.savefig(filename)
        plt.close()

        # Crop the depth map to the tree matched from panoramic image to inventory
        cropped_depth = metric_depth_resized_zoe[ymin:ymax, xmin:xmax]

        # Estimate depth as the median of the depthmap
        est_tree_depth = np.median(cropped_depth) * DEPTH_SCALING_FACTOR

        # Get index and rotation (0-360 degrees) for detected tree in the panoramic image
        tree_rotation_value = calculate_pano_tree_angle(row, width, pano_rotation_value)
        
        # Geolocate tree given pano origin, distance, and bearing
        est_tree_lat, est_tree_lon = get_point_at_distance(pano_lat, pano_lon, est_tree_depth, tree_rotation_value)

        # Capture outputs for each predicted tree as a new_row in a dataframe
        new_row = {
            'Pano_ID': pano_id,
            'Pano_Origin_Lon': pano_lon,
            'Pano_Origin_Lat': pano_lat,
            'Est_Tree_Lon': est_tree_lon,
            'Est_Tree_Lat': est_tree_lat,
            'Predicted_Genus': predicted_genus,
            'Pano_Tree_Angle': tree_rotation_value,
            'Est_Depth': est_tree_depth,
            'Bbox_Area': row['area'],
            'Bbox_Aspect': row['aspect_ratio']
        }

        # Extend the new_row dictionary with predicted genus softmax values
        new_row.update(genus_softmax_dict)

        tree_geolocation_results.append(new_row)

    return pd.DataFrame(tree_geolocation_results)
    

def find_intersections(combined_geolocation_lines, tolerance=1e-6):
    """
    Identifies intersections between lines and averages classification probabilities if they intersect.

    Parameters:
    combined_geolocation_lines (GeoDataFrame): DataFrame containing lines and associated metadata.
    tolerance (float): Tolerance for intersection checks to exclude origins.

    Returns:
    GeoDataFrame: A GeoDataFrame containing intersection points and averaged classification probabilities.
    """
    
    # Initialize an empty DataFrame for triangulated tree detections
    triangulated_tree_detections = pd.DataFrame(columns=[
        'Intersection', 'Pano_ID_1', 'Pano_Origin_Lon_1', 'Pano_Origin_Lat_1', 'Est_Depth_1', 
        'Pano_ID_2', 'Pano_Origin_Lon_2', 'Pano_Origin_Lat_2', 'Est_Depth_2'
    ])
    
    # Iterate through all detected and classified trees and triangulate detected trees
    for i in range(len(combined_geolocation_lines)):
        origin = combined_geolocation_lines.geometry[i].coords[0]
        for j in range(i + 1, len(combined_geolocation_lines)):
            if combined_geolocation_lines.geometry[i].intersects(combined_geolocation_lines.geometry[j]):
                intersection_point = combined_geolocation_lines.geometry[i].intersection(combined_geolocation_lines.geometry[j])
                if (abs(intersection_point.x - origin[0]) > tolerance or abs(intersection_point.y - origin[1]) > tolerance) and \
                   (abs(intersection_point.x - combined_geolocation_lines.geometry[j].coords[0][0]) > tolerance or 
                    abs(intersection_point.y - combined_geolocation_lines.geometry[j].coords[0][1]) > tolerance):
                    
                    pano_i = combined_geolocation_lines.iloc[i]
                    pano_j = combined_geolocation_lines.iloc[j]
                    
                    # Get Pano ID and Location For Intersection Points
                    output_data = {
                        'Intersection': intersection_point,
                        'Pano_ID_1': pano_i['Pano_ID'], 
                        'Pano_Origin_Lon_1': pano_i['Pano_Origin_Lon'], 
                        'Pano_Origin_Lat_1': pano_i['Pano_Origin_Lat'], 
                        'Est_Depth_1': pano_i['Est_Depth'],
                        'Pano_ID_2': pano_j['Pano_ID'], 
                        'Pano_Origin_Lon_2': pano_j['Pano_Origin_Lon'], 
                        'Pano_Origin_Lat_2': pano_j['Pano_Origin_Lat'], 
                        'Est_Depth_2': pano_j['Est_Depth']
                    }
                    
                    # Get the class probabilities for panos i and j
                    pano_i_genus_probs = combined_geolocation_lines.iloc[i, 10:110]
                    pano_j_genus_probs = combined_geolocation_lines.iloc[j, 10:110]
                    
                    # Average the two predicted tree class probabilities
                    combined_probs = pano_i_genus_probs.add(pano_j_genus_probs)
                    average_genus_probs = combined_probs / 2
                    
                    # Convert average_genus_probs to a dictionary and unpack into the output DataFrame
                    output_data.update(average_genus_probs.to_dict())
                    
                    triangulated_tree_detections = pd.concat([triangulated_tree_detections, pd.DataFrame([output_data])], ignore_index=True)
    
    # Convert triangulated tree intersections to a GeoDataFrame
    intersection_geometry = [Point(xy) for xy in triangulated_tree_detections['Intersection']]
    triangulated_tree_detections_gdf = gpd.GeoDataFrame(triangulated_tree_detections, geometry=intersection_geometry, crs="EPSG:4326")
    
    return triangulated_tree_detections_gdf
    
    

def filter_intersections(triangulated_tree_detections_gdf, tree_points_kdtree, max_distance=5):
    """
    Filters intersections based on the distance between estimated tree locations.

    Parameters:
    triangulated_tree_detections_gdf (GeoDataFrame): DataFrame containing intersection points and metadata.
    tree_points_kdtree (scipy.spatial.cKDTree): KDTree for efficient nearest-neighbor lookup.
    max_distance (float): Maximum distance (in meters) to search for intersections near estimated tree locations.

    Returns:
    GeoDataFrame: A filtered GeoDataFrame with unique intersection points and predicted genus.
    """
    
    filtered_intersections = []
    assigned_positions = {}
    
    for point in range(len(triangulated_tree_detections_gdf.geometry)):
        intersection_point = triangulated_tree_detections_gdf.geometry[point]
        distances, indices = tree_points_kdtree.query((intersection_point.x, intersection_point.y), 2)
        distances_meters = distances * 111139  # Convert distances to meters
        
        for i, index in enumerate(indices):
            if distances_meters.max() <= max_distance:
                if index in assigned_positions:
                    if distances_meters[i] < assigned_positions[index]['distance']:
                        assigned_positions[index] = {'distance': distances_meters[i], 'intersection': intersection_point}
                else:
                    assigned_positions[index] = {'distance': distances_meters[i], 'intersection': intersection_point}
    
    for _, info in assigned_positions.items():
        if info['intersection'] not in filtered_intersections:
            filtered_intersections.append(info['intersection'])

    df_intersections_filtered = pd.DataFrame([(point.x, point.y) for point in filtered_intersections], columns=['Lon', 'Lat'])

    print(f"Intersections before filtering:", len(triangulated_tree_detections_gdf.geometry))
    print(f"Intersections after filtering:", len(filtered_intersections))

    filtered_intersections_series = gpd.GeoSeries(filtered_intersections)
    filtered_tree_detections_gdf = triangulated_tree_detections_gdf[triangulated_tree_detections_gdf['geometry'].isin(filtered_intersections_series)]
    
    filtered_intersection_geometry = [Point(xy) for xy in filtered_tree_detections_gdf['Intersection']]
    filtered_triangulated_tree_detections_gdf = gpd.GeoDataFrame(filtered_tree_detections_gdf, geometry=filtered_intersection_geometry, crs="EPSG:4326")
    
    predicted_classes = filtered_triangulated_tree_detections_gdf.iloc[:, 9:109].idxmax(axis=1)
    predicted_classes = predicted_classes.apply(lambda x: x.split('_')[0])
    filtered_triangulated_tree_detections_gdf['predicted_genus'] = predicted_classes
    
    filtered_triangulated_tree_detections_gdf['latitude'] = filtered_triangulated_tree_detections_gdf.geometry.y
    filtered_triangulated_tree_detections_gdf['longitude'] = filtered_triangulated_tree_detections_gdf.geometry.x
    
    return filtered_triangulated_tree_detections_gdf
