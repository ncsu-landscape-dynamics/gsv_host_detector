# Module for Downloading Google Street View Panoramic Images

# Thomas Lake, June 2024

# Imports
import time
import os
import argparse
import logging
from tqdm import tqdm
from datetime import datetime
import requests
import json
import pandas as pd
import geopandas as gpd
import numpy as np
import folium
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from shapely.geometry import Point
import geopy.distance
from PIL import Image

# Imports for OSMnX Open Street Maps Networks
# https://osmnx.readthedocs.io/en/stable/user-reference.html
import networkx as nx
import osmnx as ox

# Imports for Google Street View Image downloader
# https://github.com/robolyst/streetview/tree/master
from streetview import search_panoramas, get_panorama_meta, get_streetview, get_panorama, get_panorama_async

from image_downloaders import osmnx_functions
from image_downloaders.osmnx_functions import *


# Set up logging
def setup_logging(log_path):
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Download a Google Street View Panoramic Image from Location")

    # Paths for input datasets
    parser.add_argument('--config', type=str, required=True, 
                        help="Path to the configuration file")

    # Parse arguments
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Args
    api_key_path = config['api_key_path']
    latitude = float(config['latitude'])
    longitude = float(config['longitude'])
    distance = float(config['distance'])
    output_path = config['output_path']

    # Create output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    # Setup Logging
    log_path = os.path.join(output_path, 'log-image-downloader.log')
    setup_logging(log_path)
    logging.info(f"Configuration File: {config}")
    logging.info("Image Downloader Logging File")
    logging.info(f'Searching for Images at {latitude}, {longitude}')

    # Read Secret Google Street View API Key
    with open(api_key_path, 'r') as file:
        api_key = file.read().strip()
    print("Found API Key")
    
    # Define location for road network analysis from latitude and longitude
    sel_location = (latitude, longitude)
    
    # Create road network graph from point
    graph, road = get_road_network_from_point(sel_location, distance)
    
    # Calculate summary statistics for road network graph
    road_proj = ox.project_graph(graph)
    nodes_proj = ox.graph_to_gdfs(road_proj, edges=False)
    graph_area_m = nodes_proj.unary_union.convex_hull.area

    # Output summary statistics of road network
    # To get density-based statistics, you must also pass the network's bounding area in square meters
    # Information on summary statistics: 
    stats = ox.basic_stats(road_proj, area=graph_area_m, clean_int_tol=15)
    
    print("Summary Statistics for Road Network")
    print(pd.Series(stats))
    
    print("Converting Road Network to Points")
    # Create a set of approx. equally-spaced points along the road network
    points = select_points_on_road_network(road, 25)

    # Convert points into latitude/longitude with CRS
    points_coords = points.to_crs(4326)
    points_coords.head(5)

    print("Finding Panoramic Images Near Road Points")
    # Hold point location and available panoramic image location metadata
    pano_data = []

    # Iterate over each point in the road network, and get metadata for the nearest panoramic images
    for i in tqdm(range(len(points_coords.geometry))):

        # Search for all available panoramic images closest to each point
        panos = search_panoramas(lat=points_coords.geometry.y[i], lon=points_coords.geometry.x[i])
        
        # Iterate through the closest set of panos for a given location
        # For each pano image, get the metadata by supplying the unique pano_id string
        for pano in panos:
        
            # Fetch metadata
            resp = requests.get(f"https://maps.googleapis.com/maps/api/streetview/metadata?pano={pano.pano_id}&key={GOOGLE_MAPS_API_KEY}")
            meta_data = resp.json()

            # Check if metadata is valid or if the status is ZERO_RESULTS
            if meta_data.get('status') == 'ZERO_RESULTS':
                print(f"No panorama data found for pano_id: {pano.pano_id}")
                continue  # Skip to the next panorama if no data is found
            
            meta = get_panorama_meta(pano_id=pano.pano_id, api_key=api_key)
            
            if meta.date:
                date_code = datetime.strptime(meta.date, '%Y-%m')
                
                # Append data on point and panoramic image location
                pano_data.append({'Point_Index': i,
                    'Point_Latitude': points_coords.geometry.y[i],
                    'Point_Longitude': points_coords.geometry.x[i],
                    'Panorama_ID': pano.pano_id,
                    'Panorama_Date': meta.date,
                    'Panorama_Latitude': pano.lat,
                    'Panorama_Longitude': pano.lon,
                    'Panorama_Rotation': pano.heading})


    # All available panoramic images sampled from road points
    pano_df = pd.DataFrame(pano_data)

    # Remove duplicate images based on 'Panorama_ID'
    pano_df.drop_duplicates(subset='Panorama_ID', keep='first', inplace=True)

    print(f'Total Available Panoramic Images:', len(pano_df))

    # Remove images where 'Panorama_Date' is after 2016-01
    pano_df = pano_df[pano_df['Panorama_Date'] >= '2016-01']
    pano_df.reset_index(drop=True, inplace=True)
    print(f'Total Panoramic Images After 2016:', len(pano_df))

    # Remove any panoraamic images closer than 5 meters
    pano_df_simple = remove_adjacent_panoramics(pano_df, 5)
    pano_df_simple.reset_index(drop=True, inplace=True)

    print(f'Total Panoramic Images After De-Duplication:', len(pano_df_simple))

    print('Downloading Panoramic Images')
    # Loop over panoramic images sampled in the pano_df
    for i in tqdm(range(len(pano_df_simple))):

        # Get panoramic image unique ID
        panoID = pano_df_simple['Panorama_ID'][i]
        
        # Define the file path for the image
        image_path = os.path.join(output_path, f'{panoID}.jpg')
        metadata_path = os.path.join(output_path, f'{panoID}.metadata.json')
        
        # Check if image already exists. If it does, skip that image.
        if os.path.exists(image_path):
            print(f"Image {panoID} already exists. Skipping...")
            continue
        
        try:
            print("Downloading Panoramic Image and Metadata:")
            
            # Create the matching panoramic image metadata .json file
            metadata = {
                'panoId': panoID,
                'panoDate:': pano_df_simple['Panorama_Date'][i],
                'lat': pano_df_simple['Panorama_Latitude'][i],
                'lng': pano_df_simple['Panorama_Longitude'][i],
                'rotation': pano_df_simple['Panorama_Rotation'][i]
                }
                
            print(metadata)
            
            # Write metadata to JSON file            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
                
            # Attempt to download single panoramic image
            image = get_panorama(pano_id=panoID)

            # Save the image
            image.save(image_path, "jpeg")
                
            print(f"Saved Panoramic Image and Metadata: ", panoID)
            
        except UnidentifiedImageError as e:
        
            print(f"Error downloading image {panoID}: {e}")
            # Optionally, you can add a delay before retrying
            time.sleep(2)  # Wait for 2 seconds before retrying
            continue  # Skip to the next iteration of the loop


if __name__ == '__main__':
    main()
    

