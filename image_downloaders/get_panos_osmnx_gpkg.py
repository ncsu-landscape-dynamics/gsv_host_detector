# Module for Downloading Google Street View Panoramic Images
# Configuration file: downloader_config_gpkg.json

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
    gpkg_roads_file = config['gpkg_roads_file']
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
    logging.info("Found API Key")
    
    # Read in roads graph as a geopandas dataframe from the .gpkg file
    #gdf_nodes = gpd.read_file(gpkg_roads_file, layer='nodes').set_index('osmid')
    gdf_edges = gpd.read_file(gpkg_roads_file, layer='edges').set_index(['u', 'v', 'key'])

    # Reproject roads (edges) to UTM Zone N
    roads_gdf = gdf_edges.to_crs('EPSG:32618')
    
    logging.info("Converting Road Network to Points")
    # Create a set of approx. equally-spaced points along the road network
    points = select_points_on_road_network(road, 25)
    logging.info(f"Number of sampled road points: {len(points}")

    # Convert points into latitude/longitude with CRS
    points_coords = points.to_crs(4326)

    logging.info("Finding Panoramic Images Near Road Points")
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

    logging.info(f'Total Available Panoramic Images:', len(pano_df))

    # Remove images where 'Panorama_Date' is after 2016-01
    pano_df = pano_df[pano_df['Panorama_Date'] >= '2016-01']
    pano_df.reset_index(drop=True, inplace=True)
    logging.info(f'Total Panoramic Images After 2016:', len(pano_df))

    # Remove any panoraamic images closer than 5 meters
    pano_df_simple = remove_adjacent_panoramics(pano_df, 5)
    pano_df_simple.reset_index(drop=True, inplace=True)

    logging.info(f'Total Panoramic Images After De-Duplication:', len(pano_df_simple))

    logging.info('Downloading Panoramic Images')
    # Loop over panoramic images sampled in the pano_df
    for i in tqdm(range(len(pano_df_simple))):

        # Get panoramic image unique ID
        panoID = pano_df_simple['Panorama_ID'][i]
        
        # Define the file path for the image
        image_path = os.path.join(output_path, f'{panoID}.jpg')
        metadata_path = os.path.join(output_path, f'{panoID}.metadata.json')
        
        # Check if image already exists. If it does, skip that image.
        if os.path.exists(image_path):
            logging.info(f"Image {panoID} already exists. Skipping...")
            continue
        
        try:
            logging.info("Downloading Panoramic Image and Metadata:")
            
            # Create the matching panoramic image metadata .json file
            metadata = {
                'panoId': panoID,
                'panoDate:': pano_df_simple['Panorama_Date'][i],
                'lat': pano_df_simple['Panorama_Latitude'][i],
                'lng': pano_df_simple['Panorama_Longitude'][i],
                'rotation': pano_df_simple['Panorama_Rotation'][i]
                }
                
            logging.info(metadata)
            
            # Write metadata to JSON file            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
                
            # Attempt to download single panoramic image
            image = get_panorama(pano_id=panoID)

            # Save the image
            image.save(image_path, "jpeg")
                
            logging.info(f"Saved Panoramic Image and Metadata: ", panoID)
            
        except UnidentifiedImageError as e:
        
            logging.info(f"Error downloading image {panoID}: {e}")
            # Optionally, you can add a delay before retrying
            time.sleep(2)  # Wait for 2 seconds before retrying
            continue  # Skip to the next iteration of the loop


if __name__ == '__main__':
    main()
    

