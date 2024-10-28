# Module for Downloading Google Street View Panoramic Images
# Configuration file: downloader_config_gpkg.json

# Thomas Lake, October 2024

# Imports
import time
import os
import argparse
import logging
from tqdm import tqdm
from datetime import datetime
import aiohttp
from aiohttp import ClientSession, ClientConnectorError
import asyncio
import requests
import json
import pandas as pd
import geopandas as gpd
import numpy as np
import folium
from scipy.spatial import cKDTree
from shapely.geometry import Point
from shapely import wkt
import geopy.distance
from PIL import Image
from PIL import UnidentifiedImageError

# Imports for OSMnX Open Street Maps Networks
# https://osmnx.readthedocs.io/en/stable/user-reference.html
import networkx as nx
import osmnx as ox

# Imports for Google Street View Image downloader
# https://github.com/robolyst/streetview/tree/master
from streetview import search_panoramas, get_panorama_meta, get_streetview, get_panorama, get_panorama_async


# Set up logging
def setup_logging(log_path):
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

def crop_bottom_and_right_black_border(img: Image.Image):
    """
    Crop the black border at the bottom and right of the panorama.
    When you download streetviews get_panorama(), it's common to see black borders at the bottom and right of the image.
    This is because the dimensions of the panorama are not always correct / multiple of 512, a common issue with user-contributed panoramas.
    The implementation is not perfect, but it works for most cases.
    """
    (width, height) = img.size
    bw_img = img.convert("L")
    black_luminance = 4

    # Find the bottom of the panorama
    pixel_cursor = (0, height - 1)
    valid_max_y = height - 1
    while pixel_cursor[0] < width and pixel_cursor[1] >= 0:
        pixel_color = bw_img.getpixel(pixel_cursor)

        if pixel_color > black_luminance:
            # Found a non-black pixel
            # Double check if all the pixels below this one are black
            all_pixels_below = list(
                bw_img.crop((0, pixel_cursor[1] + 1, width, height)).getdata()
            )
            all_black = True
            for pixel in all_pixels_below:
                if pixel > black_luminance:
                    all_black = False

            if all_black:
                valid_max_y = pixel_cursor[1]
                break
            else:
                # A false positive, probably the actual valid bottom pixel is very close to black
                # Reset the cursor to the next vertical line to the right
                pixel_cursor = (pixel_cursor[0] + 1, height - 1)

        else:
            pixel_cursor = (pixel_cursor[0], pixel_cursor[1] - 1)

    # Find the right of the panorama
    pixel_cursor = (width - 1, 0)
    valid_max_x = width - 1
    while pixel_cursor[1] < height and pixel_cursor[0] >= 0:
        pixel_color = bw_img.getpixel(pixel_cursor)

        if pixel_color > black_luminance:
            # Found a non-black pixel
            # Double check if all the pixels to the right of this one are black
            all_pixels_to_the_right = list(
                bw_img.crop((pixel_cursor[0] + 1, 0, width, height)).getdata()
            )
            all_black = True
            for pixel in all_pixels_to_the_right:
                if pixel > black_luminance:
                    all_black = False
            if all_black:
                valid_max_x = pixel_cursor[0]
                break
            else:
                # A false positive, probably the actual valid right pixel is very close to black
                # Reset the cursor to the next horizontal line below
                pixel_cursor = (width - 1, pixel_cursor[1] + 1)

        else:
            pixel_cursor = (pixel_cursor[0] - 1, pixel_cursor[1])

    valid_height = valid_max_y + 1
    valid_width = valid_max_x + 1

    if valid_height == height and valid_width == width:
        # No black border found
        return img

    print(
        f"Found black border. Cropping from {width}x{height} to {valid_width}x{valid_height}"
    )
    return img.crop((0, 0, valid_width, valid_height))

# Function to remove panoramic images captured within a certain distance
def remove_adjacent_panoramics(pano_df, distance):
    # Convert latitude and longitude to Cartesian coordinates for distance calculation
    coords = np.vstack([pano_df['Panorama_Longitude'], pano_df['Panorama_Latitude']]).T
    pano_kd_tree = cKDTree(coords)

    # Query the tree to find the nearest neighbor for each point
    distances, indices = pano_kd_tree.query(coords, k=2)  # Find the nearest neighbor (k=2 because the nearest point is itself)
    distances_meters = distances * 111139  # Convert distances to meters

    # Find duplicate points within a set distance (meters)
    duplicates = np.where((distances_meters[:, 1] <= distance))[0]

    # Create a list to store indices to remove
    indices_to_remove = []

    # Iterate through the clusters and randomly keep one point while removing the rest
    for duplicate in duplicates:
        cluster_indices = indices[duplicate]
        # Randomly select one index to keep
        keep_index = np.random.choice(cluster_indices)
        # Remove other indices
        remove_indices = np.setdiff1d(cluster_indices, keep_index)
        indices_to_remove.extend(remove_indices)

    # Drop the indices to remove from the DataFrame
    pano_df = pano_df.drop(index=indices_to_remove)

    return pano_df
    

# Async function to fetch metadata
async def fetch_metadata(session, pano_id, api_key, retries=3):
    url = f"https://maps.googleapis.com/maps/api/streetview/metadata?pano={pano_id}&key={api_key}"
    for attempt in range(retries):
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('status') in ['ZERO_RESULTS', 'UNKNOWN_ERROR', 'NOT_FOUND', 'DATA_NOT_AVAILABLE']:
                        logging.warning(f"No panorama data found for pano_id: {pano_id}")
                        return None
                    # Return all necessary metadata in a dictionary format for simplicity
                    return {
                        'Panorama_ID': pano_id,
                        'Panorama_Date': data.get('date'),
                        'Panorama_Latitude': data.get('location', {}).get('lat'),
                        'Panorama_Longitude': data.get('location', {}).get('lng'),
                        'Panorama_Rotation': data.get('heading')
                    }
                else:
                    logging.warning(f"Error status {resp.status} for pano_id {pano_id}")
        except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
            logging.error(f"Error fetching metadata for {pano_id}: {e}")
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
    return None  # Return None if all retries fail

# Async function to search panoramas and fetch metadata concurrently
async def get_pano_metadata_async(points_coords, api_key):
    pano_data = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        for i in range(len(points_coords.geometry)):
            lat = points_coords.geometry.y[i]
            lon = points_coords.geometry.x[i]
            panos = search_panoramas(lat=lat, lon=lon)
            
            if not panos:
                logging.warning(f"No panoramas found at coordinates ({lat}, {lon})")
                continue

            # Create a task to fetch metadata for each panorama found
            for pano in panos:
                task = asyncio.ensure_future(fetch_metadata(session, pano.pano_id, api_key, retries=3))
                tasks.append((task, i))
                await asyncio.sleep(0.1)  # Delay to avoid server overload
        
        # Run all the tasks concurrently and process results
        responses = await asyncio.gather(*[task for task, _ in tasks])
        
        for response, (_, i) in zip(responses, tasks):
            if response is None:
                continue  # Skip any failed metadata fetch
            # Append metadata to pano_data
            pano_data.append({
                'Point_Index': i,
                'Point_Latitude': points_coords.geometry.y[i],
                'Point_Longitude': points_coords.geometry.x[i],
                **response  # Unpack metadata directly
            })
    
    return pd.DataFrame(pano_data)

# Download panoramic images async
async def download_panoramic_images(pano_df_simple, output_path):
    for i in tqdm(range(len(pano_df_simple))):
        # Get panoramic image unique ID
        panoID = pano_df_simple['Panorama_ID'][i]

        # Define the file path for the image
        image_path = fr'{output_path}/{panoID}.jpg'

        # Check if image already exists. If it does, skip that image.
        if os.path.exists(image_path):
            print(f"Image {panoID} already exists. Skipping...")
            continue

        try:
            print("Downloading Panoramic Image and Metadata:")

            # Create the matching panoramic image metadata .json file
            metadata = {
                'panoId': panoID,
                'panoDate': pano_df_simple['Panorama_Date'][i],
                'lat': pano_df_simple['Panorama_Latitude'][i],
                'lng': pano_df_simple['Panorama_Longitude'][i],
                'rotation': pano_df_simple['Panorama_Rotation'][i]
            }

            # Write metadata to JSON file
            with open(fr'{output_path}/{panoID}.metadata.json', 'w') as f:
                json.dump(metadata, f)

            # Attempt to download single panoramic image
            image = await get_panorama_async(pano_id=panoID, zoom = 5)
            
            image_cropped = crop_bottom_and_right_black_border(image)

            # Save the image
            image_cropped.save(image_path, "jpeg")

            print(f"Saved Panoramic Image and Metadata: ", panoID)
        except UnidentifiedImageError as e:
            print(f"Error downloading image {panoID}: {e}")
            await asyncio.sleep(2)  # Wait for 2 seconds before retrying
            continue  # Skip to the next iteration of the loop

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
    csv_roads_points_file = config['csv_roads_points_file']
    output_path = config['output_path']

    # Create output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    # Setup Logging
    log_path = os.path.join(output_path, 'log-image-downloader.log')
    setup_logging(log_path)
    logging.info(f"Configuration File: {config}")
    logging.info("Image Downloader Logging File")

    # Read Secret Google Street View API Key
    with open(api_key_path, 'r') as file:
        api_key = file.read().strip()
    logging.info("Found API Key")
    
    # Read in .csv file with points corresponding to 25- meter spacing of roads
    points_df = pd.read_csv(csv_roads_points_file)
    points_df['geometry'] = points_df['geometry'].apply(wkt.loads)
    points_coords = gpd.GeoDataFrame(points_df, geometry='geometry', crs="EPSG:4326")  # EPSG:4326 is for WGS84 (latitude/longitude)

    logging.info("Finding Panoramic Images Near Road Points")
    
    pano_df = asyncio.run(get_pano_metadata_async(points_coords, api_key))
    
    # Save pano_df
    pano_df_output_path = os.path.join(output_path, 'panoramic_images_metadata.csv')
    pano_df.to_csv(pano_df_output_path)

    # Remove duplicate images based on 'Panorama_ID'
    pano_df.drop_duplicates(subset='Panorama_ID', keep='first', inplace=True)

    logging.info(f'Total Available Panoramic Images: {len(pano_df)}')

    # Remove images where 'Panorama_Date' is after 2016-01
    pano_df = pano_df[pano_df['Panorama_Date'] >= '2016-01']
    pano_df.reset_index(drop=True, inplace=True)
    logging.info(f'Total Panoramic Images After 2016: {len(pano_df)}')

    # Remove any panoraamic images closer than 5 meters
    pano_df_simple = remove_adjacent_panoramics(pano_df, 5)
    pano_df_simple.reset_index(drop=True, inplace=True)

    logging.info(f'Total Panoramic Images After De-Duplication: {len(pano_df_simple)}')

    logging.info('Downloading Panoramic Images')

    # Run the async function to download panoramic images
    asyncio.run(download_panoramic_images(pano_df_simple, output_path))


if __name__ == '__main__':
    main()
    

