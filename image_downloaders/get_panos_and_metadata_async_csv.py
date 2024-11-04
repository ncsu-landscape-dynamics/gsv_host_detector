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
from typing import List, Optional
from pydantic import BaseModel, ValidationError
from requests.models import Response
import re
from streetview import get_panorama_async


# Set up logging
def setup_logging(log_path):
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

class Panorama(BaseModel):
    pano_id: str
    lat: float
    lon: float
    heading: float
    pitch: Optional[float]
    roll: Optional[float]
    date: Optional[str]
    elevation: Optional[float]

class Location(BaseModel):
    lat: float
    lng: float

class MetaData(BaseModel):
    date: Optional[str]
    location: Location
    pano_id: str
    copyright: str

def get_panorama_meta(pano_id: str, api_key: str) -> Optional[MetaData]:
    """
    Returns a panorama's metadata.

    Quota: This function doesn't use up any quota or charge on your API_KEY.

    Endpoint documented at:
    https://developers.google.com/maps/documentation/streetview/metadata
    """
    url = (
        "https://maps.googleapis.com/maps/api/streetview/metadata"
        f"?pano={pano_id}&key={api_key}"
    )
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status() # Raise an exception for bad status codes
    except requests.exceptions.ReadTimeout:
        print("Timeout occurred while reading data from the server.")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
    else:
        # Process the response if no exception occurs
        resp_data = resp.json()

    # Now create the MetaData object safely
    try:
        return MetaData(**resp_data)
    except ValidationError as e:
        print(f"Validation Error for pano_id {pano_id}: {e}")
        return None  # Return None if validation fails

def search_panoramas(lat: float, lon: float) -> List[Panorama]:
    """
    Gets the closest panoramas (ids) to the GPS coordinates.
    """

    resp = search_request(lat, lon)
    pans = extract_panoramas(resp.text)
    return pans

def make_search_url(lat: float, lon: float) -> str:
    """
    Builds the URL of the script on Google's servers that returns the closest
    panoramas (ids) to a give GPS coordinate.
    """
    url = (
        "https://maps.googleapis.com/maps/api/js/"
        "GeoPhotoService.SingleImageSearch"
        "?pb=!1m5!1sapiv3!5sUS!11m2!1m1!1b0!2m4!1m2!3d{0:}!4d{1:}!2d50!3m10"
        "!2m2!1sen!2sGB!9m1!1e2!11m4!1m3!1e2!2b1!3e2!4m10!1e1!1e2!1e3!1e4"
        "!1e8!1e6!5m1!1e2!6m1!1e2"
        "&callback=callbackfunc"
    )
    return url.format(lat, lon)


def search_request(lat: float, lon: float) -> Response:
    """
    Gets the response of the script on Google's servers that returns the
    closest panoramas (ids) to a give GPS coordinate.
    """
    url = make_search_url(lat, lon)
    return requests.get(url)
    
def extract_panoramas(text: str) -> List[Panorama]:
    """
    Given a valid response from the panoids endpoint, return a list of all the
    panoids.
    """
    try:
    
        # The response is actually javascript code. It's a function with a single
        # input which is a huge deeply nested array of items.
        blob = re.findall(r"callbackfunc\( (.*) \)$", text)[0]
        data = json.loads(blob)

        if data == [[5, "generic", "Search returned no images."]]:
            return []

        subset = data[1][5][0]

        raw_panos = subset[3][0]

        if len(subset) < 9 or subset[8] is None:
            raw_dates = []
        else:
            raw_dates = subset[8]

        # For some reason, dates do not include a date for each panorama.
        # the n dates match the last n panos. Here we flip the arrays
        # so that the 0th pano aligns with the 0th date.
        raw_panos = raw_panos[::-1]
        raw_dates = raw_dates[::-1]

        dates = [f"{d[1][0]}-{d[1][1]:02d}" for d in raw_dates]

        return [
            Panorama(
                pano_id=pano[0][1],
                lat=pano[2][0][2],
                lon=pano[2][0][3],
                heading=pano[2][2][0],
                pitch=pano[2][2][1] if len(pano[2][2]) >= 2 else None,
                roll=pano[2][2][2] if len(pano[2][2]) >= 3 else None,
                date=dates[i] if i < len(dates) else None,
                elevation=pano[3][0] if len(pano) >= 4 else None,
            )
            for i, pano in enumerate(raw_panos)
        ]
    
    except (IndexError, json.JSONDecodeError) as e:
        print("Error parsing panorama")
        return []
    

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
    points_df['geometry'] = gpd.points_from_xy(points_df['Longitude'], points_df['Latitude'])
    points_coords = gpd.GeoDataFrame(points_df, geometry='geometry', crs="EPSG:4326")  # EPSG:4326 is for WGS84 (latitude/longitude)

    logging.info("Finding Panoramic Images Near Road Points")
    
    pano_data = []

    # Iterate over each point in the road network, and get metadata for the nearest panoramic images
    for i in tqdm(range(len(points_coords.geometry))):
        # Search for all available panoramic images closest to each point
        panos = search_panoramas(lat=points_coords.geometry.y[i], lon=points_coords.geometry.x[i])

        for pano in panos:
            # Fetch metadata
            meta = get_panorama_meta(pano_id=pano.pano_id, api_key=api_key)
            
            # Ensure meta is valid before proceeding
            if meta is None:
                print(f"Skipping pano_id {pano.pano_id} due to missing metadata.")
                continue

            if meta.date:
                date_code = datetime.strptime(meta.date, '%Y-%m')

                # Append data on point and panoramic image location
                pano_data.append({
                    'Point_Index': i,
                    'Point_Latitude': points_coords.geometry.y[i],
                    'Point_Longitude': points_coords.geometry.x[i],
                    'Panorama_ID': pano.pano_id,
                    'Panorama_Date': meta.date,
                    'Panorama_Latitude': pano.lat,
                    'Panorama_Longitude': pano.lon,
                    'Panorama_Rotation': pano.heading
                })


    # All available panoramic images sampled from road points
    pano_df = pd.DataFrame(pano_data)
    
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
    

