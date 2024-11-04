# Image Downloading

The `image_downloaders` module provides scripts for downloading image dat for training and evaluating a Convolutional Neural Network (CNN).

## Folder Structure

In `/image_downloaders`, you will find Python scripts for configuring and downloading GSV images.

- **get_panos_and_metadata_async_csv.py**: Script for downloading GSV images.
- **downloader_config_csv.json**: Configuration file for downloading GSV images.
- **tiny_25m_near_samples.csv**: CSV file of points along roads to download GSV images.
- **osmnx_functions.py**: Script with functions for working with OpenStreetMaps roads data.

In the `/r_scripts` directory, you will find the following key scripts and configurations:

- **eddmaps_img_dler.R**: Script for downloading images from EddMaps.
- **inat_img_dler.R**: Script for downloading images from iNaturalist.

## Usage for GSV Downloader

### 1. Export Python Path
Ensure the Python path is set to the parent directory of the `image_downloaders` folder. For example:
export PYTHONPATH=$PYTHONPATH:/path/to/parent_directory

### 2. Activate Conda Environment
source activate my_env

### 3. Modify Config File
Specify experiment name, path to API key, and locations for downloading.

### 3. Run the downloader
python -m image_downloaders.get_panos_and_metadata_async_csv --config "/path/to/train_config.json"
