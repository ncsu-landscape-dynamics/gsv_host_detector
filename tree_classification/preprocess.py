# Host Tree Classification Preprocessing

"""
The preprocessing module contains functions to create datasets for training and testing, export metadata to csv, and print directory information.
"""

# Imports
import os
import shutil
import random
import warnings
import time
import numpy as np
import pandas as pd


def print_directory_info(root_directory):
    """
    Function to print directory information. 
    
    Print the number of files in each directory in the root directory.

    Parameters:
    root_directory (str): The path to the root directory containing images of tree genera.

    Returns:
    Print statement with the number of files in each directory.
    
    Usage: print_directory_info(training_destination_root)
    
    """
    for genus_folder in os.listdir(root_directory):
        genus_path = os.path.join(root_directory, genus_folder)
        
        # Check if it's a directory
        if os.path.isdir(genus_path):
            # Count the number of files in the directory
            num_files = len([f for f in os.listdir(genus_path) if os.path.isfile(os.path.join(genus_path, f))])
            
            print(f"Directory: {genus_folder}, Number of Files: {num_files}")
    return None


def process_existing_files(root_directory, csv_path, export_csv=False):
    """
    Function to export image metadata as csv.
    
    Process existing files in the root directory and write to csv.

    Parameters:
    root_directory (str): The path to the root directory containing images of tree genera.

    Returns:
    Print statement with the file path to the csv.
    """
    existing_files = {}

    # Get the list of genus names from the root directory
    genera = os.listdir(root_directory)

    for genus in genera:
        genus_dir = os.path.join(root_directory, genus)
        genus_files = os.listdir(genus_dir)
        existing_files[genus] = genus_files

    # Create a dataframe from the dictionary
    df = pd.DataFrame(existing_files.items(), columns=['genus', 'files'])
    df = df.explode('files')

    # Add additional column for data source
    if "inat" in root_directory.lower():
        df['data_source'] = "iNaturalist"
    else:
        df['data_source'] = "Autoarborist"
    
    if export_csv:
        # Write to csv
        csv_filepath = os.path.join(csv_path, os.path.basename(root_directory) + ".csv")
        df.to_csv(csv_filepath, index=False)
        print(f"Existing files for {root_directory} written to csv.")
    else:
        return df
        
        
def create_datasets (training_ratio, max_training_images, max_testing_images, selected_genera, source_root, testing_destination_root, 
                     training_destination_root, existing_training_root, existing_testing_root, append, csv_path):
    """
    Function to create datasets for training and testing: Autoarborist or iNaturalist
    
    Create training and testing datasets for selected genera from any source image dataset.
    
    Parameters:
    training_ratio (float): The ratio of training images to total images.
    max_training_images (int): The maximum number of training images to select.
    max_testing_images (int): The maximum number of testing images to select.
    selected_genera (list): The list of selected genera to include in the training and testing datasets.
    source_root (str): The path to the source directory containing all available images of tree genera from Autoarborist.
    testing_destination_root (str): The path to the destination directory for images of tree genera as testing data.
    training_destination_root (str): The path to the destination directory for images of tree genera as training data.
    existing_training_root (str): The path to the existing directory containing images of tree genera as training data used in previous experiments.
    existing_testing_root (str): The path to the existing directory containing images of tree genera as testing data used in previous experiments.
    append (bool): A logical statement to append new images to existing training and testing data.
    
    Returns:
    None
    """
    # Iterate through the source directory
    for genus_folder in os.listdir(source_root):
        # Keep track of starting time
        start_time = time.time()
        genus_path = os.path.join(source_root, genus_folder) # Get path to images for each genera
    
        # List all images in the current genus folder. Some images are .jpg and .jpeg format.
        # If "inat" is found anywhere in the genus folder name, then the images are in the root folder.
        if "inat" in genus_path.lower():
            images = [image for image in os.listdir(genus_path) if image.lower().endswith(('.png', '.jpg', '.jpeg'))]
        else:
            images = [image for image in os.listdir(os.path.join(genus_path, 'images')) if image.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # Only select genera with >100 images.
        if len(images) > 100:
            # Check if it's a directory and if it's in the selected genera list
            if os.path.isdir(genus_path) and genus_folder in selected_genera:
                # Create destination folders for the training and testing data for the current genus
                training_destination_genus_path = os.path.join(training_destination_root, genus_folder)
                testing_destination_genus_path = os.path.join(testing_destination_root, genus_folder)
                os.makedirs(training_destination_genus_path, exist_ok=True)
                os.makedirs(testing_destination_genus_path, exist_ok=True)
                # Append new images to existing training and testing data
                if append:
                    print(f"Copying existing images for {genus_folder} to new training and testing folders.")
                    existing_training_genus_path = os.path.join(existing_training_root, genus_folder)
                    existing_testing_genus_path = os.path.join(existing_testing_root, genus_folder)

                    # Copy existing training images to the new training destination folder
                    for image in os.listdir(existing_training_genus_path):
                        source_image_path = os.path.join(existing_training_genus_path, image)
                        destination_image_path = os.path.join(training_destination_genus_path, image)
                        _= shutil.copy2(source_image_path, destination_image_path)

                    # Copy existing testing images to the new testing destination folder
                    for image in os.listdir(existing_testing_genus_path):
                        source_image_path = os.path.join(existing_testing_genus_path, image)
                        destination_image_path = os.path.join(testing_destination_genus_path, image)
                        _= shutil.copy2(source_image_path, destination_image_path)

                    # Update the max_training_images and max_testing_images
                    max_training_images = max_training_images - len(os.listdir(existing_training_genus_path))
                    max_testing_images = max_testing_images - len(os.listdir(existing_testing_genus_path))
                    print(f"Updated max_training_images: {max_training_images}, max_testing_images: {max_testing_images}")

                    # Update images to exclude the existing training and testing images
                    print(f"Total number of available images: {len(images)} for {genus_folder}...")
                    existing_training_images = set(os.listdir(existing_training_genus_path))
                    existing_testing_images = set(os.listdir(existing_testing_genus_path))
                    images = [image for image in images if image not in existing_training_images and image not in existing_testing_images]
                    print(f"Total number of images after excluding existing training and testing images: {len(images)} for {genus_folder}...")

                # Randomly select a number of images from the folder here: (900 training + 100 testing).
                if len(images) > max_training_images + max_testing_images:
                    images = random.sample(images, max_training_images + max_testing_images) # file paths for images

                # Randomly divide images into training and testing sets
                num_total_images = len(images)
                num_training_images_to_copy = min(int(num_total_images * training_ratio), max_training_images)

                # Randomly shuffle the images before moving
                random.shuffle(images)

                # Split images into training and testing sets
                training_images = images[:num_training_images_to_copy]
                testing_images = images[num_training_images_to_copy:]
            
                # Copy training images to the training destination folder
                print(f"Copying new images for {genus_folder} to new training and testing folders.")
                for image in training_images:
                    if "inat" in genus_path.lower():
                        source_image_path = os.path.join(genus_path, image)
                    else:
                        source_image_path = os.path.join(genus_path, 'images', image)
                    destination_image_path = os.path.join(training_destination_genus_path, image)
                    _= shutil.copy2(source_image_path, destination_image_path)

                # Copy testing images to the testing destination folder
                for image in testing_images:
                    if "inat" in genus_path.lower():
                        source_image_path = os.path.join(genus_path, image)
                    else:
                        source_image_path = os.path.join(genus_path, 'images', image)
                    destination_image_path = os.path.join(testing_destination_genus_path, image)
                    _= shutil.copy2(source_image_path, destination_image_path)
                # Keep track of ending time
                end_time = time.time()
                # Report time take in minutes
                total_time = (end_time - start_time) / 60
                print(f"Images copied successfully for {genus_folder}. Time taken: {total_time} minutes.")
    print(f"All images copied successfully for: {selected_genera}.")
    
    process_existing_files(training_destination_root, csv_path, export_csv=True)
    
    return None


def combine_datasets(autoarborist_dataset_root, inaturalist_dataset_root, combined_dataset_root):
    """
    Function to combine Autoarborist and iNaturalist training or testing datasets and metadata.
    Combine the training datasets from Autoarborist and iNaturalist into a single testing or training dataset.
    Combine the testing datasets from Autoarborist and iNaturalist into a single testing or training dataset.
    Combine the metadata from Autoarborist and iNaturalist into a single metadata file.

    Parameters:
    autoarborist_dataset_root (str): The path to the Autoarborist testing or training dataset.
    inaturalist_dataset_root (str): The path to the iNaturalist testing or training dataset.
    combined_dataset_root (str): The path to the combined testing or training dataset.

    Returns:
    None
    """
    # Create the combined training dataset directory
    os.makedirs(combined_dataset_root, exist_ok=True)

    # Copy the Autoarborist dataset to the combined dataset
    for genus_folder in os.listdir(autoarborist_dataset_root):
        genus_path = os.path.join(autoarborist_dataset_root, genus_folder)
        destination_genus_path = os.path.join(combined_dataset_root, genus_folder)
        os.makedirs(destination_genus_path, exist_ok=True)
        print(f"Copying Autoarborist images for {genus_folder} to the combined dataset.")

        for image in os.listdir(genus_path):
            source_image_path = os.path.join(genus_path, image)
            destination_image_path = os.path.join(destination_genus_path, image)
            _= shutil.copy2(source_image_path, destination_image_path)
        
    
    # Copy the iNaturalist dataset to the training dataset
    for genus_folder in os.listdir(inaturalist_dataset_root):
        genus_path = os.path.join(inaturalist_dataset_root, genus_folder)
        destination_genus_path = os.path.join(combined_dataset_root, genus_folder)
        os.makedirs(destination_genus_path, exist_ok=True)
        print(f"Copying iNaturalist images for {genus_folder} to the combined dataset.")

        for image in os.listdir(genus_path):
            source_image_path = os.path.join(genus_path, image)
            destination_image_path = os.path.join(destination_genus_path, image)
            _= shutil.copy2(source_image_path, destination_image_path)

    print(f"Combined dataset created successfully.")

    # Combine the autoarborist_root, inaturalist_root, and combined_root metadata
    roots = [autoarborist_dataset_root, inaturalist_dataset_root, combined_dataset_root]
    updated_roots = [os.path.join(os.path.join(root.rsplit('\\', 1)[0], os.path.basename(root) + ".csv")) for root in roots]
    
    # Pull in the existing metadata for autoarborist
    autoarborist_metadata = pd.read_csv(updated_roots[0])
    # Pull in the existing metadata for inaturalist
    inaturalist_metadata = pd.read_csv(updated_roots[1])
    # Combine the metadata
    combined_metadata = pd.concat([autoarborist_metadata, inaturalist_metadata], ignore_index=True)
    # Write to csv
    combined_metadata.to_csv(updated_roots[2], index=False)
    print(f"Combined metadata written to csv.")
    return None
