# Host Tree Classification Utilities

# Imports

import os
import pandas as pd
import numpy as np
import torch
import tqdm as tqdm
import json
import argparse
import logging


def process_image(image_path):
    """
    Function to process each image for summary statistics
    
    Parameters: image_path (str): 
    
    Returns: mean, std (float): mean and standard deviation of an image by channel
    
    """
    
    # Transforms to preprocess iNaturalist images for CNN
    transform = v2.Compose([
        v2.ToPILImage(),
        v2.Resize(size=(512, 512), antialias=True),
        v2.ToTensor(),
    ])

    try:
        img = imread(image_path)
        img = transform(img)  # Transforms image to [0-1] and channels first
        if torch.isnan(img).any():
            print(f"NaN found after transformation: {image_path}")
            return

        img = img.permute(1, 2, 0)  # Transforms image to channels last dimensions
        mean = torch.mean(img, dim=(0, 1))
        std = torch.std(img, dim=(0, 1))

        if torch.isnan(mean).any() or torch.isnan(std).any():
            print(f"NaN in statistics: {image_path}")
            return

        return mean, std
    except Exception as e:
        print(f"Error: {image_path} - {e}")
        return

def image_summary_statistics(image_root):
    """
    Function to compute average mean and standard deviation across image channels.
    Mean and standard deviation values used for normalization.
    
    Parameters: image_root (str): Path to directory containing image data
    
    Returns: summary_df (pandas df): Global mean and standard deviation per image channel normalization.
    
    Example:
    
    image root = r"D:\...\training_data_inat"
    
    Processing directories: 26it [29:13, 67.44s/it]
    Average Mean: tensor([0.5046, 0.5397, 0.4886])
    Average Std: tensor([0.2176, 0.2148, 0.2471])
    
    """
    
    # Initialize variables to accumulate mean and standard deviation values
    mean_sum = torch.zeros(3)
    std_sum = torch.zeros(3)
    num_images = 0
    
    for root, dirs, files in tqdm(os.walk(image_root), desc="Processing directories"):
        for file in tqdm(files, desc="Processing files", leave=False):
            if file.lower().endswith(('.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                result = process_image(image_path)
                if result:
                    mean, std = result
                    mean_sum += mean
                    std_sum += std
                    num_images += 1

    # Calculate and save the average mean and standard deviation
    if num_images > 0:
        avg_mean = mean_sum / num_images
        avg_std = std_sum / num_images
    else:
        print("No valid images processed.")

    print("Average Mean:", avg_mean)
    print("Average Std:", avg_std)

    # Save the average mean and standard deviation as data frame
    summary_df = pd.DataFrame({'mean': avg_mean, 'std': avg_std})
    #summary_df.to_csv(r"D:\blaginh\tree_classification\aa_inat_combined\training_data_top1000_may724_curated_mean_std.csv", index=False)
    return summary_df
