# Filtering and Exporting iNaturalist Images for Trees based on OPENAI CLIP Model.
# Tutorial: https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/image-search/Image_Search.ipynb

# Imports
import sys
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import numpy as np
import glob
import torch
import pickle
import zipfile
from tqdm import tqdm
import os

# Function to search for images nearest to a text query embedding
def search(query, img_embeddings, img_names, img_folder, out_folder, k=10):
    # Ensure the output folder exists; create it if not
    os.makedirs(out_folder, exist_ok=True)
    
    # Encode the query (text string) into an embedding
    query_emb = model.encode([query], convert_to_tensor=True, show_progress_bar=False)
    
    # Perform semantic search to find images closest to the query embedding
    hits = util.semantic_search(query_emb, img_embeddings, top_k=k)[0]
    
    # Iterate over the top-k hits and display/save the corresponding images
    for idx, hit in enumerate(hits):
        # Get the image path from img_names using corpus_id
        img_path = os.path.join(img_folder, img_names[hit['corpus_id']])
        
        # Display the image
        #print(f"Rank {idx + 1}: {img_path}")
        #display(IPImage(filename=img_path, width=200))
        
        # Save the image to out_folder with the original file name
        image_name = os.path.basename(img_path)  # Get the original file name
        out_path = os.path.join(out_folder, image_name)
        
        # Save the image
        Image.open(img_path).save(out_path)
        
        print(f"Saved result {idx + 1} to: {out_path}")

#First, we load the respective CLIP model
model = SentenceTransformer('clip-ViT-B-32')

# Get genus from command-line argument
genus = sys.argv[1]

# Set up paths
img_folder_genus = f'/rs1/researchers/c/cmjone25/auto_arborist_cvpr2022_v0.15/data/tree_classification/inat/images/original_full/{genus}/'
out_folder_genus = f'/rs1/researchers/c/cmjone25/auto_arborist_cvpr2022_v0.15/data/tree_classification/inat/images/original_full_clip_filtered/{genus}/'

# Get image file names
img_names_jpg = glob.glob(img_folder_genus + '*.jpg')
img_names_jpeg = glob.glob(img_folder_genus + '*.jpeg')
img_names = img_names_jpg + img_names_jpeg

# Determine number of images to filter
num_imgs_genus = len(img_names)
if num_imgs_genus >= 10000:
    query_k = 1000
else:
    query_k = int(np.floor(num_imgs_genus * .10))

batch_size = 1024
img_embeddings = []

for i in range(0, len(img_names), batch_size):
    valid_images = []
    valid_filepaths = []

    for filepath in img_names[i:i + batch_size]:
        try:
            with Image.open(filepath) as img:
                img.verify()
                img = Image.open(filepath)
                img.load()
                valid_images.append(img)
                valid_filepaths.append(filepath)
        except (UnidentifiedImageError, OSError) as e:
            print(f"Skipping corrupted or truncated image: {filepath}, error: {e}")

    if valid_images:
        embeddings = model.encode(valid_images, batch_size=128, convert_to_tensor=True, show_progress_bar=True)
        img_embeddings.append(embeddings)

img_embeddings = torch.cat(img_embeddings)
sample_query = "A photo of a large mature tree urban landscape"
search(sample_query, img_embeddings, img_names, img_folder_genus, out_folder_genus, k=query_k)
# EOF
