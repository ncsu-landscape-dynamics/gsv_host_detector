# Filtering and Exporting iNaturalist Images for Trees based on OPENAI CLIP Model.
# Tutorial: https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/image-search/Image_Search.ipynb

# Imports
from sentence_transformers import SentenceTransformer, util # pip install transformers; pip install sentence_transformers
from PIL import Image
import numpy as np
import glob
import torch
import pickle
import zipfile
from tqdm import tqdm
from IPython.display import display
from IPython.display import Image as IPImage
import os

# Function to search for images nearest to a text query embedding
def search(query, img_embeddings, img_names, img_folder, out_folder, k=10):
    # Ensure the output folder exists; create it if not
    os.makedirs(out_folder, exist_ok=True)
    
    # Encode the query (text string) into an embedding
    query_emb = model.encode([query], convert_to_tensor=True, show_progress_bar=False)
    
    # Perform semantic search to find images closest to the query embedding
    hits = util.semantic_search(query_emb, img_embeddings, top_k=k)[0]
    
    print("Query:")
    display(query)
    
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

# Get names for all tree genera
selected_genera = ["acer","ailanthus","betula","citrus","cupaniopsis","erythrina","fraxinus","gleditsia",
"juglans","juniperus","magnolia","phoenix","picea","pinus","prunus","pseudotsuga","pyrus",
"quercus","rhus","sequoia","taxodium","thuja","tilia","ulmus","washingtonia"]

# Iterate through all genera for image filtering based on CLIP prompt
for genus in selected_genera:
    # Set up paths
    img_folder_genus = f'D:/inat/images/original_full/{genus}/'
    out_folder_genus = f'D:/blaginh/tree_classification/inaturalist/training_data_top100_july17_clip_curated/{genus}'

    # Get image file names with .jpg and .jpeg extensions
    img_names_jpg = glob.glob(img_folder_genus + '*.jpg')
    img_names_jpeg = glob.glob(img_folder_genus + '*.jpeg')
    img_names = img_names_jpg + img_names_jpeg

    # Process images in smaller batches
    batch_size = 1024  # Adjust based on your system resources
    img_embeddings = []

    for i in range(0, len(img_names), batch_size):
        valid_images = []
        valid_filepaths = []

        # Process each image in the batch
        for filepath in img_names[i:i + batch_size]:
            try:
                with Image.open(filepath) as img:
                    img.verify()  # Verify the image is not corrupted
                    img = Image.open(filepath)  # Reopen the image for processing
                    img.load()  # Ensure the image is fully loaded
                    valid_images.append(img)
                    valid_filepaths.append(filepath)
            except (UnidentifiedImageError, OSError) as e:
                print(f"Skipping corrupted or truncated image: {filepath}, error: {e}")

        # Only encode if there are valid images
        if valid_images:
            embeddings = model.encode(valid_images, batch_size=128, convert_to_tensor=True, show_progress_bar=True)
            img_embeddings.append(embeddings)

    # Concatenate embeddings from all parts into a single tensor
    img_embeddings = torch.cat(img_embeddings)

    # Perform a search with a sample query (adjust as needed)
    sample_query = "A photo of a large mature tree urban landscape"
    search(sample_query, img_embeddings, img_names, img_folder_genus, out_folder_genus, k=100)

# EOF