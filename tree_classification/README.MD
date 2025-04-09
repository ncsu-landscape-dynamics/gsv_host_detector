# Tree Classification

The `tree_classification` module provides scripts for preprocessing data, training a Convolutional Neural Network (CNN), and evaluating the performance of the model specifically designed for host tree classification.

## Folder Structure

In the `/tree_classification` directory, you will find the following key scripts and configurations:

- **preprocess.py**: For data preprocessing, creating training and testing datasets, and combining images across datasets.
- **train.py**: For training the CNN image classification model.
- **evaluate.py**: For evaluating a trained CNN on a testing dataset.
- **models.py**: Contains model architectures for EfficientNet and helper functions.
- **utilities.py**: Functions to calculate summary statistics for images.
- **train_config.json**: Configuration file for training the image classification models.
- **eval_config.json**: Configuration file for evaluating the image classification models.
- **iNaturalist_Image_Filtering_CLIP_hpc.py**: Python script to run contrastive language image pretraining (CLIP) on iNaturalist images.
- **image_checks.sh**: Script to identify non-jpg and non-jpeg images.
- **weights_100_genera.txt**: Experimental weights to adjust loss function while training CNNs.

Notes: files named with *_hpc are for running on a high performance compute (HPC) cluster. These files also have accompanying .sh scripts to submit HPC jobs.

## Usage

### 1. Export Python Path
Ensure the Python path is set to the parent directory of the `tree_classification` folder. For example:
export PYTHONPATH=$PYTHONPATH:/path/to/parent_directory

### 2. Activate Conda Environment
source activate my_env

### 3. Modify Config File
Specify experiment name, model training and evaluation parameters, tree genera, and output paths.

### 3. Preprocess Data and Train the Model
python -m tree_classification.train --config "/path/to/train_config.json"

### 4. Evaluate the Model
python -m tree_classification.evaluate --config "/path/to/eval_config.json"

### Additional Information
For detailed information on model performance and other analyses, refer to the Jupyter notebooks and R markdown files available in the jupyter_notebooks and r_markdowns directories, respectively.

### Geopriors
The `/geopriors` directory contains exploratory analyses to modify CNN predictions based on prior information about exisitng tree genera in a geographic region.
