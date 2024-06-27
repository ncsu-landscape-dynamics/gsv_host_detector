# Tree Classification

The `tree_classification` module provides scripts for preprocessing data, training a Convolutional Neural Network (CNN), and evaluating the performance of the model specifically designed for host tree classification.

## Folder Structure

In the `/tree_classification` directory, you will find the following key scripts and configurations:

- **preprocess.py**: Scripts for data preprocessing.
- **train.py**: Script for training the CNN model.
- **evaluate.py**: Script for evaluating the trained model.
- **models.py**: Contains model architectures.
- **utilities.py**: Utility functions used across different scripts.
- **train_config.json**: Configuration file for training parameters.
- **eval_config.json**: Configuration file for evaluation parameters.

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
