# Tree Geolocation

The 'tree_geolocation' module provides Python scripts to apply tree object detection (YOLOv5) and image classification (CNN), and depth estimate (ZoeDepth) models to geolocate street trees from Google Street View imagery.

## Files

- **geolocate.py**: Python script applying the tree detection, image classification, and depth estimation models to geolocate trees.
- **geolocation_config.json**: Config file for geolocate.py to specify paths for models, images to use for geolocation, and tree genera to identify.
- **geoutilities.py**: Python function library to load models, process imagery, apply model predictions, and geolocate trees.
- **/jupyter_notebooks**: Notebooks to experiment with tree geolocation workflows.

# Usage

# Export Python Path to Parent Directory of Tree_Geolocation Folder
Ex in Bash; export PYTHONPATH=$PYTHONPATH:/path/to/parent/directory

# Load Conda Environment with Torch
source activate my_env

# Inspect geolocation_config.json File to Specify Training Parameters

# Run Train.py to Train CNN
python -m tree_geolocation.geolocate --config "path\to\geolocation_config.json"

