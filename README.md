# Host Tree Detection and Classification with Google Street View

This repository contains code for detecting and classifying host trees using Google Street View imagery. The project consists of four main components:

## Image Downloaders

The `/image_downloaders` folder contains tools for obtaining two types of images:
- iNaturalist host tree images
- Google Street View panoramic images

## Tree Classification

In the `/tree_classification` folder, you'll find scripts for developing and evaluating a convolutional neural network (CNN) tailored for host tree classification.

## Tree Detection

The `/tree_detection` folder houses scripts to apply a pre-trained object detection model (YOLOv5) to identify trees in Google Street View panoramic images.

## Tree Geolocation

The `/tree_geolocation` folder contains scripts that integrate detection and classification models into a workflow. This workflow triangulates and assigns geolocation coordinates to street trees.

![Example Tree Detection](https://github.com/ncsu-landscape-dynamics/gsv_host_detector/blob/main/yolov5-prediction-sample-tree.jpg?raw=true)
