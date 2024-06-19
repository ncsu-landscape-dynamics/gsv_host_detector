# Functions for Downloading Google Street View Panoramic Images

# Thomas Lake, Brit Laginhas, June 2024

# Imports
import argparse
import json
import os
import logging
import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
import random
import networkx as nx
import osmnx as ox



# Following implementation of OSMnx in the manuscript: https://www.sciencedirect.com/science/article/pii/S221067072400091X?via%3Dihub#bib0020
# and code implementation of OSMnx from the manuscript: https://github.com/Spatial-Data-Science-and-GEO-AI-Lab/StreetView-NatureVisibility

def get_road_network_from_point(location, dist):
    # Get the road network graph using OpenStreetMap data
    # 'network_type' argument is set to 'drive' to get the road network suitable for driving
    # 'simplify' argument is set to 'True' to simplify the road network
    G = ox.graph_from_point(location, dist, dist_type = 'network', network_type='drive', simplify=True)

    # Create a set to store unique road identifiers
    unique_roads = set()
    # Create a new graph to store the simplified road network
    G_simplified = G.copy()

    # Iterate over each road segment
    for u, v, key, data in G.edges(keys=True, data=True):
        # Check if the road segment is a duplicate
        if (v, u) in unique_roads:
            # Remove the duplicate road segment
            G_simplified.remove_edge(u, v, key)
        else:
            # Add the road segment to the set of unique roads
            unique_roads.add((u, v))
    
    # Update the graph with the simplified road network
    G = G_simplified
    
    # Project the graph from latitude-longitude coordinates to a local projection (in meters)
    G_proj = ox.project_graph(G)

    # Convert the projected graph to a GeoDataFrame
    # This function projects the graph to the UTM CRS for the UTM zone in which the graph's centroid lies
    _, edges = ox.graph_to_gdfs(G_proj) 

    return G, edges



def select_points_on_road_network(roads, N=15):
    # Get a list of points over the road map with a N distance between them
    points = []
    # Iterate over each road
    
    for row in roads.itertuples(index=True, name='Road'):
        # Get the LineString object from the geometry
        linestring = row.geometry
        index = row.Index

        # Calculate the distance along the linestring and create points every 50 meters
        for distance in range(0, int(linestring.length), N):
            # Get the point on the road at the current position
            point = linestring.interpolate(distance)

            # Add the curent point to the list of points
            points.append([point, index])
    
    # Convert the list of points to a GeoDataFrame
    gdf_points = gpd.GeoDataFrame(points, columns=["geometry", "road_index"], geometry="geometry")

    # Set the same CRS as the road dataframes for the points dataframe
    gdf_points.set_crs(roads.crs, inplace=True)

    # Drop duplicate rows based on the geometry column
    gdf_points = gdf_points.drop_duplicates(subset=['geometry'])
    gdf_points = gdf_points.reset_index(drop=True)

    return gdf_points



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




