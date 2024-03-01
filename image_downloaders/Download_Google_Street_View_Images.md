# Accessing Street-Level Imagery with Open Street Maps 

Author: Thomas Lake

Using OSMnx to create street networks from OpenStreetMap.
See research paper: Boeing, 2017. OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks.

The following notebook explores the Python OSMnx package (https://osmnx.readthedocs.io/en/stable/getting-started.html#introducing-osmnx)

You can download a street network by providing OSMnx any of the following:

- a bounding box
- a lat-long point plus a distance
- an address plus a distance
- a place name or list of place names (to automatically geocode and get the boundary of)
- a polygon of the desired street network's boundaries
- a .osm formatted xml file

You can also specify several different network types:

- 'drive' - get drivable public streets (but not service roads)
- 'drive_service' - get drivable streets, including service roads
- 'walk' - get all streets and paths that pedestrians can use (this network type ignores one-way directionality)
- 'bike' - get all streets and paths that cyclists can use
- 'all' - download all non-private OSM streets and paths (this is the default network type unless you specify a different one)
- 'all_private' - download all OSM streets and paths, including private-access ones

Once created, maps of street networks derived from Open Street Maps can be used to create regularly-spaced points along publicly-accessabile streets, and then sample Google Street View locations from points along a road network.

This process is outlined in the research paper: Vazquez Sanchez and Labib, 2024. Accessing eye-level greenness visibility from open-source street view images: A methodological development and implementation in multi-city and multi-country contexts. See: https://github.com/Spatial-Data-Science-and-GEO-AI-Lab/StreetView-NatureVisibility

To programatically download google street view images from locations using the Google API, see the Python package 'streetview': https://github.com/robolyst/streetview/tree/master


# Imports


```python
# Imports for OSMnx
# https://osmnx.readthedocs.io/en/stable/user-reference.html
import networkx as nx
import osmnx as ox
print(ox.__version__) # OSMnx plans to update to version 2.0 in 2024.

# Imports for Google Street View Image downloader
# https://github.com/robolyst/streetview/tree/master
from streetview import search_panoramas, get_panorama_meta, get_streetview, get_panorama

# Other Imports

from tqdm import tqdm
from datetime import datetime
import requests
import json
import pandas as pd
import geopandas as gpd
import numpy as np
import folium
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from shapely.geometry import Point
import geopy.distance


```


```python
# Read Google Street View API Key for downloading GSV images

# Read secret API key
key_path = r'C:\Users\talake2\Desktop\auto_arborist_cvpr2022_v015\api_keys\Google_Street_View_Static_API_Key.txt'
with open(key_path, 'r') as file:
    GOOGLE_MAPS_API_KEY = file.read().strip()
```

# OSMnx Functions



```python
# Following implementation of OSMnx in the manuscript: https://www.sciencedirect.com/science/article/pii/S221067072400091X?via%3Dihub#bib0020
# and code implementation of OSMnx from the manuscript: https://github.com/Spatial-Data-Science-and-GEO-AI-Lab/StreetView-NatureVisibility

def get_road_network(location, dist):
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

    return G_simplified, edges



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
```



# Setup. Define Locations to Sample Road Networks for Tree Geolocation Analyses


```python

# Create dictionary of 20 testing locations for tree geolocation analyses
apple_park = (37.33354, -122.00567)
buffalo = (42.92952, -78.87408)
columbus = (40.15238321, -82.97449574)
la = (33.75995068, -118.286375)
montreal = (45.46182151, -73.60870962)
nyc = (40.59762804, -73.96560899)
nyc_2 = (40.67957647, -73.73748243)
pittsburgh = (40.44788537, -80.01430011)
sanfran = (37.72658289, -122.4719503)
sanfran_lombard = (37.80247707, -122.4181456)
sanjose = (37.28612748, -121.8084685)
seattle = (47.62319504, -122.3574537)
seattle_3 = (47.57606272, -122.3865466)
siouxfalls = (43.49218953, -96.72747222)
siouxfalls_2 = (43.49514608, -96.73300202)
siouxfalls_3 = (43.5491668, -96.73438307)
vancouver = (49.25261711, -123.023932)
vancouver_4 = (49.21252079, -123.0568194)
washington = (38.94273133, -76.99800397)
washington_2 = (38.88023974, -77.01280141)


sel_location = nyc

```

# Step 1. Get the road network

The first step of the code is to retrieve the road network for a specific place using OpenStreetMap data with the help of the OSMNX library. It begins by fetching the road network graph, focusing on roads that are suitable for driving. One important thing to note is that for bidirectional streets, the osmnx library returns duplicate lines. In this code, we take care to remove these duplicates and keep only the unique road segments to ensure that samples are not taken on the same road multiple times, preventing redundancy in subsequent analysis.

Following that, the code proceeds to project the graph from its original latitude-longitude coordinates to a local projection in meters. This projection is crucial for achieving accurate measurements in subsequent steps where we need to calculate distances between points. By converting the graph to a local projection, we ensure that our measurements align with the real-world distances on the ground, enabling precise analysis and calculations based on the road network data.



```python
# Set distance to get road network
distance = 300

# Create road network graph from point
graph, road = get_road_network(sel_location, distance)

# View first 5 road edges
road.head(5)

# Calculate summary statistics for road network graph
road_proj = ox.project_graph(graph)
nodes_proj = ox.graph_to_gdfs(road_proj, edges=False)
graph_area_m = nodes_proj.unary_union.convex_hull.area

# Output summary statistics of road network
# To get density-based statistics, you must also pass the network's bounding area in square meters
# Information on summary statistics: 
stats = ox.basic_stats(road_proj, area=graph_area_m, clean_int_tol=15)
pd.Series(stats)

```

    C:\Users\talake2\AppData\Local\anaconda3\envs\ox\Lib\site-packages\osmnx\utils_graph.py:513: FutureWarning: <class 'geopandas.array.GeometryArray'>._reduce will require a `keepdims` parameter in the future
      dupes = edges[mask].dropna(subset=["geometry"])
    




    n                                                                      14
    m                                                                      17
    k_avg                                                            2.428571
    edge_length_total                                                2050.389
    edge_length_avg                                                120.611118
    streets_per_node_avg                                                  4.0
    streets_per_node_counts                   {0: 0, 1: 0, 2: 0, 3: 0, 4: 14}
    streets_per_node_proportions     {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 1.0}
    intersection_count                                                     14
    street_length_total                                              2050.389
    street_segment_count                                                   17
    street_length_avg                                              120.611118
    circuity_avg                                                     1.000522
    self_loop_proportion                                                  0.0
    clean_intersection_count                                                8
    node_density_km                                                107.457599
    intersection_density_km                                        107.457599
    edge_density_km                                              15737.848529
    street_density_km                                            15737.848529
    clean_intersection_density_km                                   61.404342
    dtype: object




```python
# Plot the road map

# Reproject the GeoDataFrame to WGS84 (EPSG:4326)
road_wgs84 = road.to_crs('EPSG:4326')

# Step 1: Create a Folium map object
m = folium.Map(location=sel_location, zoom_start=15)

# Step 2: Iterate over the GeoDataFrame and add lines to the map
for index, row in road_wgs84.iterrows():
    line = row['geometry']
    coordinates = list(line.coords)
    coordinates = [(coord[1], coord[0]) for coord in coordinates]  # Swap lat and lon order
    folium.PolyLine(locations=coordinates, color='blue', weight=2).add_to(m)
# Step 3: Display the map
m
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe srcdoc="&lt;!DOCTYPE html&gt;
&lt;html&gt;
&lt;head&gt;

    &lt;meta http-equiv=&quot;content-type&quot; content=&quot;text/html; charset=UTF-8&quot; /&gt;

        &lt;script&gt;
            L_NO_TOUCH = false;
            L_DISABLE_3D = false;
        &lt;/script&gt;

    &lt;style&gt;html, body {width: 100%;height: 100%;margin: 0;padding: 0;}&lt;/style&gt;
    &lt;style&gt;#map {position:absolute;top:0;bottom:0;right:0;left:0;}&lt;/style&gt;
    &lt;script src=&quot;https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://code.jquery.com/jquery-3.7.1.min.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/js/bootstrap.bundle.min.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js&quot;&gt;&lt;/script&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/css/bootstrap.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.2.0/css/all.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/gh/python-visualization/folium/folium/templates/leaflet.awesome.rotate.min.css&quot;/&gt;

            &lt;meta name=&quot;viewport&quot; content=&quot;width=device-width,
                initial-scale=1.0, maximum-scale=1.0, user-scalable=no&quot; /&gt;
            &lt;style&gt;
                #map_09141ad31aa6d865fac8ce15e1478684 {
                    position: relative;
                    width: 100.0%;
                    height: 100.0%;
                    left: 0.0%;
                    top: 0.0%;
                }
                .leaflet-container { font-size: 1rem; }
            &lt;/style&gt;

&lt;/head&gt;
&lt;body&gt;


            &lt;div class=&quot;folium-map&quot; id=&quot;map_09141ad31aa6d865fac8ce15e1478684&quot; &gt;&lt;/div&gt;

&lt;/body&gt;
&lt;script&gt;


            var map_09141ad31aa6d865fac8ce15e1478684 = L.map(
                &quot;map_09141ad31aa6d865fac8ce15e1478684&quot;,
                {
                    center: [40.59762804, -73.96560899],
                    crs: L.CRS.EPSG3857,
                    zoom: 15,
                    zoomControl: true,
                    preferCanvas: false,
                }
            );





            var tile_layer_5e59447e38aad86aad375ecef1250291 = L.tileLayer(
                &quot;https://tile.openstreetmap.org/{z}/{x}/{y}.png&quot;,
                {&quot;attribution&quot;: &quot;\u0026copy; \u003ca href=\&quot;https://www.openstreetmap.org/copyright\&quot;\u003eOpenStreetMap\u003c/a\u003e contributors&quot;, &quot;detectRetina&quot;: false, &quot;maxNativeZoom&quot;: 19, &quot;maxZoom&quot;: 19, &quot;minZoom&quot;: 0, &quot;noWrap&quot;: false, &quot;opacity&quot;: 1, &quot;subdomains&quot;: &quot;abc&quot;, &quot;tms&quot;: false}
            );


            tile_layer_5e59447e38aad86aad375ecef1250291.addTo(map_09141ad31aa6d865fac8ce15e1478684);


            var poly_line_a373e18b190e00926bb99cc29babf568 = L.polyline(
                [[40.5979869, -73.9641104], [40.5979964, -73.9640219], [40.5980688, -73.9633671], [40.598096899999994, -73.9631121], [40.5981055, -73.9630344]],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;noClip&quot;: false, &quot;opacity&quot;: 1.0, &quot;smoothFactor&quot;: 1.0, &quot;stroke&quot;: true, &quot;weight&quot;: 2}
            ).addTo(map_09141ad31aa6d865fac8ce15e1478684);


            var poly_line_b39dc8f3c6f4aecc271e8f6c398e5ea0 = L.polyline(
                [[40.5979869, -73.9641104], [40.59797890000001, -73.964182], [40.59788749999999, -73.9650117], [40.59787399999999, -73.9651338], [40.5978654, -73.9652122]],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;noClip&quot;: false, &quot;opacity&quot;: 1.0, &quot;smoothFactor&quot;: 1.0, &quot;stroke&quot;: true, &quot;weight&quot;: 2}
            ).addTo(map_09141ad31aa6d865fac8ce15e1478684);


            var poly_line_5d96945682290f3b813f13c4df778c32 = L.polyline(
                [[40.5978654, -73.9652122], [40.5979466, -73.9652295], [40.599992300000004, -73.9656124], [40.60004429999999, -73.9656222], [40.6001207, -73.9656365]],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;noClip&quot;: false, &quot;opacity&quot;: 1.0, &quot;smoothFactor&quot;: 1.0, &quot;stroke&quot;: true, &quot;weight&quot;: 2}
            ).addTo(map_09141ad31aa6d865fac8ce15e1478684);


            var poly_line_66940c937c9c0be15b33a7f1bcaec4a1 = L.polyline(
                [[40.597581699999985, -73.9677765], [40.5975901, -73.9677001], [40.59762430000001, -73.9673906], [40.59765109999999, -73.9671477], [40.59767579999999, -73.9669239], [40.5976839, -73.9668503]],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;noClip&quot;: false, &quot;opacity&quot;: 1.0, &quot;smoothFactor&quot;: 1.0, &quot;stroke&quot;: true, &quot;weight&quot;: 2}
            ).addTo(map_09141ad31aa6d865fac8ce15e1478684);


            var poly_line_c8816b1734cd034bca267471724cd040 = L.polyline(
                [[40.597581699999985, -73.9677765], [40.597573499999996, -73.9678508], [40.5974883, -73.968623], [40.59747949999999, -73.9687031]],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;noClip&quot;: false, &quot;opacity&quot;: 1.0, &quot;smoothFactor&quot;: 1.0, &quot;stroke&quot;: true, &quot;weight&quot;: 2}
            ).addTo(map_09141ad31aa6d865fac8ce15e1478684);


            var poly_line_a732785d84b8535d2255a98d6167e098 = L.polyline(
                [[40.5976839, -73.9668503], [40.5976932, -73.9667664], [40.5977395, -73.9663462], [40.59778920000001, -73.9658961], [40.597797, -73.9658258], [40.59780510000001, -73.965752]],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;noClip&quot;: false, &quot;opacity&quot;: 1.0, &quot;smoothFactor&quot;: 1.0, &quot;stroke&quot;: true, &quot;weight&quot;: 2}
            ).addTo(map_09141ad31aa6d865fac8ce15e1478684);


            var poly_line_a4bbe2828af0109b5a136d1dd2a34414 = L.polyline(
                [[40.600089, -73.9659038], [40.60001369999999, -73.96588950000002], [40.59958329999999, -73.9658081], [40.59941940000001, -73.9657771], [40.5983585, -73.9655763], [40.597918899999996, -73.9654931], [40.5978358, -73.9654774]],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;noClip&quot;: false, &quot;opacity&quot;: 1.0, &quot;smoothFactor&quot;: 1.0, &quot;stroke&quot;: true, &quot;weight&quot;: 2}
            ).addTo(map_09141ad31aa6d865fac8ce15e1478684);


            var poly_line_d017db6758f7f1f7e850dd78ba033895 = L.polyline(
                [[40.600089, -73.9659038], [40.6001108, -73.96572110000001], [40.6001207, -73.9656365]],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;noClip&quot;: false, &quot;opacity&quot;: 1.0, &quot;smoothFactor&quot;: 1.0, &quot;stroke&quot;: true, &quot;weight&quot;: 2}
            ).addTo(map_09141ad31aa6d865fac8ce15e1478684);


            var poly_line_5851bf643a24adaa0c148a177e135dfe = L.polyline(
                [[40.600089, -73.9659038], [40.60006620000001, -73.9660957], [40.6000571, -73.9661725]],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;noClip&quot;: false, &quot;opacity&quot;: 1.0, &quot;smoothFactor&quot;: 1.0, &quot;stroke&quot;: true, &quot;weight&quot;: 2}
            ).addTo(map_09141ad31aa6d865fac8ce15e1478684);


            var poly_line_892162a0fd71ee617702b354bc1fb91b = L.polyline(
                [[40.5978358, -73.9654774], [40.597857399999995, -73.9652838], [40.5978654, -73.9652122]],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;noClip&quot;: false, &quot;opacity&quot;: 1.0, &quot;smoothFactor&quot;: 1.0, &quot;stroke&quot;: true, &quot;weight&quot;: 2}
            ).addTo(map_09141ad31aa6d865fac8ce15e1478684);


            var poly_line_6c05d92190d73bbae058767e8a3e2926 = L.polyline(
                [[40.5978358, -73.9654774], [40.5977569, -73.9654625], [40.5973495, -73.9653853], [40.5971177, -73.9653414], [40.596382099999985, -73.9652021], [40.59624369999999, -73.9651759], [40.5958005, -73.965092], [40.59571630000001, -73.965076]],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;noClip&quot;: false, &quot;opacity&quot;: 1.0, &quot;smoothFactor&quot;: 1.0, &quot;stroke&quot;: true, &quot;weight&quot;: 2}
            ).addTo(map_09141ad31aa6d865fac8ce15e1478684);


            var poly_line_ef2267f244be5d0eb83c61dfac1e41af = L.polyline(
                [[40.5978358, -73.9654774], [40.59781539999999, -73.9656587], [40.59780510000001, -73.965752]],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;noClip&quot;: false, &quot;opacity&quot;: 1.0, &quot;smoothFactor&quot;: 1.0, &quot;stroke&quot;: true, &quot;weight&quot;: 2}
            ).addTo(map_09141ad31aa6d865fac8ce15e1478684);


            var poly_line_8625b14c9ea1188e429c99e9a2655ecc = L.polyline(
                [[40.6000571, -73.9661725], [40.599982899999986, -73.9661586], [40.5979364, -73.9657765], [40.5978889, -73.9657677], [40.59780510000001, -73.965752]],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;noClip&quot;: false, &quot;opacity&quot;: 1.0, &quot;smoothFactor&quot;: 1.0, &quot;stroke&quot;: true, &quot;weight&quot;: 2}
            ).addTo(map_09141ad31aa6d865fac8ce15e1478684);


            var poly_line_1a2b232c48770ac3626bf76da84cc2c2 = L.polyline(
                [[40.59571630000001, -73.965076], [40.595695, -73.9652622], [40.5956848, -73.9653521]],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;noClip&quot;: false, &quot;opacity&quot;: 1.0, &quot;smoothFactor&quot;: 1.0, &quot;stroke&quot;: true, &quot;weight&quot;: 2}
            ).addTo(map_09141ad31aa6d865fac8ce15e1478684);


            var poly_line_e6e8e414a08ddff6bcd842a34db39add = L.polyline(
                [[40.59571630000001, -73.965076], [40.595734199999995, -73.9649185], [40.5957479, -73.9647984]],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;noClip&quot;: false, &quot;opacity&quot;: 1.0, &quot;smoothFactor&quot;: 1.0, &quot;stroke&quot;: true, &quot;weight&quot;: 2}
            ).addTo(map_09141ad31aa6d865fac8ce15e1478684);


            var poly_line_cf3ccdf5ab41bedd35e68ce1755d6271 = L.polyline(
                [[40.59780510000001, -73.965752], [40.597726699999996, -73.96573720000002], [40.59582149999999, -73.96537790000001], [40.59576989999999, -73.9653682], [40.5956848, -73.9653521]],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;noClip&quot;: false, &quot;opacity&quot;: 1.0, &quot;smoothFactor&quot;: 1.0, &quot;stroke&quot;: true, &quot;weight&quot;: 2}
            ).addTo(map_09141ad31aa6d865fac8ce15e1478684);


            var poly_line_046cd29b4af69acc60f9de0d6eb76137 = L.polyline(
                [[40.5957479, -73.9647984], [40.5958311, -73.9648147], [40.59774269999999, -73.9651906], [40.5977861, -73.9651988], [40.5978654, -73.9652122]],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;noClip&quot;: false, &quot;opacity&quot;: 1.0, &quot;smoothFactor&quot;: 1.0, &quot;stroke&quot;: true, &quot;weight&quot;: 2}
            ).addTo(map_09141ad31aa6d865fac8ce15e1478684);

&lt;/script&gt;
&lt;/html&gt;" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



# Step 2. Select the sample points on the road network

The second step of the code generates a list of evenly distributed points along the road network, with a specified distance between each point. This is achieved using a function that takes the road network data and an optional distance parameter N, which is set to a default value of 50 meters.

The function iterates over each road in the roads dataframe and creates points at regular intervals of the specified distance (N). By doing so, it ensures that the generated points are evenly spaced along the road network.

To maintain a consistent spatial reference, the function sets the Coordinate Reference System (CRS) of the gdf_points dataframe to match the CRS of the roads dataframe. This ensures that the points and the road network are in the same local projected CRS, measured in meters.

Furthermore, to avoid duplication and redundancy, the function removes any duplicate points in the gdf_points dataframe based on the geometry column. This ensures that each point in the resulting dataframe is unique and represents a distinct location along the road network.


```python
# Create a set of approx. equally-spaced points along the road network
points = select_points_on_road_network(road, 25)

# Convert points into latitude/longitude with CRS
points_coords = points.to_crs(4326)
points_coords.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>geometry</th>
      <th>road_index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>POINT (-73.96411 40.59799)</td>
      <td>(42485311, 42534856, 0)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>POINT (-73.96382 40.59802)</td>
      <td>(42485311, 42534856, 0)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>POINT (-73.96353 40.59805)</td>
      <td>(42485311, 42534856, 0)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>POINT (-73.96323 40.59808)</td>
      <td>(42485311, 42534856, 0)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>POINT (-73.96440 40.59795)</td>
      <td>(42485311, 2282434963, 0)</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot the points along the road map

# Reproject the GeoDataFrame to WGS84 (EPSG:4326)
points_wgs84 = points.to_crs('EPSG:4326')

# Step 1: Create a Folium map object
m = folium.Map(location=sel_location, zoom_start=15)

# Step 2: Iterate over the GeoDataFrame and add points to the map
for index, row in points_wgs84.iterrows():
    point = row['geometry']
    coordinates = (point.y, point.x)  # Swap lat and lon order
    folium.CircleMarker(location=coordinates, color='blue', radius=2).add_to(m)

# Step 3: Display the map
m
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe srcdoc="&lt;!DOCTYPE html&gt;
&lt;html&gt;
&lt;head&gt;

    &lt;meta http-equiv=&quot;content-type&quot; content=&quot;text/html; charset=UTF-8&quot; /&gt;

        &lt;script&gt;
            L_NO_TOUCH = false;
            L_DISABLE_3D = false;
        &lt;/script&gt;

    &lt;style&gt;html, body {width: 100%;height: 100%;margin: 0;padding: 0;}&lt;/style&gt;
    &lt;style&gt;#map {position:absolute;top:0;bottom:0;right:0;left:0;}&lt;/style&gt;
    &lt;script src=&quot;https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://code.jquery.com/jquery-3.7.1.min.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/js/bootstrap.bundle.min.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js&quot;&gt;&lt;/script&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/css/bootstrap.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.2.0/css/all.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/gh/python-visualization/folium/folium/templates/leaflet.awesome.rotate.min.css&quot;/&gt;

            &lt;meta name=&quot;viewport&quot; content=&quot;width=device-width,
                initial-scale=1.0, maximum-scale=1.0, user-scalable=no&quot; /&gt;
            &lt;style&gt;
                #map_514a824eae1865763ef68e8ab47424b3 {
                    position: relative;
                    width: 100.0%;
                    height: 100.0%;
                    left: 0.0%;
                    top: 0.0%;
                }
                .leaflet-container { font-size: 1rem; }
            &lt;/style&gt;

&lt;/head&gt;
&lt;body&gt;


            &lt;div class=&quot;folium-map&quot; id=&quot;map_514a824eae1865763ef68e8ab47424b3&quot; &gt;&lt;/div&gt;

&lt;/body&gt;
&lt;script&gt;


            var map_514a824eae1865763ef68e8ab47424b3 = L.map(
                &quot;map_514a824eae1865763ef68e8ab47424b3&quot;,
                {
                    center: [40.59762804, -73.96560899],
                    crs: L.CRS.EPSG3857,
                    zoom: 15,
                    zoomControl: true,
                    preferCanvas: false,
                }
            );





            var tile_layer_2a8a8540ee9ecfa6443dcd0caa9c35bc = L.tileLayer(
                &quot;https://tile.openstreetmap.org/{z}/{x}/{y}.png&quot;,
                {&quot;attribution&quot;: &quot;\u0026copy; \u003ca href=\&quot;https://www.openstreetmap.org/copyright\&quot;\u003eOpenStreetMap\u003c/a\u003e contributors&quot;, &quot;detectRetina&quot;: false, &quot;maxNativeZoom&quot;: 19, &quot;maxZoom&quot;: 19, &quot;minZoom&quot;: 0, &quot;noWrap&quot;: false, &quot;opacity&quot;: 1, &quot;subdomains&quot;: &quot;abc&quot;, &quot;tms&quot;: false}
            );


            tile_layer_2a8a8540ee9ecfa6443dcd0caa9c35bc.addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_a0f74c1bf07353d16357c7e42363f52b = L.circleMarker(
                [40.5979869, -73.9641104],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_19066f3dae4ff41e20ce7bc88f21cbfb = L.circleMarker(
                [40.59801894925538, -73.96381796381085],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_3bcc4a7aa3838253de4371c74f578393 = L.circleMarker(
                [40.59805127754107, -73.96352557982128],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_4601a069e0c077a63674b7871b20ce47 = L.circleMarker(
                [40.59808355682714, -73.96323318639772],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_4c04826f510d936df83a55cc7368dbe1 = L.circleMarker(
                [40.5979545788675, -73.96440278498903],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_25f9fa68dbced004dd0cdd25c71be2f2 = L.circleMarker(
                [40.59792236748138, -73.96469519081045],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_f281a0b291828105fb58b8d41ac4115a = L.circleMarker(
                [40.59789015535128, -73.96498759636638],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_e29b4fb67d58e49f09e4f69d817c3311 = L.circleMarker(
                [40.5978654, -73.9652122],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_f507d3d0ea9d662261ae770404a0a51a = L.circleMarker(
                [40.59808810382881, -73.96525598498913],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_35e0d93a227ff81e32e8d97c10239676 = L.circleMarker(
                [40.59831104680604, -73.9652977129983],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_95c45c2de2e0623843537ee4c306f8a9 = L.circleMarker(
                [40.5985339897631, -73.96533944126948],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_7e45c4c1ba4bfa6eb44216d6c4c4c038 = L.circleMarker(
                [40.59875693269998, -73.96538116980267],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_8e9d41c28f47473870e1f2fb32fdeb6d = L.circleMarker(
                [40.598979875616685, -73.96542289859786],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_141616fa0bd5a717c830f51f9ea89e9c = L.circleMarker(
                [40.59920281851321, -73.96546462765507],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_dd8b1278a37d79c96c0bfe82ac2fb63c = L.circleMarker(
                [40.599425761389575, -73.9655063569743],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_ffa6e44e548834c0ad2199225b661768 = L.circleMarker(
                [40.59964870424574, -73.96554808655556],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_50553d6f0b933831c3579aad7285c4b2 = L.circleMarker(
                [40.59987164708174, -73.96558981639885],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_5527cacaa11a5740975d03965eb88c01 = L.circleMarker(
                [40.60009458279777, -73.96563161156728],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_01ad8dbc91c019e8ec0f9cac22af7bdc = L.circleMarker(
                [40.597581699999985, -73.9677765],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_de2a7a06d891211b09643c068a3e7a61 = L.circleMarker(
                [40.59761396758359, -73.96748410617401],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_84c5c099e06e97b38b82364041a1d0eb = L.circleMarker(
                [40.59764624383795, -73.96719171385722],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_82275769511304a5f5c5e805dd0136fd = L.circleMarker(
                [40.59767850529479, -73.9668993186054],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_388230a53a8dc61cb8e7c0abdd3a2a3d = L.circleMarker(
                [40.597549437268704, -73.96806889463747],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_5044942771fb4c611f23149aeb7057f6 = L.circleMarker(
                [40.59751717620304, -73.96836128946663],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_5ce8a282750edf21a435ccb82927aa0d = L.circleMarker(
                [40.59748492867994, -73.96865368673704],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_bba6771483bcea443fadd96034c4b5cd = L.circleMarker(
                [40.5976839, -73.9668503],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_263ce136d4681dc5fea97335e2106a8f = L.circleMarker(
                [40.597716173148534, -73.96655790684339],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_1583e18905bb588905bfe287a83c974d = L.circleMarker(
                [40.59774841028091, -73.96626550670504],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_7a8095b6b7542d5965a2ecbd9e83479a = L.circleMarker(
                [40.597780696065925, -73.96597311567832],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_1762e7387eb9eaeb7be2f300d441909d = L.circleMarker(
                [40.600089, -73.9659038],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_bf521b5eddbc550abf6444428de73617 = L.circleMarker(
                [40.599866110035045, -73.96586158672928],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_c460667e55f4a75a1d6bd7def180fc3a = L.circleMarker(
                [40.59964321372049, -73.96581943120589],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_c41733448857a7f7545dee64f760428f = L.circleMarker(
                [40.59942031764745, -73.96577727356318],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_8ab3bbfdd72854d085b5c87957c00924 = L.circleMarker(
                [40.599197424921634, -73.96573508554964],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_237fbc860a6ad48d188f0cbbe1988821 = L.circleMarker(
                [40.59897453218888, -73.96569289767812],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_14db95b5f91dacadcb7e62f6b1801b29 = L.circleMarker(
                [40.598751639435655, -73.96565071007161],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_f8a987d4d4007e4f231a5e21fafb543e = L.circleMarker(
                [40.59852874666197, -73.96560852273012],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_5b10ca7e51dd9b97cf15daed05d23756 = L.circleMarker(
                [40.59830585383245, -73.96556633597504],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_8b0c3ba9c8e297fa623f70cd8d39b0f2 = L.circleMarker(
                [40.59808296086818, -73.96552415052426],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_88714639994025eab1bd05291ec1192d = L.circleMarker(
                [40.59786006577763, -73.96548198450527],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_0798684ce33109850635c714fdbdcda5 = L.circleMarker(
                [40.5978358, -73.9654774],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_81ae9a6ff47a63b533f8a8829747fd93 = L.circleMarker(
                [40.59761290711412, -73.96543521405829],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_ef971248b82ed90c442814422e6910e1 = L.circleMarker(
                [40.59739001972133, -73.96539297821816],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_275b5e7b3b77d9a833cf607ec011532c = L.circleMarker(
                [40.59716713023909, -73.9653507614402],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_366f59eccaf253e3b13d837cc8f57e89 = L.circleMarker(
                [40.5969442399857, -73.96530855174797],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_68d75af0df0b094e1d7d59f436de997f = L.circleMarker(
                [40.59672134962893, -73.96526634307439],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_cb450ee722c01e0c79f0a3db0c56a64d = L.circleMarker(
                [40.59649845925166, -73.96522413466593],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_887826bd7f917f151d5643b4d44beefd = L.circleMarker(
                [40.596275568159356, -73.965181932836],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_30a160801ae0f924e4fa403f907e91a4 = L.circleMarker(
                [40.5960526762935, -73.96513973811653],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_17385afad23d471ab601877e06c99115 = L.circleMarker(
                [40.59582978440809, -73.96509754365366],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_587ca1e65d51df3fea884bc99d74ec61 = L.circleMarker(
                [40.6000571, -73.9661725],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_f68d1ad8d23dcbf3c00c661d17a7a46a = L.circleMarker(
                [40.59983415101987, -73.96613082647158],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_8b52e6bea1fe97d823444e8600494c4c = L.circleMarker(
                [40.59961119714515, -73.96608919806582],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_dd236a1ac1a252ec84829fa8549ce18f = L.circleMarker(
                [40.59938824325032, -73.96604756992144],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_01cad5b94ea88893333f70bf7a4210fc = L.circleMarker(
                [40.59916528933537, -73.96600594203842],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_d1df7cb237b79b153b5377619598dba3 = L.circleMarker(
                [40.5989423354003, -73.96596431441678],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_9ab970d8b529217989a24408832ce266 = L.circleMarker(
                [40.59871938144512, -73.96592268705649],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_3001ea3c231e76e0a334fd2f95a5e438 = L.circleMarker(
                [40.59849642746983, -73.96588105995757],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_706c383850c11e0a457a47652217cce5 = L.circleMarker(
                [40.59827347347443, -73.96583943311998],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_76b80f10c3bfa52f8eef0171e44dc64d = L.circleMarker(
                [40.59805051945892, -73.96579780654376],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_e25dbe597bc2aad69e670b12e1ff86f5 = L.circleMarker(
                [40.59782756240844, -73.9657562083473],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_d0115eff618e5629edf0e960f4326379 = L.circleMarker(
                [40.59571630000001, -73.965076],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_c6ff8fae2b10ebb91890c726dd4c9798 = L.circleMarker(
                [40.59780510000001, -73.965752],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_6458eee4f59d28fbb6cf4a963b918266 = L.circleMarker(
                [40.59758219252683, -73.96570994677634],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_226969021bfb4ddcf128e30d280e2b02 = L.circleMarker(
                [40.59735928350875, -73.96566790772265],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_30c0bef033ccf38acc01c6b60529950d = L.circleMarker(
                [40.597136374470324, -73.96562586893297],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_b1ee61e8ce8870ca890088904e6dfce4 = L.circleMarker(
                [40.59691346541152, -73.96558383040734],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_dbdbe68a96f399a6a87e20fe3208cc3c = L.circleMarker(
                [40.59669055633233, -73.96554179214574],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_40c0a83112f399c0cc11ec8c3191ec34 = L.circleMarker(
                [40.59646764723278, -73.96549975414813],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_9c31431d3d83b9ea1463aa68cc4eb7e6 = L.circleMarker(
                [40.596244738112844, -73.96545771641455],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_cb3eb64ede0ff5d4534d1df6b1939a17 = L.circleMarker(
                [40.59602182897253, -73.96541567894498],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_3e1684a94387bd69b915d75f7dea96ee = L.circleMarker(
                [40.59579891836107, -73.96537365500023],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_9dcfb0ab131f17fe7f31e10783aeca0d = L.circleMarker(
                [40.5957479, -73.9647984],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_55b231fa64b25291f5498aaf6b51314c = L.circleMarker(
                [40.59597061929558, -73.96484213459439],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_b7ad2ea30197e3879b5e092b20dc6d3a = L.circleMarker(
                [40.596193331864725, -73.96488592825537],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_05d4262a29204a6d92010ce9dd0512c9 = L.circleMarker(
                [40.596416044412344, -73.96492972219184],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_acc1e3f9627aebc65ac4feb815485bd4 = L.circleMarker(
                [40.59663875693846, -73.96497351640376],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_3d0f45125b8a80e8344e00b7bbe8e110 = L.circleMarker(
                [40.596861469443034, -73.96501731089117],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_8ecab229cac2f4894972d4ab225f196f = L.circleMarker(
                [40.597084181926085, -73.96506110565406],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_d951b68427c38152c9527312e5432bac = L.circleMarker(
                [40.59730689438764, -73.96510490069244],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_d4f3cdd1a3af14778d358bb4d0d418e9 = L.circleMarker(
                [40.59752960682764, -73.9651486960063],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);


            var circle_marker_50e8cd657b732162b8639d5ed829a249 = L.circleMarker(
                [40.59775232737836, -73.96519241899688],
                {&quot;bubblingMouseEvents&quot;: true, &quot;color&quot;: &quot;blue&quot;, &quot;dashArray&quot;: null, &quot;dashOffset&quot;: null, &quot;fill&quot;: false, &quot;fillColor&quot;: &quot;blue&quot;, &quot;fillOpacity&quot;: 0.2, &quot;fillRule&quot;: &quot;evenodd&quot;, &quot;lineCap&quot;: &quot;round&quot;, &quot;lineJoin&quot;: &quot;round&quot;, &quot;opacity&quot;: 1.0, &quot;radius&quot;: 2, &quot;stroke&quot;: true, &quot;weight&quot;: 3}
            ).addTo(map_514a824eae1865763ef68e8ab47424b3);

&lt;/script&gt;
&lt;/html&gt;" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



# Step 3: Query panoramic images near road points


The next step in the pipeline focuses on finding the closest images for each point.

To calculate the distances between the features and the points, a k-dimensional tree (KDTree) approach is employed using the local projected crs in meters. The KDTree is built using the geometry coordinates of the feature points. By querying the KDTree, the nearest neighbors of the points in the points dataframe are identified. The closest feature and distance information are then assigned to each point accordingly.


```python
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
```


```python

# Hold point location and available panoramic image locations
pano_data = []

# Iterate over each point in the road network, and get metadata for the nearest panoramic images
for i in tqdm(range(len(points_coords.geometry))):
    
    # Find the most recent panoramic image
    most_recent_date = None
    
    # Search for all available panoramic images closest to each point
    panos = search_panoramas(lat=points_coords.geometry.y[i], lon=points_coords.geometry.x[i])
    
    # Iterate through the closest set of panos for a given location
    # For each pano image, get the metadata by supplying the unique pano_id string
    # Return the most recent dated panoramic image and its unique ID.
    for pano in panos:
        meta = get_panorama_meta(pano_id=pano.pano_id, api_key=GOOGLE_MAPS_API_KEY)
                
        if meta.date:
            date_code = datetime.strptime(meta.date, '%Y-%m')
            
            # Append data on point and panoramic image location
            pano_data.append({'Point_Index': i,
                'Point_Latitude': points_coords.geometry.y[i],
                'Point_Longitude': points_coords.geometry.x[i],
                'Panorama_ID': pano.pano_id,
                'Panorama_Date': meta.date,
                'Panorama_Latitude': pano.lat,
                'Panorama_Longitude': pano.lon,
                'Panorama_Rotation': pano.heading})


# All available panoramic images sampled from road points
pano_df = pd.DataFrame(pano_data)

# Remove duplicate images based on 'Panorama_ID'
pano_df.drop_duplicates(subset='Panorama_ID', keep='first', inplace=True)

print(f'Total Available Panoramic Images:', len(pano_df))

# Remove images where 'Panorama_Date' is after 2016-01
pano_df = pano_df[pano_df['Panorama_Date'] >= '2016-01']

print(f'Total Panoramic Images After 2016:', len(pano_df))

# Reset dataframe index
pano_df.reset_index(drop=True, inplace=True)

# Remove any panoraamic images closer than 10 meters
pano_df_simple = remove_adjacent_panoramics(pano_df, 10)

print(f'Total Panoramic Images After De-Duplication:', len(pano_df_simple))

```

    100%|██████████████████████████████████████████████████████████████████████████████████| 82/82 [09:26<00:00,  6.91s/it]

    Total Available Panoramic Images: 587
    Total Panoramic Images After 2016: 457
    Total Panoramic Images After De-Duplication: 206
    

    
    


```python
len(pano_df_simple)
```




    206




```python

```


```python

```


```python

```


```python

```


```python
# Plot Pano Locations and Street Sampling Points

from eomaps import Maps
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.io import img_tiles
from shapely.geometry import Point, LineString
%matplotlib inline


# Define the bounding box for the plotting area
bbox = [
    pano_df['Point_Longitude'].min() - 0.0005,
    pano_df['Point_Longitude'].max() + 0.0005,
    pano_df['Point_Latitude'].min() - 0.0005,
    pano_df['Point_Latitude'].max() + 0.0005
]

# Convert panos_df to a GeoDataFrame
point_geometry = [Point(xy) for xy in zip(pano_df['Point_Longitude'], pano_df['Point_Latitude'])]
point_locations_gdf = gpd.GeoDataFrame(pano_df, geometry=point_geometry, crs="EPSG:4326")

# Convert cropped_tree_inventory to a Geodataframe
pano_geometry = [Point(xy) for xy in zip(pano_df['Panorama_Longitude'], pano_df['Panorama_Latitude'])]
pano_locations_gdf = gpd.GeoDataFrame(pano_df, geometry=pano_geometry, crs="EPSG:4326")

# Convert cropped_tree_inventory to a Geodataframe
sel_pano_geometry = [Point(xy) for xy in zip(pano_df_simple['Panorama_Longitude'], pano_df_simple['Panorama_Latitude'])]
sel_pano_locations_gdf = gpd.GeoDataFrame(pano_df_simple, geometry=sel_pano_geometry, crs="EPSG:4326")


# Plotting:
# Define a map and extent
m = Maps(crs=Maps.CRS.Mercator.GOOGLE, figsize=(20, 20))
m.set_extent((bbox[0], bbox[1], bbox[2], bbox[3]))

# Plot the GeoDataFrame onto the map
m.add_gdf(pano_locations_gdf, marker='x', color='blue', alpha=0.80, markersize = 250, label='Available Panoramic Images')
m.add_gdf(sel_pano_locations_gdf, marker='+', color='white', alpha=0.80, markersize = 250, label='Filtered Panoramic Images')
m.add_gdf(point_locations_gdf, marker='o', color='red', alpha=0.80, markersize = 250, label='Street Points')

m.add_wms.ESRI_ArcGIS.SERVICES.World_Imagery.add_layer.xyz_layer()

# Show the map
#m.savefig(f"C:/Users/talake2/Desktop/tree-geolocation/geolocation-panos-testing-cities-maps/geolocation-pano-testing-{testing_city}.png")
m.show();

```


    
![png](output_23_0.png)
    



    <Figure size 640x480 with 0 Axes>



```python
pano_df_simple
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Point_Index</th>
      <th>Point_Latitude</th>
      <th>Point_Longitude</th>
      <th>Panorama_ID</th>
      <th>Panorama_Date</th>
      <th>Panorama_Latitude</th>
      <th>Panorama_Longitude</th>
      <th>Panorama_Rotation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>40.597987</td>
      <td>-73.964110</td>
      <td>lQ2EKXWRVLToxLJPdDHZHQ</td>
      <td>2018-06</td>
      <td>40.598015</td>
      <td>-73.963696</td>
      <td>79.625259</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>40.597987</td>
      <td>-73.964110</td>
      <td>ywE3mhrm0hiYdGZoPur9og</td>
      <td>2019-10</td>
      <td>40.597969</td>
      <td>-73.964392</td>
      <td>260.682404</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>40.597987</td>
      <td>-73.964110</td>
      <td>zPJaZ6lJffc0KeHKfIWPVg</td>
      <td>2019-10</td>
      <td>40.597958</td>
      <td>-73.964507</td>
      <td>263.732849</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>40.597987</td>
      <td>-73.964110</td>
      <td>5yNf6FYIHTKEkMiqSzii7w</td>
      <td>2022-07</td>
      <td>40.597596</td>
      <td>-73.964028</td>
      <td>170.130783</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>40.597987</td>
      <td>-73.964110</td>
      <td>-Jl1_mx76HkeklFP1zxoAw</td>
      <td>2022-07</td>
      <td>40.597505</td>
      <td>-73.964008</td>
      <td>169.998337</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>444</th>
      <td>71</td>
      <td>40.595799</td>
      <td>-73.965374</td>
      <td>wYisxiBZ2Mi2cP0LWoX1EA</td>
      <td>2018-06</td>
      <td>40.595634</td>
      <td>-73.965711</td>
      <td>80.290596</td>
    </tr>
    <tr>
      <th>446</th>
      <td>72</td>
      <td>40.595748</td>
      <td>-73.964798</td>
      <td>GlkuYUG9tUv8-pe3fifpzg</td>
      <td>2018-06</td>
      <td>40.595773</td>
      <td>-73.964420</td>
      <td>81.660408</td>
    </tr>
    <tr>
      <th>447</th>
      <td>72</td>
      <td>40.595748</td>
      <td>-73.964798</td>
      <td>HZJvAZBMGfQBGbf1ayIoKQ</td>
      <td>2018-06</td>
      <td>40.595786</td>
      <td>-73.964302</td>
      <td>81.254898</td>
    </tr>
    <tr>
      <th>448</th>
      <td>72</td>
      <td>40.595748</td>
      <td>-73.964798</td>
      <td>IFsSMwfw3GE99dwYc4jtEQ</td>
      <td>2018-06</td>
      <td>40.595800</td>
      <td>-73.964186</td>
      <td>80.422897</td>
    </tr>
    <tr>
      <th>450</th>
      <td>74</td>
      <td>40.596193</td>
      <td>-73.964886</td>
      <td>Uf5ycgLR6mJPuCFf2Z58EA</td>
      <td>2017-09</td>
      <td>40.596189</td>
      <td>-73.964881</td>
      <td>353.775208</td>
    </tr>
  </tbody>
</table>
<p>206 rows × 8 columns</p>
</div>




```python
# Download Panoramic Images
```


```python
# Downloading Panoramic Images

# Loop over 'pano_df_simple' dataFrame
for i in tqdm(range(len(pano_df_simple))):

    # Get panoramic image unique ID
    panoID = pano_df_simple['Panorama_ID'][i]
    
    # Attempt to download single panoramic image
    image = get_panorama(pano_id = panoID)

    # Save the image
    image.save(fr'C:/Users/talake2/Desktop/tree-geolocation/geolocation-pano-testing-cities/geolocation-pano-testing-experiments/{panoID}.jpg', "jpeg")
            
    # Create the matching panoramic image metadata .json file
    metadata = {
        'panoId': panoID,
        'lat': pano_df_simple['Panorama_Latitude'][i],
        'lng': pano_df_simple['Panorama_Longitude'][i],
        'rotation': pano_df_simple['Panorama_Rotation'][i]
        }
        
    # Write metadata to JSON file
    with open(fr'C:/Users/talake2/Desktop/tree-geolocation/geolocation-pano-testing-cities/geolocation-pano-testing-experiments/{panoID}.metadata.json', 'w') as f:
        json.dump(metadata, f)
            
    print(f"Saved Panoramic Image and Metadata: ", panoID)
```

      0%|▍                                                                              | 1/206 [02:00<6:50:12, 120.06s/it]

    Saved Panoramic Image and Metadata:  lQ2EKXWRVLToxLJPdDHZHQ
    


```python

```
