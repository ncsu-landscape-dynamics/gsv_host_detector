# 1. rename lat/long columns
# 2. take data set, make new dataframe w/ just 
# species name and lat/lng
# 3. round to 6 decimals
# x = rounded_lng
# y = rounded_lat and 

# coordindate
# POINT (-114.1020403 51.00288)
# POINT (-86.542234300422 39.165948831758)


# vancouver - coordindate - 
# -123.105315 49.283639

# step one, make a new dataframe 
  
import pandas as pd
from EcoNameTranslator import to_scientific, to_common, to_species
import os
import re

# folderpath for the coordinates you are taking
folder_path = "inventory_copy_coords"


def cleanCoordinates(coord):
    return re.sub(r'[A-Za-z()]+', '', coord).strip()


# Iterate through the files
# Create a master dataframe to add everything to
mainDF = pd.DataFrame()
for i, filename in enumerate(os.listdir(folder_path)):
  file_path = os.path.join(folder_path, filename)
  print(file_path)
  cityDF = pd.read_csv(file_path)

  # do string operations for incorrect formatting
  if not {'rounded_lng', 'rounded_lat'}.issubset(cityDF.columns):
    cityDF['coordinates'] = cityDF['coordinates'].astype(str)
    cityDF['coordinates'] = cityDF['coordinates'].apply(cleanCoordinates)
    cityDF[['rounded_lng', 'rounded_lat']] = cityDF['coordinates'].str.split(expand=True)
    cityDF['rounded_lng'] = cityDF['rounded_lng'].astype(float)
    cityDF['rounded_lat'] = cityDF['rounded_lat'].astype(float)

  # take only what we need and add it to a main dataframe
  cityDF = cityDF[['rounded_lng', 'rounded_lat', 'species_name']]

  mainDF = pd.concat([mainDF, cityDF], ignore_index=True)

mainDF['rounded_lng'] = mainDF['rounded_lng'].round(6)
mainDF['rounded_lat'] = mainDF['rounded_lat'].round(6)

# Change this for the path of the AutoArborist 
AAPath =  "tree_locations/tree_locations_tfrecord_idx_merged_subset_4dl.csv"
AADF = cityDF = pd.read_csv(AAPath)

AADF['rounded_lng'] = AADF['rounded_lng'].round(6)
AADF['rounded_lat'] = AADF['rounded_lat'].round(6)
AADF = pd.merge(AADF, mainDF[['rounded_lng', 'rounded_lat', 'species_name']], 
                    on=['rounded_lng', 'rounded_lat'], 
                    how='left')
filename = "AutoArboristData.csv"
AADF.to_csv('tree_locations/' + filename, index=False)
