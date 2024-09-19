import pandas as pd
from EcoNameTranslator import to_scientific, to_common
import requests
from EcoNameTranslator import to_common
from EcoNameTranslator import to_scientific
import os

# scientific name to common name
i = 0
noneCount = 0
sNaValues = set()
scientificNamesDict = {}
def getCommonNames(scientificName):
    global i
    global noneCount
    global sNaValues

    if i % 100 == 0:
        print(i)
    i += 1
    if scientificName in sNaValues:
        noneCount += 1
        return None
    elif scientificName in scientificNamesDict:
        return scientificNamesDict[scientificName]
    try:
        common_names = to_common([scientificName])[scientificName][1]
        if len(common_names) == 0:
            raise ValueError()
        
        scientificNamesDict[scientificName] = common_names
        
        return common_names
    except:
        noneCount += 1 
        sNaValues.add(scientificName)
        return None

# common name to scientific Name
i = 0
noneCount = 0
cNaValues = set()
common_namesDict = {}
def getScientificNames(common_name):
    global i
    global noneCount
    global cNaValues
    if i % 100 == 0:
        print(i)
    i += 1
    if common_name in cNaValues:
        noneCount += 1
        return None
    elif common_name in common_namesDict:
        return common_namesDict[common_name]
    try:
        common_names = to_scientific([common_name])[common_name][1]
        if len(common_names) == 0:
            raise ValueError()
        
        common_namesDict[common_name] = common_names
        
        return common_names
    except:
        noneCount += 1 
        cNaValues.add(common_name)
        return None

# reverse comma ordering
def reorganizeComma(name):
    if isinstance(name, str):

        if ',' in name:
            parts = name.split(',', 1)
            preComma = parts[0].strip()
            postComma = parts[1].strip()
            return postComma + ' ' + preComma
        else:
            return name
    else:
        return name
    
# Handle NA Values
def handleNA(value):
    # questions --- common name as array, or string?
    naValues = {'', 'a.', 'nan', 'other', 'n/a'}
    containsNaValues = {'unidentified', 'unsuitable', 'vacant', '*', '_', '-', 'proposed', 'unknown', '#', 'other ', 'no ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}

    if value.strip() in naValues or any(word in value for word in containsNaValues):
        return 'NA'
    else:
        # If the value contains there characters, include only the elements before
        if "'" in value:
            return value.split("'")[0]
        elif "(" in value:
            return value.split("(")[0]
        elif ":" in value:
            return value.split(":")[0]
        elif "`" in value:
            return value.split("`")[0]
        elif "‘" in value:
            return value.split("‘")[0]

        return value

# ==================================================================================================================================
# Get city and create columns

# Find the folder path and create a dataframe from it
folder_path = "../tree_inventories_original_records"
df = pd.DataFrame(columns=['species_name', 'common_name'])
unorganizedData = []
# Iterate through the files
for i, filename in enumerate(os.listdir(folder_path)):
    print("FILE #", i)
    file_path = os.path.join(folder_path, filename)
    if not filename.endswith('.csv'):
        continue

    # The coulmns needed fom the data
    required_columns = ['species_name', 'common_name']


    cityDF = pd.read_csv(file_path)
    missing_columns = [col for col in required_columns if col not in cityDF.columns]
    if missing_columns:
        unorganizedData.append(filename)
        continue
    
    # select only the columns we need while also handling possible comma interruptions
    cityDF = cityDF.map(reorganizeComma)
    cityDF = cityDF[['species_name', 'common_name']]


    # Adding in common names using EcoNameTranslator
    # print("NA Values in common name before: ", cityDF['common_name'].isna().sum())
    # na_indices = cityDF[cityDF['common_name'].isna()].index
    # print(cityDF.iloc[na_indices])

    # cityDF['common_name'] = cityDF.apply(
    #     lambda row: getCommonNames(row['species_name']) if pd.isna(row['common_name']) else row['common_name'],
    #     axis=1
    # )

    # print("NA Values in common name after: ", cityDF['common_name'].isna().sum())
    # print(cityDF.iloc[na_indices])



    # Adding in scientific names using EcoNameTranslator
    # print("NA Values in species name before: ", cityDF['species_name'].isna().sum())
    # na_indices = cityDF[cityDF['species_name'].isna()].index
    # print(cityDF.iloc[na_indices])

    # cityDF['species_name'] = cityDF.apply(
    #     lambda row: getScientificNames(row['common_name']) if pd.isna(row['species_name']) else row['species_name'],
    #     axis=1
    # )

    # print("NA Values in species name after: ", cityDF['species_name'].isna().sum())
    # print(cityDF.iloc[na_indices])

    # Drop all na values and format the table to a string
    cityDF.dropna(how='all')
    cityDF = cityDF[['species_name', 'common_name']]
    cityDF['species_name'] = cityDF['species_name'].astype(str)
    cityDF['common_name'] = cityDF['common_name'].astype(str)

    # handle space removal and lowercase conversion
    cityDF['species_name'] = cityDF['species_name'].str.strip()
    cityDF['species_name'] = cityDF['species_name'].str.lower()

    cityDF['common_name'] = cityDF['common_name'].str.strip()
    cityDF['common_name'] = cityDF['common_name'].str.lower()

    cityDF = cityDF.map(handleNA)



    # add this files dataframe to a new dataframe
    df = pd.concat([df, cityDF])
    df.drop_duplicates( inplace=True)
    duplicated_column2 = df['common_name'].duplicated(keep=False)


# Reformat the dataframe to be put into the excel spreadsheet
df.rename(columns={'species_name': 'unique_sciname', 'common_name': 'unique_common_name'}, inplace=True)
df = df.map(handleNA)

# Ensure all spaces are the same
df = df.apply(lambda x: x.str.replace('\xa0', ' ', regex=True) if x.dtype == "object" else x)

# handle duplicates, including NA duplicates
cleaned_df = df.drop_duplicates()
cleaned_df = cleaned_df[~((cleaned_df['unique_sciname'] == 'NA') & cleaned_df['unique_common_name'].duplicated(keep=False))]

cleaned_df = cleaned_df[~((cleaned_df['unique_common_name'] == 'NA') & cleaned_df['unique_sciname'].duplicated(keep=False))]

cleaned_df.to_csv('cleaned_cityDF.csv', index=False)

