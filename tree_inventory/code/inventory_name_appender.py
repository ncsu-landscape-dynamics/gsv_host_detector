import pandas as pd
from EcoNameTranslator import to_scientific, to_common, to_species
import os
import csv
from pytaxize import scicomm
from pygbif import species
usdaDF = pd.read_csv('G:/Shared drives/host_tree_cnn/cleaning_species_names/addl_data/usda_code.txt')
print(usdaDF.columns.tolist())
usdaDF['Scientific Name with Author'] = usdaDF['Scientific Name with Author'].str.lower()
usdaDF['Common Name'] = usdaDF['Common Name'].str.lower()

# ['Symbol', 'Synonym Symbol', 'Scientific Name with Author', 'Common Name', 'Family']
prevFindings = {}
# clean up the scientific names
def cleanScientific(scientificName):
    global prevFindings
    global iterations
    if (scientificName == 'NA'):
        iterations += 1
        return ["NA", "NA"]
    # if iterations % 100 == 0 or iterations == 1:
    #     print(str(i) + "/" + str(numRows))
    if scientificName in prevFindings:
        return prevFindings[scientificName]
    try:
        index = to_species([scientificName])
        values = index[scientificName][0].split()
        if len(values) > 2:
            print(values)
        prevFindings[scientificName] = [values[-2], values[-1]]
        return values[-2], values[-1]
    except:
        prevFindings[scientificName] = ["NA", "NA"]
        return ["NA", "NA"]


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
print(usdaDF.columns.tolist())
# ['Symbol', 'Synonym Symbol', 'Scientific Name with Author', 'Common Name', 'Family']
noneCount = 0
cNaValues = set()
common_namesDict = {}
def getScientificNames(common_name):
    global i
    global noneCount
    global cNaValues
    # if i % 100 == 0:
    #     print(i)
    common_name = common_name.lower()
    print(common_name)

    i += 1
    if common_name in cNaValues:
        noneCount += 1
        return None
    elif common_name in common_namesDict:
        print(common_namesDict[common_name])
        return common_namesDict[common_name]
    try:
        sci_name = to_scientific([common_name])[common_name][1]
        if len(sci_name) == 0:
            raise ValueError()
        
        common_namesDict[common_name] = sci_name[0]
        print(sci_name[0])
        return sci_name[0]
    except:
      result = usdaDF.loc[usdaDF['Common Name'] == common_name, 'Scientific Name with Author'].values
      if len(result) > 0:
          print("RESULT")
          print(result[0])
          common_namesDict[common_name] = result[0]
          return result[0]

      print("NONE")
      noneCount += 1 
      cNaValues.add(common_name)
      return "NA"

        # result = species.name_suggest(common_name)
        # print(result)
        # if True:
        #   result = scicomm.sci2comm(common_name, db="itis")[common_name][0]
        #   print(result)
        #   thing = species.name_suggest(result)
        #   print(thing)
        # # except:
        #     print("fail")
        # if result:
        #     sci_name = result[0]['scientificName']
        #     if len(sci_name) == 0:
        #         raise ValueError()
        #     print(sci_name[0])
        #     common_namesDict[common_name] = sci_name[0]
        #     return sci_name

        # else:
        #     raise ValueError()
      # except:

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
    value = value.replace('.', '')
    # questions --- common name as array, or string?
    naValues = {'', 'a', 'nan', 'other', 'n/a', ' ', 'not specified', 'na', 'none', 'p', 'f'}
    containsNaValues = {'not specified', 'unidentified', 'unsuitable', 'vacant', '*', '_', '-', 'proposed', 'unknown', '#', 'other ', 'no ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}

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
folder_path = "G:/Shared drives/host_tree_cnn/cleaning_species_names/og_inventories_modified_labels"
# currentCities = {"columbus_inventory.csv"}
# Iterate through the files

for i, filename in enumerate(os.listdir(folder_path)):
    if filename == 'usda_code.csv':
        continue
    if filename != 'columbus_inventory.csv':
        continue
    # if filename not in currentCities:
    #     continue
    iterations = 1
    print(filename)
    print("FILE #", i)
    file_path = os.path.join(folder_path, filename)
    if not filename.endswith('.csv'):
        continue

    cityDF = pd.read_csv(file_path)
    
    cityDF = cityDF.map(reorganizeComma)
    
    
    if filename == 'columbus_inventory.csv':
        symbolDict = {}
        synonymDict = {}
        symbolDF = usdaDF.set_index('Symbol')
        synonymDF = usdaDF.set_index('Synonym Symbol')

        symbolDict = symbolDF['Common Name'].to_dict()
        synonymDict = synonymDF['Common Name'].to_dict()

        commonDict = symbolDict.copy()  
        commonDict.update(synonymDict)


        symbolDict = symbolDF['Scientific Name with Author'].to_dict()
        synonymDict = synonymDF['Scientific Name with Author'].to_dict()

        sciDict = symbolDict.copy()  
        sciDict.update(synonymDict)

        cityDF['common_name'] = cityDF['SP_CODE'].map(commonDict)
        cityDF['species_name'] = cityDF['SP_CODE'].map(sciDict)
        print(cityDF['common_name'])
        na_percentage = cityDF['common_name'].isna().mean() * 100
        print(f"Percentage of NaN values in 'common_name': {na_percentage:.2f}%")

        na_percentage = cityDF['species_name'].isna().mean() * 100
        print(f"Percentage of NaN values in 'species_name': {na_percentage:.2f}%")

    # Drop all na values and format the table to a string
    # cityDF.dropna(how='all')
    cityDF['unique_common_name'] = cityDF['common_name'].astype(str)

    # In case you need to get scientific names from ecotranslator
    if filename == 'bloomington_inventory.csv':
      print(cityDF[['unique_common_name']])
      cityDF['species_name'] = cityDF['unique_common_name'].apply(getScientificNames)
      print(cityDF)
    cityDF['unique_sciname'] = cityDF['species_name'].astype(str)

    # handle space removal and lowercase conversion
    cityDF['unique_sciname'] = cityDF['unique_sciname'].str.strip() 
    cityDF['unique_sciname'] = cityDF['unique_sciname'].str.lower()

    cityDF['unique_common_name'] = cityDF['unique_common_name'].str.strip()
    cityDF['unique_common_name'] = cityDF['unique_common_name'].str.lower()

    cityDF['unique_sciname'] = cityDF['unique_sciname'].map(handleNA)
    cityDF['unique_common_name'] = cityDF['unique_common_name'].map(handleNA)
    
    
    cityDF[['genus_name', 'species_name']] = cityDF.apply(
        lambda row: pd.Series(cleanScientific(row['unique_sciname'])),
        axis=1,
        result_type='expand'
    )
    cols = [col for col in cityDF.columns if col not in ['unique_common_name', 'unique_sciname', 'genus_name', 'species_name']]
    # Append the new columns at the end
    cols += ['unique_common_name', 'unique_sciname', 'genus_name', 'species_name']

    # Reorder the DataFrame
    cityDF = cityDF[cols]





    # no longer need to handle duplicates
    # Ensure all spaces are the same
    # cityDF = cityDF.apply(lambda x: x.str.replace('/xa0', ' ', regex=True) if x.dtype == "object" else x)

    # handle duplicates, including NA duplicates = longer need to drop dupliacates
    # cleaned_df = cityDF.drop_duplicates()
    # cleaned_df = cleaned_df[~((cleaned_df['unique_sciname'] == 'NA') & cleaned_df['unique_common_name'].duplicated(keep=False))]

    # cleaned_df = cleaned_df[~((cleaned_df['unique_common_name'] == 'NA') & cleaned_df['unique_sciname'].duplicated(keep=False))]
    cityDF.to_csv("G:/Shared drives/host_tree_cnn/cleaning_species_names/og_inventories_w_names_appended/" + filename, index=False)

