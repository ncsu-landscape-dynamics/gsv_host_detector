import pandas as pd
from EcoNameTranslator import to_scientific, to_common, to_species

 

cityDF = pd.read_csv("SpeciesCommon.csv")
print(cityDF.shape[0])
numRows = cityDF.shape[0]
i = 0
prevFindings = {}
def cleanScientific(scientificName):
    global i
    global numRows
    i += 1
    if i % 300 == 0 or i == 1:
        print(str(i) + "/" + str(numRows))
    if scientificName in prevFindings:
        return prevFindings[scientificName]
    try:
        index = to_species([scientificName])
        print(len(values))
        values = index[scientificName][0].split()
        if len(values) > 2:
            print(values)
        prevFindings[scientificName] = [values[-2], values[-1]]
        return values[-2], values[-1]
    except:
        return ["NA", "NA"]
        


cityDF = pd.read_csv("SpeciesCommon.csv")
cityDF[['genus_name', 'species_name']] = cityDF.apply(
    lambda row: pd.Series(cleanScientific(row['unique_sciname'])),
    axis=1,
    result_type='expand'
)
cityDF.to_csv('genusSpeciesSheet.csv', index=False)

# species_name, genus_name
# generated using econameparser