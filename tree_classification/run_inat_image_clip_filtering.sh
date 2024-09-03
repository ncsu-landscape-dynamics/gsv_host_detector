#!/bin/bash
#BSUB -n 4
#BSUB -W 600
#BSUB -J clip_filtering[1-98]
#BSUB -o stdout.%J
#BSUB -e stderr.%J
#BSUB -R span[hosts=1]
#BSUB -R rusage[mem=64GB]

# Define the genera array
genera=("acer" "fraxinus" "quercus" "ulmus" "prunus" "tilia" "pyrus" "gleditsia" "malus"
"platanus" "liquidambar" "pinus" "magnolia" "picea" "ginkgo" "zelkova" "celtis"
"crataegus" "populus" "carpinus" "syringa" "lagerstroemia" "betula" "amelanchier"
"cornus" "cercis" "gymnocladus" "washingtonia" "aesculus" "ficus" "eucalyptus"
"pistacia" "cinnamomum" "koelreuteria" "syagrus" "juniperus" "robinia" "cupressus"
"liriodendron" "catalpa" "ligustrum" "thuja" "jacaranda" "cercidiphyllum"
"ceratonia" "fagus" "morus" "schinus" "phoenix" "pittosporum" "parrotia" "sorbus"
"olea" "cupaniopsis" "melaleuca" "juglans" "arbutus" "nyssa" "acacia"
"cedrus" "podocarpus" "styrax" "metrosideros" "lophostemon" "eriobotrya" "ailanthus"
"metasequoia" "cladrastis" "styphnolobium" "casuarina" "maytenus" "sequoia" "pseudotsuga"
"taxodium" "citrus" "nerium" "alnus" "ostrya" "chamaecyparis" "triadica" "rhamnus"
"salix" "corylus" "myoporum" "albizia" "phellodendron" "ilex" "rhus" "elaeagnus"
"persea" "larix" "abies" "carya" "hibiscus" "chionanthus" "tsuga" "taxus" "castanea")

# Get the genus for the current array job
genus=${genera[$((LSB_JOBINDEX-1))]}

# Load conda and activate environment
module load conda
source activate /usr/local/usrapps/rkmeente/talake2/pytorch_env

# Run the Python script with the selected genus
python iNaturalist_Image_Filtering_CLIP.py $genus

conda deactivate
