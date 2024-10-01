#!/bin/bash
#BSUB -n 4
#BSUB -W 1000
#BSUB -J eval_cnn
#BSUB -o stdout.%J
#BSUB -e stderr.%J
#BSUB -q gpu
#BSUB -R span[hosts=1]
#BSUB -R "select[a100]"
#BSUB -gpu "num=1:mode=shared:mps=no"
#BSUB -R rusage[mem=64GB]


module load conda
source activate /usr/local/usrapps/rkmeente/talake2/pytorch_env
export PYTHONPATH=$PYTHONPATH:/home/talake2
python -m tree_classification.evaluate_cities_test --config "/home/talake2/tree_classification/eval_config_hpc.json"
conda deactivate
