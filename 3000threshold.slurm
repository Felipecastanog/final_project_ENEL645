#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=23:59:00
#SBATCH --mem=60GB
#SBTACH --job-name=finalproject
#SBATCH --mail-type=END
#SBATCH --mail-user=felipe.castanogonzal@ucalgary.ca
#SBATCH --output=3000threshold_%j.out 

date

# Initialize Conda for this session
eval "$(/home/felipe.castanogonzal/software/miniconda3/bin/conda shell.bash hook)"

conda activate felipe_finalproject

python 3000threshold.py

date
