#!/bin/bash -l
#SBATCH -M snowy
#SBATCH -A snic2022-22-1060
#SBATCH -p node -N 1
#SBATCH -t 1:00:00
#SBATCH --qos=short
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH -J generate_images_test
#SBATCH -D ./

module load python3/3.9.5
module load python_ML_packages/3.9.5-gpu

source venv/bin/activate

echo "Generate images..."

python3 generate_images.py -n 2

echo " "
echo "Finished calculations"
