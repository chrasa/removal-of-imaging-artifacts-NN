#!/bin/bash -l
#SBATCH -M snowy
#SBATCH -A snic2022-22-1060
#SBATCH -p core -N 1
#SBATCH -t 5:00:00
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH -J generate_training_data
#SBATCH -D ./

module load python3/3.9.5
module load python_ML_packages/3.9.5-gpu

source venv/bin/activate

nvidia-smi

echo "Train networks..."

python3 cnn.py ConvNN sobel 5 noload

echo " "
echo "Finished calculations"