#!/bin/bash -l
#SBATCH -M snowy
#SBATCH -A snic2022-22-1060
#SBATCH -p core
#SBATCH -n 2
#SBATCH -t 15:00
#SBATCH --qos=short -t 14:00
#SBATCH -J merge_and_normalize_data
#SBATCH -D ./

conda activate tf

CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib

echo "Merge data..."
python3 merge_data.py

echo "Normalize data..."
python3 normalize_data.py
