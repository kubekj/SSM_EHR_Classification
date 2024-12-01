#!/bin/sh
#BSUB -J dssm_train
#BSUB -o hpc_scripts/dssm_train.out
#BSUB -e hpc_scripts/dssm_train.err
#BSUB -q gpuv-100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 5GB
#BSUB -W 24:00
#BSUB -N

# Load required modules
module load python3/3.9.19
module load cuda/11.8
module load cudnn/v8.8.0-prod-cuda-11.X

if [ ! -d ".venv" ]; then
    echo "Creating new virtual environment..."
    python -m venv .venv
fi


# Activate virtual environment
source .venv/bin/activate

echo "Installing requirements..."
pip install --no-cache-dir -r requirements.txt

# Print environment information
python --version
pip list
nvidia-smi

echo "Starting training..."
python random_search.py

# Deactivate virtual environment
deactivate