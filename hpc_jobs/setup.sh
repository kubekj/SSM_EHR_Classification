#!/bin/bash
cd /dtu/blackhole/10/203216/SSM_EHR_Classification  # Main project directory
module load python3/3.9.19
module load cuda/11.8
module load cudnn/v8.8.0-prod-cuda-11.X

if [ ! -d "venv" ]; then
    echo "Creating new virtual environment..."
    python -m venv ./venv
fi

source ./venv/bin/activate

echo "Installing requirements..."
pip install -r requirements.txt
pip install torch_scatter --extra-index-url https://data.pyg.org/whl/torch-2.2.0+cu118.html