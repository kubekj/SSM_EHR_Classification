#!/bin/sh
#BSUB -J grud_train
#BSUB -o hpc_scripts/grud_train.out
#BSUB -e hpc_scripts/grud_train.err
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 9GB
#BSUB -W 24:00
#BSUB -N

# Load modules and setup environment
module load python3/3.9.19
module load cuda/11.8
module load cudnn/v8.8.0-prod-cuda-11.X

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating new virtual environment..."
    python -m venv .venv
fi

source .venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r ../requirements.txt
pip install torch_scatter --extra-index-url https://data.pyg.org/whl/torch-2.2.0+cu118.html

python cli.py \
    --output_path=grud_output \
    --model_type=grud \
    --epochs=100 \
    --batch_size=32 \
    --lr=0.0001 \
    --recurrent_dropout=0.2 \
    --recurrent_n_units=128