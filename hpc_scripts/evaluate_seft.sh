#!/bin/sh
#BSUB -J seft_train
#BSUB -o hpc_scripts/seft_train.out
#BSUB -e hpc_scripts/seft_train.err
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
    python -m venv ./venv
fi

source .venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt
pip install torch_scatter --extra-index-url https://data.pyg.org/whl/torch-2.2.0+cu118.html

python cli.py \
    --output_path=seft_output \
    --model_type=seft \
    --epochs=100 \
    --batch_size=128 \
    --dropout=0.4 \
    --attn_dropout=0.3 \
    --heads=2 \
    --lr=0.01 \
    --seft_dot_prod_dim=512 \
    --seft_n_phi_layers=1 \
    --seft_n_psi_layers=5 \
    --seft_n_rho_layers=2 \
    --seft_phi_dropout=0.3 \
    --seft_phi_width=512 \
    --seft_psi_width=32 \
    --seft_psi_latent_width=128 \
    --seft_latent_width=64 \
    --seft_rho_dropout=0.0 \
    --seft_rho_width=256 \
    --seft_max_timescales=1000 \
    --seft_n_positional_dims=16