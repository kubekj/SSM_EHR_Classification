#!/bin/sh
#BSUB -J seft_evaluation
#BSUB -o seft_evaluation.out
#BSUB -e seft_evaluation.err
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 09GB
#BSUB -W 24:00
#BSUB -N

# Define project root
PROJECT_ROOT="/dtu/blackhole/10/203216/SSM_EHR_Classification"

# Run setup script with project root as parameter
$PROJECT_ROOT setup.sh

# Verify cli.py exists
CLI_PATH="$PROJECT_ROOT/scripts/cli.py"
if [ ! -f "$CLI_PATH" ]; then
    echo "Error: cli.py not found at $CLI_PATH"
    exit 1
fi

echo "Running training script from $(pwd)"
python "$CLI_PATH" \
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