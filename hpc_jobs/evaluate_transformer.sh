#!/bin/sh
#BSUB -J transformer_evaluation
#BSUB -o transformer_evaluation.out
#BSUB -e transformer_evaluation.err
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
    --output_path=transformer_output \
    --model_type=transformer \
    --epochs=100 \
    --batch_size=16 \
    --dropout=0.2 \
    --attn_dropout=0.1 \
    --layers=3 \
    --heads=1 \
    --pooling=max \
    --lr=0.0001