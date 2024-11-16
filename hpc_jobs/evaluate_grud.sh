#!/bin/sh
#BSUB -J grud_evaluation
#BSUB -o grud_evaluation.out
#BSUB -e grud_evaluation.err
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
    --output_path=grud_output \
    --model_type=grud \
    --epochs=100 \
    --batch_size=32 \
    --lr=0.0001 \
    --recurrent_dropout=0.2 \
    --recurrent_n_units=128