#!/bin/sh
#BSUB -J ipnets_evaluation
#BSUB -o ipnets_evaluation.out
#BSUB -e ipnets_evaluation.err
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 09GB
#BSUB -W 24:00
#BSUB -N

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/setup.sh"

CLI_PATH="$PROJECT_ROOT/cli.py"
if [ ! -f "$CLI_PATH" ]; then
    echo "Error: cli.py not found at $CLI_PATH"
    exit 1
fi

echo "Running training script from $(pwd)"
python "$CLI_PATH" \
    --output_path=ipnets_output \
    --model_type=ipnets \
    --epochs=100 \
    --batch_size=32 \
    --lr=0.001 \
    --ipnets_imputation_stepsize=1 \
    --ipnets_reconst_fraction=0.75 \
    --recurrent_dropout=0.3 \
    --recurrent_n_units=32