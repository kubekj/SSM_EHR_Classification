#!/bin/sh
#BSUB -J ipnets_evaluation
#BSUB -o ipnets_evaluation_%J.out
#BSUB -e ipnets_evaluation_%J.err
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 10GB
#BSUB -W 24:00
#BSUB -N

# Run setup script
. setup.sh

# Run the evaluation
python cli.py \
    --output_path=ipnets_output \
    --model_type=ipnets \
    --epochs=100 \
    --batch_size=32 \
    --lr=0.001 \
    --ipnets_imputation_stepsize=1 \
    --ipnets_reconst_fraction=0.75 \
    --recurrent_dropout=0.3 \
    --recurrent_n_units=32