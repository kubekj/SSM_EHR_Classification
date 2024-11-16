#!/bin/sh
#BSUB -J grud_evaluation
#BSUB -o grud_evaluation_%J.out
#BSUB -e grud_evaluation_%J.err
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
    --output_path=grud_output \
    --model_type=grud \
    --epochs=100 \
    --batch_size=32 \
    --lr=0.0001 \
    --recurrent_dropout=0.2 \
    --recurrent_n_units=128