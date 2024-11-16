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

source ./hpc_jobs/setup.sh

# Run the evaluation
python ./scripts/cli.py \
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