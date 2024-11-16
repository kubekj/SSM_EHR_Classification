#!/bin/sh
#BSUB -J transformer_train
#BSUB -o transformer_train_%J.out
#BSUB -e transformer_train_%J.err
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 9GB
#BSUB -W 24:00
#BSUB -N

cd /dtu/blackhole/10/203216/SSM_EHR_Classification

# Load modules and setup environment
module load python3/3.9.19
module load cuda/11.8
module load cudnn/v8.8.0-prod-cuda-11.X
python -m venv ./venv
./venv/bin/activate
pip install -r requirements.txt
pip install torch_scatter --extra-index-url https://data.pyg.org/whl/torch-2.2.0+cu118.html

python scripts/cli.py \
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