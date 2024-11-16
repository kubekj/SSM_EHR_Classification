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

module load python3/3.9.19
module load cuda/11.8
module load cudnn/v8.8.0-prod-cuda-11.X

python -m venv ./venv
./venv/bin/activate

pip install -r requirements.txt
pip install torch_scatter --extra-index-url https://data.pyg.org/whl/torch-2.2.0+cu118.html

# Run the evaluation
python cli.py \
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