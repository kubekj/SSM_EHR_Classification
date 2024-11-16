#!/bin/sh
#BSUB -J ipnets_train
#BSUB -o ipnets_train_%J.out
#BSUB -e ipnets_train_%J.err
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 20GB
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
    --output_path=ipnets_output \
    --model_type=ipnets \
    --epochs=100 \
    --batch_size=32 \
    --lr=0.001 \
    --ipnets_imputation_stepsize=1 \
    --ipnets_reconst_fraction=0.75 \
    --recurrent_dropout=0.3 \
    --recurrent_n_units=32