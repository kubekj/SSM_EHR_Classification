# Deep State Space Model for Mortality Classification using Electronic Health Records (EHRs).
Project no. 24 for 02456 taken at Autumn 2024.

# Project Description
Electronic Health Records (EHRs) are an abundant resource for clinical research and are produced in increasing numbers around the world. Unfortunately, they are also noisy, collected at irregular times, and full of missing data points, making them challenging to learn from. Despite these issues, EHR data contains valuable information for patient outcome forecasting, disease classification, and imputation. As a result, many models have been developed to handle these tasks.
For classification, most state-of-the-art models are RNNS, particularly variations of the LSTM and GRU architectures (ex: GRU-D, P-LSTM). The ability to handle variable input lengths and retain context from previous measurements is widely speculated to contribute to their dominance, but they are relatively slow and computationally expensive.

Recent advances in deep-state space models (SSMs) have great potential for modelling time series. Similarly to RNNS, SSMs capture temporal dependencies and handle missing values by estimating hidden states from previous data. By learning the underlying state of the system (ex., patient health) from the observed measurements (ex., vital signs, lab results), SSMs can model the hidden dynamics of a patient’s condition while accounting for noise and irregularity. But unlike RNNs, SSMs do this in a more structured and interpretable way by relying on probabilistic transitions between states. They are also more computationally efficient and scalable.
        
This project focuses on implementing a deep-state space model for EHR mortality classification and comparing it to current state-of-the-art models. The model can be an adaptation of EHRMamba (easier) or implemented yourself (challenging). Preprocessed data from Physionet 2012, MIMIC-III, and code for several SOTA models will be readily available.
        
Supervised by; Rachael Marie De Vries (rachael.devries@bio.ku.dk)

# Background
This repository allows you to train and test a variety of electronic health record (EHR) classification models on mortality prediction for the Physionet 2012 Challenge (`P12`) dataset. More information on the dataset can be found here (https://physionet.org/content/challenge-2012/1.0.0/). Note that the data in the repository has already been preprocessed (outliers removed, normalized) in accordance with https://github.com/ExpectationMax/medical_ts_datasets/tree/master and saved as 5 randomized splits of train/validation/test data. Adam is used for optimization. Additionally, it is worth mentioning that the state-of-the-art model we're basing this task on is called Mamba and can be found here https://github.com/VectorInstitute/odyssey/tree/main/odyssey/models/ehr_mamba.

# Create Environment
The dependencies are listed for python 3.9.

To create an environment and install required packages, run one of the following: 

## Venv way
```
# CD into the project folder
module load python3/3.9.19
module load cuda/11.8
module load cudnn/v8.8.0-prod-cuda-11.X
python -m venv ./venv
source ./venv/bin/activate
pip install -r requirements.txt
pip install torch_scatter --extra-index-url https://data.pyg.org/whl/torch-2.2.0+cu118.html
```

## Conda way
```
# CD into the project folder
module load cuda/11.8
module load cudnn/v8.8.0-prod-cuda-11.X
conda create --name <your-env-name> python=3.9
conda activate <your-env-name> 
pip install -r requirements.txt
pip install torch_scatter --extra-index-url https://data.pyg.org/whl/torch-2.2.0+cu118.html
```





# Run models 
4 baseline models have been implemented in `Pytorch` and can be trained/tested on `P12`. Each has a unique set of hyperparameters that can be modified, but I've gotten the best performance by running the following commands (_Note: you should unzip the data files before running these, and change the output paths in the commands_):

`transformer` (https://arxiv.org/abs/1706.03762):

`python cli.py --output_path=your/path/here --epochs=100 --batch_size=16 --model_type=transformer --dropout=0.2 --attn_dropout=0.1 --layers=3 --heads=1 --pooling=max --lr=0.0001` 


`seft` (https://github.com/BorgwardtLab/Set_Functions_for_Time_Series):

`python cli.py --output_path=your/path/here --model_type=seft --epochs=100 --batch_size=128 --dropout=0.4 --attn_dropout=0.3 --heads=2 --lr=0.01 --seft_dot_prod_dim=512 --seft_n_phi_layers=1 --seft_n_psi_layers=5 --seft_n_rho_layers=2 --seft_phi_dropout=0.3 --seft_phi_width=512 --seft_psi_width=32 --seft_psi_latent_width=128 --seft_latent_width=64 --seft_rho_dropout=0.0 --seft_rho_width=256 --seft_max_timescales=1000 --seft_n_positional_dims=16`

`grud` (https://github.com/PeterChe1990/GRU-D/blob/master/README.md):

`python cli.py --output_path=your/path/here --model_type=grud --epochs=100 --batch_size=32 --lr=0.0001 --recurrent_dropout=0.2 --recurrent_n_units=128`

`ipnets` (https://github.com/mlds-lab/interp-net):

`python cli.py --output_path=your/path/here --model_type=ipnets --epochs=100 --batch_size=32 --lr=0.001 --ipnets_imputation_stepsize=1 --ipnets_reconst_fraction=0.75 --recurrent_dropout=0.3 --recurrent_n_units=32` 


# DIY
You are welcome to fork the repository and make your own modifications :) 
