#!/bin/bash

PROJECT_ROOT="/dtu/blackhole/10/203216/SSM_EHR_Classification"

# Check if project directory exists
if [ ! -d "$PROJECT_ROOT" ]; then
    echo "Error: Project directory $PROJECT_ROOT does not exist"
    exit 1
fi

# Change to project directory
cd "$PROJECT_ROOT" || exit
echo "Changed to directory: $(pwd)"

# Load modules
module load python3/3.9.19
module load cuda/11.8
module load cudnn/v8.8.0-prod-cuda-11.X

# Create and activate virtual environment
VENV_PATH="$PROJECT_ROOT/venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating new virtual environment at $VENV_PATH"
    python -m venv "$VENV_PATH"
fi

# Activate virtual environment
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
    echo "Activated virtual environment"
else
    echo "Error: Virtual environment activation script not found"
    exit 1
fi

# Install requirements if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
    pip install torch_scatter --extra-index-url https://data.pyg.org/whl/torch-2.2.0+cu118.html
else
    echo "Error: requirements.txt not found"
    exit 1
fi