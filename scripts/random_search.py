from models.dssm import DSSM
from train import define_parameter_grid, initialize_training, train_split

import numpy as np
import torch
import json
import itertools
from pathlib import Path
import random

# Assuming DSSM and other necessary imports are defined elsewhere in the file

def train_model(model_class, model_params, training_params, device, split_number=1):
    """
    Train the model with the given parameters.

    Args:
        model_class: The model class to train
        model_params: Model parameters dictionary
        training_params: Training parameters dictionary
        device: torch device to use
        split_number: Split number for cross-validation

    Returns:
        Metrics and trained model.
    """
    model, dataloaders, criterion, optimizer = initialize_training(
        model_class, model_params, training_params, split_number, device
    )

    split_metrics = train_split(
        split_number, model, dataloaders, criterion, optimizer,
        training_params, device
    )

    return split_metrics, model


def train_with_randomized_search_cv(model_class, base_model_params, base_training_params, device, n_iter=10, param_grid=None):
    """
    Perform random search manually by sampling parameters from the grid.

    Args:
        model_class: The model class to train
        base_model_params: Base model parameters dictionary
        base_training_params: Base training parameters dictionary
        device: torch device to use
        n_iter: Number of random combinations to try

    Returns:
        Best parameters, best metrics, and all results.
    """
    if param_grid is None:
        param_grid = define_parameter_grid()
    param_distributions = {
        'hidden_size': param_grid['hidden_size'],
        'num_layers': param_grid['num_layers'],
        'dropout_rate': param_grid['dropout_rate'],
        'learning_rate': param_grid['learning_rate'],
        'batch_size': param_grid['batch_size'],
        'class_weights': param_grid['class_weights'],
        'bidirectional': param_grid['bidirectional']
    }

    best_metrics = {'auroc': 0}
    best_params = {}
    results = []

    for i in range(n_iter):
        # Randomly sample a combination of parameters
        sampled_params = {
            key: random.choice(values) for key, values in param_distributions.items()
        }

        model_params = base_model_params.copy()
        training_params = base_training_params.copy()

        # Update model and training parameters with sampled values
        model_params['hidden_size'] = sampled_params['hidden_size']
        model_params['num_layers'] = sampled_params['num_layers']
        model_params['dropout_rate'] = sampled_params['dropout_rate']
        model_params['bidirectional'] = sampled_params['bidirectional']
        training_params['learning_rate'] = sampled_params['learning_rate']
        training_params['loader_params']['batch_size'] = sampled_params['batch_size']
        training_params['class_weights'] = sampled_params['class_weights']

        print(f"\nTrying combination {i + 1}/{n_iter}:")
        for key, value in sampled_params.items():
            print(f"{key}={value}")

        # Train model with these parameters
        metrics, _ = train_model(
            model_class, model_params, training_params, device, split_number=1
        )

        # Save results
        results.append({
            'params': sampled_params,
            'metrics': metrics
        })

        # Update best parameters if improved
        if metrics['auroc'] > best_metrics['auroc']:
            best_metrics = metrics
            best_params = sampled_params
            print("\nNew best parameters found!")
            print(f"New best AUROC: {best_metrics['auroc']:.4f}")

        # Save intermediate results to avoid losing progress
        save_path = 'random_search_intermediate_results.json'
        intermediate_results = {
            'completed_combinations': i + 1,
            'total_combinations': n_iter,
            'current_best_params': best_params,
            'current_best_metrics': best_metrics,
            'all_results': results
        }
        with open(save_path, 'w') as f:
            json.dump(intermediate_results, f, indent=4)

    return best_params, best_metrics, results


def random_search(n_iter=50,param_grid=None, results_path_str=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    base_model_params = {
        'input_size': 37,
        'static_input_size': 8,
        'num_classes': 2,
        'num_layers': 2,
        'bidirectional': True
    }

    base_training_params = {
        'num_epochs': 100,
        'class_weights': [1.163, 7.143],
        'loader_params': {
            'batch_size': 32,
            'shuffle': True
        },
        'early_stopping': {
            'patience': 10,
            'min_delta': 0,
            'verbose': True
        }
    }

    # Perform random search with n_iter iterations
    best_params, best_metrics, all_results = train_with_randomized_search_cv(
        DSSM,
        base_model_params,
        base_training_params,
        device,
        n_iter=n_iter,  # You can adjust this number as needed
        param_grid=param_grid
    )
    if (results_path_str is None):
        results_path_str = '../model_outputs/dssm_output/random_search_results.json'
    results_path = Path(results_path_str)
    with open(results_path, 'w') as f:
        json.dump({
            'best_params': best_params,
            'best_metrics': best_metrics,
            'all_results': all_results
        }, f, indent=4)

    print("\nBest parameters found:")
    print(json.dumps(best_params, indent=2))
    print("\nBest metrics achieved:")
    print(json.dumps(best_metrics, indent=2))
    return best_params, best_metrics, all_results


if __name__ == "__main__":
    random_search()
