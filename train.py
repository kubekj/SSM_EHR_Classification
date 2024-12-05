import wandb
import json
import itertools
import torch
import os
import argparse
import numpy as np
import torch.nn.functional as F

from pathlib import Path
from datetime import datetime

from sklearn.metrics import average_precision_score, accuracy_score, roc_auc_score
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from models.dssm import DSSM


class PhysionetDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        return (
            torch.tensor(record['ts_values'], dtype=torch.float32),
            torch.tensor(record['static'], dtype=torch.float32),
            torch.tensor(record['labels'], dtype=torch.long),
            len(record['ts_times'])
        )


class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.loss = 0.0
        self.predictions = []
        self.targets = []
        self.pred_probs = []
        self.num_samples = 0

    def update(self, loss, outputs, targets, batch_size):
        self.loss += loss * batch_size
        self.num_samples += batch_size

        pred_probs = F.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1)

        self.predictions.extend(predictions.cpu().detach().numpy())
        self.targets.extend(targets.cpu().detach().numpy())
        self.pred_probs.extend(pred_probs[:, 1].cpu().detach().numpy())

    def get_metrics(self):
        return {
            'loss': self.loss / self.num_samples,
            'accuracy': accuracy_score(self.targets, self.predictions),
            'auprc': average_precision_score(self.targets, self.pred_probs),
            'auroc': roc_auc_score(self.targets, self.pred_probs)
        }


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_epoch(model, loader, optimizer, criterion, device, tracker, epoch, split):
    model.train()
    tracker.reset()

    for batch_idx, batch in enumerate(loader):
        temporal_batch, static_batch, target_batch, seq_lengths = [
            x.to(device) for x in batch
        ]

        optimizer.zero_grad()
        outputs = model(temporal_batch, static_batch, seq_lengths)
        loss = criterion(outputs, target_batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        tracker.update(loss.item(), outputs, target_batch, temporal_batch.size(0))

        # Log every 10 batch metrics to wandb
        if batch_idx % 10 == 0:
            wandb.log({
                f"batch/train_loss_split_{split}": loss.item(),
                "epoch": epoch,
                "batch": batch_idx
            })

    metrics = tracker.get_metrics()

    wandb.log({
        f"train/loss_split_{split}": metrics['loss'],
        f"train/accuracy_split_{split}": metrics['accuracy'],
        f"train/auprc_split_{split}": metrics['auprc'],
        f"train/auroc_split_{split}": metrics['auroc'],
        "epoch": epoch
    })

    return metrics


def evaluate_model(model, loader, criterion, device, tracker, epoch, split, mode='val'):
    model.eval()
    tracker.reset()

    with torch.no_grad():
        for batch in loader:
            temporal_batch, static_batch, target_batch, seq_lengths = [
                x.to(device) for x in batch
            ]

            outputs = model(temporal_batch, static_batch, seq_lengths)
            loss = criterion(outputs, target_batch)

            tracker.update(loss.item(), outputs, target_batch, temporal_batch.size(0))

    metrics = tracker.get_metrics()

    wandb.log({
        f"{mode}/loss_split_{split}": metrics['loss'],
        f"{mode}/accuracy_split_{split}": metrics['accuracy'],
        f"{mode}/auprc_split_{split}": metrics['auprc'],
        f"{mode}/auroc_split_{split}": metrics['auroc'],
        "epoch": epoch
    })

    return metrics


def create_data_loader(data, batch_size, shuffle=True):
    dataset = PhysionetDataset(data)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )


def collate_fn(batch):
    ts_values, static, labels, lengths = zip(*batch)
    lengths = torch.tensor(lengths)
    ts_values_padded = pad_sequence(ts_values, batch_first=True)
    static = torch.stack(static)
    labels = torch.stack(labels)

    return ts_values_padded, static, labels, lengths


def load_split_data(split_number, base_path='P12data'):
    split_path = f'{base_path}/split_{split_number}'

    train_data = np.load(f'{split_path}/train_physionet2012_{split_number}.npy', allow_pickle=True)
    val_data = np.load(f'{split_path}/validation_physionet2012_{split_number}.npy', allow_pickle=True)
    test_data = np.load(f'{split_path}/test_physionet2012_{split_number}.npy', allow_pickle=True)

    return train_data, val_data, test_data


def setup_logging():
    """Setup logging directory and file"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = log_dir / f'training_log_{timestamp}.json'

    return log_path


def initialize_training(model_class, model_params, training_params, split, device):
    """Initialize all components needed for training"""
    train_data, val_data, test_data = load_split_data(split)

    dataloaders = {
        'train': create_data_loader(train_data, **training_params['loader_params']),
        'val': create_data_loader(val_data, **training_params['loader_params']),
        'test': create_data_loader(test_data, **training_params['loader_params'])
    }

    # Initialize model and training components
    model = model_class(**model_params).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(training_params['class_weights']).to(device))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_params['learning_rate']
    )

    return model, dataloaders, criterion, optimizer


def train_split(split, model, dataloaders, criterion, optimizer, training_params, device):
    """Train model on a single data split"""
    tracker = MetricsTracker()
    early_stopping = EarlyStopping(**training_params['early_stopping'])
    best_val_loss = float('inf')
    best_model_state = None

    # Initialize wandb run for this split
    run_name = f"split_{split}"
    wandb.init(
        project="DSSM-Mortality",
        name=run_name,
        config={
            "split": split,
            **training_params,
            "model_config": {
                "hidden_size": model.hidden_size,
                "num_layers": model.num_layers,
                "bidirectional": model.bidirectional
            }
        },
        reinit=True  # Allow multiple runs in the same script
    )

    # Log model architecture
    wandb.watch(model, log="all", log_freq=100)

    for epoch in range(training_params['num_epochs']):
        train_metrics = train_epoch(
            model, dataloaders['train'], optimizer, criterion,
            device, tracker, epoch, split
        )

        val_metrics = evaluate_model(
            model, dataloaders['val'], criterion, device,
            tracker, epoch, split, mode='val'
        )

        print(f"\nSplit {split}, Epoch {epoch + 1}/{training_params['num_epochs']}")
        print(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val AUROC: {val_metrics['auroc']:.4f}, Val AUPRC: {val_metrics['auprc']:.4f}")

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_state = model.state_dict().copy()

            # Log best model checkpoint to wandb
            checkpoint_path = f"best_model_split_{split}.pt"
            torch.save(best_model_state, checkpoint_path)
            wandb.save(checkpoint_path)

        # Early stopping check
        early_stopping(val_metrics['loss'], model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # Load best model for testing
    model.load_state_dict(best_model_state)

    # Final evaluation on test set
    test_metrics = evaluate_model(
        model, dataloaders['test'], criterion, device,
        tracker, epoch, split, mode='test'
    )

    # Log final test metrics
    wandb.log({
        f"test/final_loss_split_{split}": test_metrics['loss'],
        f"test/final_accuracy_split_{split}": test_metrics['accuracy'],
        f"test/final_auprc_split_{split}": test_metrics['auprc'],
        f"test/final_auroc_split_{split}": test_metrics['auroc']
    })

    # Close wandb run
    wandb.finish()

    return test_metrics


def calculate_final_metrics(splits_metrics):
    """Calculate mean and std of metrics across splits"""
    metrics_keys = ['loss', 'accuracy', 'auprc', 'auroc']
    final_metrics = {}

    for key in metrics_keys:
        values = [m[key] for m in splits_metrics]
        final_metrics[f'mean_{key}'] = float(np.mean(values))
        final_metrics[f'std_{key}'] = float(np.std(values))

    return final_metrics


def train_cross_validation(model_class, model_params, training_params, device, split_number=None):
    """Run training pipeline with wandb logging"""
    wandb.init(
        project="DSSM-Mortality",
        name="cross_validation",
        config={
            "model_params": model_params,
            "training_params": training_params,
        }
    )

    log_path = setup_logging()
    splits_metrics = []

    training_history = {
        'model_params': model_params,
        'training_params': training_params,
        'splits_metrics': [],
        'final_metrics': None
    }

    splits_to_run = [split_number] if split_number is not None else range(1, 6)

    for split in splits_to_run:
        model, dataloaders, criterion, optimizer = initialize_training(
            model_class, model_params, training_params, split, device
        )

        split_metrics = train_split(
            split, model, dataloaders, criterion, optimizer,
            training_params, device
        )
        splits_metrics.append(split_metrics)

        training_history['splits_metrics'].append({
            'split': split,
            'metrics': split_metrics
        })

        # Save intermediate results
        with open(log_path, 'w') as f:
            json.dump(training_history, f, indent=4)

    # Calculate and save final metrics
    final_metrics = calculate_final_metrics(splits_metrics)
    training_history['final_metrics'] = final_metrics

    # Log final cross-validation metrics
    wandb.log({
        "final/mean_loss": final_metrics['mean_loss'],
        "final/mean_accuracy": final_metrics['mean_accuracy'],
        "final/mean_auprc": final_metrics['mean_auprc'],
        "final/mean_auroc": final_metrics['mean_auroc'],
        "final/std_loss": final_metrics['std_loss'],
        "final/std_accuracy": final_metrics['std_accuracy'],
        "final/std_auprc": final_metrics['std_auprc'],
        "final/std_auroc": final_metrics['std_auroc']
    })

    # Close wandb
    wandb.finish()

    return final_metrics, splits_metrics


def define_parameter_grid():
    return {
        # Model parameters
        'hidden_size': [32, 64, 128, 256],  # Capacity of the model
        'num_layers': [1, 2, 3],  # Depth of LSTM
        'dropout_rate': [0.1, 0.2, 0.3, 0.4],  # Regularization strength

        # Training parameters
        'learning_rate': [0.01, 0.001, 0.0001],  # Learning rate range
        'batch_size': [16, 32, 64],  # Batch size options

        # Class weights (given high class imbalance)
        'class_weights': [
            [1.0, 7.143],  # Original weights
            [1.163, 7.143],  # Current weights
            [1.0, 6.0],  # Alternative weights
            [1.0, 5.0]  # Alternative weights #2
        ]
    }


def train_with_grid_search(model_class, base_model_params, base_training_params, device):
    param_grid = define_parameter_grid()
    best_metrics = {'auroc': 0}
    best_params = {}
    results = []

    # Initialize wandb sweep configuration
    sweep_config = {
        'method': 'grid',
        'name': 'dssm-grid-search',
        'metric': {'name': 'auroc', 'goal': 'maximize'},
        'parameters': {
            'hidden_size': {'values': param_grid['hidden_size']},
            'learning_rate': {'values': param_grid['learning_rate']},
            'dropout_rate': {'values': param_grid['dropout_rate']}
        }
    }

    wandb.sweep(sweep_config, project="DSSM-Mortality")

    # Generate all parameter combinations
    param_names = ['hidden_size', 'learning_rate', 'dropout_rate']
    param_values = [param_grid[name] for name in param_names]
    param_combinations = list(itertools.product(*param_values))

    total_combinations = len(param_combinations)
    print(f"Total parameter combinations to try: {total_combinations}")

    for i, (hidden_size, lr, dropout) in enumerate(param_combinations, 1):
        # Start a new wandb run for this combination
        wandb.init(
            project="DSSM-Mortality",
            group="grid-search",
            name=f"trial_{i}",
            config={
                'hidden_size': hidden_size,
                'learning_rate': lr,
                'dropout_rate': dropout,
                'base_model_params': base_model_params,
                'base_training_params': base_training_params
            },
            reinit=True
        )

        model_params = base_model_params.copy()
        training_params = base_training_params.copy()

        model_params['hidden_size'] = hidden_size
        model_params['dropout_rate'] = dropout
        training_params['learning_rate'] = lr

        print(f"\nTrying combination {i}/{total_combinations}:")
        print(f"hidden_size={hidden_size}, lr={lr}, dropout={dropout}")

        # Train model with these parameters
        metrics, _ = train_cross_validation(
            model_class=model_class,
            model_params=model_params,
            training_params=training_params,
            device=device,
            split_number=1
        )

        # Log metrics to wandb
        wandb.log({
            'test_loss': metrics['loss'],
            'test_accuracy': metrics['accuracy'],
            'test_auprc': metrics['auprc'],
            'test_auroc': metrics['auroc']
        })

        # Save results
        results.append({
            'params': {
                'hidden_size': hidden_size,
                'learning_rate': lr,
                'dropout_rate': dropout
            },
            'metrics': metrics
        })

        # Update best parameters if improved
        if metrics['auroc'] > best_metrics['auroc']:
            best_metrics = metrics
            best_params = {
                'hidden_size': hidden_size,
                'learning_rate': lr,
                'dropout_rate': dropout
            }

            # Log best model configuration
            wandb.run.summary['best_auroc'] = metrics['auroc']
            wandb.run.summary['best_params'] = best_params

            print("\nNew best parameters found!")
            print(f"New best AUROC: {best_metrics['auroc']:.4f}")

        # Save intermediate results
        save_path = 'grid_search_intermediate_results.json'
        intermediate_results = {
            'completed_combinations': i,
            'total_combinations': total_combinations,
            'current_best_params': best_params,
            'current_best_metrics': best_metrics,
            'all_results': results
        }
        with open(save_path, 'w') as f:
            json.dump(intermediate_results, f, indent=4)

        # Close the wandb run
        wandb.finish()

    # Create a final summary plot
    create_grid_search_summary(results)

    return best_params, best_metrics, results


def cross_validation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    model_params = {
        'input_size': 37,
        'hidden_size': 64,
        'static_input_size': 8,
        'num_classes': 2,
        'num_layers': 3,
        'dropout_rate': 0.1,
        'bidirectional': False
    }

    training_params = {
        'num_epochs': 100,
        'learning_rate': 0.001,
        'class_weights': [1.0, 6.0],
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

    final_metrics, splits_metrics = train_cross_validation(
        model_class=DSSM,
        model_params=model_params,
        training_params=training_params,
        device=device
    )

    print("\nFinal Cross-Validation Results:")
    print("Mean ± Std:")
    for metric, value in final_metrics.items():
        if metric.startswith('mean_'):
            base_metric = metric[5:]  # Remove 'mean_' prefix
            std_value = final_metrics[f'std_{base_metric}']
            print(f"{base_metric}: {value:.4f} ± {std_value:.4f}")


def grid_search():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    wandb.init(project="DSSM-Mortality", name="grid-search-main")

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

    best_params, best_metrics, all_results = train_with_grid_search(
        DSSM,
        base_model_params,
        base_training_params,
        device
    )

    wandb.log({
        'final_best_params': best_params,
        'final_best_metrics': best_metrics
    })

    results_path = Path('grid_search_results.json')
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

    wandb.finish()


def create_grid_search_summary(results):
    """Create summary visualizations of grid search results"""

    with wandb.init(project="DSSM-Mortality", name="grid-search-summary", job_type="analysis"):
        # Create a parallel coordinates plot
        parallel_coords_data = [{
            'hidden_size': r['params']['hidden_size'],
            'learning_rate': r['params']['learning_rate'],
            'dropout_rate': r['params']['dropout_rate'],
            'auroc': r['metrics']['auroc'],
            'auprc': r['metrics']['auprc'],
            'accuracy': r['metrics']['accuracy']
        } for r in results]

        wandb.log({"grid_search_results": wandb.Table(data=parallel_coords_data)})

        # Create custom plots using wandb
        wandb.log({
            "parameter_importance": wandb.plot.parallel_coordinates(
                parallel_coords_data,
                cols=['hidden_size', 'learning_rate', 'dropout_rate', 'auroc']
            )
        })


def get_wandb_key():
    wandb_key = os.getenv('WANDB_API_KEY')

    if not wandb_key:
        raise ValueError(
            "WANDB_API_KEY not found in environment variables. "
            "Please set it using: export WANDB_API_KEY='your-key' (Linux/Mac) "
            "or set WANDB_API_KEY='your-key' (Windows)"
        )

    return wandb_key


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DSSM Training')
    parser.add_argument('--mode', type=str, choices=['cross_validation', 'grid_search'],
                        default='cross_validation', help='Training mode')
    parser.add_argument('--wandb_project', type=str, default="DSSM-Mortality",
                        help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='W&B entity (username or team name)')

    args = parser.parse_args()

    # Setup wandb with key from environment
    try:
        wandb_key = get_wandb_key()
        wandb.login(key=wandb_key)
        print("Successfully logged into Weights & Biases!")
    except Exception as e:
        print(f"Error logging into Weights & Biases: {str(e)}")
        print("Continuing without W&B logging...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    if args.mode == 'cross_validation':
        model_params = {
            'input_size': 37,
            'hidden_size': 64,
            'static_input_size': 8,
            'num_classes': 2,
            'num_layers': 3,
            'dropout_rate': 0.1,
            'bidirectional': False
        }

        training_params = {
            'num_epochs': 100,
            'learning_rate': 0.001,
            'class_weights': [1.0, 6.0],
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

        # Initialize wandb for cross-validation
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name="cross_validation",
            config={
                "mode": "cross_validation",
                "model_params": model_params,
                "training_params": training_params
            }
        )

        try:
            final_metrics, splits_metrics = train_cross_validation(
                model_class=DSSM,
                model_params=model_params,
                training_params=training_params,
                device=device
            )

            print("\nFinal Cross-Validation Results:")
            print("Mean ± Std:")
            for metric, value in final_metrics.items():
                if metric.startswith('mean_'):
                    base_metric = metric[5:]
                    std_value = final_metrics[f'std_{base_metric}']
                    print(f"{base_metric}: {value:.4f} ± {std_value:.4f}")

        finally:
            wandb.finish()

    elif args.mode == 'grid_search':
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

        # Initialize wandb for grid search
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name="grid_search_main",
            config={
                "mode": "grid_search",
                "base_model_params": base_model_params,
                "base_training_params": base_training_params
            }
        )

        try:
            best_params, best_metrics, all_results = train_with_grid_search(
                DSSM,
                base_model_params,
                base_training_params,
                device
            )

            print("\nBest parameters found:")
            print(json.dumps(best_params, indent=2))
            print("\nBest metrics achieved:")
            print(json.dumps(best_metrics, indent=2))

        finally:
            wandb.finish()
