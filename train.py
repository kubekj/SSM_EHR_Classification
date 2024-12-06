import wandb
import argparse
import json
import itertools
import torch
import os
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


def train_epoch(model, loader, optimizer, criterion, device, tracker, epoch, split, use_wandb=False):
    model.train()
    tracker.reset()

    for batch in loader:
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

    metrics = tracker.get_metrics()

    if use_wandb:
        wandb.log({
            f"split_{split}/train/loss": metrics['loss'],
            f"split_{split}/train/auroc": metrics['auroc'],
            "epoch": epoch
        })

    return metrics


def evaluate_model(model, loader, criterion, device, tracker, epoch, split, phase='val', use_wandb=False):
    model.eval()
    tracker.reset()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            temporal_batch, static_batch, target_batch, seq_lengths = [
                x.to(device) for x in batch
            ]

            outputs = model(temporal_batch, static_batch, seq_lengths)
            loss = criterion(outputs, target_batch)

            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target_batch.cpu().numpy())

            tracker.update(loss.item(), outputs, target_batch, temporal_batch.size(0))

    metrics = tracker.get_metrics()

    if use_wandb and phase != 'test':  # Only log validation metrics during training
        wandb.log({
            f"split_{split}/{phase}/loss": metrics['loss'],
            f"split_{split}/{phase}/auroc": metrics['auroc'],
            f"split_{split}/{phase}/auprc": metrics['auprc'],
            "epoch": epoch
        })

        if epoch % 10 == 0:  # Log confusion matrix less frequently
            wandb.log({
                f"split_{split}/{phase}/confusion_matrix": wandb.plot.confusion_matrix(
                    y_true=all_targets,
                    preds=all_predictions,
                    class_names=['Survived', 'Deceased']
                ),
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


def train_split(split, model, dataloaders, criterion, optimizer, training_params, device, use_wandb=False):
    tracker = MetricsTracker()
    early_stopping = EarlyStopping(**training_params['early_stopping'])
    best_val_loss = float('inf')
    best_model_state = None
    best_metrics = None

    for epoch in range(training_params['num_epochs']):
        # Training phase
        train_metrics = train_epoch(
            model, dataloaders['train'], optimizer, criterion,
            device, tracker, epoch, split, use_wandb
        )

        # Validation phase
        val_metrics = evaluate_model(
            model, dataloaders['val'], criterion, device,
            tracker, epoch, split, 'val', use_wandb
        )

        print(f"\nSplit {split}, Epoch {epoch + 1}/{training_params['num_epochs']}")
        print(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val AUROC: {val_metrics['auroc']:.4f}, Val AUPRC: {val_metrics['auprc']:.4f}")

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_state = model.state_dict().copy()
            best_metrics = val_metrics

            if use_wandb:
                wandb.log({
                    f"split_{split}/best_val_auroc": val_metrics['auroc'],
                    f"split_{split}/best_val_auprc": val_metrics['auprc'],
                    "epoch": epoch
                })

        early_stopping(val_metrics['loss'], model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # Load best model and evaluate on test set
    model.load_state_dict(best_model_state)
    test_metrics = evaluate_model(
        model, dataloaders['test'], criterion, device,
        tracker, epoch, split, 'test', use_wandb
    )

    return test_metrics, best_metrics


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
    use_wandb = initialize_wandb({
        "model_params": model_params,
        "training_params": training_params,
    })

    log_path = setup_logging()
    splits_metrics = []
    val_metrics = []

    try:
        splits_to_run = [split_number] if split_number is not None else range(1, 6)

        for split in splits_to_run:
            print(f"\n=== Training on Split {split}/{len(splits_to_run)} ===")

            model, dataloaders, criterion, optimizer = initialize_training(
                model_class, model_params, training_params, split, device
            )

            test_metrics, best_val_metrics = train_split(
                split, model, dataloaders, criterion, optimizer,
                training_params, device, use_wandb
            )

            splits_metrics.append(test_metrics)
            val_metrics.append(best_val_metrics)

        final_metrics = calculate_final_metrics(splits_metrics)
        final_val_metrics = calculate_final_metrics(val_metrics)

        if use_wandb:
            # Log only final cross-validation metrics
            wandb.log({
                "final/test_auroc": final_metrics['mean_auroc'],
                "final/test_auprc": final_metrics['mean_auprc'],
                "final/test_accuracy": final_metrics['mean_accuracy'],
                "final/val_auroc": final_val_metrics['mean_auroc'],
                "final/val_auprc": final_val_metrics['mean_auprc'],
                "final/val_accuracy": final_val_metrics['mean_accuracy']
            })

    finally:
        if use_wandb:
            wandb.finish()

    return final_metrics, splits_metrics


def define_parameter_grid():
    return {
        # Model parameters
        'hidden_size': [32, 64],  # Capacity of the model
        'num_layers': [2, 3],  # Depth of LSTM
        'dropout_rate': [0.2, 0.3],   # Regularization strength

        # Training parameters
        'learning_rate': [0.001, 0.0001],  # Regularization strength
        'batch_size': [32, 64],  # Batch sizes
        'class_weights': [
            [1.0, 7.143],   # Original ratio (performed well)
            [1.0, 8.5],     # Slightly higher weight
        ]
    }


def calculate_combined_score(metrics):
    """Calculate a combined score giving higher weight to AUPRC due to class imbalance"""
    weights = {
        'mean_auroc': 0.3,
        'mean_auprc': 0.4,  # Higher weight due to class imbalance
        'mean_accuracy': 0.3
    }

    combined_score = (
            weights['mean_auroc'] * metrics['mean_auroc'] +
            weights['mean_auprc'] * metrics['mean_auprc'] +
            weights['mean_accuracy'] * metrics['mean_accuracy']
    )

    return combined_score


def train_with_grid_search(model_class, base_model_params, base_training_params, device):
    param_grid = define_parameter_grid()
    best_metrics = {'combined_score': 0}
    best_params = {}
    results = []

    param_values = [
        param_grid['hidden_size'],
        param_grid['num_layers'],
        param_grid['dropout_rate'],
        param_grid['learning_rate'],
        param_grid['batch_size'],
        param_grid['class_weights'],
        [True]
    ]
    param_combinations = list(itertools.product(*param_values))

    total_combinations = len(param_combinations)
    print(f"Total parameter combinations to try: {total_combinations}")

    for i, params in enumerate(param_combinations, 1):
        (hidden_size, num_layers, dropout, lr, batch_size,
         class_weights, bidirectional) = params

        model_params = base_model_params.copy()
        training_params = base_training_params.copy()

        model_params.update({
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout_rate': dropout,
            'bidirectional': bidirectional
        })

        training_params.update({
            'learning_rate': lr,
            'class_weights': class_weights,
            'loader_params': {
                'batch_size': batch_size,
                'shuffle': True
            }
        })

        print(f"\nTrying combination {i}/{total_combinations}:")
        print(f"Parameters: hidden_size={hidden_size}, num_layers={num_layers}, "
              f"dropout={dropout}, lr={lr}, batch_size={batch_size}, "
              f"class_weights={class_weights}, bidirectional={bidirectional}")

        metrics, splits_metrics = train_cross_validation(
            model_class=model_class,
            model_params=model_params,
            training_params=training_params,
            device=device
        )

        combined_score = calculate_combined_score(metrics)
        metrics['combined_score'] = combined_score

        results.append({
            'params': {
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'dropout_rate': dropout,
                'learning_rate': lr,
                'batch_size': batch_size,
                'class_weights': class_weights,
                'bidirectional': bidirectional
            },
            'metrics': metrics,
            'splits_metrics': splits_metrics,
            'combined_score': combined_score
        })

        if combined_score > best_metrics['combined_score']:
            best_metrics = metrics
            best_params = results[-1]['params']
            print("\nNew best parameters found!")
            print(f"New best combined score: {combined_score:.4f}")
            print(f"AUROC: {metrics['mean_auroc']:.4f}")
            print(f"AUPRC: {metrics['mean_auprc']:.4f}")
            print(f"Accuracy: {metrics['mean_accuracy']:.4f}")

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

    return best_params, best_metrics, results


def cross_validation(device):
    model_params = {
        'input_size': 37,
        'hidden_size': 32,
        'static_input_size': 8,
        'num_classes': 2,
        'num_layers': 2,
        'dropout_rate': 0.2,
        'bidirectional': True
    }

    training_params = {
        'num_epochs': 100,
        'learning_rate': 0.0001,
        'class_weights': [1.0, 7.143],
        'loader_params': {
            'batch_size': 64,
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


def grid_search(device):
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

def get_wandb_key():
    wandb_key = os.getenv('WANDB_API_KEY')
    if not wandb_key:
        raise ValueError(
            "WANDB_API_KEY not found in environment variables. "
            "Please set it using: export WANDB_API_KEY='your-key'"
        )
    return wandb_key


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DSSM Training')
    parser.add_argument('--mode', type=str, choices=['cross_validation', 'grid_search'],
                        default='cross_validation', help='Training mode')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    if args.mode == 'cross_validation':
        try:
            wandb_key = get_wandb_key()
            wandb.login(key=wandb_key)
            print("Successfully logged into Weights & Biases!")
        except Exception as e:
            print(f"Error logging into Weights & Biases: {str(e)}")
            print("Continuing without W&B logging...")

        cross_validation(device)
    elif args.mode == 'grid_search':
        grid_search(device)
