import json
import itertools
import torch
import numpy as np
import torch.nn.functional as F

from pathlib import Path
from datetime import datetime

from sklearn.metrics import average_precision_score, accuracy_score, roc_auc_score
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from models.dssm import DSSM

from scripts.balancing import downsample_majority_class, upsample_minority_class, calculate_class_ratio

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


def train_epoch(model, loader, optimizer, criterion, device, tracker):
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

    return tracker.get_metrics()


def evaluate_model(model, loader, criterion, device, tracker):
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

    return tracker.get_metrics()


def create_data_loader(data, batch_size, shuffle=True, balancing='none', ratio=1.0):
    """
    Creates a DataLoader with optional class balancing.

    Args:
        data (list of dict): The dataset.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data.
        balancing (str): Type of balancing ('none', 'upsample', 'downsample').
        ratio (float): Upsample or downsample ratio.

    Returns:
        DataLoader: A DataLoader object.
    """
    # print(f"Data ratio BEFORE balancing: {calculate_class_ratio(data)}")
    
    if balancing == 'upsample':
        data = upsample_minority_class(data, target_label=1, upsample_ratio=ratio)
    elif balancing == 'downsample':
        data = downsample_majority_class(data, target_label=0, downsample_ratio=ratio)
    
    # print(f"Data ratio AFTER balancing: {calculate_class_ratio(data)}")

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
        'train': create_data_loader(
            train_data, 
            **training_params['loader_params'],
            balancing=training_params.get('balancing', 'none'),
            ratio=training_params.get('balancing_ratio', 1.0)
        ),
        'val': create_data_loader(
            val_data, 
            **training_params['loader_params']
        ),
        'test': create_data_loader(
            test_data, 
            **training_params['loader_params']
        )
    }

    # Initialize model and training components
    model = model_class(**model_params).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(training_params['class_weights']).to(device) if training_params['class_weights'] else None)
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

    for epoch in range(training_params['num_epochs']):
        train_metrics = train_epoch(
            model, dataloaders['train'], optimizer, criterion, device, tracker
        )

        val_metrics = evaluate_model(
            model, dataloaders['val'], criterion, device, tracker
        )

        print(f"\nSplit {split}, Epoch {epoch + 1}/{training_params['num_epochs']}")
        print(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val AUROC: {val_metrics['auroc']:.4f}, Val AUPRC: {val_metrics['auprc']:.4f}")

        # Save best model (we're only comparing the loss in terms of early stopping)
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_state = model.state_dict().copy()

        # Early stopping check
        early_stopping(val_metrics['loss'], model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # Load best model for testing
    model.load_state_dict(best_model_state)

    # Final evaluation on test set
    test_metrics = evaluate_model(
        model, dataloaders['test'], criterion, device, tracker
    )

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
    """Run training pipeline, either for all splits or a specific split

    Args:
        model_class: The model class to train
        model_params: Model parameters dictionary
        training_params: Training parameters dictionary
        device: torch device to use
        split_number: Optional; If provided, trains only on that split (1-5)
    """
    log_path = setup_logging()
    splits_metrics = []

    training_history = {
        'model_params': model_params,
        'training_params': training_params,
        'splits_metrics': [],
        'final_metrics': None
    }

    # Determine which splits to run
    if split_number is not None:
        if not 1 <= split_number <= 5:
            raise ValueError("split_number must be between 1 and 5")
        splits_to_run = [split_number]
    else:
        splits_to_run = range(1, 6)

    for split in splits_to_run:
        print(f"\n=== Training on Split {split}/{len(splits_to_run)} ===")

        # Initialize training components for this split
        model, dataloaders, criterion, optimizer = initialize_training(
            model_class, model_params, training_params, split, device
        )

        # Train on this split
        split_metrics = train_split(
            split, model, dataloaders, criterion, optimizer,
            training_params, device
        )
        splits_metrics.append(split_metrics)

        # Save intermediate results
        training_history['splits_metrics'].append({
            'split': split,
            'metrics': split_metrics
        })
        with open(log_path, 'w') as f:
            json.dump(training_history, f, indent=4)

        print(f"\nSplit {split} Test Metrics:")
        for k, v in split_metrics.items():
            print(f"{k}: {v:.4f}")

    # Calculate and save final metrics
    final_metrics = calculate_final_metrics(splits_metrics) if len(splits_metrics) > 1 else splits_metrics[0]
    training_history['final_metrics'] = final_metrics

    with open(log_path, 'w') as f:
        json.dump(training_history, f, indent=4)

    return final_metrics, splits_metrics


def define_parameter_grid():
    return {
        # Model parameters
        'hidden_size': [32],  # Capacity of the model
        'num_layers': [2],  # Depth of LSTM
        'dropout_rate': [0.2],  # Regularization strength

        # Training parameters
        'learning_rate': [0.0001],  # Learning rates
        'batch_size': [64],  # Batch sizes
        'class_weights': [
            None
            # [1.0, 7.143],  # Original ratio (performed well)
            # [1.0, 8.5],  # Slightly higher weight
        ],

        # Balancing parameters
        'balancing': [
            'none', 
            # 'upsample', 
            # 'downsample'
            ],  # Balancing strategies
        'balancing_ratio': [1.0, 1.5, 2.0, 2.5, 3.0]  # Ratios for upsampling/downsampling
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
        param_grid['balancing'],
        param_grid['balancing_ratio'],
        [False]
    ]
    param_combinations = list(itertools.product(*param_values))

    total_combinations = len(param_combinations)
    print(f"Total parameter combinations to try: {total_combinations}")

    for i, params in enumerate(param_combinations, 1):
        (hidden_size, num_layers, dropout, lr, batch_size,
         class_weights, balancing, balancing_ratio, bidirectional) = params

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
            'balancing': balancing,
            'balancing_ratio': balancing_ratio,
            'loader_params': {
                'batch_size': batch_size,
                'shuffle': True
            }
        })

        print(f"\nTrying combination {i}/{total_combinations}:")
        print(f"Parameters: hidden_size={hidden_size}, num_layers={num_layers}, "
              f"dropout={dropout}, lr={lr}, batch_size={batch_size}, "
              f"class_weights={class_weights}, balancing={balancing}, ratio={balancing_ratio}, "
              f"bidirectional={bidirectional}")

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
                'balancing': balancing,
                'balancing_ratio': balancing_ratio,
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

    base_model_params = {
        'input_size': 37,
        'static_input_size': 8,
        'num_classes': 2,
        'num_layers': 2,
        'bidirectional': False
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

    results_path = Path('model_outputs/dssm_output/grid_search_results_with_bidirection.json')
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


if __name__ == "__main__":
    grid_search()
