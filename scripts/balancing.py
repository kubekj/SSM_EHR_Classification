import numpy as np
import random

def upsample_minority_class(data, target_label=0, upsample_ratio=2.0):
    """
    Upsample the minority class by duplicating existing samples to achieve a desired ratio.

    Args:
        data (list of dict): Dataset with time-series data.
        target_label (int): The label of the minority class to upsample.
        upsample_ratio (float): Desired ratio of majority to minority samples (majority / minority).

    Returns:
        list of dict: Dataset with upsampled minority class.
    """
    # Separate the minority and majority classes
    minority_data = [record for record in data if record['labels'] == target_label]
    majority_data = [record for record in data if record['labels'] != target_label]

    # Calculate the desired size of the minority class
    num_majority = len(majority_data)
    desired_minority_size = int(num_majority / upsample_ratio)

    # Calculate the number of new samples needed
    num_minority = len(minority_data)
    num_new_samples = max(0, desired_minority_size - num_minority)

    # Duplicate existing minority samples to upsample
    synthetic_data = random.choices(minority_data, k=num_new_samples)
    
    # Combine the original and synthetic data

    upsampled_data = np.concatenate((data, synthetic_data))
    random.shuffle(upsampled_data)

    return upsampled_data

def downsample_majority_class(data, target_label=0, downsample_ratio=1.0):
    """
    Downsample the majority class to achieve a desired ratio with the minority class.

    Args:
        data (list of dict): Dataset with time-series data.
        target_label (int): The label of the minority class to retain as-is.
        downsample_ratio (float): Desired ratio of majority to minority samples (majority / minority).

    Returns:
        list of dict: Dataset with the specified class distribution.
    """
    # Separate majority and minority classes
    minority_data = [record for record in data if record['labels'] != target_label]
    majority_data = [record for record in data if record['labels'] == target_label]

    # Calculate the desired size of the majority class
    num_minority = len(minority_data)
    desired_majority_size = int(num_minority * downsample_ratio)

    # Ensure we don't exceed the size of the majority class
    desired_majority_size = min(desired_majority_size, len(majority_data))

    # Randomly select a subset of the majority class
    downsampled_majority = random.sample(majority_data, desired_majority_size)

    # Combine the downsampled majority class with the minority class
    balanced_dataset = np.concatenate((minority_data, downsampled_majority))

    # Shuffle the dataset to mix classes
    random.shuffle(balanced_dataset)

    return balanced_dataset

def calculate_class_ratio(data):
    """
    Calculate the ratio of each class in the dataset.
    
    Args:
        data (list of dict): Dataset where each sample has a 'labels' key.
        
    Returns:
        dict: A dictionary with class labels as keys and their ratios as values.
    """
    # Count the occurrences of each label
    label_counts = {}
    for record in data:
        label = record['labels']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    # Calculate total number of samples
    total_samples = len(data)
    
    # Calculate the ratio for each class
    label_ratios = {label: count / total_samples for label, count in label_counts.items()}
    # print(label_counts)
    return label_ratios
