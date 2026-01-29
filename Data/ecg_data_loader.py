"""
ECG Data Loader Module
Handles loading and parsing of ECG data files
"""

import numpy as np
import os


def load_ecg_data(filepath):
    """
    Load ECG data from text file
    
    Args:
        filepath: Path to the ECG data file
    
    Returns:
        List of ECG signals (each as numpy array)
    """
    signals = []
    
    with open(filepath, 'r') as f:
        for line in f:
            # Split by pipe delimiter and convert to float
            values = [float(x) for x in line.strip().split('|') if x]
            signals.append(np.array(values))
    
    return signals


def load_dataset(data_folder, beat_types=['Normal', 'RBBB', 'LBBB']):
    """
    Load complete dataset with train and test splits
    
    Args:
        data_folder: Path to folder containing data files
        beat_types: List of beat types to load
    
    Returns:
        Dictionary containing train and test data with labels
    """
    dataset = {
        'train': {'signals': [], 'labels': []},
        'test': {'signals': [], 'labels': []}
    }
    
    # Map beat types to numeric labels
    label_map = {beat_type: idx for idx, beat_type in enumerate(beat_types)}
    
    for beat_type in beat_types:
        # Load training data
        train_file = os.path.join(data_folder, f'{beat_type}_Train.txt')
        if os.path.exists(train_file):
            train_signals = load_ecg_data(train_file)
            dataset['train']['signals'].extend(train_signals)
            dataset['train']['labels'].extend([label_map[beat_type]] * len(train_signals))
        
        # Load test data
        test_file = os.path.join(data_folder, f'{beat_type}_Test.txt')
        if os.path.exists(test_file):
            test_signals = load_ecg_data(test_file)
            dataset['test']['signals'].extend(test_signals)
            dataset['test']['labels'].extend([label_map[beat_type]] * len(test_signals))
    
    # Convert to numpy arrays
    dataset['train']['signals'] = dataset['train']['signals']
    dataset['train']['labels'] = np.array(dataset['train']['labels'])
    dataset['test']['signals'] = dataset['test']['signals']
    dataset['test']['labels'] = np.array(dataset['test']['labels'])
    
    return dataset, label_map
