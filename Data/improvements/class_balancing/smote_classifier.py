"""
SMOTE-based ECG Classifier
Uses Synthetic Minority Over-sampling Technique to balance classes
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

from ecg_data_loader import load_dataset
from ecg_preprocessing import preprocess_ecg
from ecg_feature_extraction import extract_features_from_dataset


def plot_confusion_matrix(conf_matrix, labels, save_path='confusion_matrix_smote.png'):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix - SMOTE Balanced KNN', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def train_with_smote(train_features, train_labels, k=5):
    """
    Train KNN classifier with SMOTE oversampling
    
    Args:
        train_features: Training feature matrix
        train_labels: Training labels
        k: Number of neighbors for KNN
    
    Returns:
        Trained classifier
    """
    # Apply SMOTE to balance the dataset
    print("   Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42, k_neighbors=min(5, np.min(np.bincount(train_labels)) - 1))
    train_features_balanced, train_labels_balanced = smote.fit_resample(train_features, train_labels)
    
    print(f"   Original class distribution: {dict(zip(*np.unique(train_labels, return_counts=True)))}")
    print(f"   Balanced class distribution: {dict(zip(*np.unique(train_labels_balanced, return_counts=True)))}")
    
    # Train KNN classifier
    print(f"   Training KNN classifier (k={k})...")
    classifier = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    classifier.fit(train_features_balanced, train_labels_balanced)
    
    return classifier


def main():
    """Run SMOTE-based classification"""
    print("="*60)
    print("SMOTE-BASED ECG CLASSIFICATION")
    print("="*60)
    
    # Test on both datasets
    datasets = [
        ('Normal&LBBB', ['Normal', 'LBBB']),
        ('Normal&RBBB', ['Normal', 'RBBB'])
    ]
    
    results = {}
    
    for dataset_name, beat_types in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")
        
        # Load dataset
        print("\n1. Loading dataset...")
        data_folder = os.path.join(os.path.dirname(__file__), '..', '..', dataset_name)
        dataset, label_map = load_dataset(data_folder, beat_types=beat_types)
        
        print(f"   Training samples: {len(dataset['train']['signals'])}")
        print(f"   Test samples: {len(dataset['test']['signals'])}")
        
        # Preprocess training data
        print("\n2. Preprocessing training signals...")
        train_preprocessed = [preprocess_ecg(signal) for signal in dataset['train']['signals']]
        
        # Extract training features
        print("\n3. Extracting features from training data...")
        train_features = extract_features_from_dataset(train_preprocessed)
        print(f"   Feature vector size: {train_features.shape[1]}")
        
        # Train with SMOTE
        print("\n4. Training with SMOTE balancing...")
        classifier = train_with_smote(train_features, dataset['train']['labels'], k=5)
        
        # Preprocess test data
        print("\n5. Preprocessing test signals...")
        test_preprocessed = [preprocess_ecg(signal) for signal in dataset['test']['signals']]
        
        # Extract test features
        print("\n6. Extracting features from test data...")
        test_features = extract_features_from_dataset(test_preprocessed)
        
        # Make predictions
        print("\n7. Making predictions...")
        predictions = classifier.predict(test_features)
        
        # Calculate metrics
        print("\n8. Evaluating performance...")
        accuracy = accuracy_score(dataset['test']['labels'], predictions)
        precision = precision_score(dataset['test']['labels'], predictions, average='weighted', zero_division=0)
        recall = recall_score(dataset['test']['labels'], predictions, average='weighted', zero_division=0)
        f1 = f1_score(dataset['test']['labels'], predictions, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(dataset['test']['labels'], predictions)
        
        # Store results
        results[dataset_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix
        }
        
        # Print results
        print("\n" + "="*60)
        print(f"RESULTS - {dataset_name}")
        print("="*60)
        print(f"Accuracy:  {accuracy*100:.2f}%")
        print(f"Precision: {precision*100:.2f}%")
        print(f"Recall:    {recall*100:.2f}%")
        print(f"F1-Score:  {f1*100:.2f}%")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("="*60)
        
        # Plot confusion matrix
        idx_to_label = {v: k for k, v in label_map.items()}
        labels = [idx_to_label[i] for i in range(len(label_map))]
        save_path = os.path.join(os.path.dirname(__file__), 
                                 f'confusion_matrix_smote_{dataset_name.replace("&", "_")}.png')
        plot_confusion_matrix(conf_matrix, labels, save_path)
    
    # Save results to file
    print("\n" + "="*60)
    print("Saving results...")
    results_file = os.path.join(os.path.dirname(__file__), 'smote_results.txt')
    with open(results_file, 'w') as f:
        f.write("SMOTE-Based ECG Classification Results\n")
        f.write("="*60 + "\n\n")
        
        for dataset_name, metrics in results.items():
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Accuracy:  {metrics['accuracy']*100:.2f}%\n")
            f.write(f"Precision: {metrics['precision']*100:.2f}%\n")
            f.write(f"Recall:    {metrics['recall']*100:.2f}%\n")
            f.write(f"F1-Score:  {metrics['f1_score']*100:.2f}%\n")
            f.write(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}\n")
            f.write("="*60 + "\n\n")
    
    print(f"Results saved to {results_file}")
    print("\n" + "="*60)
    print("SMOTE Classification completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
