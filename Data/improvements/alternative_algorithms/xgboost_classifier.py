"""
XGBoost ECG Classifier
Uses gradient boosting with optimized hyperparameters
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from ecg_data_loader import load_dataset
from ecg_preprocessing import preprocess_ecg
from ecg_feature_extraction import extract_features_from_dataset


def plot_confusion_matrix(conf_matrix, labels, save_path='confusion_matrix_xgb.png'):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix - XGBoost', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def train_xgboost(train_features, train_labels):
    """
    Train XGBoost classifier
    
    Args:
        train_features: Training feature matrix
        train_labels: Training labels
    
    Returns:
        Trained classifier
    """
    # Calculate scale_pos_weight for class imbalance
    unique_labels, counts = np.unique(train_labels, return_counts=True)
    if len(unique_labels) == 2:
        scale_pos_weight = counts[0] / counts[1]
    else:
        scale_pos_weight = 1
    
    print(f"   Training XGBoost (scale_pos_weight={scale_pos_weight:.2f})...")
    
    classifier = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    classifier.fit(train_features, train_labels)
    
    return classifier


def main():
    """Run XGBoost classification"""
    print("="*60)
    print("XGBOOST ECG CLASSIFICATION")
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
        
        # Train XGBoost
        print("\n4. Training XGBoost classifier...")
        classifier = train_xgboost(train_features, dataset['train']['labels'])
        
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
                                 f'confusion_matrix_xgb_{dataset_name.replace("&", "_")}.png')
        plot_confusion_matrix(conf_matrix, labels, save_path)
    
    # Save results to file
    print("\n" + "="*60)
    print("Saving results...")
    results_file = os.path.join(os.path.dirname(__file__), 'xgboost_results.txt')
    with open(results_file, 'w') as f:
        f.write("XGBoost ECG Classification Results\n")
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
    print("XGBoost Classification completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
