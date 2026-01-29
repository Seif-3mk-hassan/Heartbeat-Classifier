"""
Support Vector Machine (SVM) ECG Classifier
Uses RBF kernel with grid search optimization
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from ecg_data_loader import load_dataset
from ecg_preprocessing import preprocess_ecg
from ecg_feature_extraction import extract_features_from_dataset


def plot_confusion_matrix(conf_matrix, labels, save_path='confusion_matrix_svm.png'):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix - SVM', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def train_svm(train_features, train_labels, optimize=True):
    """
    Train SVM classifier with RBF kernel
    
    Args:
        train_features: Training feature matrix
        train_labels: Training labels
        optimize: Whether to run grid search
    
    Returns:
        Trained classifier, scaler
    """
    # Scale features (important for SVM)
    print("   Scaling features...")
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    
    if optimize:
        # Grid search for optimal parameters
        print("   Running grid search for optimal parameters...")
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'class_weight': ['balanced', None]
        }
        
        svm = SVC(kernel='rbf', random_state=42)
        grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(train_features_scaled, train_labels)
        
        print(f"   Best parameters: {grid_search.best_params_}")
        print(f"   Best CV accuracy: {grid_search.best_score_:.4f}")
        
        classifier = grid_search.best_estimator_
    else:
        # Use default parameters with balanced class weights
        print("   Training SVM with default parameters...")
        classifier = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', random_state=42)
        classifier.fit(train_features_scaled, train_labels)
    
    return classifier, scaler


def main():
    """Run SVM classification"""
    print("="*60)
    print("SVM ECG CLASSIFICATION")
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
        
        # Train SVM
        print("\n4. Training SVM classifier...")
        classifier, scaler = train_svm(train_features, dataset['train']['labels'], optimize=False)
        
        # Preprocess test data
        print("\n5. Preprocessing test signals...")
        test_preprocessed = [preprocess_ecg(signal) for signal in dataset['test']['signals']]
        
        # Extract test features
        print("\n6. Extracting features from test data...")
        test_features = extract_features_from_dataset(test_preprocessed)
        test_features_scaled = scaler.transform(test_features)
        
        # Make predictions
        print("\n7. Making predictions...")
        predictions = classifier.predict(test_features_scaled)
        
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
                                 f'confusion_matrix_svm_{dataset_name.replace("&", "_")}.png')
        plot_confusion_matrix(conf_matrix, labels, save_path)
    
    # Save results to file
    print("\n" + "="*60)
    print("Saving results...")
    results_file = os.path.join(os.path.dirname(__file__), 'svm_results.txt')
    with open(results_file, 'w') as f:
        f.write("SVM ECG Classification Results\n")
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
    print("SVM Classification completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
