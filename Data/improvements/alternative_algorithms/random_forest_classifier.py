"""
Random Forest ECG Classifier
Uses ensemble of decision trees with feature importance analysis
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from ecg_data_loader import load_dataset
from ecg_preprocessing import preprocess_ecg
from ecg_feature_extraction import extract_features_from_dataset


def plot_confusion_matrix(conf_matrix, labels, save_path='confusion_matrix_rf.png'):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Purples', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix - Random Forest', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_feature_importance(classifier, n_features=20, save_path='feature_importance.png'):
    """Plot feature importance"""
    importances = classifier.feature_importances_
    indices = np.argsort(importances)[::-1][:n_features]
    
    plt.figure(figsize=(12, 8))
    plt.title('Top 20 Feature Importances', fontweight='bold', fontsize=14)
    plt.bar(range(n_features), importances[indices])
    plt.xticks(range(n_features), indices, rotation=45)
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved to {save_path}")
    plt.close()


def train_random_forest(train_features, train_labels, n_estimators=200):
    """
    Train Random Forest classifier
    
    Args:
        train_features: Training feature matrix
        train_labels: Training labels
        n_estimators: Number of trees
    
    Returns:
        Trained classifier
    """
    print(f"   Training Random Forest ({n_estimators} trees)...")
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    classifier.fit(train_features, train_labels)
    
    return classifier


def main():
    """Run Random Forest classification"""
    print("="*60)
    print("RANDOM FOREST ECG CLASSIFICATION")
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
        
        # Train Random Forest
        print("\n4. Training Random Forest classifier...")
        classifier = train_random_forest(train_features, dataset['train']['labels'])
        
        # Plot feature importance
        importance_path = os.path.join(os.path.dirname(__file__), 
                                      f'feature_importance_{dataset_name.replace("&", "_")}.png')
        plot_feature_importance(classifier, save_path=importance_path)
        
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
                                 f'confusion_matrix_rf_{dataset_name.replace("&", "_")}.png')
        plot_confusion_matrix(conf_matrix, labels, save_path)
    
    # Save results to file
    print("\n" + "="*60)
    print("Saving results...")
    results_file = os.path.join(os.path.dirname(__file__), 'random_forest_results.txt')
    with open(results_file, 'w') as f:
        f.write("Random Forest ECG Classification Results\n")
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
    print("Random Forest Classification completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
