"""
Quick test script for ECG classifier without user input
"""

import numpy as np
from ecg_data_loader import load_dataset
from ecg_preprocessing import preprocess_ecg
from ecg_feature_extraction import extract_features_from_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(conf_matrix, labels, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def main():
    """Run ECG classification without user prompts"""
    print("="*60)
    print("ECG HEARTBEAT CLASSIFICATION SYSTEM")
    print("="*60)
    
    # Load dataset
    print("\n1. Loading dataset...")
    data_folder = r"d:\Data\Data\Normal&LBBB"
    dataset, label_map = load_dataset(data_folder, beat_types=['Normal', 'LBBB'])
    
    print(f"   Training samples: {len(dataset['train']['signals'])}")
    print(f"   Test samples: {len(dataset['test']['signals'])}")
    print(f"   Classes: {list(label_map.keys())}")
    
    # Preprocess training data
    print("\n2. Preprocessing training signals...")
    train_preprocessed = []
    for signal in dataset['train']['signals']:
        train_preprocessed.append(preprocess_ecg(signal))
    
    # Extract training features
    print("\n3. Extracting features from training data...")
    train_features = extract_features_from_dataset(train_preprocessed)
    print(f"   Feature vector size: {train_features.shape[1]}")
    
    # Train classifier
    print("\n4. Training KNN classifier (k=5)...")
    classifier = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    classifier.fit(train_features, dataset['train']['labels'])
    
    # Preprocess test data
    print("\n5. Preprocessing test signals...")
    test_preprocessed = []
    for signal in dataset['test']['signals']:
        test_preprocessed.append(preprocess_ecg(signal))
    
    # Extract test features
    print("\n6. Extracting features from test data...")
    test_features = extract_features_from_dataset(test_preprocessed)
    
    # Make predictions
    print("\n7. Making predictions...")
    predictions = classifier.predict(test_features)
    
    # Calculate metrics
    print("\n8. Evaluating performance...")
    accuracy = accuracy_score(dataset['test']['labels'], predictions)
    precision = precision_score(dataset['test']['labels'], predictions, average='weighted')
    recall = recall_score(dataset['test']['labels'], predictions, average='weighted')
    f1 = f1_score(dataset['test']['labels'], predictions, average='weighted')
    conf_matrix = confusion_matrix(dataset['test']['labels'], predictions)
    
    # Print results
    print("\n" + "="*60)
    print("CLASSIFICATION RESULTS")
    print("="*60)
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    print("="*60)
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # Plot confusion matrix
    print("\n9. Generating confusion matrix plot...")
    idx_to_label = {v: k for k, v in label_map.items()}
    labels = [idx_to_label[i] for i in range(len(label_map))]
    plot_confusion_matrix(conf_matrix, labels, 'confusion_matrix_LBBB.png')
    
    print("\n" + "="*60)
    print("Classification completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
