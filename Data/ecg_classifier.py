"""
ECG Heartbeat Classification System
Main pipeline for training and evaluating KNN classifier
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import os

# Import custom modules
from ecg_data_loader import load_dataset
from ecg_preprocessing import preprocess_ecg
from ecg_feature_extraction import extract_features_from_dataset


class ECGClassifier:
    """
    ECG Heartbeat Classifier using KNN with wavelet features
    """
    
    def __init__(self, k=5, metric='euclidean'):
        """
        Initialize classifier
        
        Args:
            k: Number of neighbors for KNN
            metric: Distance metric to use
        """
        self.k = k
        self.metric = metric
        self.classifier = KNeighborsClassifier(n_neighbors=k, metric=metric)
        self.label_map = None
        
    def preprocess_signals(self, signals):
        """
        Preprocess all signals in the dataset
        
        Args:
            signals: List of raw ECG signals
        
        Returns:
            List of preprocessed signals
        """
        preprocessed = []
        for signal in signals:
            preprocessed.append(preprocess_ecg(signal))
        return preprocessed
    
    def train(self, train_signals, train_labels):
        """
        Train the KNN classifier
        
        Args:
            train_signals: Training ECG signals
            train_labels: Training labels
        """
        print("Preprocessing training signals...")
        train_preprocessed = self.preprocess_signals(train_signals)
        
        print("Extracting features from training data...")
        train_features = extract_features_from_dataset(train_preprocessed)
        
        print(f"Training KNN classifier (k={self.k}, metric={self.metric})...")
        self.classifier.fit(train_features, train_labels)
        
        print("Training completed!")
        
    def predict(self, test_signals):
        """
        Predict labels for test signals
        
        Args:
            test_signals: Test ECG signals
        
        Returns:
            Predicted labels
        """
        print("Preprocessing test signals...")
        test_preprocessed = self.preprocess_signals(test_signals)
        
        print("Extracting features from test data...")
        test_features = extract_features_from_dataset(test_preprocessed)
        
        print("Making predictions...")
        predictions = self.classifier.predict(test_features)
        
        return predictions
    
    def evaluate(self, test_signals, test_labels, label_map):
        """
        Evaluate classifier performance
        
        Args:
            test_signals: Test ECG signals
            test_labels: True labels
            label_map: Mapping from label names to indices
        
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(test_signals)
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions, average='weighted')
        recall = recall_score(test_labels, predictions, average='weighted')
        f1 = f1_score(test_labels, predictions, average='weighted')
        conf_matrix = confusion_matrix(test_labels, predictions)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix
        }
        
        return metrics, predictions
    
    def plot_confusion_matrix(self, conf_matrix, label_map, save_path='confusion_matrix.png'):
        """
        Plot and save confusion matrix
        
        Args:
            conf_matrix: Confusion matrix
            label_map: Mapping from label names to indices
            save_path: Path to save the plot
        """
        # Create reverse mapping
        idx_to_label = {v: k for k, v in label_map.items()}
        labels = [idx_to_label[i] for i in range(len(label_map))]
        
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
        
    def plot_sample_signals(self, signals, labels, label_map, num_samples=3, save_path='sample_signals.png'):
        """
        Plot sample ECG signals for each class
        
        Args:
            signals: ECG signals
            labels: Signal labels
            label_map: Mapping from label names to indices
            num_samples: Number of samples per class to plot
            save_path: Path to save the plot
        """
        idx_to_label = {v: k for k, v in label_map.items()}
        unique_labels = sorted(np.unique(labels))
        
        fig, axes = plt.subplots(len(unique_labels), num_samples, 
                                figsize=(15, len(unique_labels) * 3))
        
        if len(unique_labels) == 1:
            axes = axes.reshape(1, -1)
        
        for i, label in enumerate(unique_labels):
            # Find indices of this class
            class_indices = np.where(labels == label)[0]
            sample_indices = np.random.choice(class_indices, 
                                            min(num_samples, len(class_indices)), 
                                            replace=False)
            
            for j, idx in enumerate(sample_indices):
                axes[i, j].plot(signals[idx], linewidth=0.8)
                axes[i, j].set_title(f'{idx_to_label[label]} - Sample {j+1}')
                axes[i, j].set_xlabel('Sample Point')
                axes[i, j].set_ylabel('Amplitude')
                axes[i, j].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample signals saved to {save_path}")
        plt.close()


def optimize_hyperparameters(train_signals, train_labels):
    """
    Optimize KNN hyperparameters using grid search
    
    Args:
        train_signals: Training signals (preprocessed)
        train_labels: Training labels
    
    Returns:
        Best parameters
    """
    print("Extracting features for hyperparameter optimization...")
    train_features = extract_features_from_dataset(train_signals)
    
    # Define parameter grid
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    
    print("Running grid search...")
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(train_features, train_labels)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
    
    return grid_search.best_params_


def main():
    """
    Main function to run the ECG classification pipeline
    """
    print("="*60)
    print("ECG HEARTBEAT CLASSIFICATION SYSTEM")
    print("="*60)
    
    # Load dataset
    print("\n1. Loading dataset...")
    data_folder = r"d:\Data\Data\Normal&LBBB"  # Change this path as needed
    dataset, label_map = load_dataset(data_folder, beat_types=['Normal', 'LBBB'])
    
    print(f"Training samples: {len(dataset['train']['signals'])}")
    print(f"Test samples: {len(dataset['test']['signals'])}")
    print(f"Classes: {list(label_map.keys())}")
    
    # Visualize sample signals
    print("\n2. Visualizing sample ECG signals...")
    classifier = ECGClassifier()
    classifier.plot_sample_signals(
        dataset['train']['signals'][:300],  # First 300 for variety
        dataset['train']['labels'][:300],
        label_map,
        num_samples=3,
        save_path='sample_ecg_signals.png'
    )
    
    # Optional: Optimize hyperparameters
    optimize = input("\nOptimize hyperparameters? (y/n): ").lower() == 'y'
    
    if optimize:
        print("\n3. Optimizing hyperparameters...")
        train_preprocessed = classifier.preprocess_signals(dataset['train']['signals'])
        best_params = optimize_hyperparameters(train_preprocessed, dataset['train']['labels'])
        classifier = ECGClassifier(k=best_params['n_neighbors'], metric=best_params['metric'])
    else:
        print("\n3. Using default parameters (k=5, metric='euclidean')")
        classifier = ECGClassifier(k=5, metric='euclidean')
    
    # Train classifier
    print("\n4. Training classifier...")
    classifier.train(dataset['train']['signals'], dataset['train']['labels'])
    
    # Evaluate on test set
    print("\n5. Evaluating on test set...")
    metrics, predictions = classifier.evaluate(
        dataset['test']['signals'],
        dataset['test']['labels'],
        label_map
    )
    
    # Print results
    print("\n" + "="*60)
    print("CLASSIFICATION RESULTS")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall:    {metrics['recall']*100:.2f}%")
    print(f"F1-Score:  {metrics['f1_score']*100:.2f}%")
    print("="*60)
    
    # Plot confusion matrix
    print("\n6. Generating visualizations...")
    classifier.plot_confusion_matrix(
        metrics['confusion_matrix'],
        label_map,
        save_path='confusion_matrix.png'
    )
    
    # Plot preprocessed samples
    print("\n7. Visualizing preprocessed signals...")
    preprocessed_samples = classifier.preprocess_signals(dataset['test']['signals'][:300])
    classifier.plot_sample_signals(
        preprocessed_samples,
        dataset['test']['labels'][:300],
        label_map,
        num_samples=3,
        save_path='preprocessed_ecg_signals.png'
    )
    
    print("\n" + "="*60)
    print("Classification completed successfully!")
    print("Check the generated PNG files for visualizations.")
    print("="*60)


if __name__ == "__main__":
    main()
