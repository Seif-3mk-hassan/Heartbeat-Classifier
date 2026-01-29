"""
Class Balancing Comparison Test
Compares baseline KNN vs SMOTE vs Weighted KNN
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

from ecg_data_loader import load_dataset
from ecg_preprocessing import preprocess_ecg
from ecg_feature_extraction import extract_features_from_dataset


def compare_methods(dataset_name, beat_types):
    """Compare all class balancing methods"""
    
    print(f"\n{'='*60}")
    print(f"Comparing Methods on {dataset_name}")
    print(f"{'='*60}")
    
    # Load dataset
    data_folder = os.path.join(os.path.dirname(__file__), '..', '..', dataset_name)
    dataset, label_map = load_dataset(data_folder, beat_types=beat_types)
    
    # Preprocess
    print("Preprocessing...")
    train_preprocessed = [preprocess_ecg(signal) for signal in dataset['train']['signals']]
    test_preprocessed = [preprocess_ecg(signal) for signal in dataset['test']['signals']]
    
    # Extract features
    print("Extracting features...")
    train_features = extract_features_from_dataset(train_preprocessed)
    test_features = extract_features_from_dataset(test_preprocessed)
    
    results = {}
    
    # Method 1: Baseline KNN
    print("\n1. Testing Baseline KNN...")
    clf_baseline = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    clf_baseline.fit(train_features, dataset['train']['labels'])
    pred_baseline = clf_baseline.predict(test_features)
    
    results['Baseline KNN'] = {
        'accuracy': accuracy_score(dataset['test']['labels'], pred_baseline),
        'precision': precision_score(dataset['test']['labels'], pred_baseline, average='weighted', zero_division=0),
        'recall': recall_score(dataset['test']['labels'], pred_baseline, average='weighted', zero_division=0),
        'f1': f1_score(dataset['test']['labels'], pred_baseline, average='weighted', zero_division=0)
    }
    
    # Method 2: SMOTE
    print("2. Testing SMOTE...")
    smote = SMOTE(random_state=42, k_neighbors=min(5, np.min(np.bincount(dataset['train']['labels'])) - 1))
    train_features_smote, train_labels_smote = smote.fit_resample(train_features, dataset['train']['labels'])
    
    clf_smote = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    clf_smote.fit(train_features_smote, train_labels_smote)
    pred_smote = clf_smote.predict(test_features)
    
    results['SMOTE'] = {
        'accuracy': accuracy_score(dataset['test']['labels'], pred_smote),
        'precision': precision_score(dataset['test']['labels'], pred_smote, average='weighted', zero_division=0),
        'recall': recall_score(dataset['test']['labels'], pred_smote, average='weighted', zero_division=0),
        'f1': f1_score(dataset['test']['labels'], pred_smote, average='weighted', zero_division=0)
    }
    
    # Method 3: Weighted KNN
    print("3. Testing Weighted KNN...")
    clf_weighted = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='euclidean')
    clf_weighted.fit(train_features, dataset['train']['labels'])
    pred_weighted = clf_weighted.predict(test_features)
    
    results['Weighted KNN'] = {
        'accuracy': accuracy_score(dataset['test']['labels'], pred_weighted),
        'precision': precision_score(dataset['test']['labels'], pred_weighted, average='weighted', zero_division=0),
        'recall': recall_score(dataset['test']['labels'], pred_weighted, average='weighted', zero_division=0),
        'f1': f1_score(dataset['test']['labels'], pred_weighted, average='weighted', zero_division=0)
    }
    
    return results


def plot_comparison(all_results, save_path='comparison_chart.png'):
    """Plot comparison bar chart"""
    
    methods = list(all_results['Normal&LBBB'].keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    datasets = list(all_results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    x = np.arange(len(methods))
    width = 0.35
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        for i, dataset in enumerate(datasets):
            values = [all_results[dataset][method][metric] * 100 for method in methods]
            ax.bar(x + i * width, values, width, label=dataset)
        
        ax.set_ylabel(f'{metric.capitalize()} (%)', fontweight='bold')
        ax.set_title(f'{metric.upper()} Comparison', fontweight='bold', fontsize=14)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(methods, rotation=15, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison chart saved to {save_path}")
    plt.close()


def main():
    """Run comparison test"""
    print("="*60)
    print("CLASS BALANCING METHODS COMPARISON")
    print("="*60)
    
    # Test on both datasets
    datasets = [
        ('Normal&LBBB', ['Normal', 'LBBB']),
        ('Normal&RBBB', ['Normal', 'RBBB'])
    ]
    
    all_results = {}
    
    for dataset_name, beat_types in datasets:
        all_results[dataset_name] = compare_methods(dataset_name, beat_types)
    
    # Print summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name}:")
        print("-" * 60)
        print(f"{'Method':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
        print("-" * 60)
        
        for method, metrics in results.items():
            print(f"{method:<20} {metrics['accuracy']*100:>9.2f}% {metrics['precision']*100:>9.2f}% "
                  f"{metrics['recall']*100:>9.2f}% {metrics['f1']*100:>9.2f}%")
    
    # Plot comparison
    save_path = os.path.join(os.path.dirname(__file__), 'comparison_chart.png')
    plot_comparison(all_results, save_path)
    
    # Save results to file
    results_file = os.path.join(os.path.dirname(__file__), 'comparison_results.txt')
    with open(results_file, 'w') as f:
        f.write("Class Balancing Methods Comparison\n")
        f.write("="*60 + "\n\n")
        
        for dataset_name, results in all_results.items():
            f.write(f"{dataset_name}:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Method':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}\n")
            f.write("-" * 60 + "\n")
            
            for method, metrics in results.items():
                f.write(f"{method:<20} {metrics['accuracy']*100:>9.2f}% {metrics['precision']*100:>9.2f}% "
                       f"{metrics['recall']*100:>9.2f}% {metrics['f1']*100:>9.2f}%\n")
            f.write("\n")
    
    print(f"\nResults saved to {results_file}")
    print("\n" + "="*60)
    print("Comparison completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
