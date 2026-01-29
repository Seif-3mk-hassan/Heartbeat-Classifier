"""
Algorithm Comparison Script
Compares all classification algorithms on both datasets
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

from ecg_data_loader import load_dataset
from ecg_preprocessing import preprocess_ecg
from ecg_feature_extraction import extract_features_from_dataset


def test_all_algorithms(dataset_name, beat_types):
    """Test all algorithms on a dataset"""
    
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # Load and preprocess
    data_folder = os.path.join(os.path.dirname(__file__), '..', '..', dataset_name)
    dataset, label_map = load_dataset(data_folder, beat_types=beat_types)
    
    print("Preprocessing and extracting features...")
    train_preprocessed = [preprocess_ecg(signal) for signal in dataset['train']['signals']]
    test_preprocessed = [preprocess_ecg(signal) for signal in dataset['test']['signals']]
    
    train_features = extract_features_from_dataset(train_preprocessed)
    test_features = extract_features_from_dataset(test_preprocessed)
    
    results = {}
    
    # 1. KNN (Baseline)
    print("Testing KNN...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_features, dataset['train']['labels'])
    pred = knn.predict(test_features)
    results['KNN'] = calculate_metrics(dataset['test']['labels'], pred)
    
    # 2. SVM
    print("Testing SVM...")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    svm = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', random_state=42)
    svm.fit(train_scaled, dataset['train']['labels'])
    pred = svm.predict(test_scaled)
    results['SVM'] = calculate_metrics(dataset['test']['labels'], pred)
    
    # 3. Random Forest
    print("Testing Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', 
                                random_state=42, n_jobs=-1)
    rf.fit(train_features, dataset['train']['labels'])
    pred = rf.predict(test_features)
    results['Random Forest'] = calculate_metrics(dataset['test']['labels'], pred)
    
    # 4. XGBoost
    print("Testing XGBoost...")
    unique_labels, counts = np.unique(dataset['train']['labels'], return_counts=True)
    scale_pos_weight = counts[0] / counts[1] if len(unique_labels) == 2 else 1
    xgboost = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                               scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1)
    xgboost.fit(train_features, dataset['train']['labels'])
    pred = xgboost.predict(test_features)
    results['XGBoost'] = calculate_metrics(dataset['test']['labels'], pred)
    
    return results


def calculate_metrics(y_true, y_pred):
    """Calculate all metrics"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }


def plot_comparison(all_results, save_path='algorithm_comparison.png'):
    """Create comprehensive comparison visualization"""
    
    algorithms = list(all_results['Normal&LBBB'].keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    datasets = list(all_results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    x = np.arange(len(algorithms))
    width = 0.35
    
    colors = ['#3498db', '#e74c3c']  # Blue for LBBB, Red for RBBB
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        for i, dataset in enumerate(datasets):
            values = [all_results[dataset][alg][metric] * 100 for alg in algorithms]
            bars = ax.bar(x + i * width, values, width, label=dataset, color=colors[i], alpha=0.8)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel(f'{metric.capitalize()} (%)', fontweight='bold', fontsize=12)
        ax.set_title(f'{metric.upper()} Comparison', fontweight='bold', fontsize=14)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(algorithms, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 105])
    
    plt.suptitle('ECG Classification Algorithm Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison chart saved to {save_path}")
    plt.close()


def main():
    """Run comprehensive algorithm comparison"""
    print("="*60)
    print("COMPREHENSIVE ALGORITHM COMPARISON")
    print("="*60)
    
    datasets = [
        ('Normal&LBBB', ['Normal', 'LBBB']),
        ('Normal&RBBB', ['Normal', 'RBBB'])
    ]
    
    all_results = {}
    
    for dataset_name, beat_types in datasets:
        all_results[dataset_name] = test_all_algorithms(dataset_name, beat_types)
    
    # Print comparison table
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name}:")
        print("-" * 80)
        print(f"{'Algorithm':<15} {'Accuracy':>12} {'Precision':>12} {'Recall':>12} {'F1-Score':>12}")
        print("-" * 80)
        
        for alg, metrics in results.items():
            print(f"{alg:<15} {metrics['accuracy']*100:>11.2f}% {metrics['precision']*100:>11.2f}% "
                  f"{metrics['recall']*100:>11.2f}% {metrics['f1']*100:>11.2f}%")
        
        # Find best algorithm
        best_alg = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\n  Best Algorithm: {best_alg[0]} ({best_alg[1]['accuracy']*100:.2f}% accuracy)")
    
    # Plot comparison
    save_path = os.path.join(os.path.dirname(__file__), 'algorithm_comparison.png')
    plot_comparison(all_results, save_path)
    
    # Save to file
    results_file = os.path.join(os.path.dirname(__file__), 'comparison_summary.txt')
    with open(results_file, 'w') as f:
        f.write("ECG Classification - Algorithm Comparison\n")
        f.write("="*80 + "\n\n")
        
        for dataset_name, results in all_results.items():
            f.write(f"{dataset_name}:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Algorithm':<15} {'Accuracy':>12} {'Precision':>12} {'Recall':>12} {'F1-Score':>12}\n")
            f.write("-" * 80 + "\n")
            
            for alg, metrics in results.items():
                f.write(f"{alg:<15} {metrics['accuracy']*100:>11.2f}% {metrics['precision']*100:>11.2f}% "
                       f"{metrics['recall']*100:>11.2f}% {metrics['f1']*100:>11.2f}%\n")
            
            best_alg = max(results.items(), key=lambda x: x[1]['accuracy'])
            f.write(f"\nBest Algorithm: {best_alg[0]} ({best_alg[1]['accuracy']*100:.2f}% accuracy)\n\n")
    
    print(f"\nResults saved to {results_file}")
    print("\n" + "="*60)
    print("Algorithm comparison completed!")
    print("="*60)


if __name__ == "__main__":
    main()
