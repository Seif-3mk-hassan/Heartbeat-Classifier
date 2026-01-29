# ECG Heartbeat Classification System

This system classifies ECG heartbeats into three categories: Normal, RBBB, and LBBB using DSP and ML techniques.

## Files

- `ecg_preprocessing.py`: Signal preprocessing (mean removal, Butterworth filter, normalization)
- `ecg_feature_extraction.py`: Wavelet-based feature extraction using Daubechies wavelets
- `ecg_data_loader.py`: Data loading and parsing utilities
- `ecg_classifier.py`: Main classification pipeline with KNN classifier

## Requirements

```bash
pip install numpy scipy pywavelets scikit-learn matplotlib seaborn
```

## Usage

### Basic Usage

```python
python ecg_classifier.py
```

### Using Different Datasets

Edit the `data_folder` variable in `ecg_classifier.py`:

```python
data_folder = r"d:\Data\Data\Normal&LBBB"  # For Normal vs LBBB
# OR
data_folder = r"d:\Data\Data\Normal&RBBB"  # For Normal vs RBBB
```

### Hyperparameter Optimization

When prompted, choose 'y' to run grid search for optimal KNN parameters.

## Pipeline Steps

1. **Data Loading**: Load ECG signals from text files
2. **Preprocessing**:
   - Remove DC component
   - Butterworth bandpass filter (0.5-40 Hz)
   - Normalize to [-1, 1]
3. **Feature Extraction**: Daubechies wavelet decomposition (db1-db4)
4. **Classification**: K-Nearest Neighbors (KNN)
5. **Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

## Output

- Classification metrics printed to console
- `sample_ecg_signals.png`: Raw ECG samples
- `preprocessed_ecg_signals.png`: Preprocessed ECG samples
- `confusion_matrix.png`: Confusion matrix heatmap
