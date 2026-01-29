"""
ECG Feature Extraction Module
Implements wavelet decomposition for feature extraction
"""

import numpy as np
import pywt


def extract_wavelet_features(signal, wavelet_types=['db1', 'db2', 'db3', 'db4'], level=3):
    """
    Extract wavelet coefficients as features using Daubechies wavelets
    
    Args:
        signal: Preprocessed ECG signal
        wavelet_types: List of wavelet types to use
        level: Decomposition level
    
    Returns:
        Feature vector combining all wavelet coefficients
    """
    features = []
    
    for wavelet in wavelet_types:
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        
        # Flatten and concatenate all coefficients
        for coeff in coeffs:
            # Extract statistical features from each coefficient level
            features.append(np.mean(coeff))
            features.append(np.std(coeff))
            features.append(np.max(coeff))
            features.append(np.min(coeff))
    
    return np.array(features)


def extract_features_from_dataset(signals, wavelet_types=['db1', 'db2', 'db3', 'db4'], level=3):
    """
    Extract features from multiple signals
    
    Args:
        signals: List of ECG signals
        wavelet_types: List of wavelet types to use
        level: Decomposition level
    
    Returns:
        Feature matrix (n_samples x n_features)
    """
    feature_matrix = []
    
    for signal in signals:
        features = extract_wavelet_features(signal, wavelet_types, level)
        feature_matrix.append(features)
    
    return np.array(feature_matrix)
