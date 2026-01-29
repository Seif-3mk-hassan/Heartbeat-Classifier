"""
Time-Domain Feature Extraction Module
Extracts morphological and temporal features from ECG signals
"""

import numpy as np
from scipy import signal
from scipy.signal import find_peaks


def detect_r_peaks(ecg_signal, sampling_rate=250):
    """
    Detect R-peaks using a modified Pan-Tompkins approach
    
    Args:
        ecg_signal: Preprocessed ECG signal
        sampling_rate: Sampling rate in Hz
    
    Returns:
        Array of R-peak indices
    """
    # Differentiate
    diff_signal = np.diff(ecg_signal)
    
    # Square
    squared_signal = diff_signal ** 2
    
    # Moving average filter
    window_size = int(0.15 * sampling_rate)  # 150ms window
    moving_avg = np.convolve(squared_signal, np.ones(window_size)/window_size, mode='same')
    
    # Find peaks
    # Use adaptive threshold
    threshold = 0.5 * np.max(moving_avg)
    min_distance = int(0.2 * sampling_rate)  # Minimum 200ms between R-peaks
    
    peaks, _ = find_peaks(moving_avg, height=threshold, distance=min_distance)
    
    return peaks


def extract_qrs_features(ecg_signal, r_peaks, sampling_rate=250):
    """
    Extract QRS complex features
    
    Args:
        ecg_signal: ECG signal
        r_peaks: R-peak indices
        sampling_rate: Sampling rate
    
    Returns:
        Dictionary of QRS features
    """
    if len(r_peaks) == 0:
        return {
            'mean_qrs_duration': 0,
            'std_qrs_duration': 0,
            'mean_qrs_amplitude': 0,
            'std_qrs_amplitude': 0
        }
    
    qrs_durations = []
    qrs_amplitudes = []
    
    # Window around R-peak (±40ms for QRS)
    qrs_window = int(0.04 * sampling_rate)
    
    for r_idx in r_peaks:
        # Extract QRS segment
        start_idx = max(0, r_idx - qrs_window)
        end_idx = min(len(ecg_signal), r_idx + qrs_window)
        
        qrs_segment = ecg_signal[start_idx:end_idx]
        
        # Duration (simplified: window size)
        qrs_durations.append(len(qrs_segment) / sampling_rate * 1000)  # in ms
        
        # Amplitude (peak-to-peak)
        qrs_amplitudes.append(np.max(qrs_segment) - np.min(qrs_segment))
    
    return {
        'mean_qrs_duration': np.mean(qrs_durations),
        'std_qrs_duration': np.std(qrs_durations),
        'mean_qrs_amplitude': np.mean(qrs_amplitudes),
        'std_qrs_amplitude': np.std(qrs_amplitudes)
    }


def extract_rr_interval_features(r_peaks, sampling_rate=250):
    """
    Extract RR interval variability features
    
    Args:
        r_peaks: R-peak indices
        sampling_rate: Sampling rate
    
    Returns:
        Dictionary of RR interval features
    """
    if len(r_peaks) < 2:
        return {
            'mean_rr_interval': 0,
            'std_rr_interval': 0,
            'rmssd': 0,
            'sdnn': 0,
            'mean_heart_rate': 0
        }
    
    # Calculate RR intervals (in ms)
    rr_intervals = np.diff(r_peaks) / sampling_rate * 1000
    
    # SDNN: Standard deviation of NN intervals
    sdnn = np.std(rr_intervals)
    
    # RMSSD: Root mean square of successive differences
    successive_diffs = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(successive_diffs ** 2))
    
    # Heart rate
    mean_rr = np.mean(rr_intervals)
    mean_hr = 60000 / mean_rr if mean_rr > 0 else 0
    
    return {
        'mean_rr_interval': mean_rr,
        'std_rr_interval': np.std(rr_intervals),
        'rmssd': rmssd,
        'sdnn': sdnn,
        'mean_heart_rate': mean_hr
    }


def extract_morphology_features(ecg_signal, r_peaks, sampling_rate=250):
    """
    Extract signal morphology features
    
    Args:
        ecg_signal: ECG signal
        r_peaks: R-peak indices
        sampling_rate: Sampling rate
    
    Returns:
        Dictionary of morphology features
    """
    features = {
        'signal_mean': np.mean(ecg_signal),
        'signal_std': np.std(ecg_signal),
        'signal_variance': np.var(ecg_signal),
        'signal_skewness': calculate_skewness(ecg_signal),
        'signal_kurtosis': calculate_kurtosis(ecg_signal),
        'signal_energy': np.sum(ecg_signal ** 2),
        'signal_power': np.mean(ecg_signal ** 2),
        'zero_crossing_rate': calculate_zero_crossings(ecg_signal),
        'peak_to_peak': np.max(ecg_signal) - np.min(ecg_signal)
    }
    
    # Add R-peak statistics
    if len(r_peaks) > 0:
        r_amplitudes = ecg_signal[r_peaks]
        features['mean_r_amplitude'] = np.mean(r_amplitudes)
        features['std_r_amplitude'] = np.std(r_amplitudes)
        features['max_r_amplitude'] = np.max(r_amplitudes)
        features['min_r_amplitude'] = np.min(r_amplitudes)
    else:
        features['mean_r_amplitude'] = 0
        features['std_r_amplitude'] = 0
        features['max_r_amplitude'] = 0
        features['min_r_amplitude'] = 0
    
    return features


def calculate_skewness(signal):
    """Calculate skewness of signal"""
    mean = np.mean(signal)
    std = np.std(signal)
    if std == 0:
        return 0
    return np.mean(((signal - mean) / std) ** 3)


def calculate_kurtosis(signal):
    """Calculate kurtosis of signal"""
    mean = np.mean(signal)
    std = np.std(signal)
    if std == 0:
        return 0
    return np.mean(((signal - mean) / std) ** 4) - 3


def calculate_zero_crossings(signal):
    """Calculate zero crossing rate"""
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    return len(zero_crossings) / len(signal)


def extract_time_domain_features(ecg_signal, sampling_rate=250):
    """
    Extract all time-domain features from ECG signal
    
    Args:
        ecg_signal: Preprocessed ECG signal
        sampling_rate: Sampling rate in Hz
    
    Returns:
        Feature vector (numpy array)
    """
    # Detect R-peaks
    r_peaks = detect_r_peaks(ecg_signal, sampling_rate)
    
    # Extract all feature groups
    qrs_features = extract_qrs_features(ecg_signal, r_peaks, sampling_rate)
    rr_features = extract_rr_interval_features(r_peaks, sampling_rate)
    morph_features = extract_morphology_features(ecg_signal, r_peaks, sampling_rate)
    
    # Combine all features
    all_features = {**qrs_features, **rr_features, **morph_features}
    
    # Convert to array
    feature_vector = np.array(list(all_features.values()))
    
    return feature_vector


def extract_features_from_dataset(signals, sampling_rate=250):
    """
    Extract time-domain features from a dataset of signals
    
    Args:
        signals: List of ECG signals
        sampling_rate: Sampling rate
    
    Returns:
        Feature matrix (n_samples x n_features)
    """
    features = []
    for signal in signals:
        feature_vector = extract_time_domain_features(signal, sampling_rate)
        features.append(feature_vector)
    
    return np.array(features)
