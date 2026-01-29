"""
ECG Signal Preprocessing Module
Implements preprocessing pipeline for ECG heartbeat signals
"""

import numpy as np
from scipy.signal import butter, filtfilt


def remove_mean(signal):
    """
    Remove the mean (DC component) from the signal
    
    Args:
        signal: Input ECG signal (numpy array)
    
    Returns:
        Signal with mean removed
    """
    return signal - np.mean(signal)


def butterworth_bandpass_filter(signal, lowcut=0.5, highcut=40, fs=360, order=5):
    """
    Apply Butterworth bandpass filter to remove noise
    
    Args:
        signal: Input ECG signal
        lowcut: Low cutoff frequency (Hz)
        highcut: High cutoff frequency (Hz)
        fs: Sampling frequency (Hz)
        order: Filter order
    
    Returns:
        Filtered signal
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Design Butterworth bandpass filter
    b, a = butter(order, [low, high], btype='band')
    
    # Apply zero-phase filtering
    filtered_signal = filtfilt(b, a, signal)
    
    return filtered_signal


def normalize_signal(signal):
    """
    Normalize signal to range [-1, 1]
    
    Args:
        signal: Input ECG signal
    
    Returns:
        Normalized signal
    """
    signal_min = np.min(signal)
    signal_max = np.max(signal)
    
    # Avoid division by zero
    if signal_max - signal_min == 0:
        return signal
    
    normalized = 2 * (signal - signal_min) / (signal_max - signal_min) - 1
    return normalized


def preprocess_ecg(signal, fs=360):
    """
    Complete preprocessing pipeline for ECG signals
    
    Steps:
    1. Remove mean
    2. Apply bandpass filter
    3. Normalize signal
    
    Args:
        signal: Raw ECG signal
        fs: Sampling frequency (default 360 Hz)
    
    Returns:
        Preprocessed ECG signal
    """
    # Step 1: Remove mean
    signal = remove_mean(signal)
    
    # Step 2: Apply bandpass filter
    signal = butterworth_bandpass_filter(signal, fs=fs)
    
    # Step 3: Normalize
    signal = normalize_signal(signal)
    
    return signal
