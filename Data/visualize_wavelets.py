"""
Wavelet Decomposition Visualization
Visualize wavelet decomposition results for ECG signals
"""

import numpy as np
import matplotlib.pyplot as plt
import pywt
from ecg_data_loader import load_ecg_data
from ecg_preprocessing import preprocess_ecg


def plot_wavelet_decomposition(signal, wavelet='db4', level=3, save_path='wavelet_decomposition.png'):
    """
    Plot wavelet decomposition of ECG signal
    
    Args:
        signal: ECG signal to decompose
        wavelet: Wavelet type (default: db4)
        level: Decomposition level
        save_path: Path to save plot
    """
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Create figure
    fig, axes = plt.subplots(level + 2, 1, figsize=(14, 10))
    
    # Plot original signal
    axes[0].plot(signal, 'b', linewidth=0.8)
    axes[0].set_title('Original ECG Signal', fontweight='bold')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # Plot approximation coefficients (cA)
    axes[1].plot(coeffs[0], 'g', linewidth=0.8)
    axes[1].set_title(f'Approximation Coefficients (cA{level})', fontweight='bold')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    
    # Plot detail coefficients (cD)
    for i in range(1, level + 1):
        axes[i + 1].plot(coeffs[i], 'r', linewidth=0.8)
        axes[i + 1].set_title(f'Detail Coefficients (cD{level - i + 1})', fontweight='bold')
        axes[i + 1].set_ylabel('Amplitude')
        axes[i + 1].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Sample Point')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Wavelet decomposition saved to {save_path}")
    plt.close()


def compare_wavelet_types(signal, wavelets=['db1', 'db2', 'db3', 'db4'], 
                          level=3, save_path='wavelet_comparison.png'):
    """
    Compare different wavelet types on the same signal
    
    Args:
        signal: ECG signal
        wavelets: List of wavelet types to compare
        level: Decomposition level
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(len(wavelets) + 1, 1, figsize=(14, 10))
    
    # Plot original signal
    axes[0].plot(signal, 'b', linewidth=0.8)
    axes[0].set_title('Original ECG Signal', fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # Plot each wavelet decomposition
    for i, wavelet in enumerate(wavelets):
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        # Reconstruct signal from coefficients
        reconstructed = pywt.waverec(coeffs, wavelet)
        
        # Handle length mismatch
        if len(reconstructed) > len(signal):
            reconstructed = reconstructed[:len(signal)]
        
        axes[i + 1].plot(reconstructed, linewidth=0.8)
        axes[i + 1].set_title(f'Reconstructed with {wavelet.upper()}', fontweight='bold', fontsize=11)
        axes[i + 1].set_ylabel('Amplitude')
        axes[i + 1].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Sample Point')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Wavelet comparison saved to {save_path}")
    plt.close()


def visualize_multiple_signals(filepath, num_signals=3, wavelet='db4', level=3):
    """
    Visualize wavelet decomposition for multiple ECG signals
    
    Args:
        filepath: Path to ECG data file
        num_signals: Number of signals to visualize
        wavelet: Wavelet type
        level: Decomposition level
    """
    # Load signals
    signals = load_ecg_data(filepath)
    
    # Limit number of signals
    num_signals = min(num_signals, len(signals))
    
    for i in range(num_signals):
        # Preprocess signal
        preprocessed = preprocess_ecg(signals[i])
        
        # Plot wavelet decomposition
        save_path = f'wavelet_decomposition_signal_{i+1}.png'
        plot_wavelet_decomposition(preprocessed, wavelet, level, save_path)
    
    # Compare wavelets on first signal
    preprocessed = preprocess_ecg(signals[0])
    compare_wavelet_types(preprocessed, save_path='wavelet_comparison.png')


if __name__ == "__main__":
    print("="*60)
    print("WAVELET DECOMPOSITION VISUALIZATION")
    print("="*60)
    
    # Visualize Normal heartbeats
    print("\nVisualizing Normal heartbeats...")
    normal_file = r"d:\Data\Data\Normal&LBBB\Normal_Train.txt"
    visualize_multiple_signals(normal_file, num_signals=2, wavelet='db4', level=3)
    
    # Visualize LBBB heartbeats
    print("\nVisualizing LBBB heartbeats...")
    lbbb_file = r"d:\Data\Data\Normal&LBBB\LBBB_Train.txt"
    visualize_multiple_signals(lbbb_file, num_signals=2, wavelet='db4', level=3)
    
    print("\n" + "="*60)
    print("Wavelet visualization completed!")
    print("="*60)
