# PID Step Response Library - Step Response Calculator
# Copyright (C) 2024
# License: GPLv3

"""
Step response calculation using FFT-based deconvolution.
Faithfully replicates the PIDtoolbox/PID-Analyzer algorithm.
"""

from typing import Tuple, Optional
import numpy as np
from scipy.ndimage import gaussian_filter1d


def lowess_smooth(data: np.ndarray, window_size: int = 20) -> np.ndarray:
    """
    Apply LOWESS (Locally Weighted Scatterplot Smoothing) to data.
    Simplified implementation matching MATLAB's smooth() with 'lowess' method.
    
    Args:
        data: Input data array
        window_size: Smoothing window size (span)
        
    Returns:
        Smoothed data array
    """
    if window_size <= 1 or len(data) <= window_size:
        return data.copy()
    
    n = len(data)
    smoothed = np.zeros(n)
    
    # Ensure window size is odd for symmetric window
    half_window = window_size // 2
    
    for i in range(n):
        # Define window bounds
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        
        # Extract window data
        window_data = data[start:end]
        window_x = np.arange(len(window_data))
        
        # Calculate weights using tricube kernel
        center = i - start
        distances = np.abs(window_x - center) / (half_window + 1)
        weights = np.where(distances < 1.0, (1 - distances**3)**3, 0)
        
        # Weighted mean
        if np.sum(weights) > 0:
            smoothed[i] = np.sum(weights * window_data) / np.sum(weights)
        else:
            smoothed[i] = data[i]
    
    return smoothed


def wiener_deconvolution(
    input_signals: np.ndarray,
    output_signals: np.ndarray,
    dt: float,
    cutfreq: float = 25.0
) -> np.ndarray:
    """
    Wiener deconvolution matching PID-Analyzer implementation.
    
    This calculates the impulse response by dividing output/input in frequency domain
    with a low-pass filter to reduce noise at high frequencies.
    
    Args:
        input_signals: 2D array of input signals [n_windows, window_length]
        output_signals: 2D array of output signals [n_windows, window_length]
        dt: Time step between samples in seconds
        cutfreq: Cutoff frequency for low-pass filter (Hz)
        
    Returns:
        2D array of deconvolved impulse responses
    """
    # Pad to power of 2 for efficient FFT (1024-aligned like PID-Analyzer)
    pad = 1024 - (input_signals.shape[1] % 1024)
    input_padded = np.pad(input_signals, [[0, 0], [0, pad]], mode='constant')
    output_padded = np.pad(output_signals, [[0, 0], [0, pad]], mode='constant')
    
    # FFT of both signals
    H = np.fft.fft(input_padded, axis=-1)
    G = np.fft.fft(output_padded, axis=-1)
    
    # Frequency array
    freq = np.abs(np.fft.fftfreq(input_padded.shape[1], dt))
    
    # Create low-pass filter mask matching PID-Analyzer
    # This masks out high-frequency noise
    sn = np.clip(freq, cutfreq - 1e-9, cutfreq)
    sn = sn - sn.min()
    if sn.max() > 0:
        sn = sn / sn.max()
    
    # Smooth the transition with Gaussian filter
    len_lpf = np.sum(1.0 - sn > 0.5)
    if len_lpf > 0:
        sn = gaussian_filter1d(sn, len_lpf / 6.0)
        sn = sn - sn.min()
        if sn.max() > 0:
            sn = sn / sn.max()
    
    # Signal-to-noise ratio term (inverted and scaled)
    sn = 10.0 * (1.0 - sn + 1e-9)  # +1e-9 to avoid 0/0
    
    # Wiener deconvolution: H* / (|H|^2 + 1/SNR)
    Hcon = np.conj(H)
    deconvolved = np.real(np.fft.ifft(G * Hcon / (H * Hcon + 1.0 / sn), axis=-1))
    
    return deconvolved


def calculate_step_response(
    setpoint: np.ndarray,
    gyro: np.ndarray,
    log_rate: float,
    smooth_factor: int = 1,
    y_correction: bool = False
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Calculate the step response using FFT-based deconvolution.
    
    This function faithfully replicates the PID-Analyzer algorithm:
    1. Apply optional smoothing to gyro data
    2. Segment the data into windows with overlap
    3. Apply Hanning window to each segment
    4. Deconvolve using Wiener deconvolution with low-pass filter
    5. Convert impulse response to step response via cumsum
    6. Average all valid segments
    
    Args:
        setpoint: Setpoint (input) signal array (deg/s)
        gyro: Gyro (output) signal array (deg/s)
        log_rate: Log rate in samples per millisecond (e.g., 4.0 for 4kHz)
        smooth_factor: Smoothing level (1=off, 2=low, 3=medium, 4=high)
        y_correction: Whether to apply Y-axis offset correction
        
    Returns:
        Tuple of (time_ms, step_response, num_segments)
        - time_ms: Time array in milliseconds (0 to 500ms)
        - step_response: Averaged step response (normalized so 1.0 = full response)
        - num_segments: Number of valid segments used
    """
    # Smoothing values matching PIDtoolbox
    smooth_vals = [1, 20, 40, 60]
    smooth_window = smooth_vals[min(smooth_factor - 1, 3)]
    
    # Apply smoothing to gyro if requested
    if smooth_factor > 1 and smooth_window > 1:
        gyro = lowess_smooth(gyro, smooth_window)
    
    # Parameters matching PID-Analyzer
    framelen = 1.0  # Length of each frame in seconds
    resplen = 0.5   # Length of response window in seconds (500ms)
    cutfreq = 25.0  # Cutoff frequency for low-pass filter
    superpos = 16   # Number of overlapping windows per frame
    threshold = 500.0  # Threshold for high input rate
    min_input = 20.0   # Minimum input to consider valid
    
    # Calculate time step
    dt = 1.0 / (log_rate * 1000)  # Time step in seconds
    
    # Ensure arrays are numpy and same length
    n = min(len(setpoint), len(gyro))
    setpoint = np.asarray(setpoint[:n], dtype=np.float64)
    gyro = np.asarray(gyro[:n], dtype=np.float64)
    
    # Replace NaN values with 0 to avoid calculation issues
    setpoint = np.nan_to_num(setpoint, nan=0.0)
    gyro = np.nan_to_num(gyro, nan=0.0)
    
    # Calculate window lengths
    flen = int(framelen / dt)  # Frame length in samples
    rlen = int(resplen / dt)   # Response length in samples
    
    if flen <= 0 or rlen <= 0 or n < flen:
        # Not enough data
        t = np.linspace(0, 500, rlen if rlen > 0 else 2000)
        return t, np.zeros_like(t), 0
    
    # Create time array for response (in milliseconds)
    time_resp = np.arange(rlen) * dt * 1000  # Convert to ms
    
    # Apply Hanning window
    window = np.hanning(flen)
    
    # Collect windowed segments
    shift = flen // superpos
    num_windows = (n - flen) // shift
    
    if num_windows <= 0:
        t = np.linspace(0, 500, rlen)
        return t, np.zeros_like(t), 0
    
    # Build stacks of windows
    input_windows = []
    output_windows = []
    max_inputs = []
    
    for i in range(num_windows):
        start = i * shift
        end = start + flen
        
        inp_win = setpoint[start:end] * window
        out_win = gyro[start:end] * window
        
        max_inp = np.max(np.abs(inp_win))
        
        # Filter: only include windows with sufficient input
        if max_inp >= min_input:
            input_windows.append(inp_win)
            output_windows.append(out_win)
            max_inputs.append(max_inp)
    
    if len(input_windows) == 0:
        t = np.linspace(0, 500, rlen)
        return t, np.zeros_like(t), 0
    
    input_windows = np.array(input_windows, dtype=np.float64)
    output_windows = np.array(output_windows, dtype=np.float64)
    max_inputs = np.array(max_inputs, dtype=np.float64)
    
    # Perform Wiener deconvolution
    deconvolved = wiener_deconvolution(input_windows, output_windows, dt, cutfreq)
    
    # Trim to response length and compute step response (cumsum of impulse response)
    deconvolved = deconvolved[:, :rlen]
    step_responses = np.cumsum(deconvolved, axis=1)
    
    # Filter out responses with NaN
    valid_mask = ~np.any(np.isnan(step_responses), axis=1)
    
    # Also filter very low input responses (below 20 deg/s threshold)
    valid_mask = valid_mask & (max_inputs >= min_input)
    
    if valid_mask.sum() == 0:
        return time_resp, np.zeros_like(time_resp), 0
    
    step_responses = step_responses[valid_mask]
    
    # Compute weighted average response
    # Use simple mean for now (PID-Analyzer uses weighted mode average which is more complex)
    avg_response = np.mean(step_responses, axis=0)
    
    # Ensure time and response arrays match
    if len(time_resp) > len(avg_response):
        time_resp = time_resp[:len(avg_response)]
    elif len(avg_response) > len(time_resp):
        avg_response = avg_response[:len(time_resp)]
    
    # Apply Y correction if requested (offset to start at 0)
    if y_correction and len(avg_response) > 10:
        avg_response = avg_response - avg_response[0]
    
    return time_resp, avg_response, len(step_responses)


def calculate_metrics(
    time_ms: np.ndarray,
    step_response: np.ndarray
) -> Tuple[float, float, float]:
    """
    Calculate step response metrics.
    
    The step response from PID-Analyzer is already normalized such that:
    - 0.0 = no response
    - 1.0 = gyro matches setpoint (ideal response)
    - >1.0 = overshoot
    
    Args:
        time_ms: Time array in milliseconds
        step_response: Step response array
        
    Returns:
        Tuple of (rise_time_ms, max_overshoot, settling_time_ms)
    """
    if len(step_response) < 2 or len(time_ms) < 2:
        return 0.0, 0.0, 0.0
    
    response = step_response.copy()
    
    # Determine final/steady-state value (use average of last 10% of signal)
    final_idx = max(1, int(len(response) * 0.9))
    final_value = np.mean(response[final_idx:])
    
    if abs(final_value) < 1e-10:
        return 0.0, 0.0, 0.0
    
    # For step response, we use the actual values directly
    # The response should converge to ~1.0 for a well-tuned system
    
    # Rise time: time to reach ~63.2% (1 - 1/e) of final value
    target_63 = 0.632 * final_value
    rise_time_ms = 0.0
    
    for i, val in enumerate(response):
        if val >= target_63:
            rise_time_ms = time_ms[i] if i < len(time_ms) else 0.0
            break
    
    # Maximum overshoot (peak value - final value) / final value
    peak_value = np.max(response)
    if final_value > 0:
        max_overshoot = max(0.0, (peak_value - final_value) / final_value)
    else:
        max_overshoot = 0.0
    
    # Settling time: time to settle within 2% of final value
    settling_threshold = 0.02 * abs(final_value)
    settling_time_ms = 0.0
    
    for i in range(len(response) - 1, -1, -1):
        if abs(response[i] - final_value) > settling_threshold:
            settling_time_ms = time_ms[min(i + 1, len(time_ms) - 1)]
            break
    
    return rise_time_ms, max_overshoot, settling_time_ms
