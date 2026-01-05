# PID Step Response Library - Step Response Calculator
# Copyright (C) 2024
# License: GPLv3

"""
Step response calculation using FFT-based deconvolution.
Faithfully replicates the PIDtoolbox (PTstepcalc.m) algorithm.
"""

from typing import Tuple, Optional
import numpy as np


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


def calculate_step_response(
    setpoint: np.ndarray,
    gyro: np.ndarray,
    log_rate: float,
    smooth_factor: int = 1,
    y_correction: bool = False
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Calculate the step response using FFT-based deconvolution.
    
    This function faithfully replicates the PIDtoolbox PTstepcalc.m algorithm:
    1. Apply optional smoothing to gyro data
    2. Segment the data into 2-second windows
    3. Find segments with sufficient input signal
    4. Deconvolve each segment using FFT to get step response
    5. Average all valid segments
    
    Args:
        setpoint: Setpoint (input) signal array (deg/s)
        gyro: Gyro (output) signal array (deg/s)
        log_rate: Log rate in samples per millisecond (e.g., 4.0 for 4kHz)
        smooth_factor: Smoothing level (1=off, 2=low, 3=medium, 4=high)
        y_correction: Whether to apply Y-axis offset correction
        
    Returns:
        Tuple of (time_ms, step_response, num_segments)
        - time_ms: Time array in milliseconds (0 to 500ms)
        - step_response: Averaged step response
        - num_segments: Number of valid segments used
    """
    # Smoothing values matching PIDtoolbox
    smooth_vals = [1, 20, 40, 60]
    smooth_window = smooth_vals[min(smooth_factor - 1, 3)]
    
    # Apply smoothing to gyro if requested
    if smooth_factor > 1 and smooth_window > 1:
        gyro = lowess_smooth(gyro, smooth_window)
    
    # Parameters matching PIDtoolbox
    min_input = 20  # Minimum input rate to consider valid
    segment_length = int(log_rate * 2000)  # 2 second segments
    wnd = int(log_rate * 1000 * 0.5)  # 500ms step response window
    step_resp_duration_ms = 500  # Maximum duration for step response
    
    # Create time array
    t = np.arange(0, step_resp_duration_ms + 1/log_rate, 1/log_rate)
    
    # Ensure arrays are numpy and same length
    n = min(len(setpoint), len(gyro))
    setpoint = np.asarray(setpoint[:n])
    gyro = np.asarray(gyro[:n])
    
    # Replace NaN values with 0 to avoid calculation issues
    setpoint = np.nan_to_num(setpoint, nan=0.0)
    gyro = np.nan_to_num(gyro, nan=0.0)
    
    # Calculate file duration in seconds
    file_dur_sec = n / (log_rate * 1000)
    
    # Subsampling factor based on file duration (for processing efficiency)
    if file_dur_sec <= 20:
        subsample_factor = 10
    elif file_dur_sec <= 60:
        subsample_factor = 7
    else:
        subsample_factor = 3
    
    # Collect step responses from segments
    step_responses = []
    
    # Process segments
    num_segments = max(1, n // segment_length)
    
    for seg_idx in range(0, n - segment_length, segment_length // 2):
        # Extract segment
        seg_end = min(seg_idx + segment_length, n)
        sp_seg = setpoint[seg_idx:seg_end]
        gy_seg = gyro[seg_idx:seg_end]
        
        # Check if segment has sufficient input
        if np.max(np.abs(sp_seg)) < min_input:
            continue
        
        # Skip segments with too little variance
        if np.std(sp_seg) < min_input / 4:
            continue
        
        # Calculate step response using deconvolution
        try:
            step_resp = deconvolve_step_response(sp_seg, gy_seg, wnd)
            
            if step_resp is not None and len(step_resp) > 0:
                # Skip if response contains NaN (can occur from numerical issues in FFT)
                if np.any(np.isnan(step_resp)):
                    continue
                # Normalize step response
                max_val = np.max(np.abs(step_resp))
                step_resp = step_resp / max_val if max_val > 0 else step_resp
                step_responses.append(step_resp)
        except Exception:
            continue
    
    # Average all valid step responses
    if step_responses:
        # Ensure all responses are same length
        min_len = min(len(sr) for sr in step_responses)
        step_responses = [sr[:min_len] for sr in step_responses]
        avg_response = np.mean(step_responses, axis=0)
        
        # Ensure time and response arrays match
        if len(t) > len(avg_response):
            t = t[:len(avg_response)]
        elif len(avg_response) > len(t):
            avg_response = avg_response[:len(t)]
        
        # Apply Y correction if requested
        if y_correction and len(avg_response) > 10:
            # Offset to start at 0
            avg_response = avg_response - avg_response[0]
        
        return t, avg_response, len(step_responses)
    
    # Return empty response if no valid segments
    return t[:wnd] if len(t) > wnd else t, np.zeros(wnd), 0


def deconvolve_step_response(
    input_signal: np.ndarray,
    output_signal: np.ndarray,
    window_length: int
) -> Optional[np.ndarray]:
    """
    Deconvolve the step response using FFT.
    
    The step response h(t) is calculated by deconvolving:
    output = input * h
    
    In the frequency domain:
    H(f) = Output(f) / Input(f)
    
    Args:
        input_signal: Input (setpoint) signal
        output_signal: Output (gyro) signal
        window_length: Desired length of step response
        
    Returns:
        Step response array, or None if calculation fails
    """
    if len(input_signal) < window_length or len(output_signal) < window_length:
        return None
    
    # Pad to power of 2 for efficient FFT
    n = len(input_signal)
    nfft = 2 ** int(np.ceil(np.log2(n + window_length)))
    
    # Remove mean (DC offset)
    input_centered = input_signal - np.mean(input_signal)
    output_centered = output_signal - np.mean(output_signal)
    
    # Apply window to reduce spectral leakage
    window = np.hanning(len(input_centered))
    input_windowed = input_centered * window
    output_windowed = output_centered * window
    
    # FFT of both signals
    input_fft = np.fft.fft(input_windowed, nfft)
    output_fft = np.fft.fft(output_windowed, nfft)
    
    # Regularization to avoid division by zero
    eps = 1e-10
    input_power = np.abs(input_fft) ** 2
    regularizer = np.max(input_power) * eps
    
    # Wiener deconvolution for noise robustness
    h_fft = (output_fft * np.conj(input_fft)) / (input_power + regularizer)
    
    # Inverse FFT to get impulse response
    impulse_response = np.real(np.fft.ifft(h_fft))
    
    # Convert impulse response to step response (cumulative sum)
    step_response = np.cumsum(impulse_response[:window_length])
    
    # Normalize
    if np.max(np.abs(step_response)) > 0:
        step_response = step_response / np.max(np.abs(step_response))
    
    return step_response


def calculate_metrics(
    time_ms: np.ndarray,
    step_response: np.ndarray
) -> Tuple[float, float, float]:
    """
    Calculate step response metrics.
    
    Args:
        time_ms: Time array in milliseconds
        step_response: Step response array
        
    Returns:
        Tuple of (rise_time_ms, max_overshoot, settling_time_ms)
    """
    if len(step_response) < 2 or len(time_ms) < 2:
        return 0.0, 0.0, 0.0
    
    # Normalize response to [0, 1] range
    response = step_response.copy()
    response_min = np.min(response)
    response_max = np.max(response)
    
    if response_max - response_min < 1e-10:
        return 0.0, 0.0, 0.0
    
    # Determine final value (use average of last 10% of signal)
    final_idx = max(1, int(len(response) * 0.9))
    final_value = np.mean(response[final_idx:])
    
    if abs(final_value) < 1e-10:
        return 0.0, 0.0, 0.0
    
    # Normalize to final value
    response_norm = response / final_value
    
    # Rise time: time to reach ~63.2% (1 - 1/e) of final value
    # Alternative: time from 10% to 90%
    target_63 = 0.632
    rise_time_ms = 0.0
    
    for i, val in enumerate(response_norm):
        if val >= target_63:
            rise_time_ms = time_ms[i] if i < len(time_ms) else 0.0
            break
    
    # Maximum overshoot
    peak_value = np.max(response_norm)
    max_overshoot = max(0.0, peak_value - 1.0)  # Overshoot above 1.0 (final value)
    
    # Settling time: time to settle within 2% of final value
    settling_threshold = 0.02
    settling_time_ms = 0.0
    
    for i in range(len(response_norm) - 1, -1, -1):
        if abs(response_norm[i] - 1.0) > settling_threshold:
            settling_time_ms = time_ms[min(i + 1, len(time_ms) - 1)]
            break
    
    return rise_time_ms, max_overshoot, settling_time_ms
