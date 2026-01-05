# PID Step Response Library - Step Response Calculator
# Copyright (C) 2024
# License: GPLv3

"""
Step response calculation using FFT-based deconvolution.
Faithfully replicates the PIDtoolbox (PTstepcalc.m) algorithm.
"""

from typing import Tuple, Optional, List
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
    y_correction: bool = True
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Calculate the step response using FFT-based deconvolution.
    
    This function faithfully replicates the PIDtoolbox PTstepcalc.m algorithm:
    1. Apply optional smoothing to gyro data (LOWESS)
    2. Segment the data into 2-second windows with overlap
    3. Find segments with sufficient input signal (>= 20 deg/s)
    4. Deconvolve each segment using FFT to get step response
    5. Apply Y-correction to normalize steady-state to 1.0
    6. Quality control: keep only segments where steady-state is between 0.5 and 3.0
    7. Average all valid segments
    
    Args:
        setpoint: Setpoint (input) signal array (deg/s)
        gyro: Gyro (output) signal array (deg/s)
        log_rate: Log rate in samples per millisecond (e.g., 4.0 for 4kHz)
        smooth_factor: Smoothing level (1=off, 2=low, 3=medium, 4=high)
        y_correction: Whether to apply Y-axis normalization (default: True)
        
    Returns:
        Tuple of (time_ms, step_response, num_segments)
        - time_ms: Time array in milliseconds (0 to 500ms)
        - step_response: Averaged step response (normalized to converge to 1.0)
        - num_segments: Number of valid segments used
    """
    # Smoothing values matching PIDtoolbox: smoothVals = [1 20 40 60]
    smooth_vals = [1, 20, 40, 60]
    smooth_window = smooth_vals[min(smooth_factor - 1, 3)]
    
    # Apply smoothing to gyro if requested
    gyro = np.asarray(gyro).copy()
    if smooth_factor > 1 and smooth_window > 1:
        gyro = lowess_smooth(gyro, smooth_window)
    
    # Parameters matching PIDtoolbox PTstepcalc.m
    min_input = 20  # Minimum input rate to consider valid segment
    segment_length = int(log_rate * 2000)  # 2 second segments
    wnd = int(log_rate * 1000 * 0.5)  # 500ms step response window
    step_resp_duration_ms = 500  # Max duration of step resp in ms for plotting
    pad_length = 100  # Zero padding on each side (matching MATLAB)
    
    # Create time array: t = 0 : 1/lograte : StepRespDuration_ms
    t = np.arange(0, step_resp_duration_ms + 1/log_rate, 1/log_rate)
    
    # Ensure arrays are numpy and same length
    n = min(len(setpoint), len(gyro))
    setpoint = np.asarray(setpoint[:n])
    gyro = gyro[:n]
    
    # Replace NaN values with 0 to avoid calculation issues
    setpoint = np.nan_to_num(setpoint, nan=0.0)
    gyro = np.nan_to_num(gyro, nan=0.0)
    
    # Calculate file duration in seconds
    file_dur_sec = n / (log_rate * 1000)
    
    # Subsampling factor based on file duration (matching PIDtoolbox)
    if file_dur_sec <= 20:
        subsample_factor = 10
    elif file_dur_sec <= 60:
        subsample_factor = 7
    else:
        subsample_factor = 3
    
    # Create segment vector: segment_vector = 1 : round(segment_length/subsampleFactor) : length(SP)
    segment_step = max(1, round(segment_length / subsample_factor))
    segment_vector = list(range(0, n, segment_step))
    
    # Find valid segments: ensure segment doesn't exceed data length
    # MATLAB: NSegs = max(find((segment_vector+segment_length) < segment_vector(end)))
    valid_segment_indices = [i for i, sv in enumerate(segment_vector) 
                            if sv + segment_length <= n]
    n_segs = len(valid_segment_indices)
    
    if n_segs == 0:
        return t[:wnd] if len(t) > wnd else t, np.zeros(wnd), 0
    
    # Collect valid segments
    sp_segments: List[np.ndarray] = []
    gy_segments: List[np.ndarray] = []
    
    for i in valid_segment_indices:
        start_idx = segment_vector[i]
        end_idx = start_idx + segment_length
        
        sp_seg = setpoint[start_idx:end_idx]
        
        # Check if segment has sufficient input: max(abs(SP(...))) >= minInput
        if np.max(np.abs(sp_seg)) >= min_input:
            sp_segments.append(sp_seg)
            gy_segments.append(gyro[start_idx:end_idx])
    
    if len(sp_segments) == 0:
        return t[:wnd] if len(t) > wnd else t, np.zeros(wnd), 0
    
    # Process each segment with FFT deconvolution
    step_responses: List[np.ndarray] = []
    
    for sp_seg, gy_seg in zip(sp_segments, gy_segments):
        # Apply Hann window (matching MATLAB: hann(length(...)))
        window = np.hanning(len(sp_seg))
        a = gy_seg * window  # a = GYseg(i,:).*hann(length(GYseg(i,:)))'
        b = sp_seg * window  # b = SPseg(i,:).*hann(length(SPseg(i,:)))'
        
        # Zero padding on both sides (matching MATLAB)
        a_padded = np.concatenate([np.zeros(pad_length), a, np.zeros(pad_length)])
        b_padded = np.concatenate([np.zeros(pad_length), b, np.zeros(pad_length)])
        
        # FFT and normalize by length (matching MATLAB)
        G = np.fft.fft(a_padded) / len(a_padded)
        H = np.fft.fft(b_padded) / len(b_padded)
        Hcon = np.conj(H)
        
        # Impulse response: imp = real(ifft((G .* Hcon) ./ (H .* Hcon + 0.0001)))
        imp = np.real(np.fft.ifft((G * Hcon) / (H * Hcon + 0.0001)))
        
        # Step response = cumulative sum of impulse response
        resptmp = np.cumsum(imp)
        
        # Y-correction: normalize so steady-state mean = 1.0
        # Find steady-state window: t > 200 & t < StepRespDuration_ms
        # The response array length depends on the padded segment length
        # Map indices based on the window (wnd) which corresponds to 500ms
        samples_per_ms = len(resptmp) / step_resp_duration_ms if step_resp_duration_ms > 0 else log_rate
        steady_state_start = int(200 * samples_per_ms)
        steady_state_end = min(int(step_resp_duration_ms * samples_per_ms), len(resptmp))
        
        if steady_state_end > len(resptmp):
            steady_state_end = len(resptmp)
        if steady_state_start >= steady_state_end:
            steady_state_start = max(0, steady_state_end - 10)
        
        steady_state_resp = resptmp[steady_state_start:steady_state_end]
        
        if len(steady_state_resp) == 0:
            continue
        
        # Apply Y-correction (matching MATLAB)
        if y_correction:
            steady_state_mean = np.nanmean(steady_state_resp)
            if steady_state_mean != 0 and not np.isnan(steady_state_mean):
                # yoffset = 1 - nanmean(steadyStateResp)
                # resptmp(i,:) = resptmp(i,:) * (yoffset+1)
                yoffset = 1 - steady_state_mean
                resptmp = resptmp * (yoffset + 1)
                # Recalculate steady state after correction
                steady_state_resp = resptmp[steady_state_start:steady_state_end]
        
        # Quality control: min(steadyStateResp) > 0.5 && max(steadyStateResp) < 3
        if np.min(steady_state_resp) > 0.5 and np.max(steady_state_resp) < 3:
            # Keep only the step response window: stepresponse(j,:)=resptmp(i,1:1+wnd)
            step_resp = resptmp[:wnd + 1]
            if len(step_resp) > 0 and not np.any(np.isnan(step_resp)):
                step_responses.append(step_resp)
    
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
    
    This is a legacy function kept for compatibility.
    The main calculation now happens in calculate_step_response.
    
    Args:
        input_signal: Input (setpoint) signal
        output_signal: Output (gyro) signal
        window_length: Desired length of step response
        
    Returns:
        Step response array, or None if calculation fails
    """
    if len(input_signal) < window_length or len(output_signal) < window_length:
        return None
    
    pad_length = 100
    
    # Apply Hann window
    window = np.hanning(len(input_signal))
    a = output_signal * window
    b = input_signal * window
    
    # Zero padding
    a_padded = np.concatenate([np.zeros(pad_length), a, np.zeros(pad_length)])
    b_padded = np.concatenate([np.zeros(pad_length), b, np.zeros(pad_length)])
    
    # FFT and normalize
    G = np.fft.fft(a_padded) / len(a_padded)
    H = np.fft.fft(b_padded) / len(b_padded)
    Hcon = np.conj(H)
    
    # Impulse response with regularization
    imp = np.real(np.fft.ifft((G * Hcon) / (H * Hcon + 0.0001)))
    
    # Step response = cumulative sum
    step_response = np.cumsum(imp)[:window_length]
    
    return step_response


def calculate_metrics(
    time_ms: np.ndarray,
    step_response: np.ndarray
) -> Tuple[float, float, float]:
    """
    Calculate step response metrics.
    
    The step response should already be normalized to converge to 1.0.
    
    Args:
        time_ms: Time array in milliseconds
        step_response: Step response array (normalized to steady-state = 1.0)
        
    Returns:
        Tuple of (rise_time_ms, max_overshoot, settling_time_ms)
    """
    if len(step_response) < 2 or len(time_ms) < 2:
        return 0.0, 0.0, 0.0
    
    response = step_response.copy()
    
    # Determine final value (use average of last 10% of signal)
    final_idx = max(1, int(len(response) * 0.9))
    final_value = np.mean(response[final_idx:])
    
    if abs(final_value) < 1e-10:
        return 0.0, 0.0, 0.0
    
    # Rise time: time to reach ~63.2% (1 - 1/e) of final value
    target_63 = 0.632 * final_value
    rise_time_ms = 0.0
    
    for i, val in enumerate(response):
        if val >= target_63:
            rise_time_ms = time_ms[i] if i < len(time_ms) else 0.0
            break
    
    # Maximum overshoot: (peak - final) / final
    peak_value = np.max(response)
    max_overshoot = max(0.0, (peak_value - final_value) / final_value)
    
    # Settling time: time to settle within 2% of final value
    settling_threshold = 0.02 * abs(final_value)
    settling_time_ms = 0.0
    
    for i in range(len(response) - 1, -1, -1):
        if abs(response[i] - final_value) > settling_threshold:
            settling_time_ms = time_ms[min(i + 1, len(time_ms) - 1)]
            break
    
    return rise_time_ms, max_overshoot, settling_time_ms
