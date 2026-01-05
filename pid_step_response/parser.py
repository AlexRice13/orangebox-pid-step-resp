# PID Step Response Library - BBL Parser
# Copyright (C) 2024
# License: GPLv3

"""
BBL file parser using orangebox library.
Extracts setpoint, gyro data, and PID parameters from blackbox logs.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from orangebox import Parser

from .models import LogData, PIDParams


def safe_float_convert(value: Any) -> float:
    """
    Safely convert a value to float, returning NaN for invalid values.
    
    Args:
        value: Value to convert (can be string, number, None, etc.)
        
    Returns:
        Float value, or np.nan if conversion fails
    """
    if value is None or value == '':
        return np.nan
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def parse_pid_string(pid_string: str) -> Tuple[float, float, float]:
    """
    Parse a PID string like "45,80,35" into P, I, D values.
    
    Args:
        pid_string: Comma-separated PID values
        
    Returns:
        Tuple of (P, I, D) values
    """
    try:
        parts = [float(x.strip()) for x in pid_string.split(',')]
        if len(parts) >= 3:
            return parts[0], parts[1], parts[2]
        return 0.0, 0.0, 0.0
    except (ValueError, AttributeError):
        return 0.0, 0.0, 0.0


def extract_pid_params(headers: Dict, axis: str) -> PIDParams:
    """
    Extract PID parameters for a specific axis from headers.
    
    Args:
        headers: Dictionary of log headers
        axis: 'roll', 'pitch', or 'yaw'
        
    Returns:
        PIDParams object with extracted values
    """
    axis_map = {'roll': 0, 'pitch': 1, 'yaw': 2}
    idx = axis_map.get(axis, 0)
    
    params = PIDParams()
    
    # Extract main PID values
    pid_key = f"{axis}PID"
    if pid_key in headers:
        pid_val = headers[pid_key]
        if isinstance(pid_val, str):
            p, i, d = parse_pid_string(pid_val)
            params.p = p
            params.i = i
            params.d = d
        elif isinstance(pid_val, list) and len(pid_val) >= 3:
            params.p = float(pid_val[0])
            params.i = float(pid_val[1])
            params.d = float(pid_val[2])
    
    # Try alternative format: separate p, i, d arrays
    if params.p == 0 and params.i == 0 and params.d == 0:
        for term, key_patterns in [
            ('p', [f'{axis}_p', f'p_{axis}', f'rollPitchYawP[{idx}]']),
            ('i', [f'{axis}_i', f'i_{axis}', f'rollPitchYawI[{idx}]']),
            ('d', [f'{axis}_d', f'd_{axis}', f'rollPitchYawD[{idx}]']),
        ]:
            for key in key_patterns:
                if key in headers:
                    setattr(params, term, float(headers[key]))
                    break
    
    # Check for combined PID format like "dterm_filter_type" etc
    # Betaflight uses arrays: [roll, pitch, yaw]
    for combined_key in ['p_term', 'i_term', 'd_term']:
        if combined_key in headers:
            val = headers[combined_key]
            if isinstance(val, list) and len(val) > idx:
                if combined_key == 'p_term':
                    params.p = float(val[idx])
                elif combined_key == 'i_term':
                    params.i = float(val[idx])
                elif combined_key == 'd_term':
                    params.d = float(val[idx])
    
    # Extract feedforward (F term)
    for ff_key in ['feedforward_weight', 'ff_weight', f'ff_{axis}']:
        if ff_key in headers:
            val = headers[ff_key]
            if isinstance(val, list) and len(val) > idx:
                params.f = float(val[idx])
                break
            elif isinstance(val, (int, float)):
                params.f = float(val)
                break
    
    # Extract D-min
    for dmin_key in ['d_min', f'd_min_{axis}']:
        if dmin_key in headers:
            val = headers[dmin_key]
            if isinstance(val, list) and len(val) > idx:
                params.d_min = float(val[idx])
                break
            elif isinstance(val, (int, float)):
                params.d_min = float(val)
                break
    
    return params


def get_field_index(field_names: List[str], *names: str) -> Optional[int]:
    """
    Find the index of a field by checking multiple possible names.
    
    Args:
        field_names: List of field names from the parser
        *names: Possible field names to search for
        
    Returns:
        Index of the first matching field, or None if not found
    """
    for name in names:
        for i, field in enumerate(field_names):
            # Handle field names with brackets like 'gyroADC[0]'
            if field == name or field.replace('[', '_').replace(']', '_').rstrip('_') == name:
                return i
            # Handle underscore variations
            if field.replace('_', '') == name.replace('_', ''):
                return i
    return None


def parse_bbl_file(file_path: str, log_index: int = 1) -> LogData:
    """
    Parse a BBL file and extract log data.
    
    Args:
        file_path: Path to the BBL file
        log_index: Index of the log within the file (1-based)
        
    Returns:
        LogData object containing extracted data
    """
    parser = Parser.load(file_path, log_index=log_index, allow_invalid_header=True)
    
    headers = parser.headers
    field_names = parser.field_names
    
    # Collect all frames
    frames_data = []
    for frame in parser.frames():
        frames_data.append(frame.data)
    
    if not frames_data:
        return LogData(log_index=log_index, headers=headers)
    
    # Convert frame data to numpy array, handling non-numeric values
    converted_frames = []
    for frame in frames_data:
        converted_frame = [safe_float_convert(val) for val in frame]
        converted_frames.append(converted_frame)
    
    data_array = np.array(converted_frames, dtype=float)
    
    # Find field indices
    time_idx = get_field_index(field_names, 'time', 'time_us', 'loopIteration')
    
    # Setpoint fields
    sp_roll_idx = get_field_index(field_names, 'setpoint[0]', 'setpoint_0', 'rcCommand[0]')
    sp_pitch_idx = get_field_index(field_names, 'setpoint[1]', 'setpoint_1', 'rcCommand[1]')
    sp_yaw_idx = get_field_index(field_names, 'setpoint[2]', 'setpoint_2', 'rcCommand[2]')
    
    # Gyro fields (filtered)
    gyro_roll_idx = get_field_index(field_names, 'gyroADC[0]', 'gyroADC_0', 'gyro[0]')
    gyro_pitch_idx = get_field_index(field_names, 'gyroADC[1]', 'gyroADC_1', 'gyro[1]')
    gyro_yaw_idx = get_field_index(field_names, 'gyroADC[2]', 'gyroADC_2', 'gyro[2]')
    
    # Extract data
    log_data = LogData(log_index=log_index, headers=headers)
    
    if time_idx is not None:
        log_data.time_us = data_array[:, time_idx]
    
    if sp_roll_idx is not None:
        log_data.setpoint_roll = data_array[:, sp_roll_idx]
    if sp_pitch_idx is not None:
        log_data.setpoint_pitch = data_array[:, sp_pitch_idx]
    if sp_yaw_idx is not None:
        log_data.setpoint_yaw = data_array[:, sp_yaw_idx]
    
    if gyro_roll_idx is not None:
        log_data.gyro_roll = data_array[:, gyro_roll_idx]
    if gyro_pitch_idx is not None:
        log_data.gyro_pitch = data_array[:, gyro_pitch_idx]
    if gyro_yaw_idx is not None:
        log_data.gyro_yaw = data_array[:, gyro_yaw_idx]
    
    # Calculate log rate (samples per millisecond)
    if len(log_data.time_us) > 1:
        # Time is in microseconds, use nanmedian to ignore NaN values
        time_diff = np.diff(log_data.time_us)
        # Filter out NaN and non-positive differences
        valid_diffs = time_diff[~np.isnan(time_diff) & (time_diff > 0)]
        if len(valid_diffs) > 0:
            dt_us = np.median(valid_diffs)
            if dt_us > 0:
                log_data.log_rate = 1000.0 / dt_us  # Convert to samples per ms
    
    # Extract PID parameters
    log_data.roll_pid = extract_pid_params(headers, 'roll')
    log_data.pitch_pid = extract_pid_params(headers, 'pitch')
    log_data.yaw_pid = extract_pid_params(headers, 'yaw')
    
    return log_data


def get_log_count(file_path: str) -> int:
    """
    Get the number of logs in a BBL file.
    
    Args:
        file_path: Path to the BBL file
        
    Returns:
        Number of logs in the file
    """
    parser = Parser.load(file_path, log_index=1, allow_invalid_header=True)
    return parser.reader.log_count


def parse_all_logs(file_path: str) -> List[LogData]:
    """
    Parse all logs from a BBL file.
    
    Args:
        file_path: Path to the BBL file
        
    Returns:
        List of LogData objects, one for each log in the file
    """
    log_count = get_log_count(file_path)
    logs = []
    
    for i in range(1, log_count + 1):
        try:
            log_data = parse_bbl_file(file_path, log_index=i)
            logs.append(log_data)
        except Exception as e:
            # Skip logs that fail to parse
            print(f"Warning: Failed to parse log {i}: {e}")
            continue
    
    return logs
