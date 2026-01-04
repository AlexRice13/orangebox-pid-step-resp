# PID Step Response Library - Main Analyzer
# Copyright (C) 2024
# License: GPLv3

"""
Main analyzer class that orchestrates BBL parsing and step response calculation.
"""

from pathlib import Path
from typing import List, Optional, Union

from .models import AxisResult, LogData, PIDParams, StepResponseResult
from .parser import get_log_count, parse_all_logs, parse_bbl_file
from .calculator import calculate_metrics, calculate_step_response


class StepResponseAnalyzer:
    """
    Main class for analyzing step response from BBL (blackbox) files.
    
    This class combines BBL file parsing with step response calculation
    to provide a complete analysis of flight controller PID tuning.
    
    Example usage:
        analyzer = StepResponseAnalyzer()
        results = analyzer.analyze("flight.bbl")
        
        for result in results:
            print(result.summary())
    """
    
    def __init__(
        self,
        smooth_factor: int = 1,
        min_input: float = 20.0,
        y_correction: bool = False
    ):
        """
        Initialize the analyzer with default parameters.
        
        Args:
            smooth_factor: Gyro smoothing level (1=off, 2=low, 3=medium, 4=high)
            min_input: Minimum input rate to consider valid (deg/s)
            y_correction: Whether to apply Y-axis offset correction
        """
        self.smooth_factor = smooth_factor
        self.min_input = min_input
        self.y_correction = y_correction
    
    def analyze(
        self,
        file_path: Union[str, Path],
        log_index: Optional[int] = None
    ) -> List[StepResponseResult]:
        """
        Analyze a BBL file and compute step response for all axes.
        
        Args:
            file_path: Path to the BBL file
            log_index: Specific log index to analyze (1-based).
                      If None, analyzes all logs in the file.
                      
        Returns:
            List of StepResponseResult objects, one per analyzed log
        """
        file_path = str(file_path)
        
        if log_index is not None:
            # Analyze specific log
            log_data = parse_bbl_file(file_path, log_index=log_index)
            result = self._analyze_log(file_path, log_data)
            return [result] if result else []
        else:
            # Analyze all logs
            logs = parse_all_logs(file_path)
            results = []
            for log_data in logs:
                result = self._analyze_log(file_path, log_data)
                if result:
                    results.append(result)
            return results
    
    def _analyze_log(
        self,
        file_path: str,
        log_data: LogData
    ) -> Optional[StepResponseResult]:
        """
        Analyze a single log and compute step response.
        
        Args:
            file_path: Path to the BBL file
            log_data: Parsed log data
            
        Returns:
            StepResponseResult object, or None if analysis fails
        """
        if log_data.sample_count < 100:
            return None
        
        result = StepResponseResult(
            file_path=file_path,
            log_index=log_data.log_index,
            log_rate=log_data.log_rate,
            duration_seconds=log_data.duration_seconds,
            sample_count=log_data.sample_count,
            smooth_factor=self.smooth_factor,
            min_input=self.min_input
        )
        
        # Analyze Roll axis
        if len(log_data.setpoint_roll) > 0 and len(log_data.gyro_roll) > 0:
            result.roll = self._analyze_axis(
                'roll',
                log_data.setpoint_roll,
                log_data.gyro_roll,
                log_data.log_rate,
                log_data.roll_pid
            )
        
        # Analyze Pitch axis
        if len(log_data.setpoint_pitch) > 0 and len(log_data.gyro_pitch) > 0:
            result.pitch = self._analyze_axis(
                'pitch',
                log_data.setpoint_pitch,
                log_data.gyro_pitch,
                log_data.log_rate,
                log_data.pitch_pid
            )
        
        # Analyze Yaw axis
        if len(log_data.setpoint_yaw) > 0 and len(log_data.gyro_yaw) > 0:
            result.yaw = self._analyze_axis(
                'yaw',
                log_data.setpoint_yaw,
                log_data.gyro_yaw,
                log_data.log_rate,
                log_data.yaw_pid
            )
        
        return result
    
    def _analyze_axis(
        self,
        axis_name: str,
        setpoint: 'np.ndarray',
        gyro: 'np.ndarray',
        log_rate: float,
        pid_params: PIDParams
    ) -> AxisResult:
        """
        Analyze a single axis and compute its step response.
        
        Args:
            axis_name: Name of the axis ('roll', 'pitch', or 'yaw')
            setpoint: Setpoint signal array
            gyro: Gyro signal array
            log_rate: Log rate in samples per ms
            pid_params: PID parameters for this axis
            
        Returns:
            AxisResult object with computed step response and metrics
        """
        import numpy as np
        
        result = AxisResult(axis_name=axis_name)
        result.pid_params = pid_params
        result.setpoint = setpoint
        result.gyro = gyro
        
        # Calculate step response
        time_ms, step_response, num_segments = calculate_step_response(
            setpoint=setpoint,
            gyro=gyro,
            log_rate=log_rate,
            smooth_factor=self.smooth_factor,
            y_correction=self.y_correction
        )
        
        result.time_ms = time_ms
        result.step_response = step_response
        result.num_segments = num_segments
        
        # Calculate metrics
        if num_segments > 0:
            rise_time, overshoot, settling_time = calculate_metrics(time_ms, step_response)
            result.rise_time_ms = rise_time
            result.max_overshoot = overshoot
            result.settling_time_ms = settling_time
        
        return result
    
    @staticmethod
    def get_log_count(file_path: Union[str, Path]) -> int:
        """
        Get the number of logs in a BBL file.
        
        Args:
            file_path: Path to the BBL file
            
        Returns:
            Number of logs in the file
        """
        return get_log_count(str(file_path))
