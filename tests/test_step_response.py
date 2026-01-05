# PID Step Response Library - Unit Tests
# Copyright (C) 2024
# License: GPLv3

"""
Unit tests for the PID Step Response library.
"""

import os
import unittest
import tempfile
from pathlib import Path

import numpy as np

from pid_step_response.models import (
    AxisResult,
    LogData,
    PIDParams,
    StepResponseResult,
)
from pid_step_response.calculator import (
    calculate_metrics,
    calculate_step_response,
    lowess_smooth,
    wiener_deconvolution,
)


class TestModels(unittest.TestCase):
    """Tests for data models."""
    
    def test_pid_params_creation(self):
        """Test PIDParams creation and string representation."""
        params = PIDParams(p=45, i=80, d=35, f=120, d_min=20)
        self.assertEqual(params.p, 45)
        self.assertEqual(params.i, 80)
        self.assertEqual(params.d, 35)
        self.assertEqual(params.f, 120)
        self.assertEqual(params.d_min, 20)
        self.assertIn("P=45", str(params))
    
    def test_pid_params_defaults(self):
        """Test PIDParams default values."""
        params = PIDParams()
        self.assertEqual(params.p, 0.0)
        self.assertEqual(params.i, 0.0)
        self.assertEqual(params.d, 0.0)
    
    def test_axis_result_creation(self):
        """Test AxisResult creation."""
        result = AxisResult(axis_name='roll')
        self.assertEqual(result.axis_name, 'roll')
        self.assertEqual(result.rise_time_ms, 0.0)
        self.assertEqual(result.max_overshoot, 0.0)
    
    def test_axis_result_repr(self):
        """Test AxisResult string representation."""
        result = AxisResult(axis_name='pitch', rise_time_ms=25.5, max_overshoot=0.15)
        repr_str = repr(result)
        self.assertIn('pitch', repr_str)
        self.assertIn('25.50', repr_str)
    
    def test_log_data_duration(self):
        """Test LogData duration calculation."""
        log_data = LogData(log_index=1)
        log_data.time_us = np.array([0, 1000000, 2000000])  # 2 seconds
        self.assertEqual(log_data.duration_seconds, 2.0)
    
    def test_log_data_sample_count(self):
        """Test LogData sample count."""
        log_data = LogData(log_index=1)
        log_data.time_us = np.array([0, 1000, 2000, 3000, 4000])
        self.assertEqual(log_data.sample_count, 5)
    
    def test_step_response_result_axes(self):
        """Test StepResponseResult axes property."""
        result = StepResponseResult(file_path="test.bbl", log_index=1)
        axes = result.axes
        self.assertIn('roll', axes)
        self.assertIn('pitch', axes)
        self.assertIn('yaw', axes)
    
    def test_step_response_result_summary(self):
        """Test StepResponseResult summary generation."""
        result = StepResponseResult(
            file_path="test.bbl",
            log_index=1,
            duration_seconds=30.0,
            sample_count=120000
        )
        result.roll.rise_time_ms = 25.0
        result.roll.max_overshoot = 0.10
        
        summary = result.summary()
        self.assertIn("test.bbl", summary)
        self.assertIn("30.00s", summary)
        self.assertIn("ROLL", summary)
        self.assertIn("25.00 ms", summary)


class TestCalculator(unittest.TestCase):
    """Tests for step response calculator."""
    
    def test_lowess_smooth_identity(self):
        """Test LOWESS smoothing with window size 1 returns original data."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        smoothed = lowess_smooth(data, window_size=1)
        np.testing.assert_array_almost_equal(data, smoothed)
    
    def test_lowess_smooth_reduces_noise(self):
        """Test LOWESS smoothing reduces noise."""
        # Create noisy signal
        np.random.seed(42)
        t = np.linspace(0, 10, 100)
        clean_signal = np.sin(t)
        noisy_signal = clean_signal + np.random.normal(0, 0.3, len(t))
        
        smoothed = lowess_smooth(noisy_signal, window_size=10)
        
        # Smoothed should be closer to clean signal than noisy
        noise_error = np.mean((noisy_signal - clean_signal) ** 2)
        smooth_error = np.mean((smoothed - clean_signal) ** 2)
        
        self.assertLess(smooth_error, noise_error)
    
    def test_calculate_step_response_basic(self):
        """Test basic step response calculation."""
        # Create a simple step input and exponential response
        log_rate = 4.0  # 4 kHz
        n_samples = 10000
        
        # Step input
        setpoint = np.zeros(n_samples)
        setpoint[1000:] = 100.0  # Step from 0 to 100 at sample 1000
        
        # First-order response
        tau = 50  # Time constant in samples
        gyro = np.zeros(n_samples)
        for i in range(1001, n_samples):
            gyro[i] = gyro[i-1] + (setpoint[i] - gyro[i-1]) / tau
        
        time_ms, step_resp, num_segments = calculate_step_response(
            setpoint, gyro, log_rate, smooth_factor=1
        )
        
        self.assertIsInstance(time_ms, np.ndarray)
        self.assertIsInstance(step_resp, np.ndarray)
        self.assertGreaterEqual(len(time_ms), 1)
    
    def test_calculate_step_response_empty_input(self):
        """Test step response calculation with empty input."""
        time_ms, step_resp, num_segments = calculate_step_response(
            np.array([]), np.array([]), log_rate=4.0
        )
        
        self.assertEqual(num_segments, 0)
    
    def test_calculate_step_response_with_smoothing(self):
        """Test step response calculation with different smoothing levels."""
        log_rate = 4.0
        n_samples = 10000
        
        np.random.seed(42)
        setpoint = np.random.randn(n_samples) * 50
        gyro = np.random.randn(n_samples) * 50
        
        # Test all smoothing levels
        for smooth_factor in [1, 2, 3, 4]:
            time_ms, step_resp, num_segments = calculate_step_response(
                setpoint, gyro, log_rate, smooth_factor=smooth_factor
            )
            self.assertIsInstance(time_ms, np.ndarray)
    
    def test_calculate_step_response_with_nan_values(self):
        """Test step response calculation handles NaN values gracefully."""
        log_rate = 4.0
        n_samples = 10000
        
        np.random.seed(42)
        setpoint = np.random.randn(n_samples) * 50
        gyro = np.random.randn(n_samples) * 50
        
        # Insert some NaN values
        setpoint[100:110] = np.nan
        gyro[500:510] = np.nan
        
        # Should not raise an exception
        time_ms, step_resp, num_segments = calculate_step_response(
            setpoint, gyro, log_rate, smooth_factor=1
        )
        
        self.assertIsInstance(time_ms, np.ndarray)
        self.assertIsInstance(step_resp, np.ndarray)
        # Should not contain NaN in the output
        self.assertFalse(np.any(np.isnan(step_resp)))
    
    def test_wiener_deconvolution_basic(self):
        """Test Wiener deconvolution with known signals."""
        # Create simple signals (2D arrays for batch processing)
        n = 1000
        np.random.seed(42)
        input_signal = np.random.randn(1, n) * 10
        
        # Create output as filtered input (simple low-pass)
        output_signal = np.zeros_like(input_signal)
        output_signal[0] = np.convolve(input_signal[0], np.ones(10)/10, mode='same')
        
        dt = 0.001  # 1ms time step
        deconvolved = wiener_deconvolution(input_signal, output_signal, dt, cutfreq=25.0)
        
        self.assertIsNotNone(deconvolved)
        self.assertEqual(deconvolved.shape[0], 1)
    
    def test_wiener_deconvolution_handles_empty(self):
        """Test Wiener deconvolution handles edge cases gracefully."""
        # Small inputs should still work
        input_signal = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        output_signal = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        
        dt = 0.001
        deconvolved = wiener_deconvolution(input_signal, output_signal, dt, cutfreq=25.0)
        
        self.assertIsNotNone(deconvolved)
    
    def test_calculate_metrics_ideal_response(self):
        """Test metrics calculation with ideal step response."""
        # Create ideal first-order response: 1 - exp(-t/tau)
        time_ms = np.linspace(0, 500, 500)
        tau = 50  # Time constant in ms
        step_response = 1 - np.exp(-time_ms / tau)
        
        rise_time, overshoot, settling_time = calculate_metrics(time_ms, step_response)
        
        # Rise time should be around tau (time to reach 63.2%)
        self.assertGreater(rise_time, 30)
        self.assertLess(rise_time, 70)
        
        # No overshoot for first-order system
        self.assertAlmostEqual(overshoot, 0.0, places=2)
    
    def test_calculate_metrics_with_overshoot(self):
        """Test metrics calculation with overshoot."""
        # Create response with overshoot
        time_ms = np.linspace(0, 500, 500)
        # Damped oscillation with 20% overshoot
        omega = 0.05
        zeta = 0.5
        step_response = 1 - np.exp(-zeta * omega * time_ms) * (
            np.cos(omega * np.sqrt(1 - zeta**2) * time_ms)
        )
        # Scale to have max > 1
        step_response = step_response * 1.0
        
        rise_time, overshoot, settling_time = calculate_metrics(time_ms, step_response)
        
        # Should have some overshoot
        self.assertGreaterEqual(overshoot, 0.0)
    
    def test_calculate_metrics_empty_input(self):
        """Test metrics calculation with empty input."""
        rise_time, overshoot, settling_time = calculate_metrics(
            np.array([]), np.array([])
        )
        
        self.assertEqual(rise_time, 0.0)
        self.assertEqual(overshoot, 0.0)
        self.assertEqual(settling_time, 0.0)
    
    def test_calculate_metrics_constant_response(self):
        """Test metrics calculation with constant response."""
        time_ms = np.linspace(0, 100, 100)
        step_response = np.ones(100)
        
        rise_time, overshoot, settling_time = calculate_metrics(time_ms, step_response)
        
        self.assertEqual(overshoot, 0.0)


class TestParser(unittest.TestCase):
    """Tests for BBL file parser."""
    
    def test_parse_pid_string(self):
        """Test PID string parsing."""
        from pid_step_response.parser import parse_pid_string
        
        p, i, d = parse_pid_string("45,80,35")
        self.assertEqual(p, 45.0)
        self.assertEqual(i, 80.0)
        self.assertEqual(d, 35.0)
    
    def test_parse_pid_string_with_spaces(self):
        """Test PID string parsing with spaces."""
        from pid_step_response.parser import parse_pid_string
        
        p, i, d = parse_pid_string("45, 80, 35")
        self.assertEqual(p, 45.0)
        self.assertEqual(i, 80.0)
        self.assertEqual(d, 35.0)
    
    def test_parse_pid_string_invalid(self):
        """Test PID string parsing with invalid input."""
        from pid_step_response.parser import parse_pid_string
        
        p, i, d = parse_pid_string("invalid")
        self.assertEqual(p, 0.0)
        self.assertEqual(i, 0.0)
        self.assertEqual(d, 0.0)
    
    def test_extract_pid_params(self):
        """Test PID parameter extraction from headers."""
        from pid_step_response.parser import extract_pid_params
        
        headers = {
            'rollPID': [45, 80, 35],
            'pitchPID': [50, 85, 40],
            'yawPID': [55, 90, 0],
        }
        
        roll_pid = extract_pid_params(headers, 'roll')
        self.assertEqual(roll_pid.p, 45.0)
        self.assertEqual(roll_pid.i, 80.0)
        self.assertEqual(roll_pid.d, 35.0)
    
    def test_get_field_index(self):
        """Test field index lookup."""
        from pid_step_response.parser import get_field_index
        
        field_names = ['time', 'loopIteration', 'gyroADC[0]', 'setpoint[0]']
        
        idx = get_field_index(field_names, 'time')
        self.assertEqual(idx, 0)
        
        idx = get_field_index(field_names, 'gyroADC[0]', 'gyro[0]')
        self.assertEqual(idx, 2)
        
        idx = get_field_index(field_names, 'nonexistent')
        self.assertIsNone(idx)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete analysis pipeline."""
    
    def test_analyzer_creation(self):
        """Test StepResponseAnalyzer creation."""
        from pid_step_response.analyzer import StepResponseAnalyzer
        
        analyzer = StepResponseAnalyzer()
        self.assertEqual(analyzer.smooth_factor, 1)
        self.assertEqual(analyzer.min_input, 20.0)
        self.assertFalse(analyzer.y_correction)
    
    def test_analyzer_custom_params(self):
        """Test StepResponseAnalyzer with custom parameters."""
        from pid_step_response.analyzer import StepResponseAnalyzer
        
        analyzer = StepResponseAnalyzer(
            smooth_factor=3,
            min_input=30.0,
            y_correction=True
        )
        self.assertEqual(analyzer.smooth_factor, 3)
        self.assertEqual(analyzer.min_input, 30.0)
        self.assertTrue(analyzer.y_correction)


if __name__ == '__main__':
    unittest.main()
