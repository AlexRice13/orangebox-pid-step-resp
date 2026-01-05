# PID Step Response Library
# A Python library for parsing BBL files and computing step response analysis
# Based on orangebox library and PIDtoolbox algorithms
#
# Copyright (C) 2024
# License: GPLv3

"""
PID Step Response Library

This library parses Betaflight blackbox log (BBL) files and computes
step response analysis for Roll, Pitch, and Yaw axes.

Features:
- Parse BBL files using orangebox
- Handle multiple logs per BBL file
- Calculate step response using FFT-based deconvolution
- Compute rise time and maximum overshoot ratio
- Extract PID parameters from headers
- Generate response curve plots
"""

from .analyzer import StepResponseAnalyzer
from .models import StepResponseResult, AxisResult, PIDParams, LogData
from .calculator import calculate_step_response, calculate_metrics
from .plotter import plot_step_response

__version__ = "0.1.0"
__all__ = [
    "StepResponseAnalyzer",
    "StepResponseResult",
    "AxisResult",
    "PIDParams",
    "LogData",
    "calculate_step_response",
    "calculate_metrics",
    "plot_step_response",
]
