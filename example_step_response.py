#!/usr/bin/env python3
# PID Step Response Library - Example Usage
# Copyright (C) 2024
# License: GPLv3

"""
Example script demonstrating how to use the PID Step Response library.

This script shows how to:
1. Parse a BBL (blackbox) file
2. Analyze step response for Roll, Pitch, Yaw axes
3. Extract rise time and overshoot metrics
4. Generate plots of the step response curves
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pid_step_response import StepResponseAnalyzer, plot_step_response


def main():
    """Main example function."""
    # Check for input file
    if len(sys.argv) < 2:
        print("Usage: python example_step_response.py <bbl_file>")
        print("\nThis script analyzes a Betaflight blackbox log file and")
        print("computes step response metrics for Roll, Pitch, and Yaw axes.")
        return
    
    bbl_file = sys.argv[1]
    
    if not Path(bbl_file).exists():
        print(f"Error: File not found: {bbl_file}")
        return
    
    # Create analyzer with default settings
    # smooth_factor: 1=off, 2=low, 3=medium, 4=high
    analyzer = StepResponseAnalyzer(smooth_factor=1)
    
    # Get the number of logs in the file
    log_count = analyzer.get_log_count(bbl_file)
    print(f"Found {log_count} log(s) in file: {bbl_file}\n")
    
    # Analyze all logs in the file
    results = analyzer.analyze(bbl_file)
    
    # Print summary for each log
    for result in results:
        print("=" * 60)
        print(result.summary())
        print()
        
        # Optionally plot the results
        try:
            # Generate step response plot
            fig = plot_step_response(
                result,
                axes=['roll', 'pitch', 'yaw'],
                show=True,  # Set to False to not display
                save_path=f"{Path(bbl_file).stem}_log{result.log_index}_step_response.png"
            )
        except ImportError:
            print("Note: matplotlib is required for plotting. Install with: pip install matplotlib")
        except Exception as e:
            print(f"Warning: Could not generate plot: {e}")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
