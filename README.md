# Orangebox

[![PyPI version](https://badge.fury.io/py/orangebox.svg)](https://badge.fury.io/py/orangebox) 
[![Documentation Status](https://readthedocs.org/projects/orangebox/badge/?version=latest)](https://orangebox.readthedocs.io/en/latest/?badge=latest)

A Cleanflight/Betaflight blackbox log parser written in Python 3 with **PID Step Response Analysis**.

This library combines the orangebox BBL parser with PID step response analysis capabilities, inspired by [PIDtoolbox](https://github.com/bw1129/PIDtoolbox). It can parse blackbox log files and compute step response metrics for Roll, Pitch, and Yaw axes.

## Features

- **BBL File Parsing**: Parse Betaflight/Cleanflight blackbox log files
- **Multiple Logs Support**: Handle BBL files with multiple flight logs
- **Step Response Analysis**: Calculate step response using FFT-based deconvolution
- **Metrics Calculation**: 
  - Rise time (time to reach 63.2% of final value)
  - Maximum overshoot ratio
  - Settling time
- **PID Parameter Extraction**: Extract P, I, D, F, and D-min values from log headers
- **Visualization**: Generate step response plots with matplotlib

## Installation

```bash
pip install -r requirements.txt
```

Or install with plotting support:

```bash
pip install numpy matplotlib
```

## Quick Start

### Step Response Analysis

```python
from pid_step_response import StepResponseAnalyzer, plot_step_response

# Create analyzer
analyzer = StepResponseAnalyzer(smooth_factor=1)  # 1=off, 2=low, 3=medium, 4=high

# Analyze BBL file (handles multiple logs automatically)
results = analyzer.analyze("flight.bbl")

# Print results for each log
for result in results:
    print(result.summary())
    
    # Access individual axis results
    print(f"Roll rise time: {result.roll.rise_time_ms:.1f} ms")
    print(f"Roll overshoot: {result.roll.max_overshoot*100:.1f}%")
    print(f"Roll PID: {result.roll.pid_params}")

# Generate plots
for result in results:
    plot_step_response(result, save_path=f"step_response_log{result.log_index}.png")
```

### Analyze a Specific Log

```python
# Analyze only log #2 from the file
results = analyzer.analyze("flight.bbl", log_index=2)
```

### Get Log Count

```python
log_count = StepResponseAnalyzer.get_log_count("flight.bbl")
print(f"File contains {log_count} logs")
```

### Basic BBL Parsing (Original Orangebox)

```python3
from orangebox import Parser

# Load a file
parser = Parser.load("btfl_all.bbl")
# or optionally select a log by index (1 is the default)
# parser = Parser.load("btfl_all.bbl", 1)

# Print headers
print("headers:", parser.headers)

# Print the names of fields
print("field names:", parser.field_names)

# Select a specific log within the file by index
print("log count:", parser.reader.log_count)
parser.set_log_index(2)

# Print field values frame by frame
for frame in parser.frames():
    print("first frame:", frame.data)
    break

# Complete list of events only available once all frames have been parsed
print("events:", parser.events)

# Selecting another log changes the header and frame data produced by the Parser
# and also clears any previous results and state
parser.set_log_index(1)
```

## Step Response Analysis Details

The step response calculation faithfully replicates the [PIDtoolbox](https://github.com/bw1129/PIDtoolbox) algorithm:

1. **Data Segmentation**: Splits flight data into 2-second segments
2. **Input Validation**: Filters segments with sufficient stick input (>20 deg/s by default)
3. **Deconvolution**: Uses FFT-based Wiener deconvolution to extract the system response
4. **Averaging**: Combines responses from all valid segments
5. **Metrics**: Calculates rise time, overshoot, and settling time

### Smoothing Options

The `smooth_factor` parameter controls gyro data smoothing:
- `1`: No smoothing (recommended for clean data)
- `2`: Low smoothing (LOWESS with window=20)
- `3`: Medium smoothing (LOWESS with window=40)
- `4`: High smoothing (LOWESS with window=60)

## Output

### StepResponseResult

Contains analysis results for a single log:
- `file_path`: Path to the BBL file
- `log_index`: Index of the log within the file
- `roll`, `pitch`, `yaw`: AxisResult objects for each axis
- `log_rate`: Sampling rate in kHz
- `duration_seconds`: Log duration
- `sample_count`: Number of samples

### AxisResult

Contains results for a single axis:
- `axis_name`: 'roll', 'pitch', or 'yaw'
- `time_ms`: Time array in milliseconds
- `step_response`: Step response array (normalized)
- `rise_time_ms`: Time to reach 50% of final value (matching PIDtoolbox)
- `max_overshoot`: Maximum overshoot ratio (0.1 = 10%)
- `settling_time_ms`: Time to settle within 2% of final value
- `pid_params`: PIDParams object with P, I, D, F, D-min values
- `num_segments`: Number of segments used in calculation

## Running Tests

```bash
python -m pytest tests/test_step_response.py -v
```

## Contributing

* Contributions are very welcome!
* Please follow the [PEP8](https://www.python.org/dev/peps/pep-0008/) Style Guide.
* [More info](https://orangebox.readthedocs.io/#development) in the docs.

## Changelog

### 0.5.0 (Current)

* **NEW**: Added PID Step Response Analysis module
* Compute step response for Roll, Pitch, Yaw axes
* Calculate rise time, overshoot, and settling time metrics
* Extract PID parameters from log headers
* Generate step response plots
* Handle multiple logs per BBL file

### 0.4.0

* Inspect log files for potentially missing required headers
* Allow skipping of badly formatted headers via the `allow_invalid_header` argument

### 0.3.1

* Add `bb2gpx` utility for converting GPS data into GPX

### 0.3.0

* Add support for GPS frames (thanks to [@tblaha](https://github.com/tblaha)!)

### 0.2.0

* Improved `Reader` class can now handle multiple logs in a single file
* Add `bbsplit` command-line script for splitting flashchip logs (thanks to [@ysoldak](https://github.com/ysoldak)!)
* Improved logging
* Added HTML documentation
* Fix parsing stats

### 0.1.1-beta

* Add `bb2csv` command-line script for converting logs into CSV

### 0.1.0-beta

* First release (with a lot of missing parts)

## Known issues

* No explicit validation of raw data against corruption (except for headers), but it's highly likely that a Python exception will be raised in these cases anyway
* Tested only on logs generated by Betaflight
* Not all event frames are parsed (see [TODO](orangebox/events.py) comments)
* Some decoders are missing (see [TODO](orangebox/decoders.py) comments)

## Acknowledgement

* Original blackbox data encoder and decoder was written by [Nicholas Sherlock](https://github.com/thenickdude).
* Step response algorithm based on [PIDtoolbox](https://github.com/bw1129/PIDtoolbox) by Brian White.

## License

This project is licensed under GPLv3.
