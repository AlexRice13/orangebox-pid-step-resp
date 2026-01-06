# PID Step Response Library - Plotter
# Copyright (C) 2024
# License: GPLv3

"""
Plotting functions for step response visualization.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np

from .models import AxisResult, StepResponseResult

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def plot_step_response(
    result: StepResponseResult,
    axes: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: tuple = (15, 6),
    single_panel: bool = False,
    y_max: float = 1.75
) -> "Figure":
    """
    Plot step response curves for the specified axes.
    
    Args:
        result: StepResponseResult object containing analysis results
        axes: List of axes to plot ('roll', 'pitch', 'yaw'). 
              If None, plots all axes with data.
        save_path: Path to save the figure. If None, figure is not saved.
        show: Whether to display the figure interactively.
        figsize: Figure size as (width, height) tuple.
        single_panel: If True, plot all axes on a single panel.
        y_max: Maximum y-axis value for step response plots.
        
    Returns:
        matplotlib Figure object
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        )
    
    # Default to all axes with data
    if axes is None:
        axes = []
        for name in ['roll', 'pitch', 'yaw']:
            axis_result = getattr(result, name)
            if axis_result.num_segments > 0:
                axes.append(name)
    
    if not axes:
        print("No valid axes to plot")
        return None
    
    # Colors for each axis
    colors = {
        'roll': '#1f77b4',   # Blue
        'pitch': '#ff7f0e',  # Orange
        'yaw': '#2ca02c'     # Green
    }
    
    # Create figure
    if single_panel:
        fig, ax_main = plt.subplots(figsize=figsize)
        axes_plots = [ax_main]
    else:
        num_axes = len(axes)
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, num_axes, height_ratios=[3, 1])
        axes_plots = []
        axes_info = []
        
        for i, axis_name in enumerate(axes):
            ax_step = fig.add_subplot(gs[0, i])
            ax_info = fig.add_subplot(gs[1, i])
            axes_plots.append(ax_step)
            axes_info.append(ax_info)
    
    # Plot each axis
    for i, axis_name in enumerate(axes):
        axis_result = getattr(result, axis_name)
        color = colors.get(axis_name, '#333333')
        
        if single_panel:
            ax = ax_main
            label = f'{axis_name.upper()}'
        else:
            ax = axes_plots[i]
            label = None
        
        # Plot step response
        if len(axis_result.time_ms) > 0 and len(axis_result.step_response) > 0:
            ax.plot(
                axis_result.time_ms,
                axis_result.step_response,
                color=color,
                linewidth=2,
                label=label
            )
        
        # Add reference lines
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Target')
        ax.axhline(y=0.0, color='gray', linestyle='-', alpha=0.3)
        
        # Add overshoot indicator if significant
        if axis_result.max_overshoot > 0.01:
            ax.axhline(
                y=1.0 + axis_result.max_overshoot,
                color=color,
                linestyle=':',
                alpha=0.5
            )
        
        # Add rise time indicator
        if axis_result.rise_time_ms > 0:
            ax.axvline(
                x=axis_result.rise_time_ms,
                color=color,
                linestyle=':',
                alpha=0.5
            )
        
        if not single_panel:
            # Configure axis
            ax.set_xlim(0, min(500, np.max(axis_result.time_ms) if len(axis_result.time_ms) > 0 else 500))
            ax.set_ylim(-0.1, y_max)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Response')
            ax.set_title(f'{axis_name.upper()} Step Response')
            ax.grid(True, alpha=0.3)
            
            # Add info panel
            ax_info_panel = axes_info[i]
            ax_info_panel.axis('off')
            
            info_text = (
                f"Rise Time: {axis_result.rise_time_ms:.1f} ms\n"
                f"Overshoot: {axis_result.max_overshoot*100:.1f}%\n"
                f"Segments: {axis_result.num_segments}\n"
                f"\nPID: P={axis_result.pid_params.p:.0f}, "
                f"I={axis_result.pid_params.i:.0f}, "
                f"D={axis_result.pid_params.d:.0f}"
            )
            
            if axis_result.pid_params.f > 0:
                info_text += f"\nFF: {axis_result.pid_params.f:.0f}"
            if axis_result.pid_params.d_min > 0:
                info_text += f"\nD-min: {axis_result.pid_params.d_min:.0f}"
            
            ax_info_panel.text(
                0.5, 0.5, info_text,
                transform=ax_info_panel.transAxes,
                fontsize=10,
                verticalalignment='center',
                horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
    
    if single_panel:
        ax_main.set_xlim(0, 500)
        ax_main.set_ylim(-0.1, y_max)
        ax_main.set_xlabel('Time (ms)')
        ax_main.set_ylabel('Response')
        ax_main.set_title('Step Response')
        ax_main.legend()
        ax_main.grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle(
        f"Step Response Analysis: {Path(result.file_path).name} "
        f"(Log #{result.log_index})\n"
        f"Duration: {result.duration_seconds:.1f}s, "
        f"Log Rate: {result.log_rate:.1f} kHz",
        fontsize=12
    )
    
    plt.tight_layout()
    
    # Save figure
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    # Show figure
    if show:
        plt.show()
    
    return fig


def plot_setpoint_gyro(
    result: StepResponseResult,
    axes: Optional[List[str]] = None,
    time_range: Optional[tuple] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: tuple = (15, 6)
) -> "Figure":
    """
    Plot setpoint vs gyro curves for the specified axes.
    
    Args:
        result: StepResponseResult object containing analysis results
        axes: List of axes to plot ('roll', 'pitch', 'yaw').
        time_range: Time range to plot as (start_ms, end_ms). 
                    If None, plots first 2 seconds.
        save_path: Path to save the figure.
        show: Whether to display the figure.
        figsize: Figure size.
        
    Returns:
        matplotlib Figure object
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting")
    
    if axes is None:
        axes = ['roll', 'pitch', 'yaw']
    
    colors_sp = {
        'roll': '#1f77b4',
        'pitch': '#ff7f0e',
        'yaw': '#2ca02c'
    }
    colors_gyro = {
        'roll': '#aec7e8',
        'pitch': '#ffbb78',
        'yaw': '#98df8a'
    }
    
    num_axes = len(axes)
    fig, ax_list = plt.subplots(num_axes, 1, figsize=figsize, sharex=True)
    
    if num_axes == 1:
        ax_list = [ax_list]
    
    for i, axis_name in enumerate(axes):
        ax = ax_list[i]
        axis_result = getattr(result, axis_name)
        
        if len(axis_result.setpoint) == 0:
            continue
        
        # Create time array in ms
        n_samples = len(axis_result.setpoint)
        time_ms = np.arange(n_samples) / result.log_rate
        
        # Apply time range
        if time_range:
            mask = (time_ms >= time_range[0]) & (time_ms <= time_range[1])
        else:
            mask = time_ms <= 2000  # First 2 seconds
        
        time_plot = time_ms[mask]
        sp_plot = axis_result.setpoint[mask]
        gyro_plot = axis_result.gyro[mask] if len(axis_result.gyro) > 0 else np.array([])
        
        # Plot setpoint and gyro
        ax.plot(time_plot, sp_plot, color=colors_sp[axis_name], 
                label='Setpoint', linewidth=1.5)
        if len(gyro_plot) > 0:
            ax.plot(time_plot, gyro_plot[:len(time_plot)], 
                    color=colors_gyro[axis_name], label='Gyro', 
                    linewidth=1, alpha=0.8)
        
        ax.set_ylabel(f'{axis_name.upper()} (deg/s)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    ax_list[-1].set_xlabel('Time (ms)')
    
    fig.suptitle(
        f"Setpoint vs Gyro: {Path(result.file_path).name} (Log #{result.log_index})",
        fontsize=12
    )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig
