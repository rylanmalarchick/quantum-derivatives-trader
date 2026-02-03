"""
Visualization utilities for options pricing and PINN training.

Provides functions for plotting option price surfaces, Greeks,
training history, and comparing PINN results with analytical solutions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D


def plot_surface(
    S: np.ndarray,
    t: np.ndarray,
    V: np.ndarray,
    title: str = "Option Price Surface",
    xlabel: str = "Spot Price ($S$)",
    ylabel: str = "Time to Expiration ($\\tau$)",
    zlabel: str = "Option Value ($V$)",
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (10, 8),
    elevation: float = 30.0,
    azimuth: float = -60.0,
) -> Figure:
    """Create a 3D surface plot of option prices.
    
    Visualizes how option prices vary with spot price and time to expiration.
    
    Args:
        S: 1D array of spot prices or 2D meshgrid.
        t: 1D array of time values or 2D meshgrid.
        V: 2D array of option values with shape (len(t), len(S)) if S, t are 1D.
        title: Plot title.
        xlabel: Label for x-axis (spot price).
        ylabel: Label for y-axis (time).
        zlabel: Label for z-axis (option value).
        cmap: Colormap name.
        figsize: Figure size in inches.
        elevation: Viewing elevation angle.
        azimuth: Viewing azimuth angle.
    
    Returns:
        matplotlib Figure object.
    
    Example:
        >>> S = np.linspace(50, 150, 50)
        >>> t = np.linspace(0, 1, 50)
        >>> V = compute_option_prices(S, t)
        >>> fig = plot_surface(S, t, V, title="Call Option Prices")
        >>> plt.show()
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid if 1D arrays provided
    if S.ndim == 1 and t.ndim == 1:
        S_mesh, t_mesh = np.meshgrid(S, t)
    else:
        S_mesh, t_mesh = S, t
    
    # Plot surface
    surf = ax.plot_surface(
        S_mesh, t_mesh, V,
        cmap=cmap,
        edgecolor='none',
        alpha=0.9,
        antialiased=True,
    )
    
    ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
    ax.set_zlabel(zlabel, fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.view_init(elev=elevation, azim=azimuth)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=zlabel)
    
    plt.tight_layout()
    return fig


def plot_greeks(
    S: np.ndarray,
    greeks_dict: Dict[str, np.ndarray],
    title: str = "Option Greeks vs Spot Price",
    figsize: Tuple[int, int] = (14, 10),
    K: Optional[float] = None,
) -> Figure:
    """Plot option Greeks as functions of spot price.
    
    Creates a multi-panel plot showing how each Greek varies with the
    underlying asset price.
    
    Args:
        S: Array of spot prices.
        greeks_dict: Dictionary mapping Greek names to their values.
            Expected keys: 'delta', 'gamma', 'theta', 'vega', 'rho'.
        title: Overall plot title.
        figsize: Figure size in inches.
        K: Optional strike price to mark with vertical line.
    
    Returns:
        matplotlib Figure object.
    
    Example:
        >>> S = np.linspace(50, 150, 100)
        >>> greeks = compute_greeks_batch(S, K=100, T=1.0, r=0.05, sigma=0.2)
        >>> fig = plot_greeks(S, greeks, title="Call Option Greeks", K=100)
        >>> plt.show()
    """
    greek_info = {
        'delta': {'ylabel': r'$\Delta$', 'color': '#2E86AB', 'title': 'Delta'},
        'gamma': {'ylabel': r'$\Gamma$', 'color': '#A23B72', 'title': 'Gamma'},
        'theta': {'ylabel': r'$\Theta$', 'color': '#F18F01', 'title': 'Theta'},
        'vega': {'ylabel': r'$\mathcal{V}$', 'color': '#C73E1D', 'title': 'Vega'},
        'rho': {'ylabel': r'$\rho$', 'color': '#6B2D5C', 'title': 'Rho'},
    }
    
    # Filter to available Greeks
    available = [g for g in greek_info if g in greeks_dict]
    n_greeks = len(available)
    
    if n_greeks == 0:
        raise ValueError("No valid Greeks found in greeks_dict")
    
    # Determine layout
    if n_greeks <= 3:
        nrows, ncols = 1, n_greeks
    elif n_greeks <= 4:
        nrows, ncols = 2, 2
    else:
        nrows, ncols = 2, 3
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()
    
    for idx, greek_name in enumerate(available):
        ax = axes[idx]
        info = greek_info[greek_name]
        values = greeks_dict[greek_name]
        
        ax.plot(S, values, color=info['color'], linewidth=2, label=info['title'])
        ax.set_xlabel('Spot Price ($S$)', fontsize=11)
        ax.set_ylabel(info['ylabel'], fontsize=12)
        ax.set_title(info['title'], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        if K is not None:
            ax.axvline(x=K, color='red', linestyle=':', alpha=0.7, label=f'K={K}')
            ax.legend(fontsize=9)
    
    # Hide unused subplots
    for idx in range(n_greeks, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "PINN Training Loss History",
    figsize: Tuple[int, int] = (12, 6),
    log_scale: bool = True,
) -> Figure:
    """Plot PINN training loss curves.
    
    Visualizes the evolution of different loss components during PINN training.
    
    Args:
        history: Dictionary containing loss arrays with expected keys:
            - 'pde': PDE residual loss
            - 'bc': Boundary condition loss
            - 'ic': Initial condition loss
            - 'total': Total combined loss
        title: Plot title.
        figsize: Figure size in inches.
        log_scale: Whether to use logarithmic y-axis scale.
    
    Returns:
        matplotlib Figure object.
    
    Example:
        >>> history = {'pde': [...], 'bc': [...], 'ic': [...], 'total': [...]}
        >>> fig = plot_training_history(history)
        >>> plt.show()
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    loss_styles = {
        'pde': {'color': '#2E86AB', 'linestyle': '-', 'label': 'PDE Residual'},
        'bc': {'color': '#A23B72', 'linestyle': '--', 'label': 'Boundary Condition'},
        'ic': {'color': '#F18F01', 'linestyle': '-.', 'label': 'Initial Condition'},
        'total': {'color': '#1B1B1E', 'linestyle': '-', 'label': 'Total Loss', 'linewidth': 2.5},
    }
    
    # Left panel: All components
    for loss_name, values in history.items():
        if loss_name in loss_styles:
            style = loss_styles[loss_name]
            epochs = np.arange(len(values))
            ax1.plot(
                epochs, values,
                color=style['color'],
                linestyle=style['linestyle'],
                linewidth=style.get('linewidth', 1.5),
                label=style['label'],
                alpha=0.9,
            )
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Components', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    if log_scale:
        ax1.set_yscale('log')
    
    # Right panel: Total loss with smoothing
    if 'total' in history:
        total_loss = np.array(history['total'])
        epochs = np.arange(len(total_loss))
        
        # Raw loss
        ax2.plot(epochs, total_loss, color='#CCCCCC', alpha=0.5, linewidth=0.8, label='Raw')
        
        # Smoothed loss (exponential moving average)
        window = min(50, len(total_loss) // 10)
        if window > 1:
            smoothed = _exponential_moving_average(total_loss, span=window)
            ax2.plot(epochs, smoothed, color='#1B1B1E', linewidth=2, label=f'EMA (span={window})')
        
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Total Loss', fontsize=12)
        ax2.set_title('Total Loss (Smoothed)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        if log_scale:
            ax2.set_yscale('log')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_comparison(
    S: np.ndarray,
    V_pinn: np.ndarray,
    V_analytical: np.ndarray,
    title: str = "PINN vs Analytical Solution",
    figsize: Tuple[int, int] = (14, 5),
) -> Figure:
    """Compare PINN predictions with analytical solutions.
    
    Creates a side-by-side comparison showing both solutions overlaid
    and their absolute/relative errors.
    
    Args:
        S: Array of spot prices.
        V_pinn: PINN-predicted option values.
        V_analytical: Analytical (Black-Scholes) option values.
        title: Overall plot title.
        figsize: Figure size in inches.
    
    Returns:
        matplotlib Figure object.
    
    Example:
        >>> S = np.linspace(50, 150, 100)
        >>> V_pinn = pinn_model.predict(S)
        >>> V_analytical = black_scholes(S, K, T, r, sigma)
        >>> fig = plot_comparison(S, V_pinn, V_analytical)
        >>> plt.show()
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # Panel 1: Overlay comparison
    ax1.plot(S, V_analytical, 'b-', linewidth=2, label='Analytical (BS)', alpha=0.9)
    ax1.plot(S, V_pinn, 'r--', linewidth=2, label='PINN', alpha=0.9)
    ax1.set_xlabel('Spot Price ($S$)', fontsize=12)
    ax1.set_ylabel('Option Value ($V$)', fontsize=12)
    ax1.set_title('Price Comparison', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Absolute error
    abs_error = np.abs(V_pinn - V_analytical)
    ax2.plot(S, abs_error, 'g-', linewidth=2)
    ax2.fill_between(S, 0, abs_error, alpha=0.3, color='green')
    ax2.set_xlabel('Spot Price ($S$)', fontsize=12)
    ax2.set_ylabel('Absolute Error', fontsize=12)
    ax2.set_title(f'Absolute Error (max: {abs_error.max():.4f})', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Relative error
    # Avoid division by zero for near-zero analytical values
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_error = np.abs(abs_error / V_analytical) * 100
        rel_error = np.nan_to_num(rel_error, nan=0.0, posinf=0.0, neginf=0.0)
    
    ax3.plot(S, rel_error, 'm-', linewidth=2)
    ax3.fill_between(S, 0, rel_error, alpha=0.3, color='magenta')
    ax3.set_xlabel('Spot Price ($S$)', fontsize=12)
    ax3.set_ylabel('Relative Error (%)', fontsize=12)
    mean_rel_error = np.mean(rel_error[V_analytical > 0.01])  # Exclude near-zero regions
    ax3.set_title(f'Relative Error (mean: {mean_rel_error:.2f}%)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def plot_pde_residual(
    S: np.ndarray,
    t: np.ndarray,
    residual: np.ndarray,
    title: str = "PDE Residual",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "RdBu_r",
) -> Figure:
    """Plot the PDE residual as a heatmap.
    
    Visualizes where the PINN solution violates the PDE most strongly.
    
    Args:
        S: 1D array of spot prices.
        t: 1D array of time values.
        residual: 2D array of PDE residuals.
        title: Plot title.
        figsize: Figure size.
        cmap: Colormap (diverging recommended).
    
    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Center colormap at zero
    vmax = np.abs(residual).max()
    vmin = -vmax
    
    im = ax.pcolormesh(S, t, residual, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    
    ax.set_xlabel('Spot Price ($S$)', fontsize=12)
    ax.set_ylabel('Time to Expiration ($\\tau$)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    cbar = fig.colorbar(im, ax=ax, label='Residual')
    
    plt.tight_layout()
    return fig


def plot_convergence_comparison(
    histories: Dict[str, Dict[str, List[float]]],
    metric: str = "total",
    title: str = "Training Convergence Comparison",
    figsize: Tuple[int, int] = (10, 6),
) -> Figure:
    """Compare training convergence across multiple experiments.
    
    Args:
        histories: Dictionary mapping experiment names to their history dicts.
        metric: Which loss metric to compare ('total', 'pde', 'bc', 'ic').
        title: Plot title.
        figsize: Figure size.
    
    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    
    for (name, history), color in zip(histories.items(), colors):
        if metric in history:
            values = history[metric]
            epochs = np.arange(len(values))
            ax.plot(epochs, values, label=name, color=color, linewidth=1.5)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(f'{metric.capitalize()} Loss', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def _exponential_moving_average(data: np.ndarray, span: int) -> np.ndarray:
    """Compute exponential moving average.
    
    Args:
        data: Input array.
        span: Span for EMA calculation.
    
    Returns:
        Smoothed array.
    """
    alpha = 2 / (span + 1)
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema


def set_publication_style() -> None:
    """Set matplotlib style for publication-quality figures.
    
    Configures fonts, sizes, and other parameters suitable for
    academic publications.
    """
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 1.5,
    })
