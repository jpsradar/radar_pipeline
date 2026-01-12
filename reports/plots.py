"""
reports/plots.py

Plotting utilities for the radar pipeline.

What this module does
---------------------
Provides Matplotlib-only plotting helpers used by report generators.
This module is intentionally "thin":
- It does not interpret domain logic
- It only receives x/y arrays and labels and produces files

Design goals
------------
- Deterministic outputs (same data => same plot)
- No seaborn (project rule)
- One plot per figure (project rule)
- No forced colors/styles unless explicitly required
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt


class PlotError(ValueError):
    """Raised when plotting inputs are invalid."""


def plot_xy(
    *,
    x: Sequence[float],
    y: Sequence[float],
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
    ylog: bool = False,
    xlog: bool = False,
    grid: bool = True,
) -> None:
    """
    Create a simple XY line plot and save it to disk.

    Parameters
    ----------
    x, y : sequences
        Data series.
    xlabel, ylabel, title : str
        Plot labels.
    out_path : Path
        Output file path (extension determines format, e.g., .png).
    ylog, xlog : bool
        Enable log scales.
    grid : bool
        Enable grid.

    Returns
    -------
    None
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if x_arr.ndim != 1 or y_arr.ndim != 1:
        raise PlotError("x and y must be 1D sequences")
    if x_arr.size != y_arr.size:
        raise PlotError("x and y must have the same length")
    if x_arr.size == 0:
        raise PlotError("x/y must be non-empty")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(x_arr, y_arr)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if xlog:
        plt.xscale("log")
    if ylog:
        plt.yscale("log")

    if grid:
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_pareto_scatter(
    *,
    x: Sequence[float],
    y: Sequence[float],
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
    highlight_idx: Optional[Iterable[int]] = None,
    grid: bool = True,
) -> None:
    """
    Scatter plot for Pareto results with optional highlighting.

    Notes
    -----
    We do not enforce colors; Matplotlib defaults are used.
    Highlighting is done by plotting the highlighted points on top.

    Parameters
    ----------
    x, y : sequences
        Scatter coordinates.
    highlight_idx : iterable[int] | None
        Indices to emphasize (e.g., Pareto front).
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if x_arr.ndim != 1 or y_arr.ndim != 1:
        raise PlotError("x and y must be 1D sequences")
    if x_arr.size != y_arr.size or x_arr.size == 0:
        raise PlotError("x and y must have same non-zero length")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.scatter(x_arr, y_arr)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if highlight_idx is not None:
        idx = np.asarray(list(highlight_idx), dtype=int)
        idx = idx[(idx >= 0) & (idx < x_arr.size)]
        if idx.size > 0:
            plt.scatter(x_arr[idx], y_arr[idx])

    if grid:
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()