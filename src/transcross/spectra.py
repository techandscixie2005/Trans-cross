"""IR resampling and NMR binning utilities."""

from typing import Optional

import numpy as np


def make_ir_grid(min_cm: float, max_cm: float, step_cm: float) -> np.ndarray:
    """Create a fixed IR wavenumber grid.

    Returns 1D array from min_cm to max_cm (inclusive) with step_cm spacing.
    """
    num = int(round((max_cm - min_cm) / step_cm)) + 1
    return np.linspace(min_cm, max_cm, num)


def _clean_xy(x: np.ndarray, y: np.ndarray):
    """Remove NaN/inf, sort by x, remove duplicates by averaging y."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) == 0:
        return x, y
    order = np.argsort(x)
    x, y = x[order], y[order]
    # Average y for duplicate x values
    unique_x, indices = np.unique(x, return_inverse=True)
    if len(unique_x) == len(x):
        return x, y
    unique_y = np.zeros_like(unique_x)
    np.add.at(unique_y, indices, y)
    counts = np.bincount(indices)
    unique_y /= counts
    return unique_x, unique_y


def resample_ir(
    x: np.ndarray, y: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    """Resample an IR spectrum to a fixed wavenumber grid.

    Uses linear interpolation. Values outside the observed x-range
    are filled with the nearest boundary value.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x, y = _clean_xy(x, y)
    if len(x) < 2:
        return np.full_like(grid, y[0] if len(y) > 0 else 0.0)
    interp_y = np.interp(grid, x, y, left=y[0], right=y[-1])
    return interp_y


def normalize_minmax(y: np.ndarray) -> np.ndarray:
    """Per-spectrum min-max normalization to [0, 1].

    If the spectrum is constant, returns zeros.
    """
    y = np.asarray(y, dtype=np.float64)
    ymin, ymax = y.min(), y.max()
    if ymax - ymin < 1e-12:
        return np.zeros_like(y)
    return (y - ymin) / (ymax - ymin)


def make_nmr_grid(
    min_ppm: float, max_ppm: float, step_ppm: float
) -> np.ndarray:
    """Create a fixed NMR chemical shift grid.

    Returns 1D array from min_ppm to max_ppm (inclusive) with step_ppm spacing.
    """
    num = int(round((max_ppm - min_ppm) / step_ppm)) + 1
    return np.linspace(min_ppm, max_ppm, num)


def _find_nearest_bin(peaks: np.ndarray, grid: np.ndarray):
    """Find the nearest grid index for each peak value."""
    grid_min, grid_step = grid[0], grid[1] - grid[0]
    indices = np.round((peaks - grid_min) / grid_step).astype(int)
    return indices


def bin_nmr_peaks(
    peaks: list[float],
    grid: np.ndarray,
    mode: str = "binary",
    sigma: Optional[float] = None,
) -> np.ndarray:
    """Bin NMR peak positions into a fixed chemical shift grid.

    Args:
        peaks: Chemical shift values (ppm).
        grid: Fixed grid from make_nmr_grid().
        mode: "binary" sets bin=1 where a peak falls.
              "gaussian" applies Gaussian smearing with given sigma.
        sigma: Gaussian width in ppm (only used in gaussian mode).

    Returns:
        1D numpy array of shape (len(grid),) with binned values.
    """
    result = np.zeros(len(grid), dtype=np.float32)
    if not peaks:
        return result

    peaks = np.asarray(peaks, dtype=np.float64)
    # Filter peaks outside grid
    grid_min, grid_max = grid[0], grid[-1]
    grid_step = grid[1] - grid[0]
    mask = (peaks >= grid_min) & (peaks <= grid_max)
    peaks = peaks[mask]
    if len(peaks) == 0:
        return result

    if mode == "binary":
        indices = _find_nearest_bin(peaks, grid)
        valid = (indices >= 0) & (indices < len(grid))
        result[indices[valid]] = 1.0

    elif mode == "gaussian":
        if sigma is None:
            sigma = grid_step * 2.0
        for peak in peaks:
            result += np.exp(-0.5 * ((grid - peak) / sigma) ** 2)
        # Clip to 1.0 max per bin
        result = np.minimum(result, 1.0)

    return result
