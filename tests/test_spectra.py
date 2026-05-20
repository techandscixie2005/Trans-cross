"""Tests for IR resampling and NMR binning."""

import numpy as np
import pytest

from src.transcross.spectra import (
    make_ir_grid,
    resample_ir,
    normalize_minmax,
    make_nmr_grid,
    bin_nmr_peaks,
)


class TestMakeIrGrid:
    def test_basic(self):
        grid = make_ir_grid(400.0, 4000.0, 2.0)
        assert grid[0] == 400.0
        assert grid[-1] == 4000.0
        assert len(grid) == 1801

    def test_inclusive(self):
        grid = make_ir_grid(0.0, 10.0, 2.0)
        assert grid[0] == 0.0
        assert grid[-1] == 10.0
        assert len(grid) == 6


class TestResampleIr:
    def test_basic_interpolation(self):
        grid = np.array([500.0, 550.0, 600.0])
        x = np.array([500.0, 600.0])
        y = np.array([10.0, 20.0])
        result = resample_ir(x, y, grid)
        assert len(result) == 3
        assert result[0] == 10.0  # exact match
        assert result[2] == 20.0  # exact match
        assert 10.0 < result[1] < 20.0  # interpolated

    def test_outside_range_fills_boundary(self):
        grid = np.array([400.0, 500.0, 600.0])
        x = np.array([450.0, 550.0])
        y = np.array([10.0, 20.0])
        result = resample_ir(x, y, grid)
        assert result[0] == 10.0  # left fill
        assert result[2] == 20.0  # right fill

    def test_nan_removal(self):
        grid = np.array([500.0, 600.0])
        x = np.array([500.0, np.nan, 600.0])
        y = np.array([10.0, 999.0, 20.0])
        result = resample_ir(x, y, grid)
        assert not np.any(np.isnan(result))
        assert len(result) == 2

    def test_single_point(self):
        grid = np.array([500.0, 600.0])
        x = np.array([550.0])
        y = np.array([15.0])
        result = resample_ir(x, y, grid)
        assert len(result) == 2
        # Single point fills the entire grid with that value
        assert np.allclose(result, 15.0)


class TestNormalizeMinmax:
    def test_basic(self):
        y = np.array([10.0, 20.0, 30.0])
        result = normalize_minmax(y)
        assert np.isclose(result.min(), 0.0)
        assert np.isclose(result.max(), 1.0)

    def test_constant(self):
        y = np.array([5.0, 5.0, 5.0])
        result = normalize_minmax(y)
        assert np.all(result == 0.0)

    def test_already_normalized(self):
        y = np.array([0.0, 0.5, 1.0])
        result = normalize_minmax(y)
        assert np.isclose(result[0], 0.0)
        assert np.isclose(result[-1], 1.0)


class TestMakeNmrGrid:
    def test_1h_grid(self):
        grid = make_nmr_grid(0.0, 15.0, 0.01)
        assert grid[0] == 0.0
        assert grid[-1] == 15.0
        assert len(grid) == 1501

    def test_13c_grid(self):
        grid = make_nmr_grid(0.0, 220.0, 0.1)
        assert grid[0] == 0.0
        assert grid[-1] == 220.0
        assert len(grid) == 2201


class TestBinNmrPeaks:
    def test_binary_mode(self):
        grid = make_nmr_grid(0.0, 10.0, 1.0)  # 11 bins: 0,1,...,10
        peaks = [1.2, 3.7, 5.0]
        result = bin_nmr_peaks(peaks, grid, mode="binary")
        assert len(result) == 11
        assert result[1] == 1.0  # peak at 1.2 → bin 1
        assert result[4] == 1.0  # peak at 3.7 → bin 4
        assert result[5] == 1.0  # peak at 5.0 → bin 5
        assert result.sum() == 3.0

    def test_empty_peaks(self):
        grid = make_nmr_grid(0.0, 10.0, 1.0)
        result = bin_nmr_peaks([], grid, mode="binary")
        assert result.sum() == 0.0

    def test_peaks_outside_grid(self):
        grid = make_nmr_grid(0.0, 10.0, 1.0)
        peaks = [-5.0, 15.0]  # both outside
        result = bin_nmr_peaks(peaks, grid, mode="binary")
        assert result.sum() == 0.0

    def test_gaussian_mode(self):
        grid = make_nmr_grid(0.0, 10.0, 1.0)
        peaks = [5.0]
        result = bin_nmr_peaks(peaks, grid, mode="gaussian", sigma=0.5)
        assert len(result) == 11
        # The peak at 5.0 should be highest
        assert np.argmax(result) == 5
        assert result[5] > 0.0
        # Neighbors also get some signal
        assert result[4] > 0.0
        assert result[6] > 0.0
        # Should be clamped to 1.0 max
        assert result.max() <= 1.0

    def test_binary_mode_default(self):
        grid = make_nmr_grid(0.0, 5.0, 1.0)
        peaks = [2.0]
        result = bin_nmr_peaks(peaks, grid)
        assert result[2] == 1.0
