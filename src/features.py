"""
MALLORN Feature Engineering Utilities
Feature extraction functions for TDE classification.
"""

import numpy as np
import pandas as pd
from .config_loader import get_lsst_config

WAVELENGTHS = get_lsst_config()['wavelengths']


def calculate_von_neumann(flux_series: pd.Series) -> float:
    """
    Calculate the Von Neumann ratio for a flux time series.
    Lower values indicate smoother light curves.
    """
    if len(flux_series) < 2:
        return np.nan
    
    flux = flux_series.values
    n = len(flux)
    mean = np.mean(flux)
    
    numerator = np.sum(np.diff(flux) ** 2)
    denominator = np.sum((flux - mean) ** 2)
    
    if denominator == 0:
        return np.nan
    
    return (n / (n - 1)) * (numerator / denominator)


def get_filter_wavelength(filter_name: str) -> float:
    """Get the central wavelength for a filter."""
    return WAVELENGTHS.get(filter_name, np.nan)


def calculate_weighted_mean(values: np.ndarray, errors: np.ndarray) -> float:
    """Calculate inverse-variance weighted mean."""
    weights = 1.0 / (errors ** 2 + 1e-10)
    return np.sum(weights * values) / np.sum(weights)


def calculate_basic_stats(group: pd.DataFrame) -> dict:
    """Calculate basic statistical features for a group of observations."""
    flux = group['Flux'].values
    flux_err = group['Flux_err'].values
    
    if len(flux) == 0:
        return {}
    
    return {
        'mean': np.mean(flux),
        'std': np.std(flux),
        'median': np.median(flux),
        'min': np.min(flux),
        'max': np.max(flux),
        'range': np.max(flux) - np.min(flux),
        'weighted_mean': calculate_weighted_mean(flux, flux_err),
        'n_obs': len(flux)
    }
