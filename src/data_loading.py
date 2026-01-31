"""
MALLORN Data Loading Utilities
Data loading and preprocessing functions.
"""

import os
import glob
import pandas as pd
import numpy as np
from .config_loader import get_lsst_config, get_path

EXTINCTION_COEFFS = get_lsst_config()['extinction_coefficients']


def load_lightcurves(meta_df: pd.DataFrame, input_root: str, is_train: bool = True) -> pd.DataFrame:
    """
    Load and merge lightcurve CSVs from the distributed folder structure.
    
    Args:
        meta_df: Metadata DataFrame with object_id column
        input_root: Root directory containing lightcurve folders
        is_train: Whether to load from train or test folder
        
    Returns:
        Merged DataFrame with all lightcurve data
    """
    folder = 'train_lightcurves' if is_train else 'test_lightcurves'
    lc_path = os.path.join(input_root, folder)
    
    lc_files = glob.glob(os.path.join(lc_path, '*.csv'))
    
    all_lcs = []
    for f in lc_files:
        df = pd.read_csv(f)
        all_lcs.append(df)
    
    if not all_lcs:
        return pd.DataFrame()
    
    return pd.concat(all_lcs, ignore_index=True)


def apply_extinction_correction(lc_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply extinction correction to flux values.
    
    Args:
        lc_df: Lightcurve DataFrame with object_id, Filter, Flux, Flux_err columns
        meta_df: Metadata DataFrame with object_id and MWEBV columns
        
    Returns:
        DataFrame with corrected Flux and Flux_err columns
    """
    lc_df = lc_df.merge(meta_df[['object_id', 'MWEBV']], on='object_id', how='left')
    
    def correct_row(row):
        filt = row['Filter']
        if filt in EXTINCTION_COEFFS and pd.notna(row['MWEBV']):
            correction = 10 ** (0.4 * EXTINCTION_COEFFS[filt] * row['MWEBV'])
            row['Flux'] = row['Flux'] * correction
            row['Flux_err'] = row['Flux_err'] * correction
        return row
    
    lc_df = lc_df.apply(correct_row, axis=1)
    lc_df = lc_df.drop(columns=['MWEBV'])
    
    return lc_df
