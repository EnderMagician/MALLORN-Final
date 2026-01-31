"""
Legacy Data Processing Module for MALLORN Challenge.
Contains logic extracted from data-process-mallorn.ipynb to ensure 100% reproducibility.
"""

import os
import glob
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from src.config_loader import load_config, get_project_root

# Load config to get GP constants
config = load_config()

class GPConfig:
    """Gaussian Process Hyperparameters & LSST Filter Definitions"""
    
    # LSST Central Wavelengths (in nanometers)
    WAVELENGTHS = config['lsst']['wavelengths']
    
    # 2D Kernel Length Scales
    LENGTH_SCALE = config['gp']['length_scale']
    
    # Optimization Bounds
    LENGTH_SCALE_BOUNDS = config['gp']['length_scale_bounds']
    
    # Noise Level
    NOISE_LEVEL = config['gp']['noise_level']
    NOISE_LEVEL_BOUNDS = config['gp']['noise_level_bounds']

def load_and_merge_data(meta_df, input_root, is_train=True):
    """
    Iterates through the 'split' column in metadata to load distributed lightcurves.
    """
    all_lightcurves = []
    
    # Get unique splits (e.g., 'Split_1', 'Split_2')
    unique_splits = meta_df['split'].unique()
    
    print(f"Loading lightcurves from {len(unique_splits)} splits...")
    
    for split_name in tqdm(unique_splits):
        # Construct path: Root / Split_Name / [train/test]_full_lightcurves.csv
        file_prefix = "train" if is_train else "test"
        
        # Try finding the file in the directory structure
        path = f"{input_root}/{split_name}/{file_prefix}_full_lightcurves.csv"
        
        if not os.path.exists(path):
            # Fallback: sometimes folder names differ slightly
            possible_paths = glob.glob(f"{input_root}/*{split_name}*/*{file_prefix}_full_lightcurves.csv")
            if possible_paths:
                path = possible_paths[0]
            else:
                print(f"Warning: Could not find file for split {split_name}")
                continue
                
        # Load only necessary cols to save memory
        chunk = pd.read_csv(path, usecols=['object_id', 'Time (MJD)', 'Flux', 'Flux_err', 'Filter'])
        all_lightcurves.append(chunk)
        
    if not all_lightcurves:
        raise ValueError("No lightcurve data loaded!")
        
    full_lc = pd.concat(all_lightcurves, ignore_index=True)
    
    return full_lc

def apply_extinction_correction(lc_df, meta_df):
    """
    Corrects Flux using EBV from metadata.
    Flux_corr = Flux * 10^(0.4 * R_lambda * EBV)
    """
    print("✨ Applying Extinction Correction...")
    
    # Merge EBV into lightcurves
    lc_df = lc_df.merge(meta_df[['object_id', 'EBV']], on='object_id', how='left')
    
    # Map coefficients
    extinction_coeffs = config['lsst']['extinction_coefficients']
    lc_df['R_lambda'] = lc_df['Filter'].map(extinction_coeffs)
    
    # Calculate correction factor
    # A_lambda = R_lambda * EBV
    # Correction = 10^(0.4 * A_lambda)
    lc_df['A_lambda'] = lc_df['R_lambda'] * lc_df['EBV']
    correction_factor = np.power(10, 0.4 * lc_df['A_lambda'])
    
    # Apply correction
    lc_df['Flux'] = lc_df['Flux'] * correction_factor
    
    # Clean up
    lc_df.drop(columns=['EBV', 'R_lambda', 'A_lambda'], inplace=True)
    return lc_df

def calculate_von_neumann(flux_series):
    """
    von Neumann Ratio: Mean Squared Successive Difference / Variance
    """
    if len(flux_series) < 2: return 0.0
    return np.mean(np.diff(flux_series)**2) / (np.var(flux_series) + 1e-9)

def duration(x):
    return x.max() - x.min()

def _create_nan_features(oid):
    """Create dictionary with all features set to NaN for failed fits"""
    features = {'object_id': oid}
    
    # Add all possible feature names with NaN values
    feature_names = [
        'gp_log_likelihood', 'gp_time_scale', 'gp_wavelength_scale', 'gp_noise_level',
        'gp_time_scale_normalized', 'gp_wavelength_scale_normalized',
        'gp_residual_mean', 'gp_residual_std', 'gp_residual_median', 'gp_residual_skew',
        'gp_residual_kurtosis', 'gp_max_deviation', 'gp_mae', 'gp_mse', 
        'gp_normalized_residual_std', 'gp_overall_mean', 'gp_overall_std',
        'gp_overall_range', 'gp_overall_uncertainty_mean', 'gp_overall_uncertainty_std',
        'gp_spectral_slope', 'gp_spectral_curvature'
    ]
    
    for band in ['u', 'g', 'r', 'i', 'z', 'y']:
        feature_names.extend([
            f'gp_smoothed_mean_{band}', f'gp_smoothed_max_{band}', f'gp_smoothed_min_{band}',
            f'gp_smoothed_range_{band}', f'gp_smoothed_std_{band}',
            f'gp_uncertainty_mean_{band}', f'gp_uncertainty_max_{band}', f'gp_uncertainty_std_{band}',
            f'gp_deriv_mean_{band}', f'gp_deriv_std_{band}', f'gp_deriv_max_{band}', f'gp_deriv_min_{band}',
            f'gp_peak_time_{band}', f'gp_peak_value_{band}',
            f'gp_rise_duration_{band}', f'gp_rise_rate_{band}',
            f'gp_decline_duration_{band}', f'gp_decline_rate_{band}',
            f'gp_band_mse_{band}', f'gp_band_mae_{band}'
        ])
    
    for time_label in ['early', 'mid', 'late']:
        feature_names.extend([
            f'gp_color_ug_{time_label}', f'gp_color_gr_{time_label}',
            f'gp_color_ri_{time_label}', f'gp_color_iz_{time_label}'
        ])
    
    feature_names.extend(['gp_color_ug_evolution', 'gp_color_gr_evolution'])
    
    # Cross-band correlations
    bands = ['u', 'g', 'r', 'i', 'z', 'y']
    for i, b1 in enumerate(bands):
        for b2 in bands[i+1:]:
            feature_names.append(f'gp_corr_{b1}{b2}')
    
    for name in feature_names:
        features[name] = np.nan
    
    return features

def extract_gp_features_all_bands(obj_data, oid, seed):
    """
    Extract comprehensive 2D Gaussian Process features using ALL bands together.
    This exploits the full 2D (Time, Wavelength) modeling capability.
    """
    
    # Map filters to wavelengths
    obj_data = obj_data.copy()
    obj_data['Wavelength'] = obj_data['Filter'].map(GPConfig.WAVELENGTHS)
    
    # Clean data - remove NaN/Inf
    mask = np.isfinite(obj_data['Flux']) & np.isfinite(obj_data['Flux_err'])
    obj_data = obj_data[mask]
    
    # Initialize all features with NaN
    gp_feats = {'object_id': oid}
    
    # Skip if too few points
    if len(obj_data) < 5:
        return _create_nan_features(oid)
    
    try:
        # ===== FIT 2D GP ON ALL BANDS SIMULTANEOUSLY =====
        X = obj_data[['Time (MJD)', 'Wavelength']].values
        y = obj_data['Flux'].values
        y_err = obj_data['Flux_err'].values
        y_mean = np.mean(y)
        y_std = np.std(y)
        
        # Define kernel
        kernel = Matern(length_scale=GPConfig.LENGTH_SCALE, nu=1.5) + \
                 WhiteKernel(noise_level=GPConfig.NOISE_LEVEL)
        
        # Fit GP on ALL data points (all bands together)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=(y_err**2 + 1e-5),
            normalize_y=False,
            random_state=seed
        )
        gp.fit(X, y - y_mean)
        
        # ========================================
        # FEATURE GROUP 1: KERNEL HYPERPARAMETERS
        # ========================================
        learned_params = gp.kernel_.theta
        gp_feats['gp_log_likelihood'] = gp.log_marginal_likelihood_value_
        gp_feats['gp_time_scale'] = np.exp(learned_params[0])
        gp_feats['gp_wavelength_scale'] = np.exp(learned_params[1])
        gp_feats['gp_noise_level'] = np.exp(learned_params[2])
        
        # Normalized scales (relative to data range)
        time_range = X[:, 0].max() - X[:, 0].min()
        wavelength_range = X[:, 1].max() - X[:, 1].min()
        gp_feats['gp_time_scale_normalized'] = gp_feats['gp_time_scale'] / (time_range + 1e-9)
        gp_feats['gp_wavelength_scale_normalized'] = gp_feats['gp_wavelength_scale'] / (wavelength_range + 1e-9)
        
        # ========================================
        # FEATURE GROUP 2: RESIDUAL STATISTICS
        # ========================================
        y_pred_train = gp.predict(X) + y_mean
        residuals = y - y_pred_train
        normalized_residuals = residuals / (y_std + 1e-9)
        
        gp_feats['gp_residual_mean'] = np.mean(residuals)
        gp_feats['gp_residual_std'] = np.std(residuals)
        gp_feats['gp_residual_median'] = np.median(residuals)
        gp_feats['gp_residual_skew'] = pd.Series(residuals).skew()
        gp_feats['gp_residual_kurtosis'] = pd.Series(residuals).kurtosis()
        gp_feats['gp_max_deviation'] = np.max(np.abs(residuals))
        gp_feats['gp_mae'] = np.mean(np.abs(residuals))
        gp_feats['gp_mse'] = np.mean(residuals**2)
        gp_feats['gp_normalized_residual_std'] = np.std(normalized_residuals)
        
        # ========================================
        # FEATURE GROUP 3: DENSE GRID PREDICTIONS
        # ========================================
        time_min, time_max = X[:, 0].min(), X[:, 0].max()
        time_grid = np.linspace(time_min, time_max, 30)
        
        band_predictions = {}
        band_uncertainties = {}
        band_derivatives = {}
        
        for band, wavelength in GPConfig.WAVELENGTHS.items():
            X_pred = np.column_stack([time_grid, np.full(len(time_grid), wavelength)])
            y_pred, y_std = gp.predict(X_pred, return_std=True)
            y_pred = y_pred + y_mean
            
            band_predictions[band] = y_pred
            band_uncertainties[band] = y_std
            
            # Time derivative (rate of change)
            band_derivatives[band] = np.gradient(y_pred, time_grid)
        
        # ========================================
        # FEATURE GROUP 4: PER-BAND STATISTICS
        # ========================================
        for band in ['u', 'g', 'r', 'i', 'z', 'y']:
            if band not in band_predictions:
                continue
            
            pred = band_predictions[band]
            uncert = band_uncertainties[band]
            deriv = band_derivatives[band]
            
            # Smoothed statistics
            gp_feats[f'gp_smoothed_mean_{band}'] = np.mean(pred)
            gp_feats[f'gp_smoothed_max_{band}'] = np.max(pred)
            gp_feats[f'gp_smoothed_min_{band}'] = np.min(pred)
            gp_feats[f'gp_smoothed_range_{band}'] = np.max(pred) - np.min(pred)
            gp_feats[f'gp_smoothed_std_{band}'] = np.std(pred)
            
            # Uncertainty metrics
            gp_feats[f'gp_uncertainty_mean_{band}'] = np.mean(uncert)
            gp_feats[f'gp_uncertainty_max_{band}'] = np.max(uncert)
            gp_feats[f'gp_uncertainty_std_{band}'] = np.std(uncert)
            
            # Derivative features (variability characteristics)
            gp_feats[f'gp_deriv_mean_{band}'] = np.mean(deriv)
            gp_feats[f'gp_deriv_std_{band}'] = np.std(deriv)
            gp_feats[f'gp_deriv_max_{band}'] = np.max(deriv)
            gp_feats[f'gp_deriv_min_{band}'] = np.min(deriv)
            
            # Peak timing
            peak_idx = np.argmax(pred)
            gp_feats[f'gp_peak_time_{band}'] = time_grid[peak_idx]
            gp_feats[f'gp_peak_value_{band}'] = pred[peak_idx]
        
        # ========================================
        # FEATURE GROUP 5: COLOR FEATURES
        # ========================================
        # Colors at multiple time points (early, middle, late)
        time_points = [0, len(time_grid)//2, len(time_grid)-1]
        time_labels = ['early', 'mid', 'late']
        
        for time_idx, time_label in zip(time_points, time_labels):
            if 'u' in band_predictions and 'g' in band_predictions:
                gp_feats[f'gp_color_ug_{time_label}'] = band_predictions['u'][time_idx] - band_predictions['g'][time_idx]
            if 'g' in band_predictions and 'r' in band_predictions:
                gp_feats[f'gp_color_gr_{time_label}'] = band_predictions['g'][time_idx] - band_predictions['r'][time_idx]
            if 'r' in band_predictions and 'i' in band_predictions:
                gp_feats[f'gp_color_ri_{time_label}'] = band_predictions['r'][time_idx] - band_predictions['i'][time_idx]
            if 'i' in band_predictions and 'z' in band_predictions:
                gp_feats[f'gp_color_iz_{time_label}'] = band_predictions['i'][time_idx] - band_predictions['z'][time_idx]
        
        # Color evolution (how colors change over time)
        if 'u' in band_predictions and 'g' in band_predictions:
            gp_feats['gp_color_ug_evolution'] = (band_predictions['u'][-1] - band_predictions['g'][-1]) - \
                                                 (band_predictions['u'][0] - band_predictions['g'][0])
        if 'g' in band_predictions and 'r' in band_predictions:
            gp_feats['gp_color_gr_evolution'] = (band_predictions['g'][-1] - band_predictions['r'][-1]) - \
                                                 (band_predictions['g'][0] - band_predictions['r'][0])
        
        # ========================================
        # FEATURE GROUP 6: CROSS-BAND CORRELATIONS
        # ========================================
        # Correlation between band lightcurves
        available_bands = list(band_predictions.keys())
        for i, band1 in enumerate(available_bands):
            for band2 in available_bands[i+1:]:
                corr = np.corrcoef(band_predictions[band1], band_predictions[band2])[0, 1]
                gp_feats[f'gp_corr_{band1}{band2}'] = corr
        
        # ========================================
        # FEATURE GROUP 7: TEMPORAL FEATURES
        # ========================================
        # Using all bands combined
        all_predictions = np.concatenate([band_predictions[b] for b in band_predictions.keys()])
        
        gp_feats['gp_overall_mean'] = np.mean(all_predictions)
        gp_feats['gp_overall_std'] = np.std(all_predictions)
        gp_feats['gp_overall_range'] = np.max(all_predictions) - np.min(all_predictions)
        
        # Average uncertainty across all bands
        all_uncertainties = np.concatenate([band_uncertainties[b] for b in band_uncertainties.keys()])
        gp_feats['gp_overall_uncertainty_mean'] = np.mean(all_uncertainties)
        gp_feats['gp_overall_uncertainty_std'] = np.std(all_uncertainties)
        
        # ========================================
        # FEATURE GROUP 8: RISE/DECLINE CHARACTERISTICS
        # ========================================
        # Analyze each band for rise/decline patterns
        rise_durations = {}
        decline_durations = {}
        peak_times = {}
        
        for band in band_predictions.keys():
            pred = band_predictions[band]
            peak_idx = np.argmax(pred)
            peak_times[band] = time_grid[peak_idx]
            
            # Rise metrics (before peak)
            if peak_idx > 0:
                rise_values = pred[:peak_idx+1]
                rise_times = time_grid[:peak_idx+1]
                gp_feats[f'gp_rise_duration_{band}'] = rise_times[-1] - rise_times[0]
                gp_feats[f'gp_rise_rate_{band}'] = (rise_values[-1] - rise_values[0]) / (gp_feats[f'gp_rise_duration_{band}'] + 1e-9)
                rise_durations[band] = gp_feats[f'gp_rise_duration_{band}']
            
            # Decline metrics (after peak)
            if peak_idx < len(pred) - 1:
                decline_values = pred[peak_idx:]
                decline_times = time_grid[peak_idx:]
                gp_feats[f'gp_decline_duration_{band}'] = decline_times[-1] - decline_times[0]
                gp_feats[f'gp_decline_rate_{band}'] = (decline_values[-1] - decline_values[0]) / (gp_feats[f'gp_decline_duration_{band}'] + 1e-9)
                decline_durations[band] = gp_feats[f'gp_decline_duration_{band}']
        
        # CRITICAL TDE FEATURE: Rise/Decline Asymmetry
        # TDEs have very asymmetric lightcurves (decline >> rise)
        for band in band_predictions.keys():
            if band in rise_durations and band in decline_durations:
                gp_feats[f'gp_asymmetry_ratio_{band}'] = decline_durations[band] / (rise_durations[band] + 1e-9)
        
        # Overall asymmetry (average across bands)
        asymmetry_ratios = [gp_feats[f'gp_asymmetry_ratio_{b}'] for b in band_predictions.keys() 
                           if f'gp_asymmetry_ratio_{b}' in gp_feats]
        if asymmetry_ratios:
            gp_feats['gp_mean_asymmetry_ratio'] = np.mean(asymmetry_ratios)
        
        # CRITICAL TDE FEATURE: Peak Timing Across Bands
        # TDEs: bluer bands peak earlier than redder bands
        if len(peak_times) >= 2:
            if 'u' in peak_times and 'r' in peak_times:
                gp_feats['gp_peak_timing_diff_ur'] = peak_times['u'] - peak_times['r']
            if 'g' in peak_times and 'i' in peak_times:
                gp_feats['gp_peak_timing_diff_gi'] = peak_times['g'] - peak_times['i']
            if 'r' in peak_times and 'z' in peak_times:
                gp_feats['gp_peak_timing_diff_rz'] = peak_times['r'] - peak_times['z']
            
            # Measure spread of peak times (AGNs have synchronized "peaks", TDEs staggered)
            gp_feats['gp_peak_timing_spread'] = np.std(list(peak_times.values()))
        
        # ========================================
        # FEATURE GROUP 9: SPECTRAL FEATURES
        # ========================================
        # How does flux vary with wavelength at fixed times?
        wavelengths = np.array([GPConfig.WAVELENGTHS[b] for b in band_predictions.keys()])
        
        # Spectral slope at peak time
        peak_fluxes = np.array([band_predictions[b][len(time_grid)//2] for b in band_predictions.keys()])
        if len(wavelengths) >= 2:
            spectral_slope = np.polyfit(wavelengths, peak_fluxes, 1)[0]
            gp_feats['gp_spectral_slope'] = spectral_slope
            gp_feats['gp_spectral_curvature'] = np.std(peak_fluxes - np.polyval([spectral_slope, 0], wavelengths))
        
        # ========================================
        # FEATURE GROUP 10: MODEL QUALITY METRICS
        # ========================================
        # How well does the GP fit each band?
        for band in band_predictions.keys():
            band_mask = obj_data['Filter'] == band
            if band_mask.sum() > 0:
                band_data = obj_data[band_mask]
                X_band = band_data[['Time (MJD)', 'Wavelength']].values
                y_band = band_data['Flux'].values
                y_pred_band = gp.predict(X_band) + y_mean
                
                residuals_band = y_band - y_pred_band
                gp_feats[f'gp_band_mse_{band}'] = np.mean(residuals_band**2)
                gp_feats[f'gp_band_mae_{band}'] = np.mean(np.abs(residuals_band))
        
        # ========================================
        # FEATURE GROUP 11: VARIABILITY METRICS (AGN vs TDE/SN)
        # ========================================
        # AGNs show stochastic variability, TDEs/SNe are smooth
        
        # Normalized excess variance (high for AGN, low for TDE/SN)
        all_predictions_flat = np.concatenate([band_predictions[b] for b in band_predictions.keys()])
        all_uncertainties_flat = np.concatenate([band_uncertainties[b] for b in band_uncertainties.keys()])
        
        mean_flux = np.mean(all_predictions_flat)
        var_flux = np.var(all_predictions_flat)
        mean_err_sq = np.mean(all_uncertainties_flat**2)
        
        excess_variance = (var_flux - mean_err_sq) / (mean_flux**2 + 1e-9)
        gp_feats['gp_excess_variance'] = max(0, excess_variance)  # Clip negative values
        
        # Structure function at multiple timescales (AGN-specific)
        for band in band_predictions.keys():
            pred = band_predictions[band]
            # Calculate structure function at lag = 1, 5, 10 points
            for lag in [1, 5, 10]:
                if len(pred) > lag:
                    sf = np.mean((pred[lag:] - pred[:-lag])**2)
                    gp_feats[f'gp_structure_function_lag{lag}_{band}'] = sf
        
        # ========================================
        # FEATURE GROUP 12: SECOND DERIVATIVE (CURVATURE)
        # ========================================
        # TDEs have smooth curvature, SNe exponential, AGNs irregular
        for band in band_predictions.keys():
            pred = band_predictions[band]
            second_deriv = np.gradient(np.gradient(pred, time_grid), time_grid)
            
            gp_feats[f'gp_curvature_mean_{band}'] = np.mean(second_deriv)
            gp_feats[f'gp_curvature_std_{band}'] = np.std(second_deriv)
            gp_feats[f'gp_curvature_max_{band}'] = np.max(np.abs(second_deriv))
        
        # ========================================
        # FEATURE GROUP 13: COLOR-MAGNITUDE CORRELATION
        # ========================================
        # How do colors correlate with brightness?
        if 'g' in band_predictions and 'r' in band_predictions:
            color_gr = band_predictions['g'] - band_predictions['r']
            magnitude_r = band_predictions['r']
            
            corr_color_mag = np.corrcoef(color_gr, magnitude_r)[0, 1]
            gp_feats['gp_color_magnitude_corr_gr'] = corr_color_mag
        
        if 'u' in band_predictions and 'g' in band_predictions:
            color_ug = band_predictions['u'] - band_predictions['g']
            magnitude_g = band_predictions['g']
            
            corr_color_mag = np.corrcoef(color_ug, magnitude_g)[0, 1]
            gp_feats['gp_color_magnitude_corr_ug'] = corr_color_mag
        
    except Exception as e:
        # If GP fitting fails, return NaN features
        return _create_nan_features(oid)
    
    return gp_feats


def extract_features_with_gp(lc_df, meta_df, seed, use_gp=True):
    """
    Enhanced feature extraction with comprehensive 2D GP features across all bands.
    """
    print("Extracting Features...")
    
    # ===== PART 1: ORIGINAL FEATURES =====
    aggs = {
        'Flux': ['mean', 'max', 'min', 'std', 'skew'],
        'Flux_err': ['mean'],
        'Time (MJD)': [duration, 'count']
    }
    
    feats = lc_df.groupby(['object_id', 'Filter']).agg(aggs)
    feats.columns = ['_'.join(col).strip() for col in feats.columns.values]
    feats = feats.unstack('Filter')
    feats.columns = [f"{c[0]}_{c[1]}" for c in feats.columns]
    feats.reset_index(inplace=True)
    
    # Von Neumann Ratio
    vn_ratios = []
    filters = lc_df['Filter'].unique()
    
    for f in filters:
        f_data = lc_df[lc_df['Filter'] == f]
        if f_data.empty:
            continue
        vn = f_data.groupby('object_id')['Flux'].apply(calculate_von_neumann)
        vn.name = f'Flux_VonNeumann_{f}'
        vn_ratios.append(vn)
    
    if vn_ratios:
        vn_df = pd.concat(vn_ratios, axis=1).reset_index()
        feats = pd.merge(feats, vn_df, on='object_id', how='left')
    
    # Merge Metadata
    meta_df['object_id'] = meta_df['object_id'].astype(str)
    feats['object_id'] = feats['object_id'].astype(str)
    
    merge_cols = ['object_id', 'Z']
    if 'target' in meta_df.columns:
        merge_cols.append('target')
    feats = pd.merge(feats, meta_df[merge_cols], on='object_id', how='left')
    
    # Redshift Scaling
    if 'Z' in feats.columns:
        for f in filters:
            col_mean = f'Flux_mean_{f}'
            col_max = f'Flux_max_{f}'
            if col_mean in feats.columns:
                feats[f'Flux_Z_Scaled_{f}'] = feats[col_mean] * (feats['Z']**2)
            if col_max in feats.columns:
                feats[f'Flux_Max_Z_Scaled_{f}'] = feats[col_max] * (feats['Z']**2)
    
    # ===== PART 2: COMPREHENSIVE 2D GP FEATURES =====
    if use_gp:
        print("Extracting Comprehensive 2D GP Features (All Bands)...")
        
        unique_ids = lc_df['object_id'].unique()
        gp_features_list = []
        
        for oid in tqdm(unique_ids, desc="Fitting 2D GPs"):
            obj_data = lc_df[lc_df['object_id'] == oid]
            gp_feats = extract_gp_features_all_bands(obj_data, oid, seed)
            gp_features_list.append(gp_feats)
        
        gp_feats_df = pd.DataFrame(gp_features_list)
        
        # Merge GP features
        feats = pd.merge(feats, gp_feats_df, on='object_id', how='left')
        
        n_gp_features = len(gp_feats_df.columns) - 1
        print(f"Added {n_gp_features} comprehensive GP-derived features")
    
    return feats
