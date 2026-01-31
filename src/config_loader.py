"""
MALLORN Configuration Loader
Provides functions to load and access project configuration from config.yml
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml


_config: Optional[dict] = None


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


def load_config(config_path: Optional[str] = None) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Optional path to config file. Defaults to project root config.yml.
    
    Returns:
        Configuration dictionary.
    """
    global _config
    if _config is not None:
        return _config
    
    if config_path is None:
        config_path = get_project_root() / "config.yml"
    
    with open(config_path, 'r') as f:
        _config = yaml.safe_load(f)
    return _config


def reload_config(config_path: Optional[str] = None) -> dict:
    """Force reload configuration from file."""
    global _config
    _config = None
    return load_config(config_path)


def get_path(key: str) -> Path:
    """
    Get absolute path from config key.
    
    Args:
        key: Path key from config (e.g., 'data_raw', 'data_processed', 'models')
    
    Returns:
        Absolute path.
    """
    config = load_config()
    relative_path = config['paths'][key]
    return get_project_root() / relative_path


def get_seed(legacy: bool = False) -> int:
    """
    Get random seed.
    
    Args:
        legacy: If True, returns the legacy seed (42) for data-process-mallorn.ipynb.
                If False, returns the default seed (15).
    
    Returns:
        Random seed integer.
    """
    config = load_config()
    return config['seeds']['data_processing_legacy' if legacy else 'default']


def get_lsst_config() -> dict:
    """Get LSST physics constants."""
    config = load_config()
    return config['lsst']


def get_gp_config() -> dict:
    """Get Gaussian Process hyperparameters."""
    config = load_config()
    return config['gp']


def get_cv_config() -> dict:
    """Get cross-validation settings."""
    config = load_config()
    return config['cv']


def get_model_config() -> dict:
    """Get model settings."""
    config = load_config()
    return config['models']


def get_meta_columns() -> list:
    """Get list of meta columns to exclude from features."""
    config = load_config()
    return config['meta_columns']


def get(key: str, default: Any = None) -> Any:
    """
    Get any config value by dot-notation key.
    
    Args:
        key: Dot-separated key (e.g., 'seeds.default', 'lsst.wavelengths.g')
        default: Default value if key not found.
    
    Returns:
        Config value or default.
    """
    config = load_config()
    keys = key.split('.')
    value = config
    try:
        for k in keys:
            value = value[k]
        return value
    except (KeyError, TypeError):
        return default
