"""
MALLORN Utilities
Common utility functions.
"""

import os
import re


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def clean_column_names(cols: list) -> list:
    """Remove special characters from column names for LightGBM compatibility."""
    return [re.sub(r'[^A-Za-z0-9_]+', '_', str(c)) for c in cols]
