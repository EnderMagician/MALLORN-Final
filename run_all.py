"""
Usage:
    python run_all.py
"""

import sys
import os
import time
from pathlib import Path

import papermill as pm


# ==========================================
# CONFIGURATION
# ==========================================

# List of notebooks to run in sequential order
NOTEBOOKS = [
    # 1. Base Data Processing (Legacy)
    # "data-process-mallorn.ipynb",

    # 2. Split Generation
    # "03_make_folds.ipynb",
    
    # 3. Advanced Features (Dependent on Base)
    # "further_data.ipynb",
    
    # 5. NN Preprocessing (Dependent on Further)
    # "preprocessing_nn.ipynb",

    # 6. Model Training
    # "04_XGBoost.ipynb",
    # "04_LightGBM.ipynb",
    # "04_CatBoost.ipynb",
    # "04_NN_MLP.ipynb",
    # "04_ResNet.ipynb",
    # "04_ExtraTrees.ipynb",
    # "04_FT_Transformer.ipynb",

    # 7. Ensembling
    "05_ensemble.ipynb",
    "05_ensemble_hill.ipynb",
]

PROJECT_ROOT = Path(__file__).parent.resolve()
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
EXECUTED_DIR = PROJECT_ROOT / "notebooks_executed"
CONFIG_PATH = PROJECT_ROOT / "config.yml"

def validate_environment():
    """
    Validate that the environment is ready for execution.
    Checks for config.yml, source code, and data directories.
    """
    print("\nValidating Environment...")
    
    # 1. Check Config
    if not CONFIG_PATH.exists():
        print(f"Critical: Config file missing at {CONFIG_PATH}")
        sys.exit(1)
    
    # 2. Check src directory
    src_path = PROJECT_ROOT / "src"
    if not src_path.exists():
        print(f"Critical: 'src' directory missing at {src_path}")
        sys.exit(1)
        
    # 3. Check Data Raw (Simple check)
    data_raw = PROJECT_ROOT / "data" / "raw"
    if not data_raw.exists():
        print(f" Warning: 'data/raw' not found at {data_raw}. Ensure data is mapped correctly.")
    
    # 4. Create Executed Directory
    EXECUTED_DIR.mkdir(exist_ok=True)
    
    print("Environment looks good.")

def run_notebook(notebook_name):
    """
    Execute a single notebook using Papermill.
    
    Args:
        notebook_name (str): Filename of the notebook.
    """
    input_path = NOTEBOOKS_DIR / notebook_name
    output_path = EXECUTED_DIR / notebook_name
    
    if not input_path.exists():
        print(f"Error: Notebook not found: {notebook_name}")
        sys.exit(1)
        
    print(f"\nRunning: {notebook_name}")
    print(f"   Input:  {input_path}")
    print(f"   Output: {output_path}")
    
    start_time = time.time()
    
    try:
        pm.execute_notebook(
            input_path=str(input_path),
            output_path=str(output_path),
            kernel_name="python3",
            cwd=str(NOTEBOOKS_DIR),
            progress_bar=True,
            stdout_file=sys.stdout,
            request_save_on_cell_execute=True,
            execution_timeout=None
        )
    except pm.PapermillExecutionError as e:
        print(f"\nFAIL: {notebook_name} failed during execution.")
        print(f"   Error: {str(e)}")
        print("   See executed notebook for traceback.")
        sys.exit(1) # STOP ON FAILURE
    except Exception as e:
        print(f"\nFAIL: An unexpected error occurred while running {notebook_name}.")
        print(f"   Error: {str(e)}")
        sys.exit(1)
        
    elapsed = time.time() - start_time
    print(f"SUCCESS: {notebook_name} completed in {elapsed:.2f}s")


def main():
    print(f"{'='*60}")
    print("MALLORN REPRODUCIBILITY PIPELINE")
    print(f"{'='*60}")
    
    validate_environment()
    
    print(f"\nScheduled Notebooks ({len(NOTEBOOKS)}):")
    for i, nb in enumerate(NOTEBOOKS, 1):
        print(f"   {i}. {nb}")
    
    print(f"\n{'='*60}")
    print("STARTING EXECUTION")
    print(f"{'='*60}")
    
    total_start = time.time()
    
    for notebook in NOTEBOOKS:
        run_notebook(notebook)
        
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"All {len(NOTEBOOKS)} notebooks completed successfully!")
    print(f"Total Time: {total_elapsed/60:.2f} minutes")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
