"""
Configuration management for multi-environment deployment.
Paths and settings adapt automatically based on environment.

Usage:
    from src.config import DATA_PATH, RESULTS_PATH, N_WORKERS
"""
import os
from pathlib import Path

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Detect environment
ENV = os.environ.get("PROJECT_ENV", "local")

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Core paths
DATA_PATH = Path(os.environ.get("DATA_PATH", PROJECT_ROOT / "data"))
RESULTS_PATH = Path(os.environ.get("RESULTS_PATH", PROJECT_ROOT / "results"))
RAW_DATA_PATH = DATA_PATH / "raw"
PROCESSED_DATA_PATH = DATA_PATH / "processed"

# Ensure directories exist
for path in [DATA_PATH, RESULTS_PATH, RAW_DATA_PATH, PROCESSED_DATA_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# Computation settings
if ENV == "server":
    N_WORKERS = int(os.environ.get("N_WORKERS", 8))
else:
    N_WORKERS = int(os.environ.get("N_WORKERS", 2))

# Convenience function
def get_result_path(name: str, ext: str = "") -> Path:
    """Generate a path in the results directory."""
    filename = f"{name}{ext}" if ext.startswith('.') else f"{name}.{ext}" if ext else name
    return RESULTS_PATH / filename
