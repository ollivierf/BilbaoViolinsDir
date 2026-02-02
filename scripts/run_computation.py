#!/usr/bin/env python3
"""
Main computation script.
Run with: python scripts/run_computation.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_PATH, RESULTS_PATH, N_WORKERS, ENV

def main():
    print(f"Environment: {ENV}")
    print(f"Data path: {DATA_PATH}")
    print(f"Results path: {RESULTS_PATH}")
    print(f"Workers: {N_WORKERS}")
    
    # Your computation code here
    print("\nRunning computation...")
    
    # Example: save result
    result_file = RESULTS_PATH / "output.txt"
    result_file.write_text("Computation complete\n")
    print(f"Result saved to: {result_file}")

if __name__ == "__main__":
    main()
