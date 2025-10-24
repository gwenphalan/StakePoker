#!/usr/bin/env python3
"""
Test runner for advisor module tests.

Runs all unit tests for the advisor module components.
"""

import sys
import os
import pytest
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

def run_advisor_tests():
    """Run all advisor module tests."""
    test_dir = Path(__file__).parent
    
    # Test files to run
    test_files = [
        "test_gto_loader.py",
        "test_equity_calculator.py", 
        "test_range_estimator.py",
        "test_postflop_solver.py",
        "test_decision_engine.py"
    ]
    
    # Run tests with verbose output
    args = [
        "-v",  # Verbose
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Strict marker handling
        "--disable-warnings",  # Disable warnings for cleaner output
    ]
    
    # Add test files
    for test_file in test_files:
        test_path = test_dir / test_file
        if test_path.exists():
            args.append(str(test_path))
        else:
            print(f"Warning: Test file not found: {test_path}")
    
    # Run pytest
    exit_code = pytest.main(args)
    return exit_code

if __name__ == "__main__":
    print("Running Advisor Module Tests")
    print("=" * 50)
    
    exit_code = run_advisor_tests()
    
    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code: {exit_code}")
    
    sys.exit(exit_code)
