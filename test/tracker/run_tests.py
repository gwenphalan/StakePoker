#!/usr/bin/env python3
"""
Test runner for tracker module tests.

Provides convenient functions to run tracker module tests with various options
including verbose output, coverage reporting, and specific test selection.
"""

import sys
import subprocess
from pathlib import Path


def run_tracker_tests(test_type="all", verbose=True, coverage=False):
    """
    Run tracker module tests.
    
    Args:
        test_type: Type of tests to run ("all", "hand_tracker", "hero_detector", "position_calculator", "state_machine", "turn_detector")
        verbose: Enable verbose output
        coverage: Enable coverage reporting
    
    Returns:
        Exit code from pytest
    """
    project_root = Path(__file__).parent.parent.parent
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test path based on type
    if test_type == "all":
        cmd.append(str(project_root / "test" / "tracker" / "test_tracker.py"))
    elif test_type == "hand_tracker":
        cmd.extend(["-k", "TestHandTracker", str(project_root / "test" / "tracker" / "test_tracker.py")])
    elif test_type == "hero_detector":
        cmd.extend(["-k", "TestHeroDetector", str(project_root / "test" / "tracker" / "test_tracker.py")])
    elif test_type == "position_calculator":
        cmd.extend(["-k", "TestPositionCalculator", str(project_root / "test" / "tracker" / "test_tracker.py")])
    elif test_type == "state_machine":
        cmd.extend(["-k", "TestStateMachine", str(project_root / "test" / "tracker" / "test_tracker.py")])
    elif test_type == "turn_detector":
        cmd.extend(["-k", "TestTurnDetector", str(project_root / "test" / "tracker" / "test_tracker.py")])
    else:
        print(f"Unknown test type: {test_type}")
        return 1
    
    # Add verbosity
    if verbose:
        cmd.extend(["-v", "-s"])
    
    # Add coverage
    if coverage:
        cmd.extend(["--cov=src.tracker", "--cov-report=term-missing"])
    
    # Run tests
    print(f"Running tracker tests: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode


def main():
    """Main entry point for test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run tracker module tests")
    parser.add_argument(
        "--type", 
        choices=["all", "hand_tracker", "hero_detector", "position_calculator", "state_machine", "turn_detector"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Disable verbose output"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true",
        help="Enable coverage reporting"
    )
    
    args = parser.parse_args()
    
    exit_code = run_tracker_tests(
        test_type=args.type,
        verbose=not args.quiet,
        coverage=args.coverage
    )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
