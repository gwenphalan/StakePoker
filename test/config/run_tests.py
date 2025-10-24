#!/usr/bin/env python3
"""
Test runner specifically for the config module.

This script provides a simple way to run config module tests
with various options and configurations.
"""

import sys
import subprocess
from pathlib import Path


def run_config_tests(test_type="all", verbose=True, coverage=False):
    """
    Run config module tests.
    
    Args:
        test_type: Type of tests to run ("all", "settings", "unit", "integration")
        verbose: Enable verbose output
        coverage: Enable coverage reporting
    
    Returns:
        Exit code from pytest
    """
    project_root = Path(__file__).parent.parent.parent
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test path based on type
    if test_type == "settings":
        cmd.append("test/config/test_settings.py")
    elif test_type == "unit":
        cmd.append("test/config/")
    elif test_type == "integration":
        cmd.append("test/config/")
        cmd.extend(["-m", "integration"])
    else:  # all
        cmd.append("test/config/")
    
    # Add verbosity
    if verbose:
        cmd.extend(["-v", "-s"])
    else:
        cmd.extend(["-q"])
    
    # Add coverage
    if coverage:
        cmd.extend([
            "--cov=src/config",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
    
    # Add other options
    cmd.extend([
        "--tb=short",
        "--color=yes"
    ])
    
    print(f"Running config tests: {' '.join(cmd)}")
    print("-" * 60)
    
    # Run the tests
    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test runner for config module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test/config/run_tests.py                    # Run all config tests
  python test/config/run_tests.py --type settings    # Run only Settings tests
  python test/config/run_tests.py --coverage         # Run with coverage
  python test/config/run_tests.py --quiet            # Run with minimal output
        """
    )
    
    parser.add_argument(
        "--type", "-t",
        choices=["all", "settings", "unit", "integration"],
        default="all",
        help="Type of tests to run (default: all)"
    )
    
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Enable coverage reporting"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Run with minimal output"
    )
    
    args = parser.parse_args()
    
    return run_config_tests(
        test_type=args.type,
        verbose=not args.quiet,
        coverage=args.coverage
    )


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
