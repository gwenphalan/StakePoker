#!/usr/bin/env python3
"""
Unified test runner for StakePoker.

A comprehensive CLI utility to run all tests or specific subsets with various configurations.
This replaces multiple test runner scripts with a single, powerful entry point.
"""

import argparse
import sys
import subprocess
from pathlib import Path
from typing import List, Optional


class TestRunner:
    """Unified test runner for StakePoker project."""
    
    def __init__(self):
        """Initialize the test runner."""
        self.project_root = Path(__file__).parent.parent
        self.test_dir = Path(__file__).parent
        
    def run_tests(self, 
                  modules: Optional[List[str]] = None,
                  test_type: Optional[str] = None,
                  markers: Optional[str] = None,
                  keywords: Optional[str] = None,
                  coverage: bool = False,
                  verbose: bool = True,
                  debug: bool = False,
                  summary: bool = False,
                  specific_file: Optional[str] = None,
                  specific_function: Optional[str] = None) -> int:
        """
        Run tests with specified configuration.
        
        Args:
            modules: List of module names to test (e.g., ['capture', 'parser'])
            test_type: Type of tests to run (unit, integration, performance, visual)
            markers: Additional pytest markers to filter tests
            keywords: Keywords to filter test names
            coverage: Enable coverage reporting
            verbose: Enable verbose output
            debug: Run with debugging enabled
            summary: Show only final test results summary
            specific_file: Specific test file to run
            specific_function: Specific test function to run
            
        Returns:
            Exit code from pytest
        """
        # Build pytest command
        cmd = ["python", "-m", "pytest"]
        
        # Add test path
        if specific_file:
            if specific_function:
                cmd.append(f"test/{specific_file}::{specific_function}")
            else:
                cmd.append(f"test/{specific_file}")
        elif modules:
            # Test specific modules
            for module in modules:
                cmd.append(f"test/{module}/")
        else:
            # Test all modules
            cmd.append("test/")
        
        # Add verbosity
        if summary:
            # Summary mode: minimal output, only show final results
            cmd.extend(["-q", "--tb=no"])
        elif verbose:
            cmd.extend(["-v", "-s"])
        
        # Add debugging
        if debug:
            cmd.extend(["--tb=long", "--pdb"])
        
        # Add coverage
        if coverage:
            cmd.extend([
                "--cov=src",
                "--cov-report=html",
                "--cov-report=term-missing",
                "--cov-report=xml"
            ])
        
        # Add markers
        if test_type:
            cmd.extend(["-m", test_type])
        
        if markers:
            if test_type:
                # Combine with existing marker
                marker_index = cmd.index("-m") + 1
                cmd[marker_index] = f"{cmd[marker_index]} and {markers}"
            else:
                cmd.extend(["-m", markers])
        
        # Add keywords
        if keywords:
            cmd.extend(["-k", keywords])
        
        # Add timeout and other options
        cmd.extend([
            "--tb=short",
            "--color=yes"
        ])
        
        print(f"Running: {' '.join(cmd)}")
        print("-" * 80)
        
        # Run the tests
        if summary:
            # Capture output and filter to show only summary
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
            
            # Extract and display only the summary lines
            lines = result.stdout.split('\n')
            summary_lines = []
            
            # Find the start of the summary section
            summary_started = False
            for line in lines:
                # Start collecting from the short test summary section
                if 'short test summary' in line.lower():
                    summary_started = True
                
                if summary_started:
                    # Look for summary lines (containing test counts, warnings, errors)
                    if any(keyword in line.lower() for keyword in [
                        'failed', 'passed', 'skipped', 'deselected', 'warnings', 'errors',
                        'test session', 'short test summary', 'warnings summary',
                        '============', 'failed test/parser', 'assertionerror'
                    ]):
                        summary_lines.append(line)
            
            # Also include stderr for warnings and errors
            if result.stderr:
                stderr_lines = result.stderr.split('\n')
                for line in stderr_lines:
                    if any(keyword in line.lower() for keyword in [
                        'warning', 'error', 'failed', 'exception'
                    ]):
                        summary_lines.append(line)
            
            # Display summary
            if summary_lines:
                print("Test Results Summary:")
                print("-" * 50)
                for line in summary_lines:
                    if line.strip():  # Skip empty lines
                        print(line)
                print("-" * 50)
            else:
                print("No summary information available")
            
            return result.returncode
        else:
            # Normal execution with full output
            result = subprocess.run(cmd, cwd=self.project_root)
            return result.returncode
    
    def list_available_modules(self) -> List[str]:
        """List all available test modules."""
        modules = []
        for item in self.test_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.') and item.name != '__pycache__':
                modules.append(item.name)
        return sorted(modules)
    
    def list_available_tests(self, module: Optional[str] = None) -> List[str]:
        """List all available test files."""
        if module:
            test_path = self.test_dir / module
        else:
            test_path = self.test_dir
        
        test_files = []
        for item in test_path.rglob("test_*.py"):
            relative_path = item.relative_to(self.test_dir)
            test_files.append(str(relative_path).replace("\\", "/"))
        
        return sorted(test_files)
    
    def run_parser_visual_tests(self, parser_name: str, region_name: str, num_tests: int = 10, random_mode: bool = False) -> int:
        """Run visual tests for parser components."""
        try:
            # Add test directory to path for imports
            import sys
            test_dir = Path(__file__).parent
            if str(test_dir) not in sys.path:
                sys.path.insert(0, str(test_dir))
            
            if random_mode:
                # Import random visual test runner
                from parser.visual_test import run_random_visual_tests
                run_random_visual_tests(num_tests)
            else:
                # Import visual test runner
                from parser.visual_test import run_visual_test
                
                # Special handling for transparency and table_info parsers with no region specified
                if parser_name in ["transparency", "table_info"] and region_name is None:
                    if parser_name == "transparency":
                        from parser.visual_test import run_transparency_random_regions
                        run_transparency_random_regions(num_tests)
                    elif parser_name == "table_info":
                        from parser.visual_test import run_visual_test
                        # For table_info, use the single table_info region
                        run_visual_test(parser_name, "table_info", num_tests)
                else:
                    run_visual_test(parser_name, region_name, num_tests)
            return 0
        except Exception as e:
            print(f"Error running parser visual tests: {e}")
            import traceback
            traceback.print_exc()
            return 1


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Unified test runner for StakePoker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test/main.py                           # Run all tests
  python test/main.py --modules capture         # Run only capture module tests
  python test/main.py --modules parser          # Run only parser module tests
  python test/main.py --type unit               # Run only unit tests
  python test/main.py --type integration        # Run only integration tests
  python test/main.py --coverage                # Run with coverage report
  python test/main.py --file capture/test_monitor_config  # Run specific file
  python test/main.py --file capture/test_monitor_config --function test_monitor_info_creation  # Run specific test
  python test/main.py --debug                   # Run with debugging
  python test/main.py --keywords "monitor"      # Run tests matching keyword
  python test/main.py --summary-only             # Show only final test results summary
  python test/main.py --list-modules            # List available modules
  python test/main.py --list-tests              # List available test files
  python test/main.py --parser-visual --parser-name card --region-name card_1 --num-visual-tests 20  # Run visual tests
  python test/main.py --random-visual --num-visual-tests 50  # Run random visual tests
        """
    )
    
    # Test selection
    parser.add_argument(
        "--modules", "-m",
        nargs="+",
        help="Specific modules to test (e.g., capture parser overlay)"
    )
    
    parser.add_argument(
        "--type", "-t",
        choices=["unit", "integration", "performance", "visual"],
        help="Type of tests to run"
    )
    
    parser.add_argument(
        "--file", "-f",
        help="Specific test file to run (e.g., capture/test_monitor_config)"
    )
    
    parser.add_argument(
        "--function", "-fn",
        help="Specific test function to run"
    )
    
    parser.add_argument(
        "--keywords", "-k",
        help="Run tests matching keyword expression"
    )
    
    parser.add_argument(
        "--markers", "-ma",
        help="Additional pytest markers to filter tests"
    )
    
    # Output options
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Enable coverage reporting"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Enable verbose output (default)"
    )
    
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Show only final test results summary (failed, passed, warnings, errors)"
    )
    
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Run with debugging enabled"
    )
    
    # Information options
    parser.add_argument(
        "--list-modules",
        action="store_true",
        help="List available test modules"
    )
    
    parser.add_argument(
        "--list-tests",
        action="store_true",
        help="List available test files"
    )
    
    # Parser visual test options
    parser.add_argument(
        "--parser-visual",
        action="store_true",
        help="Run parser visual tests"
    )
    
    parser.add_argument(
        "--parser-name",
        choices=["card", "dealer", "hand_id", "money", "name", "status", "table_info", "timer", "transparency"],
        help="Parser to test (for visual tests)"
    )
    
    parser.add_argument(
        "--region-name",
        help="Region to test (for visual tests)"
    )
    
    parser.add_argument(
        "--num-visual-tests",
        type=int,
        default=10,
        help="Number of visual tests to run (default: 10)"
    )
    
    parser.add_argument(
        "--random-visual",
        action="store_true",
        help="Run visual tests with random parser and region selection"
    )
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    # Handle information requests
    if args.list_modules:
        modules = runner.list_available_modules()
        print("Available test modules:")
        for module in modules:
            print(f"  {module}")
        return 0
    
    if args.list_tests:
        tests = runner.list_available_tests(args.modules[0] if args.modules else None)
        print("Available test files:")
        for test in tests:
            print(f"  {test}")
        return 0
    
    # Handle parser visual tests
    if args.parser_visual or args.random_visual:
        if args.random_visual:
            # Random mode - no parser or region needed
            return runner.run_parser_visual_tests(
                None, 
                None, 
                args.num_visual_tests,
                random_mode=True
            )
        else:
            # Specific parser mode
            if not args.parser_name:
                print("Error: --parser-visual requires --parser-name")
                return 1
            
            # For transparency and table_info parsers, allow random region selection if no region specified
            if args.parser_name in ["transparency", "table_info"] and not args.region_name:
                return runner.run_parser_visual_tests(
                    args.parser_name, 
                    None,  # Will use random regions for that parser
                    args.num_visual_tests,
                    random_mode=False
                )
            elif not args.region_name:
                print("Error: --parser-visual requires --region-name (except for transparency and table_info parsers)")
                return 1
            
            return runner.run_parser_visual_tests(
                args.parser_name, 
                args.region_name, 
                args.num_visual_tests
            )
    
    # Run tests
    return runner.run_tests(
        modules=args.modules,
        test_type=args.type,
        markers=args.markers,
        keywords=args.keywords,
        coverage=args.coverage,
        verbose=args.verbose,
        debug=args.debug,
        summary=args.summary_only,
        specific_file=args.file,
        specific_function=args.function
    )


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
