#!/usr/bin/env python3
"""
Comprehensive test runner for parser module.

This script provides a unified interface for running all parser tests including:
- Unit tests for individual parser components
- Integration tests for parser workflows
- Visual tests with interactive validation
- Performance tests and benchmarks

Usage:
    python test/parser/run_tests.py --unit
    python test/parser/run_tests.py --integration
    python test/parser/run_tests.py --visual --parser card --region card_1
    python test/parser/run_tests.py --all
    python test/parser/run_tests.py --performance
"""

import argparse
import logging
import sys
import subprocess
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from test.parser.test_utils import TestResultManager
from test.parser.visual_test import PARSER_REGISTRY


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)8s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def run_unit_tests(verbose: bool = False):
    """Run unit tests for all parser components."""
    print("Running Parser Unit Tests")
    print("=" * 50)
    
    # Test files to run
    test_files = [
        "test/parser/test_card_parser.py",
        "test/parser/test_ocr_engine.py", 
        "test/parser/test_money_parser.py",
        # Add more test files as they are created
    ]
    
    pytest_args = ["python", "-m", "pytest"]
    
    if verbose:
        pytest_args.extend(["-v", "-s"])
    else:
        pytest_args.extend(["-q"])
    
    pytest_args.extend([
        "--tb=short",
        "--color=yes",
        "-m", "unit"
    ])
    
    pytest_args.extend(test_files)
    
    print(f"Running: {' '.join(pytest_args)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(pytest_args, check=True)
        print("\n✅ Unit tests completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Unit tests failed with exit code {e.returncode}")
        return False


def run_integration_tests(verbose: bool = False):
    """Run integration tests for parser workflows."""
    print("Running Parser Integration Tests")
    print("=" * 50)
    
    # Test files to run
    test_files = [
        "test/parser/test_card_parser.py",
        "test/parser/test_ocr_engine.py",
        "test/parser/test_money_parser.py",
        # Add more test files as they are created
    ]
    
    pytest_args = ["python", "-m", "pytest"]
    
    if verbose:
        pytest_args.extend(["-v", "-s"])
    else:
        pytest_args.extend(["-q"])
    
    pytest_args.extend([
        "--tb=short",
        "--color=yes",
        "-m", "integration"
    ])
    
    pytest_args.extend(test_files)
    
    print(f"Running: {' '.join(pytest_args)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(pytest_args, check=True)
        print("\n✅ Integration tests completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Integration tests failed with exit code {e.returncode}")
        return False


def run_visual_tests(parser_name: str = None, region_name: str = None, num_tests: int = 10):
    """Run visual tests for parser validation."""
    print("Running Parser Visual Tests")
    print("=" * 50)
    
    if parser_name and region_name:
        # Run specific parser test
        print(f"Testing {parser_name} parser on region {region_name}")
        print(f"Running {num_tests} tests...")
        
        # Import and run visual test
        from test.parser.visual_test import run_visual_test
        run_visual_test(parser_name, region_name, num_tests)
        
    else:
        # Run tests for all parsers
        print("Running visual tests for all parsers...")
        
        for parser_name, (parser_class, test_class, regions) in PARSER_REGISTRY.items():
            print(f"\nTesting {parser_name} parser...")
            
            for region in regions[:2]:  # Test first 2 regions for each parser
                print(f"  Testing region: {region}")
                try:
                    from test.parser.visual_test import run_visual_test
                    run_visual_test(parser_name, region, 5)  # 5 tests per region
                except Exception as e:
                    print(f"    Error testing {parser_name} on {region}: {e}")


def run_performance_tests(verbose: bool = False):
    """Run performance tests and benchmarks."""
    print("Running Parser Performance Tests")
    print("=" * 50)
    
    # Test files to run
    test_files = [
        "test/parser/test_card_parser.py",
        "test/parser/test_ocr_engine.py",
        "test/parser/test_money_parser.py",
        # Add more test files as they are created
    ]
    
    pytest_args = ["python", "-m", "pytest"]
    
    if verbose:
        pytest_args.extend(["-v", "-s"])
    else:
        pytest_args.extend(["-q"])
    
    pytest_args.extend([
        "--tb=short",
        "--color=yes",
        "-m", "performance"
    ])
    
    pytest_args.extend(test_files)
    
    print(f"Running: {' '.join(pytest_args)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(pytest_args, check=True)
        print("\n✅ Performance tests completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Performance tests failed with exit code {e.returncode}")
        return False


def show_test_statistics():
    """Show statistics from previous test runs."""
    print("Parser Test Statistics")
    print("=" * 50)
    
    try:
        result_manager = TestResultManager()
        
        for parser_name in PARSER_REGISTRY.keys():
            stats = result_manager.get_parser_statistics(parser_name)
            
            if stats['total_tests'] > 0:
                print(f"\n{parser_name.upper()} Parser:")
                print(f"  Total Tests: {stats['total_tests']}")
                print(f"  Approved: {stats['approved']}")
                print(f"  Rejected: {stats['rejected']}")
                print(f"  Skipped: {stats['skipped']}")
                print(f"  Approval Rate: {stats['approval_rate']:.1%}")
                print(f"  Average Confidence: {stats['avg_confidence']:.3f}")
            else:
                print(f"\n{parser_name.upper()} Parser: No test data available")
    
    except Exception as e:
        print(f"Error loading statistics: {e}")


def list_available_tests():
    """List all available test options."""
    print("Available Parser Tests")
    print("=" * 50)
    
    print("\nUnit Tests:")
    print("  - CardParser: Card rank and suit detection")
    print("  - OCREngine: Text extraction and preprocessing")
    print("  - MoneyParser: Monetary amount parsing")
    print("  - NameParser: Player name detection")
    print("  - StatusParser: Player status detection")
    print("  - TableInfoParser: Table information parsing")
    print("  - TimerDetector: Turn timer detection")
    print("  - TransparencyDetector: Folded player detection")
    print("  - DealerDetector: Dealer button detection")
    print("  - HandIdParser: Hand ID number parsing")
    
    print("\nIntegration Tests:")
    print("  - Full parsing workflows")
    print("  - Multi-component interactions")
    print("  - Error handling and recovery")
    
    print("\nVisual Tests:")
    print("  - Interactive validation with real poker images")
    print("  - Manual approval/rejection of parser results")
    print("  - Test result storage and aggregation")
    
    print("\nPerformance Tests:")
    print("  - Processing speed benchmarks")
    print("  - Memory usage analysis")
    print("  - Scalability testing")
    
    print("\nAvailable Parsers for Visual Testing:")
    for parser_name, (parser_class, test_class, regions) in PARSER_REGISTRY.items():
        print(f"  {parser_name}: {', '.join(regions)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive test runner for parser module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test/parser/run_tests.py --unit
  python test/parser/run_tests.py --integration --verbose
  python test/parser/run_tests.py --visual --parser card --region card_1 --tests 20
  python test/parser/run_tests.py --performance
  python test/parser/run_tests.py --all
  python test/parser/run_tests.py --stats
  python test/parser/run_tests.py --list
        """
    )
    
    # Test type arguments
    parser.add_argument(
        '--unit', '-u',
        action='store_true',
        help='Run unit tests'
    )
    
    parser.add_argument(
        '--integration', '-i',
        action='store_true',
        help='Run integration tests'
    )
    
    parser.add_argument(
        '--visual', '-v',
        action='store_true',
        help='Run visual tests'
    )
    
    parser.add_argument(
        '--performance', '-p',
        action='store_true',
        help='Run performance tests'
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Run all test types'
    )
    
    # Visual test arguments
    parser.add_argument(
        '--parser',
        choices=list(PARSER_REGISTRY.keys()),
        help='Parser to test (for visual tests)'
    )
    
    parser.add_argument(
        '--region',
        help='Region to test (for visual tests)'
    )
    
    parser.add_argument(
        '--tests', '-t',
        type=int,
        default=10,
        help='Number of visual tests to run (default: 10)'
    )
    
    # Utility arguments
    parser.add_argument(
        '--stats', '-s',
        action='store_true',
        help='Show test statistics'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available tests'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Handle different commands
    if args.list:
        list_available_tests()
    elif args.stats:
        show_test_statistics()
    elif args.all:
        print("Running All Parser Tests")
        print("=" * 60)
        
        success = True
        success &= run_unit_tests(args.verbose)
        success &= run_integration_tests(args.verbose)
        success &= run_performance_tests(args.verbose)
        
        if success:
            print("\n🎉 All tests completed successfully!")
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)
    
    elif args.unit:
        success = run_unit_tests(args.verbose)
        sys.exit(0 if success else 1)
    
    elif args.integration:
        success = run_integration_tests(args.verbose)
        sys.exit(0 if success else 1)
    
    elif args.visual:
        run_visual_tests(args.parser, args.region, args.tests)
    
    elif args.performance:
        success = run_performance_tests(args.verbose)
        sys.exit(0 if success else 1)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

