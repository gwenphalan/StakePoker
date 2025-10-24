#!/usr/bin/env python3
"""
README for advisor module tests.

Documentation for running and understanding the advisor module test suite.
"""

# Advisor Module Test Suite

This directory contains comprehensive unit tests for the advisor module components.

## Test Files

- `test_gto_loader.py` - Tests for GTOChartLoader
- `test_equity_calculator.py` - Tests for EquityCalculator
- `test_range_estimator.py` - Tests for RangeEstimator
- `test_postflop_solver.py` - Tests for PostflopSolver
- `test_decision_engine.py` - Tests for DecisionEngine
- `test_config.py` - Test configuration and fixtures
- `run_tests.py` - Test runner script

## Running Tests

### Run All Advisor Tests

```bash
cd test/advisor
python run_tests.py
```

### Run Individual Test Files

```bash
pytest test_gto_loader.py -v
pytest test_equity_calculator.py -v
pytest test_range_estimator.py -v
pytest test_postflop_solver.py -v
pytest test_decision_engine.py -v
```

### Run Specific Test Classes

```bash
pytest test_gto_loader.py::TestGTOChartLoader -v
pytest test_equity_calculator.py::TestEquityCalculator -v
```

### Run Specific Test Methods

```bash
pytest test_gto_loader.py::TestGTOChartLoader::test_should_open -v
pytest test_equity_calculator.py::TestEquityCalculator::test_calculate_equity_vs_range -v
```

## Test Coverage

### GTOChartLoader Tests

- ✅ Initialization with valid/invalid directories
- ✅ Position file selection (6-max vs 8-max)
- ✅ Data loading and caching
- ✅ Range queries (opening, 3bet, calling, blind defense)
- ✅ Frequency calculations
- ✅ Bet size retrieval
- ✅ Preflop decision recommendations
- ✅ Hand notation normalization
- ✅ Error handling for corrupted files
- ✅ BB 6-max fallback behavior

### EquityCalculator Tests

- ✅ Card conversion to Treys format
- ✅ Equity calculation vs ranges
- ✅ Equity calculation vs specific hands
- ✅ Pot odds and implied odds calculations
- ✅ Range expansion to specific hands
- ✅ Hand strength evaluation
- ✅ Monte Carlo simulation
- ✅ Caching behavior
- ✅ Error handling for invalid inputs
- ✅ Cache management

### RangeEstimator Tests

- ✅ Range estimation for multiple opponents
- ✅ Player-specific range estimation
- ✅ GTO range integration
- ✅ Action context detection
- ✅ Range adjustments for actions
- ✅ Range tightening algorithms
- ✅ Hand strength ordering
- ✅ Default range fallbacks
- ✅ Error handling for missing data
- ✅ Player count calculations

### PostflopSolver Tests

- ✅ EV calculations for all actions
- ✅ Fold equity estimation
- ✅ Board texture analysis
- ✅ Action filtering based on game state
- ✅ Alternative action generation
- ✅ Bet amount calculations
- ✅ Confidence scoring
- ✅ Reasoning generation
- ✅ Error handling for edge cases
- ✅ Integration with game state

### DecisionEngine Tests

- ✅ Preflop decision routing
- ✅ Postflop decision routing
- ✅ Hero validation
- ✅ Hand formatting
- ✅ Position context detection
- ✅ Confidence calculations
- ✅ Pot odds calculations
- ✅ Error handling and fallbacks
- ✅ Integration with all advisor components
- ✅ Stack limit handling

## Test Fixtures

The `test_config.py` file provides:

- **AdvisorTestFixtures**: Common test data creation
- **MockAdvisorComponents**: Mock objects for testing
- **TestHelpers**: Assertion helpers
- **Pytest fixtures**: Reusable test components

## Mock Strategy

Tests use extensive mocking to:

- Isolate components under test
- Avoid external dependencies (file I/O, Treys library)
- Control test data and scenarios
- Ensure fast, reliable test execution

## Test Data

Tests use realistic poker scenarios:

- Standard hand notations (AA, AKs, KQo, etc.)
- Common positions (UTG, MP, CO, BTN, SB, BB)
- Typical bet sizes and pot sizes
- Realistic equity values and pot odds

## Error Testing

Each test file includes comprehensive error testing:

- Invalid inputs
- Missing data
- Corrupted files
- Network/IO errors
- Edge cases and boundary conditions

## Performance Considerations

Tests are designed to run quickly:

- Mocked external dependencies
- Minimal file I/O
- Efficient test data generation
- Parallel test execution support

## Continuous Integration

Tests are designed to work in CI environments:

- No external network dependencies
- Deterministic results
- Clear pass/fail criteria
- Comprehensive error reporting
