# StakePoker Test Suite

This directory contains comprehensive tests for the StakePoker application, organized to mirror the `src/` directory structure.

## Test Structure

```
test/
├── __init__.py
├── capture/                    # Tests for capture module
│   ├── __init__.py
│   ├── test_monitor_config.py  # Monitor detection & configuration
│   ├── test_screen_capture.py  # Screen capture functionality
│   ├── test_region_loader.py   # Region loading & validation
│   ├── test_region_extractor.py # Region extraction & cropping
│   ├── test_utils.py           # Test utilities & helpers
│   └── run_capture_tests.py    # Test runner for capture module
├── data/                       # Test data directory
│   └── capture/                # Capture-specific test data
│       ├── reference_images/   # Reference images for comparison
│       └── test_regions/       # Test region definitions
├── advisor/                    # Tests for advisor module
├── config/                     # Tests for config module
├── history/                    # Tests for history module
├── models/                     # Tests for models module
├── overlay/                    # Tests for overlay module
├── parser/                     # Tests for parser module
└── tracker/                    # Tests for tracker module
```

## Running Tests

### Prerequisites

Install test dependencies:

```bash
pip install -r requirements.txt
```

### Quick Start

Run all tests:

```bash
python test/main.py
```

Run specific modules:

```bash
# Run only capture module tests
python test/main.py --modules capture

# Run multiple modules
python test/main.py --modules capture parser overlay
```

Run specific test types:

```bash
# Unit tests only
python test/main.py --type unit

# Integration tests only
python test/main.py --type integration

# Performance tests only
python test/main.py --type performance
```

Run with coverage:

```bash
python test/main.py --coverage
```

Run specific files or functions:

```bash
# Run specific test file
python test/main.py --file capture/test_monitor_config

# Run specific test function
python test/main.py --file capture/test_monitor_config --function test_monitor_info_creation
```

Run with debugging:

```bash
python test/main.py --debug
```

List available options:

```bash
# List available modules
python test/main.py --list-modules

# List available test files
python test/main.py --list-tests
```

### Using pytest directly

```bash
# Run all tests
pytest test/

# Run specific test file
pytest test/capture/test_monitor_config.py

# Run specific test function
pytest test/capture/test_monitor_config.py::TestMonitorConfig::test_monitor_detection

# Run with coverage
pytest test/ --cov=src --cov-report=html --cov-report=term-missing

# Run with verbose output
pytest test/ -v -s
```

## Test Types

### Unit Tests (`@pytest.mark.unit`)

- Test individual components in isolation
- Use mocks for external dependencies
- Fast execution
- Examples: MonitorConfig, RegionModel validation

### Integration Tests (`@pytest.mark.integration`)

- Test component interactions
- May use real dependencies where appropriate
- Moderate execution time
- Examples: ScreenCapture with MonitorConfig, RegionExtractor pipeline

### Performance Tests (`@pytest.mark.performance`)

- Test execution time and resource usage
- May be slower than other tests
- Examples: Large region extraction, capture timing

### Visual Tests (`@pytest.mark.visual`)

- Require manual validation
- Use reference images for comparison
- Examples: OCR accuracy validation, image quality checks

## Test Utilities

### CaptureTestUtils

Provides common functionality for capture module tests:

- `create_test_frame()` - Generate test images
- `create_patterned_frame()` - Create frames with known patterns
- `create_sample_regions()` - Generate test region definitions
- `compare_images()` - Compare images using SSIM

### VisualTestValidator

Helper for visual testing:

- `save_reference_image()` - Save reference images
- `compare_with_reference()` - Compare against saved references
- `create_comparison_image()` - Create side-by-side comparisons

### PerformanceTracker

Track performance metrics:

- `start_timer()` / `end_timer()` - Time operations
- `assert_faster_than()` - Assert performance requirements

## Test Data

Test data is stored in `test/data/` and organized by module:

- `reference_images/` - Reference images for visual comparison
- `test_regions/` - Test region definitions
- Generated test data is created in temporary directories

## Configuration

Test configuration is in `pytest.ini`:

- Test discovery patterns
- Output formatting
- Markers for different test types
- Logging configuration
- Timeout settings

## Writing Tests

### Test Naming

- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test functions: `test_<function_name>`

### Test Structure

```python
class TestClassName:
    """Test ClassName functionality."""

    @pytest.fixture
    def sample_data(self):
        """Provide sample data for tests."""
        return {"key": "value"}

    def test_function_success(self, sample_data):
        """Test successful function execution."""
        result = function_under_test(sample_data)
        assert result == expected_value

    def test_function_error_handling(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            function_under_test(invalid_data)

    @pytest.mark.integration
    def test_integration_scenario(self):
        """Test integration scenario."""
        # Integration test code
        pass
```

### Best Practices

1. **Use fixtures** for common setup
2. **Mock external dependencies** in unit tests
3. **Test both success and failure cases**
4. **Use descriptive test names**
5. **Add appropriate markers** for test types
6. **Keep tests independent** and isolated
7. **Use parametrize** for testing multiple inputs

## Continuous Integration

Tests are designed to run in CI environments:

- No external dependencies required
- Deterministic results
- Appropriate timeouts
- Clear error messages

## Debugging Tests

Run tests with debugging:

```bash
python test/run_capture_tests.py --debug --file test_monitor_config
```

This will drop into a debugger on test failures.

## Coverage

Generate coverage reports:

```bash
python test/run_capture_tests.py --coverage
```

Coverage reports are generated in HTML format in `htmlcov/` directory.
