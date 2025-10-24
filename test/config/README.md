# Config Module Tests

This directory contains comprehensive unit tests for the `src/config/` module, specifically for the `Settings` class.

## Test Files

- **`test_settings.py`** - Complete unit tests for the Settings class
- **`run_tests.py`** - Dedicated test runner for config module tests

## Test Coverage

The tests cover all major functionality of the Settings class:

### 1. Singleton Pattern Tests (`TestSettingsSingleton`)

- ✅ Singleton instance creation
- ✅ Thread safety verification
- ✅ Single initialization guarantee

### 2. Initialization Tests (`TestSettingsInitialization`)

- ✅ Default settings file path construction
- ✅ Custom settings file path handling
- ✅ Loading from non-existent files (creates new file)
- ✅ Loading from existing files with data
- ✅ Handling of invalid JSON files

### 3. CRUD Operations Tests (`TestSettingsCRUD`)

- ✅ Creating settings with default values
- ✅ Preserving existing values when creating settings
- ✅ Updating existing settings
- ✅ Error handling for updating non-existent settings
- ✅ Getting settings with fallback values
- ✅ Getting settings with default values
- ✅ Deleting settings
- ✅ Checking if settings exist

### 4. Group Operations Tests (`TestSettingsGroups`)

- ✅ Getting all settings in a group
- ✅ Getting non-existent groups (returns empty dict)
- ✅ Getting nested groups
- ✅ Resetting groups to defaults
- ✅ Resetting non-existent groups
- ✅ Listing groups and subgroups
- ✅ Listing settings (leaf nodes)

### 5. File Persistence Tests (`TestSettingsFilePersistence`)

- ✅ Saving settings to file
- ✅ Handling permission errors
- ✅ Loading corrupted files gracefully

### 6. Edge Cases Tests (`TestSettingsEdgeCases`)

- ✅ Empty setting names
- ✅ Deeply nested settings
- ✅ Special characters in setting names
- ✅ None values
- ✅ Complex data types (lists, dicts, booleans)
- ✅ Resetting non-existent settings
- ✅ Getting all settings

## Running Tests

### Using the Main Test Runner

```bash
# Run all config tests
python test/main.py --modules config

# Run only Settings tests
python test/main.py --file config/test_settings.py

# Run with coverage
python test/main.py --modules config --coverage

# Run with summary only
python test/main.py --modules config --summary-only
```

### Using the Config Test Runner

```bash
# Run all config tests
python test/config/run_tests.py

# Run only Settings tests
python test/config/run_tests.py --type settings

# Run with coverage
python test/config/run_tests.py --coverage

# Run quietly (minimal output)
python test/config/run_tests.py --quiet
```

### Direct pytest Usage

```bash
# Run all config tests
python -m pytest test/config/

# Run only Settings tests
python -m pytest test/config/test_settings.py

# Run with verbose output
python -m pytest test/config/test_settings.py -v

# Run specific test class
python -m pytest test/config/test_settings.py::TestSettingsSingleton

# Run specific test method
python -m pytest test/config/test_settings.py::TestSettingsSingleton::test_singleton_instance
```

## Test Statistics

- **Total Tests**: 33
- **Test Classes**: 6
- **Coverage**: All public methods and edge cases
- **Execution Time**: ~0.2 seconds
- **Status**: ✅ All tests passing

## Test Design Principles

The tests follow these principles:

1. **Isolation**: Each test uses temporary files and resets singleton state
2. **Comprehensive Coverage**: Tests cover normal operation, error cases, and edge cases
3. **Realistic Scenarios**: Tests use actual file operations and data structures
4. **Clear Documentation**: Each test has descriptive names and docstrings
5. **Thread Safety**: Tests verify singleton pattern works correctly in multi-threaded environments

## Dependencies

The tests require:

- `unittest` (standard library)
- `tempfile` (standard library)
- `threading` (standard library)
- `pathlib` (standard library)
- `json` (standard library)
- `pytest` (for test discovery and execution)
- `box` (for Settings class functionality)

## Notes

- Tests use temporary directories to avoid affecting the actual project settings
- Singleton instance is reset between test classes to ensure isolation
- File operations are tested with both valid and invalid data
- Error handling is thoroughly tested for robustness
- Thread safety is verified with multi-threaded test scenarios
