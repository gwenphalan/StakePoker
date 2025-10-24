# History Module Tests

Comprehensive unit tests for the `src/history/` module covering database operations, session management, and CSV export functionality.

## Test Coverage

### HandStorage Class Tests (`TestHandStorage`)

- **Database Operations**: Table creation, session management, hand persistence
- **Data Serialization**: Card and TableInfo JSON serialization/deserialization
- **CRUD Operations**: Create, read, update, delete for sessions and hands
- **Error Handling**: Database errors, invalid data, connection failures
- **Data Integrity**: Foreign key relationships, transaction handling

**Key Test Methods:**

- `test_initialization()` - Database setup and table creation
- `test_create_session()` - Session creation and validation
- `test_save_hand()` - Hand record persistence with actions
- `test_get_hands_for_session()` - Session-based hand retrieval
- `test_card_serialization()` - Card data serialization
- `test_database_error_handling()` - Error recovery and logging

### SessionTracker Class Tests (`TestSessionTracker`)

- **Session Lifecycle**: Start, end, active state management
- **Statistics Calculation**: Profit/loss, win rates, duration metrics
- **State Management**: Session state tracking and validation
- **Error Handling**: Invalid operations, database failures
- **Integration**: Hand recording and session statistics

**Key Test Methods:**

- `test_start_session()` - Session initialization and validation
- `test_end_session()` - Session termination and statistics
- `test_record_hand_in_session()` - Hand recording and state updates
- `test_get_session_stats()` - Statistics calculation and formatting
- `test_get_current_session_summary()` - Real-time session data

### HandExporter Class Tests (`TestHandExporter`)

- **CSV Export**: Session and summary data export
- **Data Formatting**: Card strings, action sequences, statistics
- **File Operations**: Path handling, directory creation, error recovery
- **Data Transformation**: HandRecord to CSV row conversion
- **Export Types**: Single session, all hands, session summaries

**Key Test Methods:**

- `test_export_session_to_csv()` - Single session export
- `test_export_all_hands_to_csv()` - Multi-session export
- `test_export_session_summary_to_csv()` - Statistics export
- `test_format_card()` - Card string formatting
- `test_format_action_sequence()` - Action sequence formatting
- `test_flatten_hand_to_row()` - HandRecord to CSV conversion

## Running Tests

### Using the Test Runner

```bash
# Run all history tests
python test/history/run_tests.py

# Run specific test class
python test/history/run_tests.py --type hand_storage
python test/history/run_tests.py --type session_tracker
python test/history/run_tests.py --type hand_exporter

# Run with coverage
python test/history/run_tests.py --coverage

# Quiet mode
python test/history/run_tests.py --quiet
```

### Using pytest Directly

```bash
# Run all tests
python -m pytest test/history/test_history.py -v

# Run specific test class
python -m pytest test/history/test_history.py::TestHandStorage -v
python -m pytest test/history/test_history.py::TestSessionTracker -v
python -m pytest test/history/test_history.py::TestHandExporter -v

# Run with coverage
python -m pytest test/history/test_history.py --cov=src.history --cov-report=term-missing
```

### Using the Unified Test Runner

```bash
# Run all history tests
python test/main.py --modules history

# Run with coverage
python test/main.py --modules history --coverage
```

## Test Data and Mocking

### Sample Test Data

- **Cards**: Ace of Hearts, King of Spades for consistent testing
- **Table Info**: 1.0/2.0 blinds for standard testing
- **Actions**: Raise, call, fold sequences for action testing
- **Hand Records**: Complete HandRecord instances with all fields

### Mocking Strategy

- **HandStorage**: Mocked for SessionTracker and HandExporter tests
- **Database Operations**: Temporary SQLite databases for integration testing
- **File Operations**: Temporary files for CSV export testing
- **Error Simulation**: Database failures, IO errors, invalid data

## Test Design Principles

### Database Testing

- **Isolation**: Each test uses temporary database
- **Cleanup**: Proper teardown of test resources
- **Transactions**: Test transaction handling and rollback
- **Schema**: Verify table creation and indexes

### Session Management Testing

- **State Validation**: Active/inactive session states
- **Statistics Accuracy**: Profit calculations, win rates, durations
- **Error Conditions**: Invalid operations, database failures
- **Integration**: Hand recording and session tracking

### Export Testing

- **Data Integrity**: Verify CSV content matches source data
- **Format Validation**: Card strings, action sequences, statistics
- **File Handling**: Path creation, error recovery, cleanup
- **Performance**: Large dataset handling, memory usage

## Dependencies

### Required Packages

- `unittest` - Core testing framework
- `tempfile` - Temporary file/database creation
- `csv` - CSV file validation
- `json` - Data serialization testing
- `datetime` - Timestamp handling
- `pathlib` - Path operations
- `unittest.mock` - Mocking and patching

### Test Data Models

- `HandRecord` - Complete hand data structure
- `Action` - Individual player actions
- `Card` - Playing card representation
- `TableInfo` - Table configuration data

## Error Scenarios Tested

### Database Errors

- Connection failures
- Transaction rollbacks
- Invalid SQL operations
- Schema creation failures

### Session Errors

- Starting session when active
- Ending session when inactive
- Database operation failures
- Invalid session IDs

### Export Errors

- File system errors
- Invalid file paths
- CSV formatting errors
- Data serialization failures

## Performance Considerations

### Database Performance

- Index usage verification
- Query optimization testing
- Large dataset handling
- Memory usage monitoring

### Export Performance

- Large session export testing
- Memory-efficient CSV writing
- File I/O optimization
- Error recovery efficiency

## Integration Testing

### Cross-Module Integration

- HandStorage with SessionTracker
- SessionTracker with HandExporter
- Complete workflow testing
- Data consistency validation

### Real-World Scenarios

- Multi-session poker games
- Large hand history exports
- Session statistics calculation
- Error recovery workflows

## Maintenance Notes

### Test Data Updates

- Update sample data when models change
- Maintain backward compatibility
- Verify test data validity
- Update mock expectations

### Database Schema Changes

- Update table creation tests
- Verify migration compatibility
- Test new field handling
- Update serialization tests

### Export Format Changes

- Update CSV column tests
- Verify data formatting
- Test backward compatibility
- Update documentation
