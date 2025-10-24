# Tracker Module Tests

Comprehensive unit tests for the `src/tracker/` module covering game state tracking, turn detection, position calculation, and hand lifecycle management.

## Test Coverage

### HandTracker Class Tests (`TestHandTracker`)

- **Hand Lifecycle**: Start, update, finalize hand tracking
- **Action Detection**: Bet, call, raise, fold, check detection
- **Profit Calculation**: Hero profit/loss calculation and result determination
- **State Management**: Hand state tracking and validation
- **Integration**: Session tracker integration and persistence

**Key Test Methods:**

- `test_start_new_hand()` - Hand initialization and hero detection
- `test_update_hand()` - Action detection and state updates
- `test_detect_actions_*()` - Individual action type detection
- `test_finalize_hand()` - Hand completion and result calculation
- `test_calculate_hero_profit_loss()` - Profit/loss calculation
- `test_determine_hand_result()` - Result determination logic

### HeroDetector Class Tests (`TestHeroDetector`)

- **Hero Identification**: Name parsing and hero detection
- **Confidence Validation**: Confidence threshold checking
- **Multiple Candidates**: Handling multiple hero candidates
- **Settings Management**: Configuration and username management
- **Error Handling**: Invalid inputs and parsing failures

**Key Test Methods:**

- `test_detect_hero_seat_success()` - Successful hero detection
- `test_detect_hero_seat_low_confidence()` - Confidence threshold validation
- `test_detect_hero_seat_multiple_candidates()` - Multiple hero handling
- `test_detect_hero_seat_disabled()` - Disabled detection
- `test_get_hero_usernames()` - Username management
- `test_get_detection_stats()` - Statistics retrieval

### PositionCalculator Class Tests (`TestPositionCalculator`)

- **Position Mapping**: All table sizes (2-8 players)
- **Button Calculation**: Dealer button position handling
- **Position Assignment**: Correct position assignment logic
- **Edge Cases**: Invalid inputs and unsupported table sizes
- **Validation**: Input validation and error handling

**Key Test Methods:**

- `test_calculate_positions_heads_up()` - 2-player heads-up
- `test_calculate_positions_6max()` - 6-max table
- `test_calculate_positions_8max()` - 8-max table
- `test_calculate_positions_invalid_button()` - Invalid button handling
- `test_calculate_positions_unsupported_table_size()` - Fallback behavior
- `test_position_maps_coverage()` - Position map validation

### StateMachine Class Tests (`TestStateMachine`)

- **State Coordination**: Integration of all parsers and trackers
- **Hand Tracking**: New hand detection and tracking
- **Hero Detection**: Hero seat identification
- **Parsing Integration**: All parser coordination
- **State Management**: Current/previous state handling

**Key Test Methods:**

- `test_update_hand_id_parsing()` - Hand ID parsing
- `test_update_new_hand_detection()` - New hand detection
- `test_update_hero_detection()` - Hero identification
- `test_update_player_parsing()` - Player data parsing
- `test_update_community_cards_parsing()` - Community cards
- `test_update_phase_determination()` - Phase detection
- `test_update_hand_tracking()` - Hand tracking integration

### TurnDetector Class Tests (`TestTurnDetector`)

- **Turn Detection**: Hero turn identification
- **Timer Validation**: Timer state analysis
- **Confidence Checking**: Confidence threshold validation
- **Multiple Players**: Cross-player timer analysis
- **Settings Management**: Configuration handling

**Key Test Methods:**

- `test_detect_hero_turn_success()` - Successful turn detection
- `test_detect_hero_turn_not_hero()` - Non-hero turn detection
- `test_detect_hero_turn_low_confidence()` - Confidence validation
- `test_detect_hero_turn_no_active_indicators()` - No active turns
- `test_detect_hero_turn_disabled()` - Disabled detection
- `test_detect_hero_turn_no_validation()` - Validation bypass

## Running Tests

### Using the Test Runner

```bash
# Run all tracker tests
python test/tracker/run_tests.py

# Run specific test class
python test/tracker/run_tests.py --type hand_tracker
python test/tracker/run_tests.py --type hero_detector
python test/tracker/run_tests.py --type position_calculator
python test/tracker/run_tests.py --type state_machine
python test/tracker/run_tests.py --type turn_detector

# Run with coverage
python test/tracker/run_tests.py --coverage

# Quiet mode
python test/tracker/run_tests.py --quiet
```

### Using pytest Directly

```bash
# Run all tests
python -m pytest test/tracker/test_tracker.py -v

# Run specific test class
python -m pytest test/tracker/test_tracker.py::TestHandTracker -v
python -m pytest test/tracker/test_tracker.py::TestHeroDetector -v
python -m pytest test/tracker/test_tracker.py::TestPositionCalculator -v
python -m pytest test/tracker/test_tracker.py::TestStateMachine -v
python -m pytest test/tracker/test_tracker.py::TestTurnDetector -v

# Run with coverage
python -m pytest test/tracker/test_tracker.py --cov=src.tracker --cov-report=term-missing
```

### Using the Unified Test Runner

```bash
# Run all tracker tests
python test/main.py --modules tracker

# Run with coverage
python test/main.py --modules tracker --coverage
```

## Test Data and Mocking

### Sample Test Data

- **Game States**: Complete GameState instances with players, cards, pot
- **Players**: Player objects with all required fields populated
- **Cards**: Card instances for hole cards and community cards
- **Actions**: Action objects representing player actions
- **Hand Records**: Complete HandRecord instances for testing

### Mocking Strategy

- **Parsers**: All parser classes mocked for StateMachine tests
- **Session Tracker**: Mocked for HandTracker integration tests
- **Timer Detector**: Mocked for TurnDetector tests
- **Name Parser**: Mocked for HeroDetector tests
- **Settings**: Mocked for configuration testing

## Test Design Principles

### Hand Tracking Testing

- **Lifecycle Validation**: Start, update, finalize hand phases
- **Action Detection**: All poker action types and edge cases
- **Profit Calculation**: Accurate profit/loss calculation
- **State Consistency**: Hand state validation and tracking
- **Integration**: Session tracker and persistence testing

### Position Calculation Testing

- **Table Size Coverage**: All supported table sizes (2-8 players)
- **Position Accuracy**: Correct position assignment logic
- **Button Handling**: Dealer button position validation
- **Edge Cases**: Invalid inputs and error conditions
- **Fallback Behavior**: Unsupported table size handling

### Turn Detection Testing

- **Timer Analysis**: Timer state detection and validation
- **Confidence Thresholds**: Confidence-based validation
- **Cross-Player Analysis**: Multi-player timer comparison
- **False Positive Prevention**: Validation logic testing
- **Configuration**: Settings-based behavior testing

### State Machine Testing

- **Parser Integration**: All parser coordination and error handling
- **State Management**: Current/previous state tracking
- **Hand Detection**: New hand detection and tracking
- **Hero Identification**: Hero seat detection and validation
- **Error Recovery**: Graceful failure handling

## Dependencies

### Required Packages

- `unittest` - Core testing framework
- `numpy` - Array operations for mock data
- `unittest.mock` - Mocking and patching
- `datetime` - Timestamp handling
- `typing` - Type hints

### Test Data Models

- `GameState` - Complete game state structure
- `Player` - Player data with all fields
- `Card` - Playing card representation
- `Action` - Player action representation
- `HandRecord` - Hand history record
- `TableInfo` - Table configuration

## Error Scenarios Tested

### Hand Tracking Errors

- Invalid hand states
- Action detection failures
- Profit calculation errors
- Session tracker failures
- State inconsistency

### Position Calculation Errors

- Invalid button positions
- Unsupported table sizes
- Missing players
- Invalid seat numbers
- Calculation failures

### Turn Detection Errors

- Low confidence timers
- No active indicators
- Invalid hero seats
- Timer parsing failures
- Validation errors

### State Machine Errors

- Parser failures
- Invalid regions
- State inconsistency
- Hand tracking errors
- Integration failures

## Performance Considerations

### Hand Tracking Performance

- Action detection efficiency
- State update performance
- Memory usage optimization
- Large hand history handling
- Database operation efficiency

### Position Calculation Performance

- Calculation speed for all table sizes
- Memory-efficient position mapping
- Fast button position lookup
- Efficient player sorting
- Minimal computation overhead

### Turn Detection Performance

- Timer analysis speed
- Cross-player comparison efficiency
- Confidence calculation performance
- Real-time detection capability
- Resource usage optimization

## Integration Testing

### Cross-Module Integration

- HandTracker with SessionTracker
- StateMachine with all parsers
- TurnDetector with TimerDetector
- HeroDetector with NameParser
- PositionCalculator with GameState

### Real-World Scenarios

- Complete hand lifecycle
- Multi-hand sessions
- Position changes
- Turn detection accuracy
- State consistency validation

## Maintenance Notes

### Test Data Updates

- Update sample data when models change
- Maintain backward compatibility
- Verify test data validity
- Update mock expectations

### Parser Integration Changes

- Update StateMachine mocks when parsers change
- Verify parser coordination logic
- Test error handling scenarios
- Update integration tests

### Position Mapping Updates

- Update position maps when rules change
- Verify all table size coverage
- Test edge cases and fallbacks
- Update documentation

### Configuration Changes

- Update settings tests when config changes
- Verify default value testing
- Test configuration validation
- Update mock settings
