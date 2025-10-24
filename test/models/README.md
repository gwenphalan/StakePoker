# Models Module Tests

This directory contains comprehensive unit tests for the `src/models/` module, covering all Pydantic data models used throughout the StakePoker application.

## Test Files

- **`test_models.py`** - Complete unit tests for all model classes

## Test Coverage

The tests cover all major functionality of the Pydantic models:

### 1. Card Model Tests (`TestCard`)

- ✅ Valid card creation with all ranks and suits
- ✅ Invalid rank validation (1, 0, B, X, ace, king, etc.)
- ✅ Invalid suit validation (heart, diamond, club, spade, red, black, etc.)
- ✅ String representation (e.g., 'Ah' for Ace of hearts)
- ✅ Card equality comparison
- ✅ Card serialization to dictionary

### 2. Player Model Tests (`TestPlayer`)

- ✅ Valid player creation with all fields
- ✅ Seat number validation (1-8)
- ✅ Stack validation (non-negative)
- ✅ Hole cards validation (max 2 cards)
- ✅ Timer state validation (purple, red, None)
- ✅ Position validation (BTN, SB, BB, UTG, etc.)
- ✅ Current bet validation (non-negative)

### 3. TableInfo Model Tests (`TestTableInfo`)

- ✅ Valid table info creation
- ✅ Blind validation (bb > sb, both positive)
- ✅ Normalization to big blinds calculation

### 4. Decision Model Tests (`TestDecision`)

- ✅ AlternativeAction creation and validation
- ✅ Decision creation with all fields
- ✅ Action validation (fold, call, raise, check, bet)
- ✅ Confidence validation (0.0-1.0)
- ✅ Equity validation (0.0-1.0)
- ✅ Pot odds validation (non-negative)
- ✅ Decision with alternative actions

### 5. HandRecord Model Tests (`TestHandRecord`)

- ✅ Action creation and validation
- ✅ HandRecord creation with all fields
- ✅ Hero position validation (BTN, SB, BB, UTG, etc.)
- ✅ Hero seat validation (1-8)
- ✅ Result validation (won, lost, folded)
- ✅ Final pot validation (non-negative)
- ✅ HandRecord with action sequence

### 6. RegionConfig Model Tests (`TestRegionConfig`)

- ✅ Valid region creation
- ✅ Coordinate validation (non-negative)
- ✅ Dimension validation (positive width/height)
- ✅ Conversion to tuple for mss library

### 7. GameState Model Tests (`TestGameState`)

- ✅ Valid game state creation
- ✅ Phase validation (preflop, flop, turn, river, showdown)
- ✅ Community cards validation by phase
- ✅ Player count validation (2-8 players)
- ✅ Unique seat numbers validation
- ✅ Single hero validation
- ✅ Single dealer validation
- ✅ Pot validation (non-negative)
- ✅ Active player validation (1-8)
- ✅ Button position validation (1-8)
- ✅ get_hero() method
- ✅ get_player_by_seat() method

## Running Tests

### Using the Main Test Runner

```bash
# Run all models tests
python test/main.py --modules models

# Run only models tests
python test/main.py --file models/test_models.py

# Run with coverage
python test/main.py --modules models --coverage

# Run with summary only
python test/main.py --modules models --summary-only
```

### Direct pytest Usage

```bash
# Run all models tests
python -m pytest test/models/

# Run only models tests
python -m pytest test/models/test_models.py

# Run with verbose output
python -m pytest test/models/test_models.py -v

# Run specific test class
python -m pytest test/models/test_models.py::TestCard

# Run specific test method
python -m pytest test/models/test_models.py::TestCard::test_valid_card_creation
```

## Test Statistics

- **Total Tests**: 41
- **Test Classes**: 7
- **Coverage**: All public methods, validation, and edge cases
- **Execution Time**: ~0.16 seconds
- **Status**: ✅ All tests passing

## Test Design Principles

The tests follow these principles:

1. **Comprehensive Validation**: Tests cover all field validations and constraints
2. **Edge Cases**: Tests include boundary conditions and invalid inputs
3. **Cross-Field Validation**: Tests complex validation rules that span multiple fields
4. **Pydantic v2 Compatibility**: Tests use proper dictionary serialization for nested models
5. **Realistic Data**: Tests use realistic poker game data and scenarios
6. **Clear Documentation**: Each test has descriptive names and docstrings

## Key Features Tested

### Validation Rules

- **Field-level validation**: Individual field constraints and types
- **Model-level validation**: Cross-field validation using `@model_validator`
- **Custom validators**: Field-specific validation using `@field_validator`

### Data Integrity

- **Type safety**: All fields have proper type annotations
- **Constraint enforcement**: Min/max values, allowed values, etc.
- **Required fields**: Proper handling of required vs optional fields

### Business Logic

- **Poker rules**: Community cards by phase, player limits, etc.
- **Game state consistency**: Unique seats, single hero/dealer, etc.
- **Data relationships**: Proper handling of nested models

## Dependencies

The tests require:

- `unittest` (standard library)
- `datetime` (standard library)
- `pydantic` (for model validation)
- `pytest` (for test discovery and execution)

## Notes

- Tests use Pydantic v2 syntax with `model_dump()` for serialization
- Nested models are properly converted to dictionaries for validation
- All validation errors are properly tested and verified
- Tests cover both valid and invalid input scenarios
- Complex cross-field validation is thoroughly tested
- Edge cases and boundary conditions are included
