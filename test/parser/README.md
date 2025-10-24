# Parser Module Tests

This directory contains comprehensive tests for the parser module, including unit tests, integration tests, and visual tests for validating parser accuracy with real poker gameplay data.

## Test Structure

```
test/parser/
├── __init__.py                 # Parser test module
├── test_utils.py              # Common test utilities and base classes
├── visual_test.py             # Interactive visual testing script
├── run_tests.py               # Comprehensive test runner
├── test_card_parser.py        # CardParser unit tests
├── test_ocr_engine.py         # OCREngine unit tests
├── test_money_parser.py       # MoneyParser unit tests
└── README.md                  # This file
```

## Test Types

### Unit Tests

- **CardParser**: Tests card rank and suit detection
- **OCREngine**: Tests text extraction and preprocessing
- **MoneyParser**: Tests monetary amount parsing
- **NameParser**: Tests player name detection
- **StatusParser**: Tests player status detection
- **TableInfoParser**: Tests table information parsing
- **TimerDetector**: Tests turn timer detection
- **TransparencyDetector**: Tests folded player detection
- **DealerDetector**: Tests dealer button detection
- **HandIdParser**: Tests hand ID number parsing

### Integration Tests

- Full parsing workflows
- Multi-component interactions
- Error handling and recovery
- Performance benchmarks

### Visual Tests

- Interactive validation with real poker images
- Manual approval/rejection of parser results
- Test result storage and aggregation
- Random frame extraction from gameplay videos

## Running Tests

### Using the Unified Test Runner

```bash
# Run all parser unit tests
python test/main.py --modules parser --type unit

# Run all parser integration tests
python test/main.py --modules parser --type integration

# Run parser visual tests
python test/main.py --parser-visual --parser-name card --region-name card_1 --num-visual-tests 20

# Run with coverage
python test/main.py --modules parser --coverage
```

### Using the Parser-Specific Test Runner

```bash
# Run unit tests
python test/parser/run_tests.py --unit

# Run integration tests
python test/parser/run_tests.py --integration

# Run visual tests
python test/parser/run_tests.py --visual --parser card --region card_1 --tests 20

# Run performance tests
python test/parser/run_tests.py --performance

# Run all tests
python test/parser/run_tests.py --all

# Show test statistics
python test/parser/run_tests.py --stats card

# List available tests
python test/parser/run_tests.py --list
```

### Using the Visual Test Script Directly

```bash
# Test card parser on card_1 region
python test/parser/visual_test.py --parser card --region card_1 --tests 20

# Test money parser on pot region
python test/parser/visual_test.py --parser money --region pot --tests 10

# List available parsers and regions
python test/parser/visual_test.py --list-parsers
python test/parser/visual_test.py --list-regions

# Show statistics for a parser
python test/parser/visual_test.py --stats card
```

## Visual Testing

The visual testing system provides an interactive way to validate parser accuracy using real poker gameplay data:

### Features

- **Random Frame Extraction**: Automatically extracts random frames from poker videos in `data/video/`
- **Region Extraction**: Uses existing capture modules to extract specific regions from frames
- **Interactive Validation**: Shows region image, parsed value, and confidence for manual approval
- **Keybinds**:
  - `Y` - Approve result
  - `N` - Reject result
  - `S` - Skip test
  - `Q` - Quit testing
  - `H` - Show help
  - `ESC` - Quit testing
- **Result Storage**: Saves all test results to `data/tests/` for analysis
- **Statistics**: Provides approval rates and confidence metrics

### Available Parsers and Regions

| Parser         | Regions                                                                 | Description                  |
| -------------- | ----------------------------------------------------------------------- | ---------------------------- |
| `card`         | `card_1`, `card_2`, `card_3`, `card_4`, `card_5`                        | Card rank and suit detection |
| `money`        | `pot`, `stack_1`, `stack_2`, `stack_3`, `stack_4`, `stack_5`, `stack_6` | Monetary amount parsing      |
| `name`         | `name_1`, `name_2`, `name_3`, `name_4`, `name_5`, `name_6`              | Player name detection        |
| `status`       | `status_1`, `status_2`, `status_3`, `status_4`, `status_5`, `status_6`  | Player status detection      |
| `timer`        | `name_1`, `name_2`, `name_3`, `name_4`, `name_5`, `name_6`              | Turn timer detection         |
| `transparency` | `name_1`, `name_2`, `name_3`, `name_4`, `name_5`, `name_6`              | Folded player detection      |
| `dealer`       | `dealer_button`                                                         | Dealer button detection      |
| `hand_id`      | `hand_id`                                                               | Hand ID number parsing       |
| `table_info`   | `table_info`                                                            | Table information parsing    |

## Test Data

### Video Files

Place poker gameplay videos in `data/video/` directory. Supported formats:

- `.mp4`
- `.avi`
- `.mov`
- `.mkv`
- `.webm`

### Test Results

Visual test results are stored in `data/tests/` directory:

- `session_YYYYMMDD_HHMMSS.json` - Session metadata
- `results_YYYYMMDD_HHMMSS.json` - Test results
- `{parser_name}_YYYYMMDD_HHMMSS.json` - Individual test results

### Region Definitions

Region definitions are loaded from `data/regions.json` and should match the regions used in your poker videos.

## Test Utilities

### TestResultManager

Manages test result storage and aggregation:

- Save individual test results
- Save complete testing sessions
- Load and filter test results
- Generate parser statistics

### VisualTestValidator

Provides interactive validation interface:

- Display parser results with region images
- Handle keybind input
- Show help information
- Manage validation workflow

### VideoFrameExtractor

Extracts random frames from poker videos:

- Discover video files automatically
- Extract random frames with frame numbers
- Handle different video formats
- Provide frame metadata

### RegionTestExtractor

Extracts regions from frames for testing:

- Load region definitions
- Extract specific regions from frames
- Validate region bounds
- Provide region metadata

## Best Practices

### Running Visual Tests

1. **Start Small**: Begin with 5-10 tests to get familiar with the interface
2. **Focus on Problem Areas**: Test parsers that are known to have issues
3. **Use Consistent Regions**: Test the same regions across different parsers
4. **Document Issues**: Use the notes field to record specific problems
5. **Regular Testing**: Run visual tests regularly to catch regressions

### Analyzing Results

1. **Check Approval Rates**: Look for parsers with low approval rates
2. **Review Confidence Scores**: Low confidence may indicate preprocessing issues
3. **Identify Patterns**: Look for common failure modes
4. **Compare Methods**: Check which preprocessing methods work best
5. **Track Improvements**: Monitor approval rates over time

### Test Data Management

1. **Use Representative Videos**: Include various game conditions and lighting
2. **Regular Updates**: Add new videos as game client updates
3. **Backup Results**: Keep test results for historical analysis
4. **Clean Old Data**: Periodically clean up old test results

## Troubleshooting

### Common Issues

**No video files found**

- Ensure videos are in `data/video/` directory
- Check video file formats are supported
- Verify file permissions

**Region extraction fails**

- Check `data/regions.json` exists and is valid
- Verify region coordinates match your video resolution
- Ensure region bounds are within frame dimensions

**OCR not working**

- Check EasyOCR installation
- Verify GPU/CUDA setup if using GPU acceleration
- Test with simple images first

**Visual test window not responding**

- Check OpenCV installation
- Verify display settings
- Try running in different terminal

### Debug Mode

Run tests with debug mode for detailed logging:

```bash
python test/main.py --modules parser --debug
```

### Performance Issues

If tests are running slowly:

- Reduce number of visual tests
- Use smaller test images
- Check system resources
- Consider using GPU acceleration for OCR

## Contributing

When adding new parser tests:

1. **Follow Naming Conventions**: Use `test_{parser_name}.py` format
2. **Include All Test Types**: Unit, integration, and performance tests
3. **Add Visual Test Support**: Include parser in visual test registry
4. **Update Documentation**: Add parser to this README
5. **Test Edge Cases**: Include tests for error conditions
6. **Use Proper Markers**: Mark tests with `@pytest.mark.unit`, `@pytest.mark.integration`, etc.

## Dependencies

Required packages for parser tests:

- `pytest` - Test framework
- `opencv-python` - Image processing
- `numpy` - Numerical operations
- `easyocr` - OCR engine
- `pydantic` - Data validation
- `scikit-image` - Image analysis (optional)
- `hypothesis` - Property-based testing (optional)

Install with:

```bash
pip install -r requirements.txt
```

