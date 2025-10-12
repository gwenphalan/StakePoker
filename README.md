# StakePoker - Real-Time Poker Advisor

A real-time poker advisor for Stake.us that analyzes game state, calculates GTO-based decisions, and provides recommendations via an overlay interface.

## Project Status

**Current Phase:** Early Development - Capture Module Complete

### Completed Modules
- ✅ `/config/` - Configuration management and monitor detection
- ✅ `/capture/` - Screen capture and region extraction

### In Progress
- 🔄 `/parser/` - Image preprocessing and OCR parsing

### Planned Modules
- ⏳ `/models/` - Pydantic data models
- ⏳ `/tracker/` - Game state tracking and turn detection
- ⏳ `/advisor/` - GTO decision engine and equity calculation
- ⏳ `/overlay/` - Transparent recommendation display
- ⏳ `/history/` - Hand history tracking and export

## Features

### Current
- Multi-monitor detection and configuration
- Region-based screen capture from Monitor 2
- JSON-defined region extraction
- Modular architecture with clean separation of concerns

### Planned
- **Turn Detection:** Color-based detection (purple = active turn, red = time bank)
- **Position Awareness:** Automatic position calculation (BTN/SB/BB/UTG/etc)
- **GTO Recommendations:** Preflop charts by position, postflop equity-based decisions
- **Hand History:** Complete hand tracking with profit/loss
- **Table Info Parsing:** Automatic SB/BB/stakes detection
- **Hero Detection:** Configurable username matching

## Architecture

```
StakePoker/
├── src/
│   ├── capture/      # Screen capture and region extraction
│   ├── parser/       # OCR and image processing
│   ├── models/       # Data models (Pydantic)
│   ├── tracker/      # Game state tracking
│   ├── advisor/      # Decision engine (GTO + equity)
│   ├── overlay/      # UI overlay display
│   ├── history/      # Hand history storage
│   ├── config/       # Configuration management
│   └── main.py       # Application entry point
├── data/
│   ├── regions.json          # Screen region definitions
│   ├── user_config.json      # User settings
│   ├── gto_charts/           # Preflop GTO ranges
│   └── backgrounds/          # Reference images
├── archive/          # Prototype codebase (reference only)
└── requirements.txt
```

## Archive

The `archive/` directory contains a previous monolithic prototype that successfully demonstrated:
- Timer bar color detection (purple/red)
- Transparency detection for folded players
- Basic OCR and region parsing

This prototype became unmaintainable due to lack of structure. The current project is a complete rewrite with proper architecture, focusing on:
- Modular design with single-responsibility modules
- Clean interfaces between components
- Comprehensive error handling and logging
- Extensibility for future features

**Note:** Files in `archive/` end with `.py.archive` to prevent IDE confusion.

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

### Monitor Setup
The application automatically detects Monitor 2. Configure in `data/user_config.json`:
```json
{
  "monitor": 2,
  "usernames": ["YourUsername1", "YourUsername2"]
}
```

### Region Definitions
Screen regions are defined in `data/regions.json` with pixel coordinates for each element (cards, pot, player info, etc).

## Usage

```python
from src.capture.region_extractor import RegionExtractor

with RegionExtractor() as extractor:
    regions = extractor.extract_all_regions()
    # regions is dict[str, np.ndarray]
```

## Dependencies

**Core:**
- opencv-python - Image processing
- easyocr - OCR engine
- Pillow - Image manipulation
- numpy - Array operations
- mss - Screen capture

**Poker:**
- eval7 - Equity calculation
- poker - Card/hand/range parsing
- pandas - Data management

**Quality:**
- pydantic - Data validation
- pytest - Testing

## Development

The project follows a phased development approach:
1. ✅ Capture infrastructure
2. 🔄 Parsing and detection
3. ⏳ State tracking
4. ⏳ Decision engine
5. ⏳ Overlay UI
6. ⏳ Hand history

See `PLAN.md` for detailed development roadmap.

## License

Private project - not for distribution.