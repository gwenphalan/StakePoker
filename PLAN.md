# Poker Advisor Development Plan

## Project Overview
Real-time poker advisor for Stake.us: screenshots Monitor 2, parses game state, calculates GTO decisions, displays recommendations via overlay, tracks hand history.

## Core Modules & Files

### 1. capture/
- `screen_capture.py` - mss capture of Monitor 2
- `monitor_config.py` - Detect/validate Monitor 2
- `region_loader.py` - Load regions.json
- `region_extractor.py` - Crop regions from screenshot

### 2. parser/
- `image_preprocessor.py` - Background subtraction (NOT for timer detection)
- `card_parser.py` - Parse cards (rank + suit)
- `money_parser.py` - Parse pot/stacks/bets via OCR
- `status_parser.py` - Parse player status text
- `transparency_detector.py` - Detect folded players (half-transparent nameplates)
- `dealer_detector.py` - Find dealer button for position calculation
- `timer_detector.py` - Simple color presence check in stack regions
- `table_info_parser.py` - Parse table_name region for SB/BB/min/max/currency
- `hand_id_parser.py` - Parse hand_num region for unique hand tracking
- `name_parser.py` - Parse player names to identify hero seat
- `ocr_engine.py` - easyocr wrapper

**Timer Detection (Simple):**
- Check player stack region for color presence
- Purple = player's turn (normal time)
- Red = player's turn in time bank (running out)
- Simple pixel count: if >50 pixels match color → present
- RGB ranges only, no HSV/saturation/contrast nonsense
- Stack region preferred over name region (larger = more reliable)

```python
# Purple: active turn
PURPLE_RGB = ((120, 200), (50, 150), (150, 255))  # R, G, B ranges

# Red: time bank active
RED_RGB = ((180, 255), (0, 80), (0, 80))
```

**Table Info Parsing:**
- Parse table_name region (e.g., "NLH 0.10/0.20 Gold $5-$20")
- Extract: SB (0.10), BB (0.20), Currency (Gold/SC), Min buyin ($5), Max buyin ($20)
- Game type: NLH/PLO/etc

**Hero Seat Detection:**
- Config file with user's known usernames
- Match parsed names against config
- Identify which physical seat (player_1 through player_8) is hero

### 3. models/
- `card.py` - Card(rank, suit)
- `player.py` - Player(seat_number, position, stack, hole_cards, timer_state, is_dealer, current_bet, is_hero)
- `game_state.py` - GameState(players, community_cards, pot, phase, active_player, button_position, hand_id, table_info)
- `table_info.py` - TableInfo(sb, bb, currency, min_buyin, max_buyin, game_type)
- `hand_record.py` - HandRecord(hand_id, timestamp, table_info, hero_position, hero_cards, actions, result, profit_loss)
- `regions.py` - RegionConfig for regions.json
- `decision.py` - Decision(action, amount, confidence, reasoning)
- `config.py` - UserConfig(usernames: list[str]) for hero detection

**timer_state:** 'purple' / 'red' / None  
**seat_number:** 1-8 (physical seat on table)  
**position:** BTN/SB/BB/UTG/etc (relative to dealer)

### 4. tracker/
- `state_machine.py` - Track game state across frames
- `turn_detector.py` - Check hero's stack for purple/red
- `phase_detector.py` - Detect preflop/flop/turn/river transitions
- `action_detector.py` - Track action sequence
- `position_calculator.py` - Calculate BTN/SB/BB/UTG from dealer button
- `hand_tracker.py` - Track hand history by hand_id
- `hero_detector.py` - Identify hero's seat from configured usernames

**Turn Detection:** Hero has purple OR red in stack region = hero's turn  
**Hero Detection:** Match parsed names to config usernames, store seat_number  
**Hand Tracking:** New hand_id = new HandRecord, track until showdown/fold

### 5. advisor/
- `gto_loader.py` - Load preflop charts by position
- `equity_calculator.py` - eval7 integration
- `range_estimator.py` - Estimate ranges by position + actions
- `decision_engine.py` - Recommend fold/call/raise
- `postflop_solver.py` - Postflop equity + EV calc

**Logic:** Preflop uses GTO charts, postflop uses equity vs ranges + pot odds

### 6. overlay/
- `window.py` - Transparent tkinter window
- `renderer.py` - Draw recommendations
- `position_manager.py` - Position above Monitor 2

**Display:** Action, position, equity, pot odds, time bank indicator (if red), current hand_id

### 7. history/
- `hand_storage.py` - Save/load hand history (SQLite or JSON)
- `hand_exporter.py` - Export hands to CSV/PokerTracker format
- `session_tracker.py` - Track profit/loss per session

**Storage:** Each HandRecord saved with all game state, actions, and outcome

### 8. config/
- `user_config.json` - Usernames for hero detection
- `settings.py` - Load user config, validate usernames

### 9. main.py
Event loop: capture → parse → detect hero → detect hand → detect turn → calculate decision → update overlay → log hand history (500ms loop)

## Development Order

**Days 1-2:** Capture - Screenshot Monitor 2, load regions, crop regions  
**Day 3:** Preprocessing - Background subtraction for OCR (skip for timers)  
**Days 4-5:** Card + Money parsers  
**Day 6:** Table info parser + Hand ID parser  
**Day 7:** Name parser + Hero detection with config  
**Day 8:** Transparency + Dealer detection  
**Day 9:** Timer detection - Simple color presence in stack regions  
**Days 10-11:** Models + Position calculation  
**Day 12:** State tracking + Turn detection  
**Day 13:** Hand tracking and history storage  
**Days 14-16:** GTO loader + Equity calc + Decision engine  
**Days 17-18:** Overlay with hand info display  
**Days 19-20:** Integration + Testing  
**Day 21:** Hand history export and session stats

## Key Technical Notes

**Timer Detection (from old project):**
- Use stack region, not name region (larger = better)
- Simple pixel counting, no complex color processing
- Purple = normal turn, Red = time bank
- Both mean "hero's turn to act"

**Hero Detection:**
- User provides list of known usernames in config
- Each frame, parse all player names
- Match against config list
- Store hero's seat_number (1-8) for session
- If hero changes seats, detect and update

**Table Info:**
- Parse once per hand (doesn't change mid-hand)
- Use for BB normalization (stack sizes in BBs)
- Currency affects bankroll tracking (Gold separate from SC)

**Hand Tracking:**
- Unique identifier: hand_id from hand_num region
- Start tracking on new hand_id
- Record: hero seat, position, cards, all actions, showdown, result
- Calculate profit/loss at hand end
- Store to database/JSON

**Dealer Button:**
- For position calculation only (BTN/SB/BB/etc)
- Not for turn detection

**Performance:**
- 2 FPS is fine (500ms loop)
- Only calculate decision when hero's turn (purple/red detected)
- Cache GTO charts and background subtraction template
- Database writes async to avoid blocking main loop

**Old Project Migration:**
- Port timer color detection (already working)
- Port transparency detection
- Name parsing now critical (for hero detection) - improve reliability
- Add position awareness (new)
- Add GTO integration (new)
- Add hand history tracking (new)

## File Structure
```
StakePoker/
├── src/
│   ├── capture/
│   │   ├── __init__.py
│   │   ├── screen_capture.py
│   │   ├── monitor_config.py
│   │   ├── region_loader.py
│   │   └── region_extractor.py
│   ├── parser/
│   │   ├── __init__.py
│   │   ├── image_preprocessor.py
│   │   ├── card_parser.py
│   │   ├── money_parser.py
│   │   ├── status_parser.py
│   │   ├── transparency_detector.py
│   │   ├── dealer_detector.py
│   │   ├── timer_detector.py
│   │   ├── table_info_parser.py
│   │   ├── hand_id_parser.py
│   │   ├── name_parser.py
│   │   └── ocr_engine.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── card.py
│   │   ├── player.py
│   │   ├── game_state.py
│   │   ├── table_info.py
│   │   ├── hand_record.py
│   │   ├── regions.py
│   │   ├── decision.py
│   │   └── config.py
│   ├── tracker/
│   │   ├── __init__.py
│   │   ├── state_machine.py
│   │   ├── turn_detector.py
│   │   ├── phase_detector.py
│   │   ├── action_detector.py
│   │   ├── position_calculator.py
│   │   ├── hand_tracker.py
│   │   └── hero_detector.py
│   ├── advisor/
│   │   ├── __init__.py
│   │   ├── gto_loader.py
│   │   ├── equity_calculator.py
│   │   ├── range_estimator.py
│   │   ├── decision_engine.py
│   │   └── postflop_solver.py
│   ├── overlay/
│   │   ├── __init__.py
│   │   ├── window.py
│   │   ├── renderer.py
│   │   └── position_manager.py
│   ├── history/
│   │   ├── __init__.py
│   │   ├── hand_storage.py
│   │   ├── hand_exporter.py
│   │   └── session_tracker.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   └── main.py
├── data/
│   ├── regions.json
│   ├── user_config.json
│   ├── hand_history.db
│   ├── gto_charts/
│   └── backgrounds/
├── requirements.txt
└── README.md
```

## Success Metrics
- Timer detection: >95% accuracy, distinguishes purple vs red
- Turn triggers: Recommendations appear when purple OR red detected
- Hero detection: 100% accuracy when username parsed correctly
- Position calc: 100% accurate when dealer button found
- Hand tracking: Every hand recorded with unique hand_id
- Table info: SB/BB parsed correctly >90% of time
- GTO preflop: Matches charts by position
- Postflop: Equity within 2% of solver
- Runs full session without crashes
- Hand history exportable and complete

## Additional Features

**Hand History Database:**
- SQLite database with tables: hands, actions, players, sessions
- Query by date range, position, hand type, profit/loss
- Export to CSV or PokerTracker format

**Session Tracking:**
- Track cumulative profit/loss per session
- Separate tracking for Gold vs SC currency
- Show session stats in overlay (optional)

**Hero Configuration:**
- JSON config with list of usernames
- Easy to add/remove usernames
- Validates on startup

**Table Info Usage:**
- Normalize all money values to BBs
- Adjust ranges for different stack depths
- Track performance by stake level