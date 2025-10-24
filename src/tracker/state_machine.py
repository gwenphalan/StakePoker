#!/usr/bin/env python3
"""
Game state machine for tracking poker game state across frames.

Coordinates all parsers and trackers to maintain complete game state,
detect changes, and handle new hands. Simple approach - parse everything
each frame with Pydantic validation.

Usage:
    from src.tracker.state_machine import GameStateMachine
    
    state_machine = GameStateMachine()
    game_state = state_machine.update(regions)
    if game_state:
        print(f"Current phase: {game_state.phase}")
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np

# Parser imports
from src.parser.card_parser import CardParser
from src.parser.money_parser import MoneyParser
from src.parser.hand_id_parser import HandIdParser
from src.parser.table_info_parser import TableInfoParser
from src.parser.name_parser import NameParser
from src.parser.status_parser import StatusParser
from src.parser.dealer_detector import DealerDetector
from src.parser.timer_detector import TimerDetector
from src.parser.transparency_detector import TransparencyDetector

# Tracker imports
from src.tracker.hero_detector import HeroDetector
from src.tracker.turn_detector import TurnDetector
from src.tracker.position_calculator import PositionCalculator
from src.tracker.hand_tracker import HandTracker

# Model imports
from src.models.game_state import GameState
from src.models.player import Player
from src.models.card import Card
from src.models.table_info import TableInfo

logger = logging.getLogger(__name__)


class GameStateMachine:
    """
    Game state machine that coordinates all parsers and trackers.
    
    Maintains game state across frames, detects changes, and handles
    new hands. Uses simple approach - parse everything each frame.
    """
    
    def __init__(self):
        """Initialize state machine with all parsers and trackers."""
        # State storage
        self.current_state: Optional[GameState] = None
        self.previous_state: Optional[GameState] = None
        self.current_hand_id: Optional[str] = None
        self.hero_seat: Optional[int] = None
        
        # Initialize all parsers (9 total)
        self.card_parser = CardParser()
        self.money_parser = MoneyParser()
        self.hand_id_parser = HandIdParser()
        self.table_info_parser = TableInfoParser()
        self.name_parser = NameParser()
        self.status_parser = StatusParser()
        self.dealer_detector = DealerDetector()
        self.timer_detector = TimerDetector()
        self.transparency_detector = TransparencyDetector()
        
        # Initialize trackers (4 total)
        self.hero_detector = HeroDetector()
        self.turn_detector = TurnDetector()
        self.position_calculator = PositionCalculator()
        self.hand_tracker = HandTracker()
        
        logger.info("GameStateMachine initialized with all parsers and trackers")
    
    def update(self, regions: Dict[str, np.ndarray]) -> Optional[GameState]:
        """
        Parse all regions and return current GameState.
        
        Args:
            regions: Dictionary mapping region names to image arrays
            
        Returns:
            GameState if parsing successful, None if critical parsing fails
            
        Example:
            regions = extractor.extract_all_regions()
            game_state = state_machine.update(regions)
            if game_state:
                print(f"Phase: {game_state.phase}, Players: {len(game_state.players)}")
        """
        try:
            # 1. Parse hand_id (critical for hand tracking)
            hand_id = self._parse_hand_id(regions)
            if not hand_id:
                logger.warning("Failed to parse hand_id - critical failure")
                return self.current_state  # Return previous state
            
            # 2. Detect new hand if hand_id changed
            if hand_id != self.current_hand_id:
                self._handle_new_hand(hand_id)
            
            # 3. Detect hero seat (once per session)
            if self.hero_seat is None:
                self.hero_seat = self._detect_hero_seat(regions)
                if self.hero_seat:
                    logger.info(f"Hero detected in seat {self.hero_seat}")
            
            # 4. Parse all 8 players (cards, money, status, timer, transparency)
            players = self._parse_players(regions)
            if len(players) < 2:
                logger.warning(f"Too few players detected: {len(players)}")
                return self.current_state
            
            # 5. Parse community cards (5 cards)
            community_cards = self._parse_community_cards(regions)
            
            # 6. Parse pot
            pot = self._parse_pot(regions)
            
            # 7. Parse table info (stakes)
            table_info = self._parse_table_info(regions)
            
            # 8. Determine phase from community card count
            phase = self._determine_phase(community_cards)
            
            # 9. Find dealer button
            button_position = self._find_dealer_button(regions)
            
            # 10. Calculate positions
            if button_position:
                players = self.position_calculator.calculate_positions(players, button_position)
            
            # 11. Detect active player
            active_player = self._detect_active_player(regions, players)
            
            # 12. Build GameState with Pydantic validation
            try:
                new_state = GameState(
                    players=players,
                    community_cards=community_cards,
                    pot=pot,
                    phase=phase,
                    active_player=active_player,
                    button_position=button_position or 1,
                    hand_id=hand_id,
                    table_info=table_info
                )
                
                # 13. Detect changes from previous state
                changes = self._detect_changes(new_state)
                if changes:
                    logger.debug(f"State changes detected: {changes}")
                
                # 13a. Track hand updates
                if self.hand_tracker.current_hand:
                    self.hand_tracker.update_hand(new_state, self.previous_state)
                    
                    # Check completion
                    if self._is_hand_complete(new_state):
                        completed_hand = self.hand_tracker.finalize_hand(new_state)
                        if completed_hand:
                            logger.info(f"Hand {completed_hand.hand_id} completed")
                
                # 14. Store as current_state
                self.previous_state = self.current_state
                self.current_state = new_state
                
                # 15. Return GameState
                return new_state
                
            except Exception as e:
                logger.error(f"Failed to build GameState: {e}")
                return self.current_state
                
        except Exception as e:
            logger.error(f"Critical error in state machine update: {e}")
            return self.current_state
    
    def _parse_hand_id(self, regions: Dict[str, np.ndarray]) -> Optional[str]:
        """Parse hand ID from hand_num region."""
        hand_num_region = regions.get('hand_num')
        if hand_num_region is None:
            logger.debug("No hand_num region found")
            return None
        
        try:
            result = self.hand_id_parser.parse_hand_id(hand_num_region)
            if result:
                return result.hand_id
        except Exception as e:
            logger.debug(f"Hand ID parsing failed: {e}")
        
        return None
    
    def _detect_hero_seat(self, regions: Dict[str, np.ndarray]) -> Optional[int]:
        """Detect hero seat from player name regions."""
        try:
            # Build nameplate regions dict for hero detector
            nameplate_regions = {}
            for seat_num in range(1, 9):
                nameplate = self._combine_nameplate_regions(regions, seat_num)
                if nameplate is not None:
                    nameplate_regions[seat_num] = nameplate
            
            return self.hero_detector.detect_hero_seat(nameplate_regions)
        except Exception as e:
            logger.debug(f"Hero detection failed: {e}")
            return None
    
    def _parse_players(self, regions: Dict[str, np.ndarray]) -> List[Player]:
        """Parse all 8 player seats, return list of active players."""
        players = []
        
        for seat_num in range(1, 9):
            try:
                # Combine name + bank regions into nameplate
                nameplate = self._combine_nameplate_regions(regions, seat_num)
                if nameplate is None:
                    continue
                
                # Check transparency (folded/away?)
                transparency_result = self.transparency_detector.detect_transparency(nameplate)
                if transparency_result.is_transparent:
                    logger.debug(f"Player {seat_num} is folded/away (transparent)")
                    continue  # Skip folded/away players
                
                # Parse player data
                stack = self._parse_player_stack(regions, seat_num)
                hole_cards = self._parse_hole_cards(regions, seat_num)
                timer_state = self._parse_timer_state(nameplate)
                current_bet = self._parse_player_bet(regions, seat_num)
                is_dealer = self._check_dealer(regions, seat_num)
                is_hero = (seat_num == self.hero_seat)
                
                player = Player(
                    seat_number=seat_num,
                    stack=stack or 0.0,
                    hole_cards=hole_cards,
                    timer_state=timer_state,
                    is_dealer=is_dealer,
                    current_bet=current_bet or 0.0,
                    is_hero=is_hero,
                    is_active=True
                )
                players.append(player)
                logger.debug(f"Parsed player {seat_num}: stack={stack}, cards={len(hole_cards)}, timer={timer_state}")
                
            except Exception as e:
                logger.debug(f"Failed to parse player {seat_num}: {e}")
                continue
        
        return players
    
    def _combine_nameplate_regions(self, regions: Dict[str, np.ndarray], seat_num: int) -> Optional[np.ndarray]:
        """Combine name + bank regions vertically into single nameplate."""
        name_region = regions.get(f'player_{seat_num}_name')
        bank_region = regions.get(f'player_{seat_num}_bank')
        
        if name_region is None or bank_region is None:
            return None
        
        try:
            # Stack regions vertically (name on top, bank below)
            combined = np.vstack([name_region, bank_region])
            return combined
        except Exception as e:
            logger.debug(f"Failed to combine nameplate regions for seat {seat_num}: {e}")
            return None
    
    def _parse_hole_cards(self, regions: Dict[str, np.ndarray], seat_num: int) -> List[Card]:
        """Parse 2 hole cards for a player."""
        cards = []
        
        for card_num in [1, 2]:
            card_region = regions.get(f'player_{seat_num}_hole_{card_num}')
            if card_region is None:
                continue
            
            try:
                result = self.card_parser.parse_card(card_region)
                if result:
                    card = Card(rank=result.rank, suit=result.suit)
                    cards.append(card)
            except Exception as e:
                logger.debug(f"Failed to parse hole card {card_num} for seat {seat_num}: {e}")
        
        return cards
    
    def _parse_community_cards(self, regions: Dict[str, np.ndarray]) -> List[Card]:
        """Parse up to 5 community cards."""
        cards = []
        
        for card_num in range(1, 6):
            card_region = regions.get(f'community_card_{card_num}')
            if card_region is None:
                continue
            
            try:
                result = self.card_parser.parse_card(card_region)
                if result:
                    card = Card(rank=result.rank, suit=result.suit)
                    cards.append(card)
            except Exception as e:
                logger.debug(f"Failed to parse community card {card_num}: {e}")
        
        return cards
    
    def _parse_player_stack(self, regions: Dict[str, np.ndarray], seat_num: int) -> Optional[float]:
        """Parse player stack from bank region."""
        bank_region = regions.get(f'player_{seat_num}_bank')
        if bank_region is None:
            return None
        
        try:
            results = self.money_parser.parse_amounts(bank_region)
            if results:
                # Return the largest amount (likely the stack)
                return max(result.value for result in results)
        except Exception as e:
            logger.debug(f"Failed to parse stack for seat {seat_num}: {e}")
        
        return None
    
    def _parse_player_bet(self, regions: Dict[str, np.ndarray], seat_num: int) -> Optional[float]:
        """Parse player bet from bet region."""
        bet_region = regions.get(f'player_{seat_num}_bet')
        if bet_region is None:
            return None
        
        try:
            results = self.money_parser.parse_amounts(bet_region)
            if results:
                # Return the largest amount (likely the bet)
                return max(result.value for result in results)
        except Exception as e:
            logger.debug(f"Failed to parse bet for seat {seat_num}: {e}")
        
        return None
    
    def _parse_timer_state(self, nameplate: np.ndarray) -> Optional[str]:
        """Parse timer state from nameplate."""
        try:
            result = self.timer_detector.detect_timer(nameplate)
            if result.turn_state in ['turn', 'turn_overtime']:
                return 'purple' if result.turn_state == 'turn' else 'red'
        except Exception as e:
            logger.debug(f"Failed to parse timer state: {e}")
        
        return None
    
    def _check_dealer(self, regions: Dict[str, np.ndarray], seat_num: int) -> bool:
        """Check if player has dealer button."""
        dealer_region = regions.get(f'player_{seat_num}_dealer')
        if dealer_region is None:
            return False
        
        try:
            result = self.dealer_detector.detect_dealer_button(dealer_region)
            return result.has_dealer
        except Exception as e:
            logger.debug(f"Failed to check dealer for seat {seat_num}: {e}")
            return False
    
    def _parse_pot(self, regions: Dict[str, np.ndarray]) -> float:
        """Parse total pot."""
        pot_region = regions.get('pot_total')
        if pot_region is None:
            return 0.0
        
        try:
            results = self.money_parser.parse_amounts(pot_region)
            if results:
                # Return the largest amount (likely the pot)
                return max(result.value for result in results)
        except Exception as e:
            logger.debug(f"Failed to parse pot: {e}")
        
        return 0.0
    
    def _parse_table_info(self, regions: Dict[str, np.ndarray]) -> TableInfo:
        """Parse table info (stakes)."""
        table_region = regions.get('table_info')
        if table_region is None:
            # Return default stakes if parsing fails
            return TableInfo(sb=0.01, bb=0.02)
        
        try:
            result = self.table_info_parser.parse_table_info(table_region)
            if result:
                return TableInfo(sb=result.sb, bb=result.bb)
        except Exception as e:
            logger.debug(f"Failed to parse table info: {e}")
        
        # Return default stakes as fallback
        return TableInfo(sb=0.01, bb=0.02)
    
    def _determine_phase(self, community_cards: List[Card]) -> str:
        """Map card count to phase (0=preflop, 3=flop, etc)."""
        card_count = len(community_cards)
        
        if card_count == 0:
            return 'preflop'
        elif card_count == 3:
            return 'flop'
        elif card_count == 4:
            return 'turn'
        elif card_count == 5:
            return 'river'
        else:
            # Invalid card count, default to preflop
            logger.warning(f"Invalid community card count: {card_count}")
            return 'preflop'
    
    def _find_dealer_button(self, regions: Dict[str, np.ndarray]) -> Optional[int]:
        """Check all 8 dealer regions to find button position."""
        for seat_num in range(1, 9):
            if self._check_dealer(regions, seat_num):
                return seat_num
        
        return None
    
    def _detect_active_player(self, regions: Dict[str, np.ndarray], players: List[Player]) -> Optional[int]:
        """Find who has timer active (hero's turn)."""
        if self.hero_seat is None:
            return None
        
        try:
            # Build nameplate regions for turn detector
            nameplate_regions = {}
            for player in players:
                nameplate = self._combine_nameplate_regions(regions, player.seat_number)
                if nameplate is not None:
                    nameplate_regions[player.seat_number] = nameplate
            
            # Check if it's hero's turn
            if self.turn_detector.detect_hero_turn(nameplate_regions, self.hero_seat):
                return self.hero_seat
        except Exception as e:
            logger.debug(f"Failed to detect active player: {e}")
        
        return None
    
    def _detect_changes(self, new_state: GameState) -> Dict[str, Any]:
        """Compare new vs previous state, return dict of changes."""
        if not self.previous_state:
            return {'initial_state': True}
        
        changes = {}
        
        # Check hand_id change
        if new_state.hand_id != self.previous_state.hand_id:
            changes['hand_id'] = {
                'old': self.previous_state.hand_id,
                'new': new_state.hand_id
            }
        
        # Check phase change
        if new_state.phase != self.previous_state.phase:
            changes['phase'] = {
                'old': self.previous_state.phase,
                'new': new_state.phase
            }
        
        # Check pot change
        if abs(new_state.pot - self.previous_state.pot) > 0.01:
            changes['pot'] = {
                'old': self.previous_state.pot,
                'new': new_state.pot
            }
        
        # Check community cards change
        old_cards = [str(card) for card in self.previous_state.community_cards]
        new_cards = [str(card) for card in new_state.community_cards]
        if old_cards != new_cards:
            changes['community_cards'] = {
                'old': old_cards,
                'new': new_cards
            }
        
        # Check active player change
        if new_state.active_player != self.previous_state.active_player:
            changes['active_player'] = {
                'old': self.previous_state.active_player,
                'new': new_state.active_player
            }
        
        return changes
    
    def _handle_new_hand(self, hand_id: str):
        """Handle new hand detection - start tracking."""
        logger.info(f"New hand detected: {hand_id}")
        
        # Finalize previous hand if exists
        if self.hand_tracker.current_hand and self.current_state:
            completed_hand = self.hand_tracker.finalize_hand(self.current_state)
            if completed_hand:
                logger.info(f"Previous hand finalized: {completed_hand.result}, P/L: {completed_hand.net_profit}")
        
        # Start new hand
        self.current_hand_id = hand_id
        if self.current_state:
            self.hand_tracker.start_new_hand(hand_id, self.current_state, self.hero_seat)
    
    def get_current_state(self) -> Optional[GameState]:
        """Get the current game state."""
        return self.current_state
    
    def get_hero_seat(self) -> Optional[int]:
        """Get the hero's seat number."""
        return self.hero_seat
    
    def is_hero_turn(self) -> bool:
        """Check if it's currently the hero's turn."""
        if not self.current_state or not self.hero_seat:
            return False
        
        return self.current_state.active_player == self.hero_seat
    
    def _is_hand_complete(self, game_state: GameState) -> bool:
        """Check if hand is complete."""
        active_players = [p for p in game_state.players if p.is_active]
        return len(active_players) <= 1 or game_state.phase == 'showdown'
    
    def get_current_hand_record(self) -> Optional[HandRecord]:
        """Get current hand being tracked."""
        return self.hand_tracker.get_current_hand()
    
    def get_completed_hands(self) -> List[HandRecord]:
        """Get all completed hands from this session."""
        return self.hand_tracker.get_completed_hands()
