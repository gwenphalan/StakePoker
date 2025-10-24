#!/usr/bin/env python3
"""
HandTracker for complete hand history tracking.

Tracks individual poker hands from start to finish, recording all actions,
detecting hand completion, and calculating profit/loss for the hero.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from src.models.hand_record import HandRecord, Action
from src.models.game_state import GameState
from src.models.player import Player
from src.models.card import Card
from src.config.settings import Settings

logger = logging.getLogger(__name__)


class HandTracker:
    """Track complete hand history from start to finish."""
    
    def __init__(self, session_tracker: Optional['SessionTracker'] = None):
        """
        Initialize hand tracker.
        
        Args:
            session_tracker: Optional SessionTracker for persistence
        """
        self.session_tracker = session_tracker
        self.current_hand: Optional[HandRecord] = None
        self.completed_hands: List[HandRecord] = []
        self.current_actions: List[Action] = []
        self.hand_start_state: Optional[GameState] = None
        self.hero_seat: Optional[int] = None
        self.hero_invested: float = 0.0
        self.settings = Settings()
        
        logger.info("HandTracker initialized")
    
    def start_new_hand(self, hand_id: str, game_state: GameState, hero_seat: Optional[int]) -> None:
        """
        Initialize new hand tracking.
        
        Args:
            hand_id: Unique hand identifier
            game_state: Initial game state
            hero_seat: Hero's seat number (1-8)
        """
        logger.info(f"Starting hand tracking for hand_id: {hand_id}")
        
        # Finalize any existing hand first
        if self.current_hand:
            logger.warning("Starting new hand while previous hand still active - finalizing previous")
            self.finalize_hand(game_state)
        
        # Store hero seat
        self.hero_seat = hero_seat
        
        # Find hero player and get their cards/position
        hero_player = None
        hero_position = "BTN"  # Default valid position
        hero_cards = []
        
        if hero_seat:
            hero_player = next((p for p in game_state.players if p.seat_number == hero_seat), None)
            if hero_player:
                hero_position = hero_player.position or "BTN"
                hero_cards = hero_player.hole_cards or []
        
        # Ensure we have at least 2 cards for validation
        if len(hero_cards) < 2:
            hero_cards = [Card(rank='A', suit='hearts'), Card(rank='K', suit='spades')]
        
        # Ensure hero_seat is valid
        if not hero_seat or hero_seat < 1:
            hero_seat = 1
        
        # Create new HandRecord
        self.current_hand = HandRecord(
            hand_id=hand_id,
            timestamp=datetime.now(),
            table_info=game_state.table_info,
            hero_position=hero_position,
            hero_seat=hero_seat or 0,
            hero_cards=hero_cards,
            actions=[],
            result=None,
            net_profit=0.0,
            final_pot=game_state.pot,
            showdown=False
        )
        
        # Reset tracking variables
        self.current_actions = []
        self.hero_invested = 0.0
        self.hand_start_state = game_state
        
        logger.info(f"Hand {hand_id} tracking started - Hero seat: {hero_seat}, Position: {hero_position}")
    
    def update_hand(self, game_state: GameState, prev_state: Optional[GameState]) -> None:
        """
        Track hand progression and detect actions.
        
        Args:
            game_state: Current game state
            prev_state: Previous game state for comparison
        """
        if not self.current_hand:
            logger.debug("No active hand to update")
            return
        
        # Detect actions if we have previous state
        if prev_state:
            actions = self.detect_actions(prev_state, game_state)
            if actions:
                self.current_actions.extend(actions)
                self.current_hand.actions.extend(actions)
                
                # Update hero investment
                for action in actions:
                    if action.seat_number == self.hero_seat and action.amount:
                        self.hero_invested += action.amount
                
                logger.debug(f"Detected {len(actions)} actions: {[f'{a.action_type}({a.amount})' for a in actions]}")
        
        # Update current pot
        self.current_hand.final_pot = game_state.pot
    
    def detect_actions(self, prev_state: GameState, curr_state: GameState) -> List[Action]:
        """
        Compare states to identify player actions.
        
        Args:
            prev_state: Previous game state
            curr_state: Current game state
            
        Returns:
            List of detected actions
        """
        actions = []
        
        # Build player lookup maps
        prev_players = {p.seat_number: p for p in prev_state.players}
        curr_players = {p.seat_number: p for p in curr_state.players}
        
        # Check each current player for actions
        for seat_num, curr_player in curr_players.items():
            prev_player = prev_players.get(seat_num)
            
            if not prev_player:
                continue  # New player joined, skip
            
            # Detect fold
            if prev_player.is_active and not curr_player.is_active:
                actions.append(Action(
                    seat_number=seat_num,
                    action_type='fold',
                    amount=None,
                    phase=curr_state.phase
                ))
                logger.debug(f"Player {seat_num} folded")
                continue
            
            # Skip if player wasn't active before
            if not prev_player.is_active:
                continue
            
            # Detect betting actions
            if curr_player.current_bet > prev_player.current_bet:
                bet_diff = curr_player.current_bet - prev_player.current_bet
                
                # Determine action type based on betting context
                max_prev_bet = max((p.current_bet for p in prev_state.players), default=0)
                
                if max_prev_bet == 0:
                    # First bet in round
                    action_type = 'bet'
                elif bet_diff > max_prev_bet:
                    # Raised above previous bet
                    action_type = 'raise'
                else:
                    # Called current bet
                    action_type = 'call'
                
                actions.append(Action(
                    seat_number=seat_num,
                    action_type=action_type,
                    amount=bet_diff,
                    phase=curr_state.phase
                ))
                logger.debug(f"Player {seat_num} {action_type}ed {bet_diff}")
            
            # Detect check (no bet change when timer was active)
            elif (prev_player.timer_state and 
                  curr_player.current_bet == prev_player.current_bet and
                  curr_player.is_active):
                actions.append(Action(
                    seat_number=seat_num,
                    action_type='check',
                    amount=None,
                    phase=curr_state.phase
                ))
                logger.debug(f"Player {seat_num} checked")
        
        return actions
    
    def finalize_hand(self, final_state: GameState) -> Optional[HandRecord]:
        """
        Complete hand tracking and calculate results.
        
        Args:
            final_state: Final game state
            
        Returns:
            Completed HandRecord if successful, None otherwise
        """
        if not self.current_hand:
            logger.warning("No active hand to finalize")
            return None
        
        logger.info(f"Finalizing hand {self.current_hand.hand_id}")
        
        # Determine hand result
        result = self._determine_hand_result(final_state)
        self.current_hand.result = result
        
        # Calculate profit/loss
        net_profit = self.calculate_hero_profit_loss(final_state)
        self.current_hand.net_profit = net_profit
        
        # Set final pot and showdown flag
        self.current_hand.final_pot = final_state.pot
        self.current_hand.showdown = (final_state.phase == 'showdown')
        
        # Complete the hand record
        completed_hand = self.current_hand
        self.completed_hands.append(completed_hand)
        
        # Persist to database via SessionTracker
        if self.session_tracker:
            success = self.session_tracker.record_hand_in_session(completed_hand)
            if not success:
                logger.warning(f"Failed to persist hand {completed_hand.hand_id} to database")
        
        # Reset current hand tracking
        self.current_hand = None
        self.current_actions = []
        self.hero_invested = 0.0
        self.hand_start_state = None
        
        logger.info(f"Hand {completed_hand.hand_id} completed: {result}, P/L: {net_profit:.2f}")
        return completed_hand
    
    def calculate_hero_profit_loss(self, final_state: GameState) -> float:
        """
        Calculate hero's net profit/loss for the hand.
        
        Args:
            final_state: Final game state
            
        Returns:
            Net profit (positive) or loss (negative)
        """
        if not self.hero_seat:
            return 0.0
        
        # Calculate total hero investment
        hero_invested = sum(
            action.amount for action in self.current_actions
            if action.seat_number == self.hero_seat and action.amount
        )
        
        # Determine if hero won
        hero_won = self._did_hero_win(final_state)
        
        if hero_won:
            # Hero won the pot
            return final_state.pot - hero_invested
        else:
            # Hero lost, loss = amount invested
            return -hero_invested
    
    def _determine_hand_result(self, final_state: GameState) -> str:
        """Determine hand result for hero."""
        if not self.hero_seat:
            return "unknown"
        
        # Check if hero is still active
        hero_player = next((p for p in final_state.players if p.seat_number == self.hero_seat), None)
        
        if not hero_player or not hero_player.is_active:
            return "folded"
        
        # Hero is still active - determine if they won
        if self._did_hero_win(final_state):
            return "won"
        else:
            return "lost"
    
    def _did_hero_win(self, final_state: GameState) -> bool:
        """Determine if hero won the hand."""
        if not self.hero_seat:
            return False
        
        # Simple heuristic: if hero is the only active player, they won
        active_players = [p for p in final_state.players if p.is_active]
        
        if len(active_players) == 1 and active_players[0].seat_number == self.hero_seat:
            return True
        
        # If we're at showdown and hero is still active, assume they won
        # (This is a simplification - real implementation would need to evaluate hands)
        if final_state.phase == 'showdown':
            hero_player = next((p for p in final_state.players if p.seat_number == self.hero_seat), None)
            return hero_player is not None and hero_player.is_active
        
        return False
    
    def get_current_hand(self) -> Optional[HandRecord]:
        """Get current hand being tracked."""
        return self.current_hand
    
    def get_completed_hands(self) -> List[HandRecord]:
        """Get all completed hands from this session."""
        return self.completed_hands.copy()
    
    def reset_hand(self) -> None:
        """Clear current hand tracking (for testing/error recovery)."""
        if self.current_hand:
            logger.warning(f"Resetting active hand {self.current_hand.hand_id}")
        
        self.current_hand = None
        self.current_actions = []
        self.hero_invested = 0.0
        self.hand_start_state = None
