#!/usr/bin/env python3
"""
Position calculator module for determining relative poker positions.

Calculates BTN/SB/BB/UTG/MP/CO positions based on dealer button and table size.
Pure calculation module - no parsing or image processing.

Usage:
    from src.tracker.position_calculator import PositionCalculator
    
    calculator = PositionCalculator()
    players = calculator.calculate_positions(players, button_position)
"""

import logging
from typing import List, Optional
from src.models.player import Player

logger = logging.getLogger(__name__)


class PositionCalculator:
    """
    Calculate relative poker positions based on dealer button.
    
    Pure calculation module that takes player seat numbers and button position,
    then assigns standard poker positions (BTN/SB/BB/UTG/MP/CO).
    """
    
    # Position mappings for each table size (2-8 players)
    POSITION_MAPS = {
        2: ['SB', 'BB'],  # Heads-up: SB is also BTN
        3: ['BTN', 'SB', 'BB'],
        4: ['UTG', 'BTN', 'SB', 'BB'],
        5: ['UTG', 'CO', 'BTN', 'SB', 'BB'],
        6: ['UTG', 'MP', 'CO', 'BTN', 'SB', 'BB'],
        7: ['UTG', 'UTG+1', 'MP', 'CO', 'BTN', 'SB', 'BB'],
        8: ['UTG', 'UTG+1', 'UTG+2', 'MP', 'CO', 'BTN', 'SB', 'BB']
    }
    
    def __init__(self):
        """Initialize position calculator."""
        logger.info("PositionCalculator initialized")
    
    def calculate_positions(self, players: List[Player], button_position: int) -> List[Player]:
        """
        Calculate and assign relative positions to all players.
        
        Args:
            players: List of Player objects with seat_number populated
            button_position: Dealer button seat number (1-8)
            
        Returns:
            Updated list of Player objects with position field set
            
        Example:
            calculator = PositionCalculator()
            players = [Player(seat_number=1, ...), Player(seat_number=2, ...)]
            updated_players = calculator.calculate_positions(players, button_position=1)
        """
        if not players:
            logger.warning("Empty player list provided")
            return []
        
        if not self._validate_button_position(button_position):
            logger.error(f"Invalid button position: {button_position}")
            return players
        
        try:
            # Sort players by seat number
            sorted_players = sorted(players, key=lambda p: p.seat_number)
            
            # Get table size
            table_size = len(sorted_players)
            
            # Get position map for this table size
            position_map = self._get_position_map(table_size)
            
            # Find button index in sorted players
            button_index = self._find_button_index(sorted_players, button_position)
            
            if button_index is None:
                logger.warning(f"Button position {button_position} not found in active players")
                return players
            
            # Assign positions
            self._assign_positions(sorted_players, button_index, position_map)
            
            logger.debug(f"Calculated positions for {table_size} players, button at seat {button_position}")
            
            return sorted_players
            
        except Exception as e:
            logger.error(f"Error calculating positions: {e}")
            return players
    
    def _validate_button_position(self, button_position: int) -> bool:
        """Validate button position is in valid range."""
        return 1 <= button_position <= 8
    
    def _get_position_map(self, table_size: int) -> List[str]:
        """Get position map for given table size."""
        if table_size not in self.POSITION_MAPS:
            logger.warning(f"Unsupported table size: {table_size}, using 6-max")
            return self.POSITION_MAPS[6]
        
        return self.POSITION_MAPS[table_size]
    
    def _find_button_index(self, sorted_players: List[Player], button_position: int) -> Optional[int]:
        """Find index of button in sorted player list."""
        for i, player in enumerate(sorted_players):
            if player.seat_number == button_position:
                return i
        return None
    
    def _assign_positions(self, sorted_players: List[Player], button_index: int, 
                         position_map: List[str]) -> None:
        """Assign positions to players starting from button."""
        table_size = len(sorted_players)
        
        # Find BTN position in the position map
        btn_position_index = next((i for i, pos in enumerate(position_map) if pos == 'BTN'), 0)
        
        # For heads-up, BTN is SB
        if table_size == 2:
            btn_position_index = 0
        
        # Assign positions clockwise from button
        for i in range(table_size):
            # Calculate position index in the map
            position_index = (i - button_index + btn_position_index) % table_size
            
            # Assign position to player
            sorted_players[i].position = position_map[position_index]
            
            logger.debug(f"Seat {sorted_players[i].seat_number} -> {sorted_players[i].position}")
