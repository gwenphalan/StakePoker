#!/usr/bin/env python3
"""
Range Estimator for estimating opponent hand ranges.

Estimates opponent ranges based on position, actions, betting patterns,
and game state to provide realistic ranges for equity calculations.
"""

import logging
from typing import List, Dict, Optional
from src.models.game_state import GameState
from src.models.player import Player
from src.advisor.gto_loader import GTOChartLoader
from src.config.settings import Settings

logger = logging.getLogger(__name__)


class RangeEstimator:
    """Estimates opponent hand ranges based on position and actions."""
    
    def __init__(self):
        """Initialize range estimator with GTO loader."""
        self.settings = Settings()
        self.gto_loader = GTOChartLoader()
        
        # Configuration
        self.settings.create("advisor.ranges.tight_adjustment", default=0.8)
        self.settings.create("advisor.ranges.loose_adjustment", default=1.2)
        self.settings.create("advisor.ranges.default_fold_equity", default=0.3)
        
        self.tight_adjustment = self.settings.get("advisor.ranges.tight_adjustment")
        self.loose_adjustment = self.settings.get("advisor.ranges.loose_adjustment")
        
        logger.info("Initialized range estimator")
    
    def estimate_ranges(self, game_state: GameState) -> List[str]:
        """
        Estimate combined range for all active opponents.
        
        Args:
            game_state: Current game state
            
        Returns:
            Combined list of hand notations representing all opponent ranges
        """
        all_hands = []
        hero = game_state.get_hero()
        
        if not hero:
            logger.warning("No hero found in game state")
            return []
        
        for player in game_state.players:
            if player.is_hero or not player.is_active:
                continue
            
            # Estimate range for this opponent
            player_range = self._estimate_player_range(player, game_state)
            all_hands.extend(player_range)
        
        # Remove duplicates and return
        unique_hands = list(set(all_hands))
        logger.debug(f"Estimated combined range: {len(unique_hands)} unique hands")
        return unique_hands
    
    def estimate_player_range(self, player: Player, game_state: GameState) -> List[str]:
        """
        Estimate range for a specific player (public method).
        
        Args:
            player: Player to estimate range for
            game_state: Current game state
            
        Returns:
            List of hand notations in player's estimated range
        """
        return self._estimate_player_range(player, game_state)
    
    def _estimate_player_range(self, player: Player, game_state: GameState) -> List[str]:
        """
        Estimate range for a specific player.
        
        Args:
            player: Player to estimate range for
            game_state: Current game state
            
        Returns:
            List of hand notations in player's estimated range
        """
        if not player.position:
            logger.warning(f"Player at seat {player.seat_number} has no position")
            return self._get_default_range()
        
        # Start with GTO range for position
        base_range = self._get_gto_range(player, game_state)
        
        if not base_range:
            logger.warning(f"No GTO range found for {player.position}")
            return self._get_default_range()
        
        # Adjust based on actions
        adjusted_range = self._adjust_for_actions(base_range, player, game_state)
        
        # Adjust for player tendencies (placeholder for future)
        final_range = self._adjust_for_tendencies(adjusted_range, player)
        
        logger.debug(f"Estimated range for seat {player.seat_number} ({player.position}): {len(final_range)} hands")
        return final_range
    
    def _get_gto_range(self, player: Player, game_state: GameState) -> List[str]:
        """Get GTO range for player's position and action context."""
        player_count = len([p for p in game_state.players if p.is_active])
        
        # Determine action context based on betting
        action_context = self._get_action_context(player, game_state)
        
        try:
            if action_context == "opening":
                return self.gto_loader.get_opening_range(player.position, player_count)
            elif action_context == "calling":
                return self.gto_loader.get_calling_range(player.position, player_count)
            elif action_context == "3betting":
                return self.gto_loader.get_3bet_range(player.position, player_count)
            elif action_context == "4betting":
                return self.gto_loader.get_4bet_range(player.position, player_count)
            elif action_context == "5betting":
                return self.gto_loader.get_5bet_range(player.position, player_count)
            elif action_context == "blind_defense":
                return self.gto_loader.get_blind_defense_range(player.position, player_count)
            else:
                # Default to opening range
                return self.gto_loader.get_opening_range(player.position, player_count)
        except Exception as e:
            logger.error(f"Error getting GTO range for {player.position}: {e}")
            return self._get_default_range()
    
    def _get_action_context(self, player: Player, game_state: GameState) -> str:
        """
        Determine what action context this player is in.
        
        This is simplified for now - would need full action tracking to be accurate.
        """
        # Check if player has made a bet
        if player.current_bet > 0:
            # Determine if this is a raise, 3bet, 4bet, etc.
            # For now, we'll use a simplified approach
            
            # Check how many players have bet before this player
            bets_before = sum(1 for p in game_state.players 
                            if p.current_bet > 0 and p.seat_number != player.seat_number)
            
            if bets_before == 0:
                return "opening"
            elif bets_before == 1:
                # Could be calling or 3betting
                if player.current_bet > game_state.table_info.bb * 2:
                    return "3betting"
                else:
                    return "calling"
            elif bets_before == 2:
                return "4betting"
            elif bets_before >= 3:
                return "5betting"
            else:
                return "calling"
        
        # Check if player is in blinds facing action
        elif player.position in ['SB', 'BB'] and self._is_facing_open(player, game_state):
            return "blind_defense"
        
        # Default to opening range
        return "opening"
    
    def _adjust_for_actions(self, base_range: List[str], player: Player, 
                          game_state: GameState) -> List[str]:
        """Adjust range based on player's actions."""
        adjusted_range = base_range.copy()
        
        # If player is not active (folded), they likely had a weak hand
        if not player.is_active:
            # Remove premium hands from range (they wouldn't fold these)
            premium_hands = ['AA', 'KK', 'QQ', 'JJ', 'TT', 'AKs', 'AKo', 'AQs', 'AQo']
            adjusted_range = [h for h in adjusted_range if h not in premium_hands]
            logger.debug(f"Removed premium hands from folded player's range")
        
        # If player raised significantly, tighten range
        elif player.current_bet > game_state.table_info.bb * 5:
            # Large raise = tighter range
            adjusted_range = self._tighten_range(adjusted_range, 0.6)
            logger.debug(f"Tightened range for large raise")
        
        return adjusted_range
    
    def _adjust_for_tendencies(self, range_list: List[str], player: Player) -> List[str]:
        """
        Adjust range based on player tendencies.
        
        This is a placeholder for future implementation with player tracking.
        Would use historical data about player tendencies (tight/loose, aggressive/passive).
        """
        # For now, return range as-is
        return range_list
    
    def _tighten_range(self, range_list: List[str], factor: float) -> List[str]:
        """
        Tighten a range by removing weaker hands.
        
        Args:
            range_list: Original range
            factor: Percentage of hands to keep (0.0 to 1.0)
            
        Returns:
            Tightened range with only strongest hands
        """
        if factor >= 1.0:
            return range_list
        
        # Sort hands by strength and keep top percentage
        hand_strength = self._get_hand_strength_order()
        
        # Keep only the strongest hands
        keep_count = max(1, int(len(range_list) * factor))
        sorted_hands = sorted(
            range_list, 
            key=lambda h: hand_strength.get(h, 0), 
            reverse=True
        )
        
        return sorted_hands[:keep_count]
    
    def _get_hand_strength_order(self) -> Dict[str, int]:
        """
        Get hand strength ordering for range adjustments.
        
        Returns:
            Dictionary mapping hand notation to strength score (higher = stronger)
        """
        strength_map = {}
        
        # Premium pocket pairs (highest strength)
        premium_pairs = ['AA', 'KK', 'QQ', 'JJ', 'TT']
        for i, hand in enumerate(premium_pairs):
            strength_map[hand] = 100 - i
        
        # Premium suited hands
        premium_suited = ['AKs', 'AQs', 'AJs', 'KQs', 'ATs']
        for i, hand in enumerate(premium_suited):
            strength_map[hand] = 95 - i
        
        # Premium offsuit hands
        premium_offsuit = ['AKo', 'AQo', 'AJo', 'KQo']
        for i, hand in enumerate(premium_offsuit):
            strength_map[hand] = 90 - i
        
        # Medium pocket pairs
        medium_pairs = ['99', '88', '77', '66', '55']
        for i, hand in enumerate(medium_pairs):
            strength_map[hand] = 85 - i
        
        # Medium suited hands
        medium_suited = ['KJs', 'QJs', 'JTs', 'T9s', '98s', 'A9s', 'A8s', 'A7s', 'A6s', 'A5s']
        for i, hand in enumerate(medium_suited):
            strength_map[hand] = 80 - i
        
        # Medium offsuit hands
        medium_offsuit = ['ATo', 'KJo', 'QJo', 'JTo']
        for i, hand in enumerate(medium_offsuit):
            strength_map[hand] = 70 - i
        
        # Small pocket pairs
        small_pairs = ['44', '33', '22']
        for i, hand in enumerate(small_pairs):
            strength_map[hand] = 65 - i
        
        # Weak suited hands
        weak_suited = ['A4s', 'A3s', 'A2s', 'K9s', 'Q9s', 'J9s', 'T8s', '97s', '87s', '76s']
        for i, hand in enumerate(weak_suited):
            strength_map[hand] = 60 - i
        
        # Weak offsuit hands
        weak_offsuit = ['A9o', 'A8o', 'A7o', 'A6o', 'A5o', 'KTo', 'QTo']
        for i, hand in enumerate(weak_offsuit):
            strength_map[hand] = 50 - i
        
        return strength_map
    
    def _get_default_range(self) -> List[str]:
        """
        Get a default range when GTO range is not available.
        
        Returns:
            Conservative default range
        """
        # Return a conservative range of premium hands
        return [
            'AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77',
            'AKs', 'AKo', 'AQs', 'AQo', 'AJs', 'AJo', 'ATs', 'ATo',
            'KQs', 'KQo', 'KJs', 'KJo', 'QJs', 'QJo', 'JTs'
        ]
    
    def _is_facing_raise(self, player: Player, game_state: GameState) -> bool:
        """Check if player is facing a raise."""
        # Check if any other player has bet more than the blinds
        for p in game_state.players:
            if p.seat_number != player.seat_number and p.current_bet > game_state.table_info.bb:
                return True
        return False
    
    def _is_facing_open(self, player: Player, game_state: GameState) -> bool:
        """Check if player is facing an open raise."""
        return self._is_facing_raise(player, game_state)

