#!/usr/bin/env python3
"""
GTOChartLoader for loading and querying GTO preflop charts.

Loads preflop ranges from individual position JSON files and provides convenient
query methods for decision-making based on player count and position.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from src.config.settings import Settings

logger = logging.getLogger(__name__)


class GTOChartLoader:
    """Load and query GTO preflop charts with support for 6-max and 8-max."""
    
    def __init__(self, chart_dir: Optional[str] = None):
        """
        Initialize GTO chart loader.
        
        Args:
            chart_dir: Path to GTO charts directory. If None, uses Settings.
        """
        self.settings = Settings()
        
        # Create settings with defaults
        self.settings.create("advisor.gto.chart_path", default="data/gto_charts")
        
        if chart_dir is None:
            chart_dir = self.settings.get("advisor.gto.chart_path")
        
        self.chart_dir = Path(chart_dir)
        
        if not self.chart_dir.exists():
            raise FileNotFoundError(f"GTO charts directory not found: {chart_dir}")
        
        # Load metadata
        metadata_path = self.chart_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        # Cache for loaded position data
        self._position_cache: Dict[str, Dict] = {}
        
        logger.info(f"Initialized GTO chart loader from {self.chart_dir}")
    
    def _get_position_file(self, position: str, player_count: int) -> Path:
        """
        Get the appropriate position file based on player count.
        
        Args:
            position: Position name (UTG, MP, CO, BTN, SB, BB)
            player_count: Number of players at table
            
        Returns:
            Path to position file
        """
        if player_count <= 6:
            filename = f"{position}_6max.json"
        else:
            filename = f"{position}.json"
        
        return self.chart_dir / filename
    
    def _load_position_data(self, position: str, player_count: int) -> Dict:
        """
        Load position data from JSON file with caching.
        
        Args:
            position: Position name
            player_count: Number of players at table
            
        Returns:
            Position data dictionary
        """
        cache_key = f"{position}_{player_count}"
        
        if cache_key in self._position_cache:
            return self._position_cache[cache_key]
        
        position_file = self._get_position_file(position, player_count)
        
        if not position_file.exists():
            # Special handling for BB in 6-max (BB_6max.json doesn't exist)
            if position == "BB" and player_count <= 6:
                logger.info(f"BB_6max.json not found - using 8-max BB data for 6-max table")
                # Fall back to 8-max BB data
                fallback_file = self.chart_dir / "BB.json"
                if fallback_file.exists():
                    try:
                        with open(fallback_file, 'r') as f:
                            data = json.load(f)
                        self._position_cache[cache_key] = data
                        return data
                    except Exception as e:
                        logger.error(f"Error loading fallback BB file {fallback_file}: {e}")
                return {}
            else:
                logger.warning(f"Position file not found: {position_file}")
                return {}
        
        try:
            with open(position_file, 'r') as f:
                data = json.load(f)
            
            self._position_cache[cache_key] = data
            return data
            
        except Exception as e:
            logger.error(f"Error loading position file {position_file}: {e}")
            return {}
    
    def should_open(self, position: str, hand: str, player_count: int) -> bool:
        """
        Check if a hand should be opened from a position.
        
        Args:
            position: Position name (UTG, MP, CO, BTN, SB, BB)
            hand: Hand in notation (e.g., "AKs", "99", "AKo")
            player_count: Number of players at table
            
        Returns:
            True if hand should be opened, False otherwise
        """
        data = self._load_position_data(position, player_count)
        
        if not data or "opening" not in data:
            return False
        
        normalized_hand = self._normalize_hand(hand)
        hands_list = data["opening"].get("hands", [])
        
        return normalized_hand in hands_list
    
    def should_3bet(self, position: str, hand: str, player_count: int) -> bool:
        """
        Check if a hand should be 3-bet from a position.
        
        Args:
            position: Position name
            hand: Hand in notation
            player_count: Number of players at table
            
        Returns:
            True if hand should be 3-bet, False otherwise
        """
        data = self._load_position_data(position, player_count)
        
        if not data or "3bet" not in data:
            return False
        
        normalized_hand = self._normalize_hand(hand)
        hands_list = data["3bet"].get("hands", [])
        
        return normalized_hand in hands_list
    
    def should_4bet(self, position: str, hand: str, player_count: int) -> bool:
        """
        Check if a hand should be 4-bet from a position.
        
        Args:
            position: Position name
            hand: Hand in notation
            player_count: Number of players at table
            
        Returns:
            True if hand should be 4-bet, False otherwise
        """
        data = self._load_position_data(position, player_count)
        
        if not data or "4bet" not in data:
            return False
        
        normalized_hand = self._normalize_hand(hand)
        hands_list = data["4bet"].get("hands", [])
        
        return normalized_hand in hands_list
    
    def should_5bet(self, position: str, hand: str, player_count: int) -> bool:
        """
        Check if a hand should be 5-bet from a position.
        
        Args:
            position: Position name
            hand: Hand in notation
            player_count: Number of players at table
            
        Returns:
            True if hand should be 5-bet, False otherwise
        """
        data = self._load_position_data(position, player_count)
        
        if not data or "5bet" not in data:
            return False
        
        normalized_hand = self._normalize_hand(hand)
        hands_list = data["5bet"].get("hands", [])
        
        return normalized_hand in hands_list
    
    def should_call(self, position: str, hand: str, player_count: int) -> bool:
        """
        Check if a hand should call from a position.
        
        Args:
            position: Position name
            hand: Hand in notation
            player_count: Number of players at table
            
        Returns:
            True if hand should call, False otherwise
        """
        data = self._load_position_data(position, player_count)
        
        if not data or "calling" not in data:
            return False
        
        normalized_hand = self._normalize_hand(hand)
        hands_list = data["calling"].get("hands", [])
        
        return normalized_hand in hands_list
    
    def should_defend_blind(self, position: str, hand: str, player_count: int) -> bool:
        """
        Check if a hand should defend the blind from a position.
        
        Args:
            position: Position name (SB or BB)
            hand: Hand in notation
            player_count: Number of players at table
            
        Returns:
            True if hand should defend blind, False otherwise
        """
        if position not in ["SB", "BB"]:
            return False
        
        data = self._load_position_data(position, player_count)
        
        if not data or "blind_defense" not in data:
            return False
        
        normalized_hand = self._normalize_hand(hand)
        hands_list = data["blind_defense"].get("hands", [])
        
        return normalized_hand in hands_list
    
    def get_opening_frequency(self, position: str, hand: str, player_count: int) -> float:
        """
        Get the opening frequency for a specific hand from a position.
        
        Args:
            position: Position name
            hand: Hand in notation
            player_count: Number of players at table
            
        Returns:
            Opening frequency (0.0 to 1.0), or 0.0 if hand not in range
        """
        data = self._load_position_data(position, player_count)
        
        if not data or "opening" not in data:
            return 0.0
        
        normalized_hand = self._normalize_hand(hand)
        frequencies = data["opening"].get("frequencies", {})
        
        return frequencies.get(normalized_hand, 0.0)
    
    def get_3bet_frequency(self, position: str, hand: str, player_count: int) -> float:
        """
        Get the 3-bet frequency for a specific hand from a position.
        
        Args:
            position: Position name
            hand: Hand in notation
            player_count: Number of players at table
            
        Returns:
            3-bet frequency (0.0 to 1.0), or 0.0 if hand not in range
        """
        data = self._load_position_data(position, player_count)
        
        if not data or "3bet" not in data:
            return 0.0
        
        normalized_hand = self._normalize_hand(hand)
        frequencies = data["3bet"].get("frequencies", {})
        
        return frequencies.get(normalized_hand, 0.0)
    
    def get_opening_range(self, position: str, player_count: int) -> List[str]:
        """
        Get list of all hands that should be opened from a position.
        
        Args:
            position: Position name
            player_count: Number of players at table
            
        Returns:
            List of hand notations
        """
        data = self._load_position_data(position, player_count)
        
        if not data or "opening" not in data:
            return []
        
        return data["opening"].get("hands", [])
    
    def get_3bet_range(self, position: str, player_count: int) -> List[str]:
        """
        Get list of all hands that should be 3-bet from a position.
        
        Args:
            position: Position name
            player_count: Number of players at table
            
        Returns:
            List of hand notations
        """
        data = self._load_position_data(position, player_count)
        
        if not data or "3bet" not in data:
            return []
        
        return data["3bet"].get("hands", [])
    
    def get_4bet_range(self, position: str, player_count: int) -> List[str]:
        """
        Get list of all hands that should be 4-bet from a position.
        
        Args:
            position: Position name
            player_count: Number of players at table
            
        Returns:
            List of hand notations
        """
        data = self._load_position_data(position, player_count)
        
        if not data or "4bet" not in data:
            return []
        
        return data["4bet"].get("hands", [])
    
    def get_5bet_range(self, position: str, player_count: int) -> List[str]:
        """
        Get list of all hands that should be 5-bet from a position.
        
        Args:
            position: Position name
            player_count: Number of players at table
            
        Returns:
            List of hand notations
        """
        data = self._load_position_data(position, player_count)
        
        if not data or "5bet" not in data:
            return []
        
        return data["5bet"].get("hands", [])
    
    def get_calling_range(self, position: str, player_count: int) -> List[str]:
        """
        Get list of all hands that should call from a position.
        
        Args:
            position: Position name
            player_count: Number of players at table
            
        Returns:
            List of hand notations
        """
        data = self._load_position_data(position, player_count)
        
        if not data or "calling" not in data:
            return []
        
        return data["calling"].get("hands", [])
    
    def get_blind_defense_range(self, position: str, player_count: int) -> List[str]:
        """
        Get list of all hands that should defend the blind from a position.
        
        Args:
            position: Position name (SB or BB)
            player_count: Number of players at table
            
        Returns:
            List of hand notations
        """
        if position not in ["SB", "BB"]:
            return []
        
        data = self._load_position_data(position, player_count)
        
        if not data or "blind_defense" not in data:
            return []
        
        return data["blind_defense"].get("hands", [])
    
    def get_bet_size(self, position: str, action_type: str, player_count: int) -> float:
        """
        Get the recommended bet size for an action from a position.
        
        Args:
            position: Position name
            action_type: Type of action (opening, 3bet, 4bet, 5bet)
            player_count: Number of players at table
            
        Returns:
            Bet size in BBs, or default if not specified
        """
        data = self._load_position_data(position, player_count)
        
        if not data or action_type not in data:
            # Return default bet sizes
            defaults = {
                "opening": 2.5,
                "3bet": 11.0,
                "4bet": 24.0,
                "5bet": 44.0
            }
            return defaults.get(action_type, 2.5)
        
        return data[action_type].get("bet_size", 2.5)
    
    def get_preflop_decision(self, position: str, hand: str, player_count: int, 
                            action_context: str = "opening") -> Tuple[str, float, str]:
        """
        Get preflop decision recommendation.
        
        Args:
            position: Position name
            hand: Hand in notation
            player_count: Number of players at table
            action_context: Context of decision (opening, vs_open, vs_3bet, vs_4bet, blind_defense)
            
        Returns:
            Tuple of (action, bet_size, reasoning)
        """
        normalized_hand = self._normalize_hand(hand)
        
        if action_context == "opening":
            if self.should_open(position, hand, player_count):
                bet_size = self.get_bet_size(position, "opening", player_count)
                return "raise", bet_size, f"GTO opening range for {position}"
            else:
                return "fold", 0.0, f"Not in GTO opening range for {position}"
        
        elif action_context == "vs_open":
            if self.should_3bet(position, hand, player_count):
                bet_size = self.get_bet_size(position, "3bet", player_count)
                return "3bet", bet_size, f"GTO 3-bet range for {position}"
            elif self.should_call(position, hand, player_count):
                return "call", 0.0, f"GTO calling range for {position}"
            else:
                return "fold", 0.0, f"Not in GTO range for {position}"
        
        elif action_context == "vs_3bet":
            if self.should_4bet(position, hand, player_count):
                bet_size = self.get_bet_size(position, "4bet", player_count)
                return "4bet", bet_size, f"GTO 4-bet range for {position}"
            elif self.should_call(position, hand, player_count):
                return "call", 0.0, f"GTO calling range for {position}"
            else:
                return "fold", 0.0, f"Not in GTO range for {position}"
        
        elif action_context == "vs_4bet":
            if self.should_5bet(position, hand, player_count):
                bet_size = self.get_bet_size(position, "5bet", player_count)
                return "5bet", bet_size, f"GTO 5-bet range for {position}"
            elif self.should_call(position, hand, player_count):
                return "call", 0.0, f"GTO calling range for {position}"
            else:
                return "fold", 0.0, f"Not in GTO range for {position}"
        
        elif action_context == "blind_defense":
            if self.should_defend_blind(position, hand, player_count):
                return "call", 0.0, f"GTO blind defense for {position}"
            else:
                return "fold", 0.0, f"Not in GTO blind defense for {position}"
        
        else:
            return "fold", 0.0, f"Unknown action context: {action_context}"
    
    def _normalize_hand(self, hand: str) -> str:
        """
        Normalize hand format to standard notation.
        
        Args:
            hand: Hand in various formats (e.g., "AhKd", "AKs", "A♠K♠")
            
        Returns:
            Normalized hand (e.g., "AKs", "99", "AKo")
        """
        # Check if already in standard format (e.g., "AKs", "AKo", "AA")
        if len(hand) == 3 and hand[2] in ['s', 'o']:
            # Already normalized with s/o indicator
            return hand
        elif len(hand) == 2 and hand[0] == hand[1]:
            # Pocket pair (e.g., "AA")
            return hand
        
        # Extract the suits before removing them
        suits = []
        for char in hand:
            if char in ['h', 'd', 'c', 's', '♠', '♥', '♦', '♣']:
                if char in ['h', '♥']:
                    suits.append('h')
                elif char in ['d', '♦']:
                    suits.append('d')
                elif char in ['c', '♣']:
                    suits.append('c')
                elif char in ['s', '♠']:
                    suits.append('s')
        
        # Remove suits from the hand string
        clean_hand = hand.replace('h', '').replace('d', '').replace('c', '').replace('s', '')
        clean_hand = clean_hand.replace('♠', '').replace('♥', '').replace('♦', '').replace('♣', '')
        
        if len(clean_hand) >= 2:
            rank1 = clean_hand[0]
            rank2 = clean_hand[1] if len(clean_hand) > 1 else clean_hand[0]
            
            # Determine if suited, pair, or offsuit
            if rank1 == rank2:
                return f"{rank1}{rank2}"  # Pair
            elif len(suits) == 2 and suits[0] == suits[1]:
                return f"{rank1}{rank2}s"  # Suited (same suits)
            else:
                return f"{rank1}{rank2}o"  # Offsuit or default
        
        return hand
    
    def get_available_positions(self) -> List[str]:
        """Get list of available positions in the charts."""
        return ["UTG", "MP", "CO", "BTN", "SB", "BB"]
    
    def get_available_positions_6max(self) -> List[str]:
        """Get list of available positions for 6-max tables."""
        return ["UTG", "MP", "CO", "BTN", "SB"]  # BB handled via fallback
    
    def get_available_positions_8max(self) -> List[str]:
        """Get list of available positions for 8-max tables."""
        return ["UTG", "MP", "CO", "BTN", "SB", "BB"]
    
    def get_chart_info(self) -> Dict:
        """Get metadata about loaded charts."""
        return {
            'format': self.metadata.get('format', 'Unknown'),
            'solver': self.metadata.get('solver', 'Unknown'),
            'stack_depth': self.metadata.get('stack_depth', 'Unknown'),
            'source': self.metadata.get('source', 'Unknown'),
            'positions': self.get_available_positions(),
            'positions_6max': self.get_available_positions_6max(),
            'positions_8max': self.get_available_positions_8max(),
            'chart_dir': str(self.chart_dir),
            'notes': {
                'bb_6max_fallback': 'BB_6max.json not available - uses BB.json (8-max) for 6-max BB decisions',
                'bb_6max_reason': 'BB is not an opening position in 6-max - handled via blind defense ranges'
            }
        }