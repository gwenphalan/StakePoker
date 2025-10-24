#!/usr/bin/env python3
"""
GTOChartLoader for loading and querying GTO preflop charts.

Loads preflop opening ranges from JSON files and provides convenient
query methods for decision-making.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, List
from src.config.settings import Settings

logger = logging.getLogger(__name__)


class GTOChartLoader:
    """Load and query GTO preflop opening charts."""
    
    def __init__(self, chart_path: Optional[str] = None):
        """
        Initialize GTO chart loader.
        
        Args:
            chart_path: Path to GTO chart JSON file. If None, uses Settings.
        """
        self.settings = Settings()
        
        if chart_path is None:
            chart_path = self.settings.get(
                "advisor.gto.chart_path",
                "data/gto_charts/texassolver_8max_100bb.json"
            )
        
        self.chart_path = Path(chart_path)
        
        if not self.chart_path.exists():
            # Fall back to hardcoded charts
            fallback = Path("data/gto_charts/gto_8max_100bb.json")
            if fallback.exists():
                logger.warning(f"Chart file {chart_path} not found, using fallback: {fallback}")
                self.chart_path = fallback
            else:
                raise FileNotFoundError(f"No GTO charts found at {chart_path} or {fallback}")
        
        # Load charts
        with open(self.chart_path, 'r') as f:
            self.charts = json.load(f)
        
        self.positions = self.charts.get('positions', {})
        self.format = self.charts.get('format', 'Unknown')
        self.solver = self.charts.get('solver', 'Unknown')
        
        logger.info(f"Loaded GTO charts from {self.chart_path} ({self.solver}, {self.format})")
    
    def get_opening_range(self, position: str) -> str:
        """
        Get opening range notation for a position.
        
        Args:
            position: Position name (UTG, MP, LJ, HJ, CO, BTN, SB, BB)
            
        Returns:
            Range notation string (e.g., "66+, A9s+, KTs+, AQo+")
        """
        if position not in self.positions:
            logger.warning(f"Position {position} not found in charts")
            return ""
        
        return self.positions[position].get('notation', '')
    
    def should_open(self, position: str, hand: str) -> bool:
        """
        Check if a hand should be opened from a position.
        
        Args:
            position: Position name
            hand: Hand in notation (e.g., "AKs", "99", "AKo")
            
        Returns:
            True if hand should be opened, False otherwise
        """
        if position not in self.positions:
            logger.warning(f"Position {position} not found in charts")
            return False
        
        normalized_hand = self._normalize_hand(hand)
        hands_list = self.positions[position].get('hands', [])
        
        return normalized_hand in hands_list
    
    def get_open_frequency(self, position: str, hand: str) -> float:
        """
        Get the opening frequency for a specific hand from a position.
        
        Args:
            position: Position name
            hand: Hand in notation
            
        Returns:
            Opening frequency (0.0 to 1.0), or 0.0 if hand not in range
        """
        if position not in self.positions:
            logger.warning(f"Position {position} not found in charts")
            return 0.0
        
        normalized_hand = self._normalize_hand(hand)
        frequencies = self.positions[position].get('frequencies', {})
        
        return frequencies.get(normalized_hand, 0.0)
    
    def get_opening_percentage(self, position: str) -> float:
        """
        Get the total opening percentage for a position.
        
        Args:
            position: Position name
            
        Returns:
            Percentage of hands opened (0.0 to 100.0)
        """
        if position not in self.positions:
            logger.warning(f"Position {position} not found in charts")
            return 0.0
        
        return self.positions[position].get('percentage', 0.0)
    
    def get_all_opening_hands(self, position: str) -> List[str]:
        """
        Get list of all hands that should be opened from a position.
        
        Args:
            position: Position name
            
        Returns:
            List of hand notations
        """
        if position not in self.positions:
            logger.warning(f"Position {position} not found in charts")
            return []
        
        return self.positions[position].get('hands', [])
    
    def _normalize_hand(self, hand: str) -> str:
        """
        Normalize hand format.
        
        Args:
            hand: Hand in various formats (e.g., "AhKd", "AKs", "A♠K♠")
            
        Returns:
            Normalized hand (e.g., "AKs", "99", "AKo")
        """
        # Remove suits and convert to standard notation
        # This is a simplified version - you may need to enhance it
        hand = hand.replace('h', '').replace('d', '').replace('c', '').replace('s', '')
        hand = hand.replace('♠', '').replace('♥', '').replace('♦', '').replace('♣', '')
        
        if len(hand) >= 2:
            rank1 = hand[0]
            rank2 = hand[1] if len(hand) > 1 else hand[0]
            
            # Determine if suited, pair, or offsuit
            if rank1 == rank2:
                return f"{rank1}{rank2}"  # Pair
            # Need more sophisticated logic here for actual card parsing
        
        return hand
    
    def get_available_positions(self) -> List[str]:
        """Get list of available positions in the charts."""
        return list(self.positions.keys())
    
    def get_chart_info(self) -> Dict:
        """Get metadata about loaded charts."""
        return {
            'format': self.format,
            'solver': self.solver,
            'stack_depth': self.charts.get('stack_depth', 'Unknown'),
            'source': self.charts.get('source', 'Unknown'),
            'positions': list(self.positions.keys())
        }

