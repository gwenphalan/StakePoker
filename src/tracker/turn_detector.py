#!/usr/bin/env python3
"""
Turn detector module for identifying when it's the hero's turn to act.

Detects hero's turn by checking timer colors across all players with
confidence-based validation to prevent false positives.

Usage:
    from src.tracker.turn_detector import TurnDetector
    
    detector = TurnDetector()
    is_hero_turn = detector.detect_hero_turn(all_nameplate_regions, hero_seat=1)
    if is_hero_turn:
        print("Hero's turn - trigger decision engine")
"""

import logging
from typing import Dict, Optional
import numpy as np

from src.parser.timer_detector import TimerDetector
from src.config.settings import Settings

logger = logging.getLogger(__name__)


class TurnDetector:
    """
    Detect when it's the hero's turn to act.
    
    Analyzes timer colors across all players and validates that the hero
    has the strongest timer signal to prevent false positives.
    """
    
    def __init__(self):
        """Initialize turn detector with timer detector and settings."""
        self.settings = Settings()
        self.timer_detector = TimerDetector()
        
        # Create turn detection settings
        self.settings.create("tracker.turn_detection.enabled", default=True)
        self.settings.create("tracker.turn_detection.min_confidence", default=0.6)
        self.settings.create("tracker.turn_detection.require_validation", default=True)
        
        logger.info("TurnDetector initialized")
    
    def detect_hero_turn(self, nameplate_regions: Dict[int, np.ndarray], hero_seat: int) -> bool:
        """
        Detect if it's the hero's turn to act.
        
        Checks timer colors for all players and validates that the hero has
        the strongest timer signal to prevent false positives from artifacts.
        
        Args:
            nameplate_regions: Dict mapping seat numbers (1-8) to nameplate BGR images
            hero_seat: Hero's seat number (1-8)
            
        Returns:
            bool: True if it's the hero's turn to act, False otherwise
            
        Example:
            nameplate_regions = {1: region1, 2: region2, ...}
            is_turn = detector.detect_hero_turn(nameplate_regions, hero_seat=1)
            if is_turn:
                # Trigger decision engine
        """
        if not self.settings.get("tracker.turn_detection.enabled"):
            logger.debug("Turn detection disabled")
            return False
        
        if hero_seat not in nameplate_regions:
            logger.warning(f"Hero seat {hero_seat} not in provided regions")
            return False
        
        # Get timer results for all players
        timer_results = {}
        for seat, region in nameplate_regions.items():
            if region is None or region.size == 0:
                continue
            timer_results[seat] = self.timer_detector.detect_timer(region)
        
        if not timer_results:
            logger.debug("No valid timer results")
            return False
        
        # Find players with active turn indicators
        active_players = [
            (seat, result) for seat, result in timer_results.items()
            if result.turn_state in ['turn', 'turn_overtime']
        ]
        
        if not active_players:
            logger.debug("No active turn indicators detected")
            return False
        
        # Validation: Check if hero has strongest signal
        if self.settings.get("tracker.turn_detection.require_validation"):
            # Sort by confidence to find strongest signal
            active_players.sort(key=lambda x: x[1].confidence, reverse=True)
            strongest_seat, strongest_result = active_players[0]
            
            # Check minimum confidence threshold
            min_confidence = self.settings.get("tracker.turn_detection.min_confidence")
            if strongest_result.confidence < min_confidence:
                logger.debug(f"Strongest signal confidence {strongest_result.confidence:.3f} "
                           f"below threshold {min_confidence}")
                return False
            
            # Hero's turn only if hero has strongest signal
            is_hero_turn = strongest_seat == hero_seat
            
            if is_hero_turn:
                logger.info(f"Hero's turn detected (seat {hero_seat}, "
                          f"state={strongest_result.turn_state}, "
                          f"confidence={strongest_result.confidence:.3f})")
            else:
                logger.debug(f"Turn detected but not hero's (seat {strongest_seat}, "
                           f"confidence={strongest_result.confidence:.3f})")
            
            return is_hero_turn
        
        # No validation: Just check if hero has any turn indicator
        else:
            hero_result = timer_results.get(hero_seat)
            if hero_result and hero_result.turn_state in ['turn', 'turn_overtime']:
                logger.info(f"Hero's turn detected (no validation, "
                          f"state={hero_result.turn_state})")
                return True
            return False

