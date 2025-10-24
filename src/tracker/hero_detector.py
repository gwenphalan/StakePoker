#!/usr/bin/env python3
"""
Hero detector module for identifying the hero's seat position.

Uses name_parser.py to detect player names and identify which player
is the hero based on configured usernames with fuzzy matching.

Usage:
    from src.tracker.hero_detector import HeroDetector
    
    detector = HeroDetector()
    hero_seat = detector.detect_hero_seat(nameplate_regions)
    if hero_seat:
        print(f"Hero is in seat {hero_seat}")
"""

import logging
from typing import Dict, Optional
import numpy as np

from src.parser.name_parser import NameParser
from src.config.settings import Settings

logger = logging.getLogger(__name__)


class HeroDetector:
    """
    Detect which player is the hero based on configured usernames.
    
    Uses name_parser.py to extract player names and match them against
    configured hero usernames with fuzzy matching support.
    """
    
    def __init__(self):
        """Initialize hero detector with name parser and settings."""
        self.settings = Settings()
        self.name_parser = NameParser()
        
        # Create hero detection settings
        self.settings.create("tracker.hero_detection.enabled", default=True)
        self.settings.create("tracker.hero_detection.min_confidence", default=0.7)
        self.settings.create("tracker.hero_detection.require_high_confidence", default=True)
        
        logger.info("HeroDetector initialized")
    
    def detect_hero_seat(self, nameplate_regions: Dict[int, np.ndarray]) -> Optional[int]:
        """
        Detect which seat contains the hero player.
        
        Parses all player names and identifies which one matches the configured
        hero usernames with fuzzy matching.
        
        Args:
            nameplate_regions: Dict mapping seat numbers (1-8) to nameplate BGR images
            
        Returns:
            int: Hero's seat number (1-8) if found, None otherwise
            
        Example:
            nameplate_regions = {1: region1, 2: region2, ...}
            hero_seat = detector.detect_hero_seat(nameplate_regions)
            if hero_seat:
                print(f"Hero is in seat {hero_seat}")
        """
        if not self.settings.get("tracker.hero_detection.enabled"):
            logger.debug("Hero detection disabled")
            return None
        
        if not nameplate_regions:
            logger.warning("No nameplate regions provided for hero detection")
            return None
        
        # Parse names for all players
        name_results = {}
        for seat, region in nameplate_regions.items():
            if region is None or region.size == 0:
                continue
            
            try:
                result = self.name_parser.parse_player_name(region, seat)
                if result:
                    name_results[seat] = result
                    logger.debug(f"Seat {seat}: '{result.name}' (hero: {result.is_hero}, conf: {result.confidence:.3f})")
                else:
                    logger.debug(f"Seat {seat}: No name detected")
            except Exception as e:
                logger.error(f"Error parsing name for seat {seat}: {e}")
                continue
        
        if not name_results:
            logger.warning("No player names detected")
            return None
        
        # Find hero candidates
        hero_candidates = [
            (seat, result) for seat, result in name_results.items()
            if result.is_hero
        ]
        
        if not hero_candidates:
            logger.debug("No hero candidates found")
            return None
        
        # If multiple candidates, pick the one with highest confidence
        if len(hero_candidates) > 1:
            logger.warning(f"Multiple hero candidates found: {[seat for seat, _ in hero_candidates]}")
            hero_candidates.sort(key=lambda x: x[1].confidence, reverse=True)
        
        hero_seat, hero_result = hero_candidates[0]
        
        # Check minimum confidence threshold
        min_confidence = self.settings.get("tracker.hero_detection.min_confidence")
        if hero_result.confidence < min_confidence:
            logger.warning(f"Hero detection confidence {hero_result.confidence:.3f} "
                         f"below threshold {min_confidence}")
            return None
        
        logger.info(f"Hero detected in seat {hero_seat}: '{hero_result.name}' "
                   f"(confidence: {hero_result.confidence:.3f})")
        
        return hero_seat
    
    def get_hero_usernames(self) -> list[str]:
        """
        Get the list of configured hero usernames.
        
        Returns:
            List of hero usernames from settings
        """
        return self.settings.get("parser.names.hero_usernames") or ["GalacticAce"]
    
    def update_hero_usernames(self, usernames: list[str]) -> None:
        """
        Update the list of hero usernames.
        
        Args:
            usernames: New list of hero usernames
        """
        self.settings.update("parser.names.hero_usernames", usernames)
        logger.info(f"Updated hero usernames: {usernames}")
    
    def get_detection_stats(self) -> Dict[str, any]:
        """
        Get statistics about hero detection.
        
        Returns:
            Dictionary with detection statistics
        """
        return {
            'enabled': self.settings.get("tracker.hero_detection.enabled"),
            'min_confidence': self.settings.get("tracker.hero_detection.min_confidence"),
            'require_high_confidence': self.settings.get("tracker.hero_detection.require_high_confidence"),
            'hero_usernames': self.get_hero_usernames()
        }
