#!/usr/bin/env python3
"""
Timer detector module for identifying active player turns.

Uses HSV color mask detection to identify purple (active turn) and red (overtime)
timer progress bars on player nameplates. Based on proven HSV mask detection from
card parser, adapted for timer progress bar detection.

Usage:
    from src.parser.timer_detector import TimerDetector
    
    detector = TimerDetector()
    result = detector.detect_timer(nameplate_region)
    if result.turn_state == 'turn':
        print(f"Active turn detected (confidence: {result.confidence:.3f})")
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

from src.config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class TimerResult:
    """Result of timer detection with turn state and confidence."""
    turn_state: str  # 'normal', 'turn', 'turn_overtime'
    confidence: float
    purple_pixels: int  # For debugging
    red_pixels: int     # For debugging


class TimerDetector:
    """
    Detect timer progress bars using HSV color mask detection.
    
    Analyzes player nameplate regions for purple (active turn) or red (overtime)
    timer colors using HSV color space and pixel counting. More reliable than
    BGR Euclidean distance for detecting progress bar colors.
    """
    
    def __init__(self):
        """Initialize timer detector with settings."""
        self.settings = Settings()
        
        # Create timer detection settings
        self._create_settings()
        
        # Load color ranges
        self._load_color_ranges()
        
        logger.info("TimerDetector initialized")
    
    def _create_settings(self) -> None:
        """Create all timer detector settings with defaults."""
        # Enable/disable detection
        self.settings.create("parser.timer.enabled", default=True)
        
        # Pixel threshold for detection
        self.settings.create("parser.timer.min_pixel_threshold", default=10)
        
        # HSV range for purple timer (normal turn)
        # Purple: H=130-150 (260-300 degrees), S=100-255, V=100-255
        # BGR equivalent from archive: (250, 0, 141) = #8d00fa
        self.settings.create("parser.timer.purple_hsv_range", 
                           default=[[130, 100, 100], [150, 255, 255]])
        
        # HSV ranges for red timer (overtime/time bank)
        # Red wraps around in HSV: 0-10 and 170-180
        # BGR equivalent from archive: (99, 65, 237) = #ed4163
        self.settings.create("parser.timer.red_hsv_ranges", 
                           default=[
                               [[0, 100, 100], [10, 255, 255]],      # Red range 1
                               [[170, 100, 100], [180, 255, 255]]    # Red range 2 (wrap-around)
                           ])
        
        logger.debug("Timer detector settings created")
    
    def _load_color_ranges(self) -> None:
        """Load HSV color ranges from settings."""
        self.purple_range = self.settings.get("parser.timer.purple_hsv_range")
        self.red_ranges = self.settings.get("parser.timer.red_hsv_ranges")
        
        logger.debug("Timer color ranges loaded from settings")
    
    def detect_timer(self, nameplate_region: np.ndarray) -> TimerResult:
        """
        Detect timer state from nameplate region using HSV color masks.
        
        Args:
            nameplate_region: BGR image of player nameplate region (name + bank combined)
            
        Returns:
            TimerResult with turn state, confidence, and pixel counts
            
        Example:
            result = detector.detect_timer(nameplate_region)
            if result.turn_state == 'turn':
                print(f"Active turn (confidence: {result.confidence:.3f})")
        """
        # Check if detection is enabled
        if not self.settings.get("parser.timer.enabled"):
            return TimerResult(
                turn_state='normal',
                confidence=0.0,
                purple_pixels=0,
                red_pixels=0
            )
        
        # Validate input
        if nameplate_region is None or nameplate_region.size == 0:
            logger.warning("Empty region provided to timer detector")
            return TimerResult(
                turn_state='normal',
                confidence=0.0,
                purple_pixels=0,
                red_pixels=0
            )
        
        try:
            # Convert to HSV color space
            hsv_image = cv2.cvtColor(nameplate_region, cv2.COLOR_BGR2HSV)
            
            # Count pixels for each timer color
            purple_pixels = self._count_purple_pixels(hsv_image)
            red_pixels = self._count_red_pixels(hsv_image)
            
            # Determine turn state based on pixel counts
            turn_state, confidence = self._evaluate_timer_state(
                purple_pixels, 
                red_pixels,
                nameplate_region.size
            )
            
            # Log result
            if turn_state != 'normal':
                logger.info(f"Timer detected: {turn_state} (confidence={confidence:.3f}, "
                          f"purple={purple_pixels}px, red={red_pixels}px)")
            else:
                logger.debug(f"No timer detected (purple={purple_pixels}px, red={red_pixels}px)")
            
            return TimerResult(
                turn_state=turn_state,
                confidence=confidence,
                purple_pixels=purple_pixels,
                red_pixels=red_pixels
            )
            
        except Exception as e:
            logger.error(f"Timer detection failed: {e}")
            return TimerResult(
                turn_state='normal',
                confidence=0.0,
                purple_pixels=0,
                red_pixels=0
            )
    
    def _count_purple_pixels(self, hsv_image: np.ndarray) -> int:
        """
        Count pixels matching purple timer color.
        
        Args:
            hsv_image: Image in HSV color space
            
        Returns:
            Number of pixels matching purple range
        """
        # Purple uses single range (no wrap-around)
        purple_mask = cv2.inRange(
            hsv_image,
            np.array(self.purple_range[0]),
            np.array(self.purple_range[1])
        )
        purple_pixels = cv2.countNonZero(purple_mask)
        
        logger.debug(f"Purple pixels detected: {purple_pixels}")
        return purple_pixels
    
    def _count_red_pixels(self, hsv_image: np.ndarray) -> int:
        """
        Count pixels matching red timer color.
        
        Red requires two ranges due to wrap-around in HSV color space.
        
        Args:
            hsv_image: Image in HSV color space
            
        Returns:
            Number of pixels matching red ranges
        """
        red_pixels = 0
        
        # Red wraps around in HSV, so check both ranges
        for range_pair in self.red_ranges:
            mask = cv2.inRange(
                hsv_image,
                np.array(range_pair[0]),
                np.array(range_pair[1])
            )
            red_pixels += cv2.countNonZero(mask)
        
        logger.debug(f"Red pixels detected: {red_pixels}")
        return red_pixels
    
    def _evaluate_timer_state(self, purple_pixels: int, red_pixels: int, 
                             total_pixels: int) -> tuple[str, float]:
        """
        Evaluate timer state based on pixel counts.
        
        Args:
            purple_pixels: Number of purple pixels detected
            red_pixels: Number of red pixels detected
            total_pixels: Total pixels in region
            
        Returns:
            Tuple of (turn_state, confidence)
        """
        min_threshold = self.settings.get("parser.timer.min_pixel_threshold")
        
        # Check if either color meets threshold
        purple_valid = purple_pixels >= min_threshold
        red_valid = red_pixels >= min_threshold
        
        # Neither color detected
        if not purple_valid and not red_valid:
            return 'normal', 0.0
        
        # Purple takes priority if both detected (shouldn't happen in practice)
        if purple_valid and purple_pixels >= red_pixels:
            confidence = self._calculate_confidence(purple_pixels, red_pixels, total_pixels)
            return 'turn', confidence
        elif red_valid:
            confidence = self._calculate_confidence(red_pixels, purple_pixels, total_pixels)
            return 'turn_overtime', confidence
        else:
            return 'normal', 0.0
    
    def _calculate_confidence(self, primary_pixels: int, secondary_pixels: int,
                             total_pixels: int) -> float:
        """
        Calculate confidence score for timer detection.
        
        Confidence is based on:
        - Pixel dominance (primary vs secondary color)
        - Absolute pixel count
        
        Args:
            primary_pixels: Pixel count for detected timer color
            secondary_pixels: Pixel count for other timer color
            total_pixels: Total pixels in region
            
        Returns:
            Confidence score (0.7-1.0 for detected timers)
        """
        # Base confidence for meeting threshold
        base_confidence = 0.7
        
        # Bonus for pixel dominance (how much more than the other color)
        total_timer_pixels = primary_pixels + secondary_pixels
        if total_timer_pixels > 0:
            dominance_ratio = primary_pixels / total_timer_pixels
            dominance_bonus = (dominance_ratio - 0.5) * 0.4  # 0.0 to 0.2 bonus
        else:
            dominance_bonus = 0.0
        
        # Bonus for strong pixel count (more pixels = more confident)
        min_threshold = self.settings.get("parser.timer.min_pixel_threshold")
        if primary_pixels > min_threshold * 5:  # 5x threshold
            count_bonus = 0.1
        elif primary_pixels > min_threshold * 2:  # 2x threshold
            count_bonus = 0.05
        else:
            count_bonus = 0.0
        
        # Calculate final confidence
        confidence = base_confidence + dominance_bonus + count_bonus
        
        # Clamp to valid range
        return max(0.0, min(1.0, confidence))
    
    def get_detection_metrics(self, name_region: np.ndarray) -> Dict[str, Any]:
        """
        Get detailed detection metrics for debugging.
        
        Args:
            name_region: BGR image region to analyze
            
        Returns:
            Dictionary with pixel counts, percentages, and threshold checks
        """
        if name_region is None or name_region.size == 0:
            return {}
        
        try:
            # Convert to HSV
            hsv_image = cv2.cvtColor(name_region, cv2.COLOR_BGR2HSV)
            
            # Count pixels
            purple_pixels = self._count_purple_pixels(hsv_image)
            red_pixels = self._count_red_pixels(hsv_image)
            total_pixels = name_region.size
            
            # Calculate percentages
            purple_percentage = purple_pixels / total_pixels if total_pixels > 0 else 0.0
            red_percentage = red_pixels / total_pixels if total_pixels > 0 else 0.0
            
            # Get threshold
            min_threshold = self.settings.get("parser.timer.min_pixel_threshold")
            
            # Evaluate state
            turn_state, confidence = self._evaluate_timer_state(
                purple_pixels, red_pixels, total_pixels
            )
            
            return {
                'region_shape': name_region.shape,
                'total_pixels': total_pixels,
                'purple_pixels': purple_pixels,
                'red_pixels': red_pixels,
                'purple_percentage': purple_percentage,
                'red_percentage': red_percentage,
                'min_threshold': min_threshold,
                'turn_state': turn_state,
                'confidence': confidence,
                'criteria_met': {
                    'purple_threshold': purple_pixels >= min_threshold,
                    'red_threshold': red_pixels >= min_threshold
                },
                'hsv_ranges': {
                    'purple': self.purple_range,
                    'red': self.red_ranges
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get detection metrics: {e}")
            return {}
    
    def update_color_ranges(self, purple_range: Optional[list] = None,
                           red_ranges: Optional[list] = None) -> None:
        """
        Update HSV color ranges for timer detection.
        
        Useful for tuning detection based on different lighting conditions
        or game client versions.
        
        Args:
            purple_range: New HSV range for purple timer [[h1,s1,v1], [h2,s2,v2]]
            red_ranges: New HSV ranges for red timer [[[h1,s1,v1], [h2,s2,v2]], ...]
        """
        if purple_range is not None:
            self.settings.update("parser.timer.purple_hsv_range", purple_range)
            self.purple_range = purple_range
            logger.info(f"Updated purple timer range: {purple_range}")
        
        if red_ranges is not None:
            self.settings.update("parser.timer.red_hsv_ranges", red_ranges)
            self.red_ranges = red_ranges
            logger.info(f"Updated red timer ranges: {red_ranges}")

