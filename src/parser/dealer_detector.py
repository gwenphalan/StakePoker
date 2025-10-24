#!/usr/bin/env python3
"""
Dealer button detector module for identifying dealer position.

Uses pixel brightness analysis to detect the dealer button (white circle
with black "D") without relying on OCR. Based on proven logic from archive.

Usage:
    from src.parser.dealer_detector import DealerDetector
    
    detector = DealerDetector()
    result = detector.detect_dealer_button(dealer_region)
    if result.has_dealer:
        print(f"Dealer button detected (confidence: {result.confidence:.3f})")
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

from src.config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class DealerResult:
    """Result of dealer button detection with status and confidence."""
    has_dealer: bool
    confidence: float


class DealerDetector:
    """
    Detect dealer buttons using pixel brightness analysis.
    
    Analyzes brightness distribution to identify white circle with black "D"
    characteristic of dealer buttons. More reliable than OCR for this use case.
    """
    
    def __init__(self):
        """Initialize dealer detector with settings."""
        self.settings = Settings()
        
        # Create dealer detection settings
        self.settings.create("parser.dealer.enabled", default=True)
        self.settings.create("parser.dealer.min_confidence", default=0.7)
        self.settings.create("parser.dealer.bright_threshold", default=200)
        self.settings.create("parser.dealer.dark_threshold", default=50)
        self.settings.create("parser.dealer.min_bright_ratio", default=0.2)
        self.settings.create("parser.dealer.min_dark_ratio", default=0.05)
        self.settings.create("parser.dealer.max_dark_ratio", default=0.8)
        self.settings.create("parser.dealer.min_contrast_ratio", default=0.3)
        
        logger.info("DealerDetector initialized")
    
    def detect_dealer_button(self, region: np.ndarray) -> DealerResult:
        """
        Detect if a region contains a dealer button.
        
        Args:
            region: BGR image region to analyze
            
        Returns:
            DealerResult with detection status and confidence
            
        Example:
            result = detector.detect_dealer_button(dealer_region)
            if result.has_dealer:
                print(f"Dealer found: {result.confidence:.3f}")
        """
        # Check if detection is enabled
        if not self.settings.get("parser.dealer.enabled"):
            return DealerResult(has_dealer=False, confidence=1.0)  # High confidence for no dealer when disabled
        
        # Validate input
        if region is None or region.size == 0:
            logger.debug("Empty region provided to dealer detector - returning high confidence for no dealer")
            return DealerResult(has_dealer=False, confidence=1.0)
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            
            # Analyze pixel distribution
            metrics = self._analyze_pixel_distribution(gray)
            
            # Check if criteria are met
            has_dealer = self._meets_dealer_criteria(metrics)
            
            # Calculate confidence
            confidence = self._calculate_confidence(metrics)
            
            # Log result
            if has_dealer:
                logger.info(f"Dealer button detected (confidence={confidence:.3f})")
            else:
                logger.debug(f"No dealer button (confidence={confidence:.3f})")
            
            return DealerResult(has_dealer=has_dealer, confidence=confidence)
            
        except Exception as e:
            logger.error(f"Dealer detection failed: {e}")
            return DealerResult(has_dealer=False, confidence=1.0)  # High confidence for no dealer on error
    
    def get_detection_metrics(self, region: np.ndarray) -> Dict[str, Any]:
        """
        Get detailed detection metrics for debugging.
        
        Args:
            region: BGR image region to analyze
            
        Returns:
            Dictionary with pixel counts, ratios, and threshold checks
        """
        if region is None or region.size == 0:
            return {}
        
        try:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            metrics = self._analyze_pixel_distribution(gray)
            
            # Load thresholds for debug info
            bright_threshold = self.settings.get("parser.dealer.bright_threshold")
            dark_threshold = self.settings.get("parser.dealer.dark_threshold")
            min_bright = self.settings.get("parser.dealer.min_bright_ratio")
            min_dark = self.settings.get("parser.dealer.min_dark_ratio")
            max_dark = self.settings.get("parser.dealer.max_dark_ratio")
            min_contrast = self.settings.get("parser.dealer.min_contrast_ratio")
            
            return {
                'region_shape': region.shape,
                'total_pixels': metrics['total_pixels'],
                'bright_pixels': metrics['bright_pixels'],
                'dark_pixels': metrics['dark_pixels'],
                'bright_ratio': metrics['bright_ratio'],
                'dark_ratio': metrics['dark_ratio'],
                'contrast_ratio': metrics['contrast_ratio'],
                'avg_brightness': metrics['avg_brightness'],
                'std_brightness': metrics['std_brightness'],
                'criteria_met': {
                    'bright_ratio': metrics['bright_ratio'] > min_bright,
                    'dark_ratio_min': metrics['dark_ratio'] > min_dark,
                    'dark_ratio_max': metrics['dark_ratio'] < max_dark,
                    'contrast': metrics['contrast_ratio'] > min_contrast
                },
                'thresholds': {
                    'bright_threshold': bright_threshold,
                    'dark_threshold': dark_threshold,
                    'min_bright_ratio': min_bright,
                    'min_dark_ratio': min_dark,
                    'max_dark_ratio': max_dark,
                    'min_contrast_ratio': min_contrast
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get detection metrics: {e}")
            return {}
    
    def _analyze_pixel_distribution(self, gray: np.ndarray) -> Dict[str, float]:
        """Calculate brightness statistics from grayscale image."""
        # Load thresholds from settings
        bright_threshold = self.settings.get("parser.dealer.bright_threshold")
        dark_threshold = self.settings.get("parser.dealer.dark_threshold")
        
        # Count pixels
        very_bright = np.sum(gray > bright_threshold)
        very_dark = np.sum(gray < dark_threshold)
        total_pixels = gray.size
        
        # Calculate ratios
        bright_ratio = very_bright / total_pixels
        dark_ratio = very_dark / total_pixels
        contrast_ratio = bright_ratio + dark_ratio
        
        # Additional statistics
        avg_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        return {
            'total_pixels': total_pixels,
            'bright_pixels': very_bright,
            'dark_pixels': very_dark,
            'bright_ratio': bright_ratio,
            'dark_ratio': dark_ratio,
            'contrast_ratio': contrast_ratio,
            'avg_brightness': avg_brightness,
            'std_brightness': std_brightness
        }
    
    def _meets_dealer_criteria(self, metrics: Dict) -> bool:
        """Check if pixel distribution meets dealer button criteria."""
        # Load threshold settings
        min_bright = self.settings.get("parser.dealer.min_bright_ratio")
        min_dark = self.settings.get("parser.dealer.min_dark_ratio")
        max_dark = self.settings.get("parser.dealer.max_dark_ratio")
        min_contrast = self.settings.get("parser.dealer.min_contrast_ratio")
        
        # Check all criteria
        criteria_met = (
            metrics['bright_ratio'] > min_bright and
            metrics['dark_ratio'] > min_dark and
            metrics['dark_ratio'] < max_dark and
            metrics['contrast_ratio'] > min_contrast
        )
        
        # Log detailed criteria check at debug level
        logger.debug(f"Criteria check: bright={metrics['bright_ratio']:.3f}>{min_bright}, "
                    f"dark_min={metrics['dark_ratio']:.3f}>{min_dark}, "
                    f"dark_max={metrics['dark_ratio']:.3f}<{max_dark}, "
                    f"contrast={metrics['contrast_ratio']:.3f}>{min_contrast} -> {criteria_met}")
        
        return criteria_met
    
    def _calculate_confidence(self, metrics: Dict) -> float:
        """Calculate confidence score based on how well metrics match ideal."""
        # Check if criteria are met first
        if not self._meets_dealer_criteria(metrics):
            # High confidence that there's no dealer button
            return 1.0
        
        # Start with base confidence
        confidence = 0.9
        
        # Adjust based on ratio quality
        # Ideal bright ratio: ~0.3-0.5 (30-50% white)
        ideal_bright = 0.4
        bright_deviation = abs(metrics['bright_ratio'] - ideal_bright)
        if bright_deviation < 0.1:  # Close to ideal
            confidence += 0.05
        elif bright_deviation > 0.2:  # Far from ideal
            confidence -= 0.1
        
        # Ideal dark ratio: ~0.1-0.2 (10-20% black)
        ideal_dark = 0.15
        dark_deviation = abs(metrics['dark_ratio'] - ideal_dark)
        if dark_deviation < 0.05:  # Close to ideal
            confidence += 0.05
        elif dark_deviation > 0.15:  # Far from ideal
            confidence -= 0.1
        
        # Factor in contrast quality
        if metrics['std_brightness'] > 50:  # Good contrast
            confidence += 0.05
        elif metrics['std_brightness'] < 20:  # Poor contrast
            confidence -= 0.1
        
        # Clamp confidence between 0.0 and 1.0
        return max(0.0, min(1.0, confidence))
