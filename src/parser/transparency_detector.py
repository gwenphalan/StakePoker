#!/usr/bin/env python3
"""
Transparency detector module for identifying folded player states.

Uses direct pixel analysis to detect when player name plates appear
transparent, indicating a folded player state. Based on proven logic
from the archive implementation.

Usage:
    from src.parser.transparency_detector import TransparencyDetector
    
    detector = TransparencyDetector()
    
    # Detect transparency using both name and bank regions
    result = detector.detect_player_transparency(name_region, bank_region)
    if result.is_transparent:
        print(f"Player is folded (confidence: {result.confidence:.3f})")
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

from src.config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class TransparencyResult:
    """Result of transparency detection with status and confidence."""
    is_transparent: bool
    confidence: float


class TransparencyDetector:
    """
    Detect player name plate transparency using pixel analysis.
    
    Uses multi-criteria analysis (contrast, saturation, brightness) to
    identify when player name plates appear transparent, indicating
    a folded player state.
    """
    
    def __init__(self):
        """Initialize transparency detector with settings."""
        self.settings = Settings()
        
        # Create transparency detection settings
        self.settings.create("parser.transparency.enabled", default=True)
        self.settings.create("parser.transparency.contrast_threshold", default=15)
        self.settings.create("parser.transparency.saturation_threshold", default=30)
        self.settings.create("parser.transparency.brightness_min", default=50)
        self.settings.create("parser.transparency.brightness_max", default=80)
        self.settings.create("parser.transparency.require_all_criteria", default=True)
        self.settings.create("parser.transparency.multi_region_threshold", default=2.0)
        
        logger.info("TransparencyDetector initialized")
    
    def detect_transparency(self, region: np.ndarray) -> TransparencyResult:
        """
        Detect if a single region appears transparent.
        
        Args:
            region: BGR image region to analyze
            
        Returns:
            TransparencyResult with transparency status and confidence
        """
        if not self.settings.get("parser.transparency.enabled"):
            return TransparencyResult(
                is_transparent=False,
                confidence=0.0
            )
        
        if region is None or region.size == 0:
            logger.warning("Empty region provided to transparency detector")
            return TransparencyResult(
                is_transparent=False,
                confidence=0.0
            )
        
        try:
            # Analyze transparency metrics
            metrics = self._analyze_transparency_metrics(region)
            
            # Apply detection criteria
            is_transparent, confidence = self._evaluate_transparency(metrics)
            
            logger.debug(f"Transparency detection: {is_transparent} (confidence={confidence:.3f})")
            
            return TransparencyResult(
                is_transparent=is_transparent,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Transparency detection failed: {e}")
            return TransparencyResult(
                is_transparent=False,
                confidence=0.0
            )
    
    def detect_player_transparency(self, name_region: np.ndarray, bank_region: np.ndarray = None) -> TransparencyResult:
        """
        Detect transparency using both name and bank regions for better accuracy.
        
        Args:
            name_region: Player name plate region
            bank_region: Player bank/stack region (optional)
            
        Returns:
            TransparencyResult with combined transparency analysis
            
        Example:
            result = detector.detect_player_transparency(name_region, bank_region)
            if result.is_transparent:
                print(f"Player is folded (confidence: {result.confidence:.3f})")
        """
        if not self.settings.get("parser.transparency.enabled"):
            return TransparencyResult(
                is_transparent=False,
                confidence=0.0
            )
        
        regions_to_check = [name_region]
        if bank_region is not None:
            regions_to_check.append(bank_region)
        
        transparency_scores = []
        confidences = []
        
        for i, region in enumerate(regions_to_check):
            if region is None or region.size == 0:
                continue
                
            try:
                metrics = self._analyze_transparency_metrics(region)
                is_transparent, confidence = self._evaluate_transparency(metrics)
                
                # Convert boolean to score (0 or 1)
                transparency_scores.append(1 if is_transparent else 0)
                confidences.append(confidence)
                
                logger.debug(f"Region {i+1} transparency: {is_transparent} (confidence={confidence:.3f})")
                
            except Exception as e:
                logger.error(f"Failed to analyze region {i+1}: {e}")
                continue
        
        if not transparency_scores:
            return TransparencyResult(
                is_transparent=False,
                confidence=0.0
            )
        
        # Calculate combined result
        avg_transparency_score = sum(transparency_scores) / len(transparency_scores)
        avg_confidence = sum(confidences) / len(confidences)
        
        # Use configurable threshold for multi-region decision
        threshold = self.settings.get("parser.transparency.multi_region_threshold")
        is_transparent = avg_transparency_score >= (threshold / len(regions_to_check))
        
        logger.debug(f"Multi-region transparency: {is_transparent} (avg_score={avg_transparency_score:.3f}, threshold={threshold/len(regions_to_check):.3f})")
        
        return TransparencyResult(
            is_transparent=is_transparent,
            confidence=avg_confidence
        )
    
    def _analyze_transparency_metrics(self, region: np.ndarray) -> Dict[str, float]:
        """
        Analyze transparency indicators in a region.
        
        Args:
            region: BGR image region
            
        Returns:
            Dictionary of transparency metrics
        """
        # Convert to grayscale and HSV
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Calculate key metrics
        contrast = np.std(gray)  # Standard deviation for contrast
        avg_brightness = np.mean(gray)
        saturation = np.mean(hsv[:, :, 1])  # S channel
        brightness_variance = np.var(gray)
        
        return {
            'contrast': float(contrast),
            'avg_brightness': float(avg_brightness),
            'saturation': float(saturation),
            'brightness_variance': float(brightness_variance)
        }
    
    def _evaluate_transparency(self, metrics: Dict[str, float]) -> tuple[bool, float]:
        """
        Evaluate transparency based on metrics and thresholds.
        
        Args:
            metrics: Transparency metrics dictionary
            
        Returns:
            Tuple of (is_transparent, confidence)
        """
        # Get thresholds from settings
        contrast_threshold = self.settings.get("parser.transparency.contrast_threshold")
        saturation_threshold = self.settings.get("parser.transparency.saturation_threshold")
        brightness_min = self.settings.get("parser.transparency.brightness_min")
        brightness_max = self.settings.get("parser.transparency.brightness_max")
        require_all = self.settings.get("parser.transparency.require_all_criteria")
        
        # Apply criteria
        very_low_contrast = metrics['contrast'] < contrast_threshold
        very_low_saturation = metrics['saturation'] < saturation_threshold
        very_mid_brightness = brightness_min < metrics['avg_brightness'] < brightness_max
        
        # Calculate transparency score
        criteria_met = sum([very_low_contrast, very_low_saturation, very_mid_brightness])
        total_criteria = 3
        
        if require_all:
            # Conservative: require ALL criteria
            is_transparent = criteria_met >= total_criteria
            confidence = criteria_met / total_criteria
        else:
            # Lenient: require majority of criteria
            is_transparent = criteria_met >= 2
            confidence = criteria_met / total_criteria
        
        return is_transparent, confidence
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about transparency detection performance.
        
        Returns:
            Dictionary with detection statistics
        """
        return {
            'settings': {
                'enabled': self.settings.get("parser.transparency.enabled"),
                'contrast_threshold': self.settings.get("parser.transparency.contrast_threshold"),
                'saturation_threshold': self.settings.get("parser.transparency.saturation_threshold"),
                'brightness_range': f"{self.settings.get('parser.transparency.brightness_min')}-{self.settings.get('parser.transparency.brightness_max')}",
                'require_all_criteria': self.settings.get("parser.transparency.require_all_criteria"),
                'multi_region_threshold': self.settings.get("parser.transparency.multi_region_threshold")
            }
        }

