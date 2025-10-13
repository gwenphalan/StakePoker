#!/usr/bin/env python3
"""
Background subtraction module for isolating foreground elements.

Provides multiple mathematical approaches to remove background images
and extract transparent/foreground elements using color operations.

Usage:
    from src.parser.background_subtractor import BackgroundSubtractor
    
    subtractor = BackgroundSubtractor()
    
    # Method 1: Simple pixel difference
    foreground = subtractor.simple_difference(composite_image, background_image)
    
    # Method 2: Statistical subtraction
    foreground = subtractor.statistical_subtraction(composite_image, background_image)
    
    # Method 3: Correlation-based detection
    foreground_mask = subtractor.correlation_detection(composite_image, background_image)
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

from src.config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class SubtractionResult:
    """Result of background subtraction operation."""
    foreground_image: np.ndarray
    foreground_mask: np.ndarray
    confidence: float
    method_used: str


class BackgroundSubtractor:
    """
    Background subtraction using various mathematical approaches.
    
    Provides multiple methods to isolate foreground elements from
    composite images by comparing against known background images.
    """
    
    def __init__(self):
        """Initialize background subtractor with settings."""
        self.settings = Settings()
        
        # Create background subtraction settings
        self.settings.create("parser.background_subtraction.enabled", default=True)
        self.settings.create("parser.background_subtraction.pixel_threshold", default=30)
        self.settings.create("parser.background_subtraction.statistical_multiplier", default=2.0)
        self.settings.create("parser.background_subtraction.correlation_threshold", default=0.8)
        self.settings.create("parser.background_subtraction.hsv_saturation_threshold", default=30)
        self.settings.create("parser.background_subtraction.hsv_brightness_threshold", default=40)
        
        logger.info("BackgroundSubtractor initialized")
    
    def simple_difference(self, composite_image: np.ndarray, background_image: np.ndarray) -> SubtractionResult:
        """
        Simple pixel-wise difference method.
        
        Args:
            composite_image: Image with foreground elements on background
            background_image: Pure background image
            
        Returns:
            SubtractionResult with isolated foreground
        """
        if not self._validate_inputs(composite_image, background_image):
            return self._empty_result("simple_difference")
        
        try:
            # Convert to float for better precision
            comp_float = composite_image.astype(np.float32)
            bg_float = background_image.astype(np.float32)
            
            # Calculate absolute difference
            diff = np.abs(comp_float - bg_float)
            
            # Create mask based on threshold
            threshold = self.settings.get("parser.background_subtraction.pixel_threshold")
            foreground_mask = np.any(diff > threshold, axis=2)
            
            # Apply mask to get foreground
            foreground_image = composite_image.copy()
            foreground_image[~foreground_mask] = [0, 0, 0]  # Set background pixels to black
            
            # Calculate confidence based on difference magnitude
            max_diff = np.max(diff)
            confidence = min(max_diff / 255.0, 1.0)
            
            logger.debug(f"Simple difference: {np.sum(foreground_mask)} foreground pixels (confidence={confidence:.3f})")
            
            return SubtractionResult(
                foreground_image=foreground_image,
                foreground_mask=foreground_mask.astype(np.uint8) * 255,
                confidence=confidence,
                method_used="simple_difference"
            )
            
        except Exception as e:
            logger.error(f"Simple difference subtraction failed: {e}")
            return self._empty_result("simple_difference")
    
    def statistical_subtraction(self, composite_image: np.ndarray, background_image: np.ndarray) -> SubtractionResult:
        """
        Statistical background subtraction using mean and standard deviation.
        
        Args:
            composite_image: Image with foreground elements on background
            background_image: Pure background image
            
        Returns:
            SubtractionResult with isolated foreground
        """
        if not self._validate_inputs(composite_image, background_image):
            return self._empty_result("statistical_subtraction")
        
        try:
            # Calculate background statistics
            bg_mean = np.mean(background_image, axis=(0, 1))
            bg_std = np.std(background_image, axis=(0, 1))
            
            # Avoid division by zero
            bg_std = np.maximum(bg_std, 1.0)
            
            # Calculate normalized difference
            comp_float = composite_image.astype(np.float32)
            diff = np.abs(comp_float - bg_mean)
            
            # Create mask using statistical threshold
            multiplier = self.settings.get("parser.background_subtraction.statistical_multiplier")
            threshold = bg_std * multiplier
            
            # Check if any channel exceeds threshold
            foreground_mask = np.any(diff > threshold, axis=2)
            
            # Apply mask
            foreground_image = composite_image.copy()
            foreground_image[~foreground_mask] = [0, 0, 0]
            
            # Calculate confidence based on statistical significance
            max_normalized_diff = np.max(diff / bg_std)
            confidence = min(max_normalized_diff / (multiplier * 3), 1.0)
            
            logger.debug(f"Statistical subtraction: {np.sum(foreground_mask)} foreground pixels (confidence={confidence:.3f})")
            
            return SubtractionResult(
                foreground_image=foreground_image,
                foreground_mask=foreground_mask.astype(np.uint8) * 255,
                confidence=confidence,
                method_used="statistical_subtraction"
            )
            
        except Exception as e:
            logger.error(f"Statistical subtraction failed: {e}")
            return self._empty_result("statistical_subtraction")
    
    def correlation_detection(self, composite_image: np.ndarray, background_image: np.ndarray) -> SubtractionResult:
        """
        Correlation-based foreground detection using structural similarity.
        
        Args:
            composite_image: Image with foreground elements on background
            background_image: Pure background image
            
        Returns:
            SubtractionResult with correlation-based mask
        """
        if not self._validate_inputs(composite_image, background_image):
            return self._empty_result("correlation_detection")
        
        try:
            # Convert to grayscale for correlation
            comp_gray = cv2.cvtColor(composite_image, cv2.COLOR_BGR2GRAY)
            bg_gray = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
            
            # Calculate local correlation using sliding window
            window_size = 15
            step_size = 5
            
            h, w = comp_gray.shape
            correlation_map = np.zeros((h, w), dtype=np.float32)
            
            for y in range(0, h - window_size, step_size):
                for x in range(0, w - window_size, step_size):
                    # Extract windows
                    comp_window = comp_gray[y:y+window_size, x:x+window_size]
                    bg_window = bg_gray[y:y+window_size, x:x+window_size]
                    
                    # Calculate correlation coefficient
                    correlation = cv2.matchTemplate(comp_window, bg_window, cv2.TM_CCOEFF_NORMED)[0, 0]
                    
                    # Fill correlation map
                    correlation_map[y:y+window_size, x:x+window_size] = correlation
            
            # Create foreground mask based on correlation threshold
            threshold = self.settings.get("parser.background_subtraction.correlation_threshold")
            foreground_mask = correlation_map < threshold
            
            # Apply mask
            foreground_image = composite_image.copy()
            foreground_image[~foreground_mask] = [0, 0, 0]
            
            # Calculate confidence based on correlation distribution
            avg_correlation = np.mean(correlation_map)
            confidence = 1.0 - avg_correlation
            
            logger.debug(f"Correlation detection: {np.sum(foreground_mask)} foreground pixels (confidence={confidence:.3f})")
            
            return SubtractionResult(
                foreground_image=foreground_image,
                foreground_mask=foreground_mask.astype(np.uint8) * 255,
                confidence=confidence,
                method_used="correlation_detection"
            )
            
        except Exception as e:
            logger.error(f"Correlation detection failed: {e}")
            return self._empty_result("correlation_detection")
    
    def hsv_analysis(self, composite_image: np.ndarray, background_image: np.ndarray) -> SubtractionResult:
        """
        HSV color space analysis for transparency detection.
        
        Args:
            composite_image: Image with foreground elements on background
            background_image: Pure background image
            
        Returns:
            SubtractionResult with HSV-based mask
        """
        if not self._validate_inputs(composite_image, background_image):
            return self._empty_result("hsv_analysis")
        
        try:
            # Convert to HSV
            comp_hsv = cv2.cvtColor(composite_image, cv2.COLOR_BGR2HSV)
            bg_hsv = cv2.cvtColor(background_image, cv2.COLOR_BGR2HSV)
            
            # Calculate differences in saturation and brightness
            sat_diff = np.abs(comp_hsv[:, :, 1].astype(np.float32) - bg_hsv[:, :, 1].astype(np.float32))
            bright_diff = np.abs(comp_hsv[:, :, 2].astype(np.float32) - bg_hsv[:, :, 2].astype(np.float32))
            
            # Create mask based on HSV thresholds
            sat_threshold = self.settings.get("parser.background_subtraction.hsv_saturation_threshold")
            bright_threshold = self.settings.get("parser.background_subtraction.hsv_brightness_threshold")
            
            # Low differences indicate transparency (background visible)
            transparency_mask = (sat_diff < sat_threshold) & (bright_diff < bright_threshold)
            foreground_mask = ~transparency_mask
            
            # Apply mask
            foreground_image = composite_image.copy()
            foreground_image[transparency_mask] = [0, 0, 0]
            
            # Calculate confidence based on HSV differences
            avg_sat_diff = np.mean(sat_diff)
            avg_bright_diff = np.mean(bright_diff)
            confidence = min((avg_sat_diff + avg_bright_diff) / 200.0, 1.0)
            
            logger.debug(f"HSV analysis: {np.sum(foreground_mask)} foreground pixels (confidence={confidence:.3f})")
            
            return SubtractionResult(
                foreground_image=foreground_image,
                foreground_mask=foreground_mask.astype(np.uint8) * 255,
                confidence=confidence,
                method_used="hsv_analysis"
            )
            
        except Exception as e:
            logger.error(f"HSV analysis failed: {e}")
            return self._empty_result("hsv_analysis")
    
    def multi_method_subtraction(self, composite_image: np.ndarray, background_image: np.ndarray) -> SubtractionResult:
        """
        Combine multiple methods for robust foreground extraction.
        
        Args:
            composite_image: Image with foreground elements on background
            background_image: Pure background image
            
        Returns:
            SubtractionResult with combined analysis
        """
        if not self._validate_inputs(composite_image, background_image):
            return self._empty_result("multi_method_subtraction")
        
        try:
            # Run all methods
            methods = [
                self.simple_difference,
                self.statistical_subtraction,
                self.correlation_detection,
                self.hsv_analysis
            ]
            
            results = []
            for method in methods:
                result = method(composite_image, background_image)
                results.append(result)
            
            # Combine masks using majority voting
            masks = [result.foreground_mask for result in results]
            combined_mask = np.zeros_like(masks[0], dtype=np.float32)
            
            for mask in masks:
                combined_mask += mask.astype(np.float32) / 255.0
            
            # Majority threshold (at least 2 methods agree)
            final_mask = combined_mask >= 0.5
            
            # Apply final mask
            foreground_image = composite_image.copy()
            foreground_image[~final_mask] = [0, 0, 0]
            
            # Calculate average confidence
            avg_confidence = np.mean([result.confidence for result in results])
            
            logger.debug(f"Multi-method subtraction: {np.sum(final_mask)} foreground pixels (confidence={avg_confidence:.3f})")
            
            return SubtractionResult(
                foreground_image=foreground_image,
                foreground_mask=final_mask.astype(np.uint8) * 255,
                confidence=avg_confidence,
                method_used="multi_method_subtraction"
            )
            
        except Exception as e:
            logger.error(f"Multi-method subtraction failed: {e}")
            return self._empty_result("multi_method_subtraction")
    
    def _validate_inputs(self, composite_image: np.ndarray, background_image: np.ndarray) -> bool:
        """Validate input images."""
        if composite_image is None or background_image is None:
            logger.warning("One or both input images are None")
            return False
        
        if composite_image.size == 0 or background_image.size == 0:
            logger.warning("One or both input images are empty")
            return False
        
        if composite_image.shape != background_image.shape:
            logger.warning(f"Image shapes don't match: {composite_image.shape} vs {background_image.shape}")
            return False
        
        return True
    
    def _empty_result(self, method_name: str) -> SubtractionResult:
        """Return empty result for failed operations."""
        return SubtractionResult(
            foreground_image=np.array([]),
            foreground_mask=np.array([]),
            confidence=0.0,
            method_used=method_name
        )
    
    def transparency_simulation(self, composite_image: np.ndarray, background_image: np.ndarray) -> SubtractionResult:
        """
        Simulate transparent elements as if rendered over black background.
        
        This method extracts the transparency effect by calculating what the
        transparent elements would look like if they were composited over
        a black background instead of the poker table background.
        
        Args:
            composite_image: Image with transparent elements on background
            background_image: Pure background image
            
        Returns:
            SubtractionResult with transparency-simulated image
            
        Mathematical approach:
        For each pixel: if composite ≈ background, it's transparent (black)
        If composite ≠ background, calculate the "pure" foreground color
        that would produce the composite when blended with black background
        """
        if not self._validate_inputs(composite_image, background_image):
            return self._empty_result("transparency_simulation")
        
        try:
            # Convert to float for precise calculations
            comp_float = composite_image.astype(np.float32) / 255.0
            bg_float = background_image.astype(np.float32) / 255.0
            
            # Calculate alpha (transparency) for each pixel
            # Alpha represents how much of the foreground is visible
            alpha = np.zeros(comp_float.shape[:2], dtype=np.float32)
            
            # For each pixel, calculate alpha based on how different it is from background
            diff = np.abs(comp_float - bg_float)
            max_diff = np.max(diff, axis=2)  # Maximum difference across color channels
            
            # Alpha calculation: higher difference = higher alpha (more opaque)
            # Use a threshold to determine if pixel is transparent or not
            threshold = self.settings.get("parser.background_subtraction.pixel_threshold") / 255.0
            
            # Calculate alpha for non-transparent pixels
            # Alpha = (composite - background) / (foreground - background)
            # But since we don't know foreground, we estimate it
            for y in range(comp_float.shape[0]):
                for x in range(comp_float.shape[1]):
                    pixel_diff = max_diff[y, x]
                    
                    if pixel_diff > threshold:
                        # This pixel has foreground content
                        # Calculate what the "pure" foreground color would be
                        # when composited over black background
                        
                        # Estimate alpha based on difference magnitude
                        alpha[y, x] = min(pixel_diff * 3.0, 1.0)  # Scale factor for alpha estimation
                    else:
                        # This pixel is transparent/background
                        alpha[y, x] = 0.0
            
            # Create the simulated image: foreground elements over black background
            simulated_image = np.zeros_like(comp_float)
            
            # For pixels with alpha > 0, calculate the foreground color
            # that would produce the composite when blended with black
            mask = alpha > 0
            
            if np.any(mask):
                # Calculate foreground color: F = (C - B*(1-alpha)) / alpha
                # Where C = composite, B = background, F = foreground
                # But since we want it over black, we use: F = C / alpha
                foreground_colors = comp_float[mask] / np.stack([alpha[mask]] * 3, axis=1)
                
                # Clamp to valid range
                foreground_colors = np.clip(foreground_colors, 0, 1)
                
                # Apply alpha blending with black background
                # Result = foreground * alpha + black * (1-alpha)
                simulated_image[mask] = foreground_colors * np.stack([alpha[mask]] * 3, axis=1)
            
            # Convert back to uint8
            simulated_image = (simulated_image * 255).astype(np.uint8)
            
            # Create mask for transparency detection
            transparency_mask = (alpha < 0.1).astype(np.uint8) * 255
            
            # Calculate confidence based on how well we can reconstruct
            reconstruction_error = np.mean(np.abs(comp_float - bg_float))
            confidence = min(reconstruction_error * 2, 1.0)
            
            logger.debug(f"Transparency simulation: {np.sum(alpha > 0)} opaque pixels (confidence={confidence:.3f})")
            
            return SubtractionResult(
                foreground_image=simulated_image,
                foreground_mask=transparency_mask,
                confidence=confidence,
                method_used="transparency_simulation"
            )
            
        except Exception as e:
            logger.error(f"Transparency simulation failed: {e}")
            return self._empty_result("transparency_simulation")
    
    def alpha_extraction(self, composite_image: np.ndarray, background_image: np.ndarray) -> SubtractionResult:
        """
        Extract alpha channel and foreground colors for perfect reconstruction.
        
        This method calculates the exact alpha and foreground colors that would
        allow perfect reconstruction of the original composite image.
        
        Args:
            composite_image: Image with transparent elements on background
            background_image: Pure background image
            
        Returns:
            SubtractionResult with alpha-extracted image
        """
        if not self._validate_inputs(composite_image, background_image):
            return self._empty_result("alpha_extraction")
        
        try:
            # Convert to float
            comp_float = composite_image.astype(np.float32) / 255.0
            bg_float = background_image.astype(np.float32) / 255.0
            
            # Calculate alpha for each pixel
            # Alpha = 1 - (similarity to background)
            diff = np.abs(comp_float - bg_float)
            pixel_diff = np.mean(diff, axis=2)  # Average difference across channels
            
            # Convert difference to alpha (0 = transparent, 1 = opaque)
            alpha = np.clip(pixel_diff * 4.0, 0, 1)  # Scale factor for alpha
            
            # Calculate foreground colors
            # F = (C - B*(1-alpha)) / alpha
            # Where C = composite, B = background, F = foreground
            foreground = np.zeros_like(comp_float)
            
            # Avoid division by zero
            alpha_safe = np.maximum(alpha, 0.001)
            
            for c in range(3):  # For each color channel
                foreground[:, :, c] = (comp_float[:, :, c] - bg_float[:, :, c] * (1 - alpha)) / alpha_safe
            
            # Clamp foreground colors
            foreground = np.clip(foreground, 0, 1)
            
            # Create simulated image: foreground over black background
            simulated = foreground * np.stack([alpha] * 3, axis=2)
            simulated = (simulated * 255).astype(np.uint8)
            
            # Create mask
            mask = (alpha > 0.1).astype(np.uint8) * 255
            
            # Calculate confidence
            confidence = np.mean(alpha)
            
            logger.debug(f"Alpha extraction: {np.sum(alpha > 0.1)} opaque pixels (confidence={confidence:.3f})")
            
            return SubtractionResult(
                foreground_image=simulated,
                foreground_mask=mask,
                confidence=confidence,
                method_used="alpha_extraction"
            )
            
        except Exception as e:
            logger.error(f"Alpha extraction failed: {e}")
            return self._empty_result("alpha_extraction")
    
    def get_settings(self) -> Dict[str, Any]:
        """Get current settings for background subtraction."""
        return {
            'enabled': self.settings.get("parser.background_subtraction.enabled"),
            'pixel_threshold': self.settings.get("parser.background_subtraction.pixel_threshold"),
            'statistical_multiplier': self.settings.get("parser.background_subtraction.statistical_multiplier"),
            'correlation_threshold': self.settings.get("parser.background_subtraction.correlation_threshold"),
            'hsv_saturation_threshold': self.settings.get("parser.background_subtraction.hsv_saturation_threshold"),
            'hsv_brightness_threshold': self.settings.get("parser.background_subtraction.hsv_brightness_threshold")
        }
