#!/usr/bin/env python3
"""
OCR engine module for text extraction from poker game images.

Uses EasyOCR with configurable preprocessing to maximize text recognition accuracy.
Automatically tries multiple preprocessing methods and selects the result with
highest confidence.

Usage:
    from src.parser.ocr_engine import OCREngine
    
    ocr = OCREngine()
    
    # Extract text with confidence and method info
    text, confidence, method = ocr.extract_text(image)
    print(f"Extracted '{text}' with {confidence:.2f} confidence using {method}")
    
    # Simple extraction (just text)
    text = ocr.extract_text_simple(image)
"""

import easyocr
import numpy as np
import logging
import re
from typing import Optional, Tuple, List, Dict
from collections import defaultdict

from src.parser.image_preprocessor import ImagePreprocessor
from src.config.settings import Settings

logger = logging.getLogger(__name__)


class OCREngine:
    """
    OCR engine for text extraction with automatic preprocessing optimization.
    
    Uses EasyOCR with multiple preprocessing methods to maximize accuracy.
    Tracks which preprocessing methods work best for future optimization.
    """
    
    def __init__(self):
        """Initialize easyocr reader and preprocessor."""
        self.settings = Settings()
        
        # Create OCR settings with defaults
        self.settings.create("parser.ocr.languages", default=["en"])
        self.settings.create("parser.ocr.gpu", default=True)
        self.settings.create("parser.ocr.min_confidence", default=0.5)
        self.settings.create("parser.ocr.preprocessing_enabled", default=True)
        self.settings.create("parser.ocr.paragraph", default=False)
        
        # Load settings
        languages = self.settings.get("parser.ocr.languages")
        gpu = self.settings.get("parser.ocr.gpu")
        
        # Initialize EasyOCR reader
        try:
            logger.info(f"Initializing EasyOCR reader (languages={languages}, gpu={gpu})")
            self.reader = easyocr.Reader(languages, gpu=gpu)
            logger.info("EasyOCR reader initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR reader: {e}")
            raise
        
        # Initialize preprocessor
        self.preprocessor = ImagePreprocessor()
        
        # Track which preprocessing methods work best
        self.method_success_count: Dict[str, int] = defaultdict(int)
        self.method_total_confidence: Dict[str, float] = defaultdict(float)
        
        logger.info("OCREngine initialized successfully")
    
    def extract_text(self, image: np.ndarray) -> Tuple[str, float, str]:
        """
        Extract text from image with confidence score and preprocessing method.
        
        Tries multiple preprocessing methods if enabled and returns the result
        with the highest confidence score.
        
        Args:
            image: Input image (BGR or grayscale numpy array)
        
        Returns:
            Tuple of (text, confidence, preprocessing_method_used)
            - text: Extracted text string (empty if no text found)
            - confidence: Confidence score (0.0 to 1.0)
            - preprocessing_method_used: Name of the preprocessing method that worked best
        
        Example:
            text, conf, method = ocr.extract_text(image)
            if conf > 0.7:
                print(f"High confidence: '{text}' via {method}")
        """
        preprocessing_enabled = self.settings.get("parser.ocr.preprocessing_enabled")
        min_confidence = self.settings.get("parser.ocr.min_confidence")
        
        # Check for empty or invalid image
        if image is None or image.size == 0:
            logger.debug("Empty or None image provided to OCR engine")
            return "", 1.0, "none"  # High confidence that there's no text in empty image
        
        best_text = ""
        best_confidence = 0.0
        best_method = "none"
        
        if preprocessing_enabled:
            # Try all preprocessing variants
            variants = self.preprocessor.preprocess_all_methods(image)
            logger.debug(f"Trying {len(variants)} preprocessing variants")
            
            # Set best_method to first method if we have variants
            if variants:
                best_method = variants[0][0]  # First method name
            
            methods_tried = 0
            methods_succeeded = 0
            
            for method_name, processed_image in variants:
                methods_tried += 1
                try:
                    text, confidence = self._run_ocr(processed_image)
                    methods_succeeded += 1
                    
                    logger.debug(f"Method '{method_name}': text='{text}', confidence={confidence:.3f}")
                    
                    # Track this method's performance
                    if text:  # Only count if text was found
                        self.method_total_confidence[method_name] += confidence
                        if confidence >= min_confidence:
                            self.method_success_count[method_name] += 1
                    
                    # Calculate adjusted confidence (penalize truncated text)
                    adjusted_confidence = confidence
                    
                    # Penalize methods that extract truncated text
                    # Look for incomplete patterns that suggest truncation
                    truncation_patterns = [
                        r'Stakes:\s*\d+$',  # "Stakes: 100" (missing /200)
                        r'Blinds:\s*\d+$',  # "Blinds: 100" (missing /200)
                        r'\d+\s*$',  # Ends with just a number
                    ]
                    
                    for pattern in truncation_patterns:
                        if re.search(pattern, text.strip()):
                            adjusted_confidence *= 0.7  # 30% penalty for truncated text
                            logger.debug(f"Applied truncation penalty to method '{method_name}': '{text}'")
                            break
                    
                    # Keep best result (use adjusted confidence for selection)
                    if adjusted_confidence > best_confidence:
                        best_text = text
                        best_confidence = adjusted_confidence
                        best_method = method_name
                
                except Exception as e:
                    logger.error(f"OCR failed for method '{method_name}': {e}")
                    continue
            
            # If all methods failed, return high confidence for empty result
            if methods_succeeded == 0:
                logger.debug("All OCR methods found no text - returning high confidence for empty result")
                return "", 1.0, "none"
        else:
            # Preprocessing disabled, use original image only
            logger.debug("Preprocessing disabled, using original image")
            try:
                best_text, best_confidence = self._run_ocr(image)
                best_method = "original"
            except Exception as e:
                logger.error(f"OCR failed: {e}")
                return "", 1.0, "none"  # High confidence that there's no text if OCR fails
        
        # Log result
        if best_confidence >= min_confidence:
            logger.info(f"OCR success: '{best_text}' (confidence={best_confidence:.3f}, method={best_method})")
        else:
            logger.warning(f"OCR low confidence: '{best_text}' (confidence={best_confidence:.3f}, method={best_method}, threshold={min_confidence})")
        
        return best_text, best_confidence, best_method
    
    def extract_text_simple(self, image: np.ndarray) -> str:
        """
        Extract text from image (simple interface).
        
        Convenience wrapper that returns only the text string.
        
        Args:
            image: Input image (BGR or grayscale numpy array)
        
        Returns:
            Extracted text string (empty if no text found)
        
        Example:
            text = ocr.extract_text_simple(image)
            if text:
                print(f"Found: {text}")
        """
        text, _, _ = self.extract_text(image)
        return text
    
    def _run_ocr(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Run OCR on a single image.
        
        Internal method that calls EasyOCR and processes results.
        
        Args:
            image: Input image (preprocessed)
        
        Returns:
            Tuple of (text, confidence)
            - text: Extracted text string
            - confidence: Average confidence score (0.0 to 1.0)
        """
        paragraph = self.settings.get("parser.ocr.paragraph")
        
        # Run EasyOCR
        results = self.reader.readtext(image, paragraph=paragraph)
        
        if not results:
            # No text detected - return low confidence for empty result
            logger.debug("No text detected in image - returning low confidence for empty result")
            return "", 0.0
        
        # Extract text and confidence
        if paragraph:
            # Paragraph mode returns single result
            text = results[0][1] if results else ""
            confidence = results[0][2] if results else 0.0
        else:
            # Word mode returns multiple results
            # Concatenate all text and average confidence
            texts = [result[1] for result in results]
            confidences = [result[2] for result in results]
            
            text = " ".join(texts)
            confidence = round(sum(confidences) / len(confidences), 3) if confidences else 0.0
        
        return text.strip(), confidence
    
    def reload_settings(self) -> None:
        """
        Reload settings and reinitialize reader if needed.
        
        Useful if language or GPU settings change at runtime.
        """
        logger.info("Reloading OCR settings")
        
        # Get current settings
        old_languages = self.settings.get("parser.ocr.languages")
        old_gpu = self.settings.get("parser.ocr.gpu")
        
        # Reload settings from file
        self.settings._load_from_file()
        
        # Get new settings
        new_languages = self.settings.get("parser.ocr.languages")
        new_gpu = self.settings.get("parser.ocr.gpu")
        
        # Reinitialize reader if languages or GPU changed
        if new_languages != old_languages or new_gpu != old_gpu:
            logger.info(f"OCR settings changed, reinitializing reader (languages={new_languages}, gpu={new_gpu})")
            try:
                self.reader = easyocr.Reader(new_languages, gpu=new_gpu)
                logger.info("EasyOCR reader reinitialized successfully")
            except Exception as e:
                logger.error(f"Failed to reinitialize EasyOCR reader: {e}")
                raise
        
        # Reload preprocessor settings
        self.preprocessor = ImagePreprocessor()
        
        logger.info("OCR settings reloaded successfully")
    
    def get_best_preprocessing_methods(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Get the preprocessing methods with best success rates.
        
        Useful for optimizing preprocessing method selection.
        
        Args:
            top_n: Number of top methods to return
        
        Returns:
            List of (method_name, average_confidence) tuples, sorted by success rate
        
        Example:
            best_methods = ocr.get_best_preprocessing_methods(3)
            for method, avg_conf in best_methods:
                print(f"{method}: {avg_conf:.3f} avg confidence")
        """
        # Calculate average confidence for each method
        method_stats = []
        for method_name in self.method_success_count.keys():
            success_count = self.method_success_count[method_name]
            if success_count > 0:
                avg_confidence = self.method_total_confidence[method_name] / success_count
                method_stats.append((method_name, avg_confidence))
        
        # Sort by average confidence (descending)
        method_stats.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N
        return method_stats[:top_n]
    
    def log_preprocessing_stats(self) -> None:
        """
        Log statistics about preprocessing method performance.
        
        Useful for debugging and optimization.
        """
        logger.info("=== OCR Preprocessing Statistics ===")
        
        best_methods = self.get_best_preprocessing_methods(top_n=10)
        
        if not best_methods:
            logger.info("No preprocessing statistics available yet")
            return
        
        for i, (method_name, avg_confidence) in enumerate(best_methods, 1):
            success_count = self.method_success_count[method_name]
            logger.info(f"{i}. {method_name}: {success_count} successes, {avg_confidence:.3f} avg confidence")
        
        logger.info("====================================")