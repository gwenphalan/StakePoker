#!/usr/bin/env python3
"""
Hand ID parser module for detecting hand ID numbers from poker game images.

Uses OCR for text extraction and parses hand ID numbers in the format "#[digits]".
Supports flexible regex patterns to handle OCR spacing variations.

Usage:
    from src.parser.hand_id_parser import HandIdParser
    
    parser = HandIdParser()
    
    # Parse hand ID from an image
    result = parser.parse_hand_id(image)
    if result:
        print(f"Hand ID: {result.hand_id}, Confidence: {result.confidence}")
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from src.parser.ocr_engine import OCREngine
from src.config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class HandIdResult:
    """Result of hand ID parsing with hand ID and confidence."""
    hand_id: str
    confidence: float


class HandIdParser:
    """
    Hand ID parser for detecting hand ID numbers from poker game images.
    
    Uses OCR for text extraction and flexible regex patterns to parse hand ID
    numbers in the format "#[digits]". Supports configurable validation ranges.
    """
    
    def __init__(self):
        """Initialize hand ID parser with OCR engine and settings."""
        self.settings = Settings()
        self.ocr_engine = OCREngine()
        
        # Create hand ID parser settings
        self._create_settings()
        
        # Load settings values
        self._load_settings()
        
        logger.info("HandIdParser initialized")
    
    def _create_settings(self) -> None:
        """Create all hand ID parser settings with defaults."""
        # Basic detection settings
        self.settings.create("parser.hand_id.min_length", default=6)
        self.settings.create("parser.hand_id.max_length", default=15)
        
        logger.debug("Hand ID parser settings created")
    
    def _load_settings(self) -> None:
        """Load settings values into instance variables."""
        self.min_length = self.settings.get("parser.hand_id.min_length")
        self.max_length = self.settings.get("parser.hand_id.max_length")
        
        logger.debug("Hand ID parser settings loaded")
    
    def parse_hand_id(self, image) -> Optional[HandIdResult]:
        """
        Parse hand ID from an image.
        
        Args:
            image: Image containing hand ID to parse
            
        Returns:
            HandIdResult with hand ID and confidence if detected successfully, None otherwise
            
        Example:
            result = parser.parse_hand_id(image)
            if result:
                print(f"Hand ID: {result.hand_id}, Confidence: {result.confidence}")
        """
        if image is None or image.size == 0:
            logger.warning("Empty image provided for hand ID parsing")
            return None
        
        try:
            # Extract text from image using OCR engine
            text, ocr_confidence, method = self._extract_text_from_image(image)
            if not text:
                logger.debug("No text extracted from image")
                return None
            
            # Parse hand ID from the extracted text
            result = self._parse_hand_id_text(text, ocr_confidence)
            
            if result:
                logger.debug(f"Successfully parsed hand ID: #{result.hand_id} (confidence: {result.confidence:.3f})")
            else:
                logger.debug(f"Failed to parse hand ID from text: '{text}'")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during hand ID parsing: {e}")
            return None
    
    def _extract_text_from_image(self, image) -> Tuple[str, float, str]:
        """
        Extract text from image using OCR engine.
        
        Args:
            image: Image to extract text from
            
        Returns:
            Tuple of (text, confidence, method) from OCR engine
        """
        try:
            # Use OCR engine to extract text with confidence and method
            text, confidence, method = self.ocr_engine.extract_text(image)
            return text.strip(), confidence, method
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return "", 0.0, "none"
    
    def _parse_hand_id_text(self, text: str, ocr_confidence: float) -> Optional[HandIdResult]:
        """
        Parse hand ID from text string.
        
        Args:
            text: Text string containing hand ID
            ocr_confidence: Confidence from OCR extraction
            
        Returns:
            HandIdResult object or None if parsing fails
        """
        if not text:
            return None
        
        # Normalize text to handle OCR spacing issues
        normalized_text = self._normalize_text(text)
        
        # Parse using regex pattern
        result = self._parse_with_regex(normalized_text, ocr_confidence)
        
        return result
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text by cleaning up spacing issues from OCR.
        
        Args:
            text: Raw text from OCR
            
        Returns:
            Normalized text string
        """
        # Remove extra whitespace and normalize spacing
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Fix common OCR spacing issues around # symbol
        text = re.sub(r'\s*#\s*', '#', text)  # Remove spaces around #
        
        return text.strip()
    
    def _parse_with_regex(self, text: str, ocr_confidence: float) -> Optional[HandIdResult]:
        """
        Parse hand ID using regex pattern.
        
        Args:
            text: Normalized text string
            ocr_confidence: OCR confidence score
            
        Returns:
            HandIdResult or None if parsing fails
        """
        # Regex pattern to find # followed by digits
        # Handles spacing variations: "#12345", "# 12345", etc.
        pattern = r'#\s*(\d+)'
        
        match = re.search(pattern, text)
        if not match:
            logger.debug(f"Regex pattern did not match text: '{text}'")
            return None
        
        try:
            # Extract hand ID digits
            hand_id = match.group(1)
            
            # Validate the hand ID
            if not self._validate_hand_id(hand_id):
                return None
            
            # Calculate final confidence
            confidence = self._calculate_confidence(ocr_confidence, True)
            
            return HandIdResult(
                hand_id=hand_id,
                confidence=confidence
            )
            
        except (ValueError, TypeError) as e:
            logger.debug(f"Error parsing regex groups from text '{text}': {e}")
            return None
    
    def _validate_hand_id(self, hand_id: str) -> bool:
        """
        Validate hand ID format and length.
        
        Args:
            hand_id: Hand ID string to validate
            
        Returns:
            True if hand ID is valid, False otherwise
        """
        if not hand_id:
            logger.debug("Empty hand ID provided")
            return False
        
        # Check if all characters are digits
        if not hand_id.isdigit():
            logger.debug(f"Hand ID contains non-digit characters: '{hand_id}'")
            return False
        
        # Check length constraints
        length = len(hand_id)
        if length < self.min_length:
            logger.debug(f"Hand ID too short: {length} digits (minimum: {self.min_length})")
            return False
        
        if length > self.max_length:
            logger.debug(f"Hand ID too long: {length} digits (maximum: {self.max_length})")
            return False
        
        return True
    
    def _calculate_confidence(self, ocr_confidence: float, validation_passed: bool) -> float:
        """
        Calculate final confidence based on OCR confidence and validation.
        
        Args:
            ocr_confidence: Confidence from OCR extraction
            validation_passed: Whether hand ID validation passed
            
        Returns:
            Final confidence score
        """
        if not validation_passed:
            return 0.0
        
        # Base confidence on OCR confidence
        # Could be enhanced with additional factors in the future
        return ocr_confidence
