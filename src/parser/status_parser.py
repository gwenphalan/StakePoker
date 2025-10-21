#!/usr/bin/env python3
"""
Status parser module for detecting player status text from poker game images.

Uses OCR for text extraction and validates against known valid statuses.
Handles common OCR errors like "aii-in" → "all-in".

Usage:
    from src.parser.status_parser import StatusParser
    
    parser = StatusParser()
    result = parser.parse_status(status_region_image)
    if result:
        print(f"Status: {result.status}, Confidence: {result.confidence}")
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from src.parser.ocr_engine import OCREngine
from src.config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class StatusResult:
    """Result of status parsing with status text and confidence."""
    status: str
    confidence: float


class StatusParser:
    """
    Status parser for detecting player status text from poker game images.
    
    Uses OCR for text extraction and validates against known valid statuses.
    Handles common OCR errors and normalizes text for consistent results.
    """
    
    def __init__(self):
        """Initialize status parser with OCR engine and settings."""
        self.settings = Settings()
        self.ocr_engine = OCREngine()
        
        # Create status parser settings
        self._create_settings()
        
        # Load settings values
        self._load_settings()
        
        logger.info("StatusParser initialized")
    
    def _create_settings(self) -> None:
        """Create all status parser settings with defaults."""
        # Basic detection settings
        self.settings.create("parser.status.min_confidence", default=0.7)
        self.settings.create("parser.status.valid_statuses", default={
            '', 'call', 'raise', 'check', 'bb', 'sb', 'fold', 'bet',
            'all-in', 'away', 'straddle', 'show cards', 'decline straddle'
        })
        self.settings.create("parser.status.ocr_error_corrections", default={
            'aii-in': 'all-in', 
            'ai-in': 'all-in'
        })
        self.settings.create("parser.status.confidence_penalty_invalid", default=0.5)
        
        logger.debug("Status parser settings created")
    
    def _load_settings(self) -> None:
        """Load settings values into instance variables."""
        self.min_confidence = self.settings.get("parser.status.min_confidence")
        self.valid_statuses = self.settings.get("parser.status.valid_statuses")
        self.ocr_error_corrections = self.settings.get("parser.status.ocr_error_corrections")
        self.confidence_penalty_invalid = self.settings.get("parser.status.confidence_penalty_invalid")
        
        logger.debug("Status parser settings loaded")
    
    def parse_status(self, image) -> Optional[StatusResult]:
        """
        Parse player status from status region image.
        
        Args:
            image: BGR image of player status region
            
        Returns:
            StatusResult with status text and confidence if detected successfully, None otherwise
            
        Example:
            result = parser.parse_status(status_region)
            if result:
                print(f"Status: {result.status}, Confidence: {result.confidence}")
        """
        if image is None or image.size == 0:
            logger.warning("Empty image provided for status parsing")
            return None
        
        try:
            # Extract text from image using OCR engine
            text, ocr_confidence, method = self._extract_text_from_image(image)
            if not text:
                logger.debug("No text extracted from status region image")
                return None
            
            # Parse status from the extracted text
            result = self._parse_status_text(text, ocr_confidence)
            
            if result:
                logger.debug(f"Successfully parsed status: '{result.status}' (confidence: {result.confidence:.3f})")
            else:
                logger.debug(f"Failed to parse status from text: '{text}'")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during status parsing: {e}")
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
    
    def _parse_status_text(self, text: str, ocr_confidence: float) -> Optional[StatusResult]:
        """
        Parse player status from text string.
        
        Args:
            text: Text string containing player status
            ocr_confidence: Confidence from OCR extraction
            
        Returns:
            StatusResult object or None if parsing fails
        """
        if not text:
            return None
        
        # Normalize text to handle OCR spacing issues
        normalized_text = self._normalize_text(text)
        
        # Validate the status
        validation_passed = self._validate_status(normalized_text)
        
        # Calculate final confidence
        confidence = self._calculate_confidence(ocr_confidence, validation_passed)
        
        return StatusResult(
            status=normalized_text,
            confidence=confidence
        )
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text by cleaning up spacing issues from OCR.
        
        Args:
            text: Raw text from OCR
            
        Returns:
            Normalized text string
        """
        # Convert to lowercase and strip
        text = text.lower().strip()
        
        # Apply OCR error corrections from settings
        for error, correction in self.ocr_error_corrections.items():
            if error in text:
                text = correction
        
        # Remove extra whitespace and normalize spacing
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _validate_status(self, status: str) -> bool:
        """
        Validate that extracted status is in the valid statuses set.
        
        Args:
            status: Status string to validate
            
        Returns:
            True if status is valid, False otherwise
        """
        return status in self.valid_statuses
    
    def _calculate_confidence(self, ocr_confidence: float, validation_passed: bool) -> float:
        """
        Calculate final confidence based on OCR confidence and validation.
        
        Args:
            ocr_confidence: Confidence from OCR extraction
            validation_passed: Whether status validation passed
            
        Returns:
            Final confidence score
        """
        if not validation_passed:
            # Apply penalty for invalid status but still return result
            return ocr_confidence * self.confidence_penalty_invalid
        
        # Base confidence on OCR confidence
        return ocr_confidence
