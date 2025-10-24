#!/usr/bin/env python3
"""
Money parser module for detecting monetary amounts from poker game images.

Uses OCR for text extraction and parses amounts including abbreviations (K, M) and currency symbols.
Supports multiple amounts per image for split pots and handles common OCR errors.

Usage:
    from src.parser.money_parser import MoneyParser
    
    parser = MoneyParser()
    
    # Parse amounts from an image
    results = parser.parse_amounts(image)
    for result in results:
        print(f"Amount: {result.value}, Confidence: {result.confidence}")
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

from src.parser.ocr_engine import OCREngine
from src.config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class AmountResult:
    """Result of money parsing with value and confidence."""
    value: float
    confidence: float


class MoneyParser:
    """
    Money parser for detecting monetary amounts from poker game images.
    
    Uses OCR for text extraction and parses amounts including abbreviations (K, M) 
    and currency symbols. Supports multiple amounts per image for split pots.
    """
    
    def __init__(self):
        """Initialize money parser with OCR engine and settings."""
        self.settings = Settings()
        self.ocr_engine = OCREngine()
        
        # Create money parser settings
        self._create_settings()
        
        # Load settings values
        self._load_settings()
        
        logger.info("MoneyParser initialized")
    
    def _create_settings(self) -> None:
        """Create all money parser settings with defaults."""
        # Basic detection settings
        self.settings.create("parser.money.min_reasonable_amount", default=0.01)
        self.settings.create("parser.money.max_reasonable_amount", default=1000000000.0)  # 1 billion for high-stakes Gold games
        self.settings.create("parser.money.currency_symbols", default=['$', '€', '£', 'G', 'S'])
        self.settings.create("parser.money.abbreviation_multipliers", default={'K': 1000, 'M': 1000000})
        self.settings.create("parser.money.ocr_error_corrections", default={'111K': '1.11K', '1M1K': '1.11K'})
        
        logger.debug("Money parser settings created")
    
    def _load_settings(self) -> None:
        """Load settings values into instance variables."""
        self.min_reasonable_amount = self.settings.get("parser.money.min_reasonable_amount")
        self.max_reasonable_amount = self.settings.get("parser.money.max_reasonable_amount")
        self.currency_symbols = self.settings.get("parser.money.currency_symbols")
        self.abbreviation_multipliers = self.settings.get("parser.money.abbreviation_multipliers")
        self.ocr_error_corrections = self.settings.get("parser.money.ocr_error_corrections")
        
        logger.debug("Money parser settings loaded")
    
    def parse_amounts(self, image) -> List[AmountResult]:
        """
        Parse monetary amounts from an image.
        
        Args:
            image: Image containing monetary amounts to parse
            
        Returns:
            List of AmountResult objects with detected amounts and confidences.
            Returns empty list if no amounts detected.
            
        Example:
            results = parser.parse_amounts(image)
            for result in results:
                print(f"Amount: {result.value}, Confidence: {result.confidence}")
        """
        if image is None or image.size == 0:
            logger.warning("Empty image provided for money parsing")
            return []
        
        try:
            # Extract text from image using OCR engine
            text, ocr_confidence = self._extract_text_from_image(image)
            if not text:
                logger.debug("No text extracted from image")
                return []
            
            # Parse amounts from the extracted text
            amounts = self._parse_amounts_from_text(text, ocr_confidence)
            
            logger.debug(f"Parsed {len(amounts)} amounts from text: '{text}'")
            return amounts
            
        except Exception as e:
            logger.error(f"Error during money parsing: {e}")
            return []
    
    def _extract_text_from_image(self, image) -> Tuple[str, float]:
        """
        Extract text from image using OCR engine.
        
        Args:
            image: Image to extract text from
            
        Returns:
            Tuple of (text, confidence) from OCR engine
        """
        try:
            # Use OCR engine to extract text with confidence
            text, confidence, method = self.ocr_engine.extract_text(image)
            return text.strip(), confidence
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return "", 0.0
    
    def _parse_amounts_from_text(self, text: str, ocr_confidence: float) -> List[AmountResult]:
        """
        Parse multiple amounts from text string.
        
        Args:
            text: Text string containing amounts
            ocr_confidence: Confidence from OCR extraction
            
        Returns:
            List of AmountResult objects
        """
        # Split text by spaces to find multiple amounts
        words = text.split()
        amounts = []
        
        for word in words:
            # Check if word contains digits or abbreviations
            if any(char.isdigit() for char in word) or any(abbr in word.upper() for abbr in ['K', 'M']):
                amount_result = self._parse_single_amount(word, ocr_confidence)
                if amount_result:
                    amounts.append(amount_result)
        
        return amounts
    
    def _parse_single_amount(self, text: str, ocr_confidence: float) -> Optional[AmountResult]:
        """
        Parse a single amount from text string.
        
        Args:
            text: Text string containing a single amount
            ocr_confidence: Confidence from OCR extraction
            
        Returns:
            AmountResult object or None if parsing fails
        """
        if not text:
            return None
        
        original_text = text.strip()
        text = original_text.upper()
        
        # Handle special cases
        if 'ALL-IN' in text or 'ALLIN' in text or 'AII-IN' in text:
            return AmountResult(
                value=0.0,
                confidence=ocr_confidence
            )
        
        if 'DISCONNECTED' in text:
            return AmountResult(
                value=0.0,
                confidence=ocr_confidence
            )
        
        # Fix common OCR errors
        text = self._normalize_amount_text(text)
        
        # Remove currency symbols
        clean_text = self._remove_currency_symbols(text)
        
        # Parse the amount
        try:
            parsed_value = self._parse_abbreviations(clean_text)
            
            # Validate the amount
            if self._validate_amount(parsed_value):
                # Calculate final confidence
                confidence = self._calculate_confidence(ocr_confidence, True)
                
                return AmountResult(
                    value=parsed_value,
                    confidence=confidence
                )
            else:
                logger.debug(f"Amount validation failed for: {original_text}")
                return None
                
        except (ValueError, TypeError) as e:
            logger.debug(f"Error parsing amount '{original_text}': {e}")
            return None
    
    def _normalize_amount_text(self, text: str) -> str:
        """
        Normalize amount text by fixing common OCR errors.
        
        Args:
            text: Raw text from OCR
            
        Returns:
            Normalized text string
        """
        # Apply OCR error corrections from settings
        for error, correction in self.ocr_error_corrections.items():
            text = text.replace(error, correction)
        
        return text
    
    def _remove_currency_symbols(self, text: str) -> str:
        """
        Remove currency symbols from text.
        
        Args:
            text: Text containing currency symbols
            
        Returns:
            Text with currency symbols removed
        """
        clean_text = text
        for symbol in self.currency_symbols:
            clean_text = clean_text.replace(symbol, '')
        
        return clean_text.strip()
    
    def _parse_abbreviations(self, text: str) -> float:
        """
        Parse text with abbreviations (K, M) into numeric value.
        
        Args:
            text: Clean text without currency symbols
            
        Returns:
            Numeric value
        """
        # Remove commas from formatted numbers
        text = text.replace(',', '')
        
        # Handle abbreviations
        for abbr, multiplier in self.abbreviation_multipliers.items():
            if abbr in text:
                # Handle decimal amounts like 1.5K
                num_part = text.replace(abbr, '')
                num = float(num_part)
                return num * multiplier
        
        # Handle regular numbers
        return float(text)
    
    def _validate_amount(self, value: float) -> bool:
        """
        Validate that amount is within reasonable ranges.
        
        Args:
            value: Numeric amount to validate
            
        Returns:
            True if amount is valid, False otherwise
        """
        return self.min_reasonable_amount <= value <= self.max_reasonable_amount
    
    def _calculate_confidence(self, ocr_confidence: float, validation_passed: bool) -> float:
        """
        Calculate final confidence based on OCR confidence and validation.
        
        Args:
            ocr_confidence: Confidence from OCR extraction
            validation_passed: Whether amount validation passed
            
        Returns:
            Final confidence score
        """
        if not validation_passed:
            return 0.0
        
        # Base confidence on OCR confidence
        # Could be enhanced with additional factors in the future
        return ocr_confidence
