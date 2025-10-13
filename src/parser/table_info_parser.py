#!/usr/bin/env python3
"""
Table info parser module for detecting table information from poker game images.

Uses OCR for text extraction and parses room name, table number, currency, and stakes.
Supports flexible regex patterns to handle OCR spacing variations.

Usage:
    from src.parser.table_info_parser import TableInfoParser
    
    parser = TableInfoParser()
    
    # Parse table info from an image
    result = parser.parse_table_info(image)
    if result:
        print(f"Room: {result.room}, Table: {result.table_number}")
        print(f"Stakes: {result.currency} {result.sb}/{result.bb}")
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from src.parser.ocr_engine import OCREngine
from src.config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class TableInfoResult:
    """Result of table info parsing with room, table number, currency, stakes, and confidence."""
    room: str
    table_number: int
    currency: str  # 'G' or 'S'
    sb: float
    bb: float
    confidence: float


class TableInfoParser:
    """
    Table info parser for detecting room, table number, currency, and stakes from poker game images.
    
    Uses OCR for text extraction and flexible regex patterns to parse table information.
    Supports configurable validation ranges and currency symbols.
    """
    
    def __init__(self):
        """Initialize table info parser with OCR engine and settings."""
        self.settings = Settings()
        self.ocr_engine = OCREngine()
        
        # Create table info parser settings
        self._create_settings()
        
        # Load settings values
        self._load_settings()
        
        logger.info("TableInfoParser initialized")
    
    def _create_settings(self) -> None:
        """Create all table info parser settings with defaults."""
        # Basic detection settings
        self.settings.create("parser.table_info.min_stake", default=0.01)
        self.settings.create("parser.table_info.max_stake", default=10000.0)
        self.settings.create("parser.table_info.valid_currencies", default=['G', 'S'])
        
        logger.debug("Table info parser settings created")
    
    def _load_settings(self) -> None:
        """Load settings values into instance variables."""
        self.min_stake = self.settings.get("parser.table_info.min_stake")
        self.max_stake = self.settings.get("parser.table_info.max_stake")
        self.valid_currencies = self.settings.get("parser.table_info.valid_currencies")
        
        logger.debug("Table info parser settings loaded")
    
    def parse_table_info(self, image) -> Optional[TableInfoResult]:
        """
        Parse table information from an image.
        
        Args:
            image: Image containing table information to parse
            
        Returns:
            TableInfoResult with room, table number, currency, stakes, and confidence if detected successfully, None otherwise
            
        Example:
            result = parser.parse_table_info(image)
            if result:
                print(f"Room: {result.room}, Table: {result.table_number}")
                print(f"Stakes: {result.currency} {result.sb}/{result.bb}")
        """
        if image is None or image.size == 0:
            logger.warning("Empty image provided for table info parsing")
            return None
        
        try:
            # Extract text from image using OCR engine
            text, ocr_confidence, method = self._extract_text_from_image(image)
            if not text:
                logger.debug("No text extracted from image")
                return None
            
            # Parse table info from the extracted text
            result = self._parse_table_text(text, ocr_confidence)
            
            if result:
                logger.debug(f"Successfully parsed table info: {result.room} #{result.table_number} {result.currency} {result.sb}/{result.bb} (confidence: {result.confidence:.3f})")
            else:
                logger.debug(f"Failed to parse table info from text: '{text}'")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during table info parsing: {e}")
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
    
    def _parse_table_text(self, text: str, ocr_confidence: float) -> Optional[TableInfoResult]:
        """
        Parse table information from text string.
        
        Args:
            text: Text string containing table information
            ocr_confidence: Confidence from OCR extraction
            
        Returns:
            TableInfoResult object or None if parsing fails
        """
        if not text:
            return None
        
        # Normalize text to handle OCR spacing issues
        normalized_text = self._normalize_text(text)
        
        # Parse using flexible regex pattern
        result = self._parse_with_regex(normalized_text, ocr_confidence)
        
        return result
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text by cleaning up spacing issues from OCR concatenation.
        
        Args:
            text: Raw text from OCR
            
        Returns:
            Normalized text string
        """
        # Remove extra whitespace and normalize spacing
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Fix common OCR spacing issues around special characters
        text = re.sub(r'\s*#\s*', ' #', text)  # Normalize # spacing
        text = re.sub(r'\s*\(\s*', ' (', text)  # Normalize ( spacing
        text = re.sub(r'\s*\)\s*', ') ', text)  # Normalize ) spacing
        text = re.sub(r'\s*/\s*', '/', text)    # Normalize / spacing
        
        return text.strip()
    
    def _parse_with_regex(self, text: str, ocr_confidence: float) -> Optional[TableInfoResult]:
        """
        Parse table info using flexible regex pattern.
        
        Args:
            text: Normalized text string
            ocr_confidence: OCR confidence score
            
        Returns:
            TableInfoResult or None if parsing fails
        """
        # Flexible regex pattern to handle OCR spacing variations
        # Example: "Tennessee #2 Hold'em No Limit Stakes: (G) 100/200"
        pattern = r'(?P<room>[A-Za-z\s]+?)\s*#\s*(?P<table_num>\d+)\s+.*?Stakes:\s*\(\s*(?P<currency>[GS])\s*\)\s*(?P<sb>[\d.]+)\s*/\s*(?P<bb>[\d.]+)'
        
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            logger.debug(f"Regex pattern did not match text: '{text}'")
            return None
        
        try:
            # Extract components
            room = match.group('room').strip()
            table_number = int(match.group('table_num'))
            currency = match.group('currency').upper()
            sb = float(match.group('sb'))
            bb = float(match.group('bb'))
            
            # Validate parsed values
            if not self._validate_stakes(sb, bb, currency, table_number, room):
                return None
            
            # Calculate final confidence
            confidence = self._calculate_confidence(ocr_confidence, True)
            
            return TableInfoResult(
                room=room,
                table_number=table_number,
                currency=currency,
                sb=sb,
                bb=bb,
                confidence=confidence
            )
            
        except (ValueError, TypeError) as e:
            logger.debug(f"Error parsing regex groups from text '{text}': {e}")
            return None
    
    def _validate_stakes(self, sb: float, bb: float, currency: str, table_number: int, room: str) -> bool:
        """
        Validate parsed stakes and other components.
        
        Args:
            sb: Small blind amount
            bb: Big blind amount
            currency: Currency symbol
            table_number: Table number
            room: Room name
            
        Returns:
            True if all validations pass, False otherwise
        """
        # Validate stakes
        if sb <= 0 or bb <= 0:
            logger.debug(f"Invalid stakes: sb={sb}, bb={bb} (must be > 0)")
            return False
        
        if bb <= sb:
            logger.debug(f"Invalid stakes: bb={bb} must be > sb={sb}")
            return False
        
        if sb < self.min_stake or bb > self.max_stake:
            logger.debug(f"Stakes out of range: sb={sb}, bb={bb} (range: {self.min_stake}-{self.max_stake})")
            return False
        
        # Validate currency
        if currency not in self.valid_currencies:
            logger.debug(f"Invalid currency: {currency} (valid: {self.valid_currencies})")
            return False
        
        # Validate table number
        if table_number <= 0:
            logger.debug(f"Invalid table number: {table_number} (must be > 0)")
            return False
        
        # Validate room name
        if not room or not room.strip():
            logger.debug(f"Invalid room name: '{room}' (must not be empty)")
            return False
        
        return True
    
    def _calculate_confidence(self, ocr_confidence: float, validation_passed: bool) -> float:
        """
        Calculate final confidence based on OCR confidence and validation.
        
        Args:
            ocr_confidence: Confidence from OCR extraction
            validation_passed: Whether all validations passed
            
        Returns:
            Final confidence score
        """
        if not validation_passed:
            return 0.0
        
        # Base confidence on OCR confidence
        # Could be enhanced with additional factors in the future
        return ocr_confidence
