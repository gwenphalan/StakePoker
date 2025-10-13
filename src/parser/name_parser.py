#!/usr/bin/env python3
"""
Name parser module for detecting player names from poker game images.

Uses OCR for text extraction and fuzzy matching for hero detection.
Supports configurable hero usernames and validation thresholds.

Usage:
    from src.parser.name_parser import NameParser
    
    parser = NameParser()
    
    # Parse player name from an image
    result = parser.parse_player_name(name_region_image, 1)
    if result:
        print(f"Player 1: {result.name} (hero: {result.is_hero}, conf: {result.confidence})")
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from src.parser.ocr_engine import OCREngine
from src.config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class NameResult:
    """Result of name parsing with name, confidence, and hero status."""
    name: str
    confidence: float
    is_hero: bool
    is_transparent: bool  # Placeholder for future transparency detection


class NameParser:
    """
    Name parser for detecting player names from poker game images.
    
    Uses OCR for text extraction and fuzzy matching for hero detection.
    Supports configurable hero usernames and validation thresholds.
    """
    
    def __init__(self):
        """Initialize name parser with OCR engine and settings."""
        self.settings = Settings()
        self.ocr_engine = OCREngine()
        
        # Create name parser settings
        self._create_settings()
        
        # Load settings values
        self._load_settings()
        
        logger.info("NameParser initialized")
    
    def _create_settings(self) -> None:
        """Create all name parser settings with defaults."""
        # Basic detection settings
        self.settings.create("parser.names.min_confidence", default=0.7)
        self.settings.create("parser.names.hero_usernames", default=["GalacticAce"])
        self.settings.create("parser.names.fuzzy_match_threshold", default=0.8)
        self.settings.create("parser.names.min_name_length", default=2)
        self.settings.create("parser.names.max_name_length", default=20)
        
        logger.debug("Name parser settings created")
    
    def _load_settings(self) -> None:
        """Load settings values into instance variables."""
        self.min_confidence = self.settings.get("parser.names.min_confidence")
        self.hero_usernames = self.settings.get("parser.names.hero_usernames")
        self.fuzzy_match_threshold = self.settings.get("parser.names.fuzzy_match_threshold")
        self.min_name_length = self.settings.get("parser.names.min_name_length")
        self.max_name_length = self.settings.get("parser.names.max_name_length")
        
        logger.debug("Name parser settings loaded")
    
    def parse_player_name(self, name_region_image, player_number: int) -> Optional[NameResult]:
        """
        Parse player name from nameplate region image.
        
        Args:
            name_region_image: BGR image of player nameplate region
            player_number: Player seat number (1-8)
            
        Returns:
            NameResult with name, confidence, hero status, and transparency if detected successfully, None otherwise
            
        Example:
            result = parser.parse_player_name(name_region, 1)
            if result:
                print(f"Player 1: {result.name} (hero: {result.is_hero}, conf: {result.confidence})")
        """
        if name_region_image is None or name_region_image.size == 0:
            logger.warning("Empty name region image provided")
            return None
        
        try:
            # Extract text from image using OCR engine
            text, ocr_confidence, method = self._extract_text_from_image(name_region_image)
            if not text:
                logger.debug("No text extracted from name region image")
                return None
            
            # Parse name from the extracted text
            result = self._parse_name_text(text, ocr_confidence)
            
            if result:
                logger.debug(f"Successfully parsed player {player_number} name: '{result.name}' (hero: {result.is_hero}, confidence: {result.confidence:.3f})")
            else:
                logger.debug(f"Failed to parse player {player_number} name from text: '{text}'")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during player {player_number} name parsing: {e}")
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
    
    def _parse_name_text(self, text: str, ocr_confidence: float) -> Optional[NameResult]:
        """
        Parse player name from text string.
        
        Args:
            text: Text string containing player name
            ocr_confidence: Confidence from OCR extraction
            
        Returns:
            NameResult object or None if parsing fails
        """
        if not text:
            return None
        
        # Normalize text to handle OCR spacing issues
        normalized_text = self._normalize_text(text)
        
        # Validate the name
        if not self._validate_name(normalized_text):
            logger.debug(f"Name validation failed for: '{normalized_text}'")
            return None
        
        # Check if this is a hero name
        is_hero = self._is_hero_name(normalized_text)
        
        # Calculate final confidence
        confidence = self._calculate_confidence(ocr_confidence, True, is_hero)
        
        return NameResult(
            name=normalized_text,
            confidence=confidence,
            is_hero=is_hero,
            is_transparent=False  # Placeholder for future transparency detection
        )
    
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
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _validate_name(self, name: str) -> bool:
        """
        Validate that extracted name is reasonable.
        
        Args:
            name: Name string to validate
            
        Returns:
            True if name is valid, False otherwise
        """
        if not name or not name.strip():
            return False
        
        name = name.strip()
        
        # Check length
        if len(name) < self.min_name_length or len(name) > self.max_name_length:
            logger.debug(f"Name length out of range: {len(name)} (range: {self.min_name_length}-{self.max_name_length})")
            return False
        
        # Check for obviously invalid characters
        # Allow alphanumeric, spaces, underscores, hyphens
        if not re.match(r'^[a-zA-Z0-9\s_-]+$', name):
            logger.debug(f"Name contains invalid characters: '{name}'")
            return False
        
        # Filter out common OCR garbage
        invalid_patterns = ['lll', 'III', '000', '111', '222', '333', '444', '555', '666', '777', '888', '999']
        for pattern in invalid_patterns:
            if pattern in name:
                logger.debug(f"Name contains OCR garbage pattern '{pattern}': '{name}'")
                return False
        
        return True
    
    def _is_hero_name(self, detected_name: str) -> bool:
        """
        Check if detected name matches any configured hero usernames with fuzzy matching.
        
        Args:
            detected_name: Name extracted from OCR
            
        Returns:
            True if name matches a hero username (with fuzzy matching), False otherwise
        """
        if not detected_name or not self.hero_usernames:
            return False
        
        for hero_name in self.hero_usernames:
            # Use simple character-based similarity for now
            # Could be enhanced with Levenshtein distance or difflib
            similarity = self._calculate_name_similarity(detected_name, hero_name)
            if similarity >= self.fuzzy_match_threshold:
                logger.debug(f"Hero match found: '{detected_name}' matches '{hero_name}' (similarity: {similarity:.3f})")
                return True
        
        return False
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """
        Calculate similarity between two names (0.0-1.0).
        
        Simple implementation - could be enhanced with more sophisticated algorithms.
        
        Args:
            name1: First name to compare
            name2: Second name to compare
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not name1 or not name2:
            return 0.0
        
        # Convert to lowercase for comparison
        name1, name2 = name1.lower().strip(), name2.lower().strip()
        
        # Exact match
        if name1 == name2:
            return 1.0
        
        # Character overlap ratio
        common_chars = set(name1) & set(name2)
        total_chars = set(name1) | set(name2)
        
        if not total_chars:
            return 0.0
        
        return len(common_chars) / len(total_chars)
    
    def _calculate_confidence(self, ocr_confidence: float, validation_passed: bool, is_hero: bool) -> float:
        """
        Calculate final confidence based on OCR confidence, validation, and hero status.
        
        Args:
            ocr_confidence: Confidence from OCR extraction
            validation_passed: Whether name validation passed
            is_hero: Whether this is detected as a hero name
            
        Returns:
            Final confidence score
        """
        if not validation_passed:
            return 0.0
        
        # Base confidence on OCR confidence
        confidence = ocr_confidence
        
        # Boost confidence slightly if this is a hero match (indicates good OCR)
        if is_hero:
            confidence = min(confidence * 1.1, 1.0)
        
        return confidence
