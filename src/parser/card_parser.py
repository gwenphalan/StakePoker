#!/usr/bin/env python3
"""
Card parser module for detecting card ranks and suits from poker game images.

Uses OCR for rank detection and HSV color analysis for suit detection.
Supports all standard poker cards (A, K, Q, J, T, 9-2) and suits (hearts, diamonds, clubs, spades).

Usage:
    from src.parser.card_parser import CardParser
    
    parser = CardParser()
    
    # Parse a complete card
    result = parser.parse_card(card_image)
    if result:
        rank, suit = result
        print(f"Detected: {rank}{suit}")
    
    # Parse individual components
    rank = parser.detect_rank(card_image)
    suit = parser.detect_suit(card_image)
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

from src.parser.ocr_engine import OCREngine
from src.config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class CardResult:
    """Result of card parsing with rank, suit, and confidence."""
    rank: Optional[str]
    suit: Optional[str]
    confidence: float


class CardParser:
    """
    Card parser for detecting ranks and suits from poker card images.
    
    Uses OCR for rank detection and HSV color analysis for suit detection.
    Supports configurable color ranges and detection thresholds.
    """
    
    # Valid card ranks
    VALID_RANKS = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
    
    # Valid suit names
    SUIT_NAMES = ['hearts', 'diamonds', 'clubs', 'spades']
    
    def __init__(self):
        """Initialize card parser with OCR engine and color settings."""
        self.settings = Settings()
        self.ocr_engine = OCREngine()
        
        # Create card parser settings
        self._create_settings()
        
        # Load HSV ranges from settings
        self._load_color_ranges()
        
        logger.info("CardParser initialized")
    
    def _create_settings(self) -> None:
        """Create all card parser settings with defaults."""
        # Basic detection settings
        self.settings.create("parser.cards.min_pixel_threshold", default=10)
        self.settings.create("parser.cards.suit_detection_enabled", default=True)
        self.settings.create("parser.cards.rank_detection_enabled", default=True)
        
        # HSV ranges for each suit (based on archive analysis)
        self.settings.create("parser.cards.hearts_hsv_ranges", default=[
            [[0, 100, 100], [10, 255, 255]],      # Red range 1
            [[160, 100, 100], [180, 255, 255]]    # Red range 2 (wraps around)
        ])
        self.settings.create("parser.cards.diamonds_hsv_range", default=[[100, 100, 100], [130, 255, 255]])
        self.settings.create("parser.cards.clubs_hsv_range", default=[[40, 100, 100], [80, 255, 255]])
        self.settings.create("parser.cards.spades_hsv_range", default=[[0, 0, 0], [180, 255, 50]])
        
        logger.debug("Card parser settings created")
    
    def _load_color_ranges(self) -> None:
        """Load HSV color ranges from settings."""
        self.hearts_ranges = self.settings.get("parser.cards.hearts_hsv_ranges")
        self.diamonds_range = self.settings.get("parser.cards.diamonds_hsv_range")
        self.clubs_range = self.settings.get("parser.cards.clubs_hsv_range")
        self.spades_range = self.settings.get("parser.cards.spades_hsv_range")
        
        logger.debug("Color ranges loaded from settings")
    
    def parse_card(self, card_image: np.ndarray) -> Optional[CardResult]:
        """
        Parse a complete card image to extract rank and suit.
        
        Args:
            card_image: BGR image of a single card
            
        Returns:
            CardResult with rank, suit, and confidence if both detected successfully, None otherwise
            
        Example:
            result = parser.parse_card(card_image)
            if result:
                print(f"Card: {result.rank}{result.suit}, Confidence: {result.confidence}")
        """
        if card_image is None or card_image.size == 0:
            logger.debug("Empty card image provided - returning high confidence for no card")
            return CardResult(rank=None, suit=None, confidence=1.0)
        
        # Detect rank and suit
        rank = self.detect_rank(card_image) if self.settings.get("parser.cards.rank_detection_enabled") else None
        suit = self.detect_suit(card_image) if self.settings.get("parser.cards.suit_detection_enabled") else None
        
        # Validate both components
        if rank is None or suit is None:
            logger.debug(f"No card detected - rank: {rank}, suit: {suit} - returning high confidence for no card")
            return CardResult(rank=None, suit=None, confidence=1.0)
        
        # Calculate confidence
        confidence = self._calculate_card_confidence(rank, suit, card_image)
        
        logger.debug(f"Successfully parsed card: {rank}{suit} (confidence: {confidence:.3f})")
        return CardResult(rank=rank, suit=suit, confidence=confidence)
    
    def detect_suit(self, card_image: np.ndarray) -> Optional[str]:
        """
        Detect card suit using HSV color analysis.
        
        Args:
            card_image: BGR image of a single card
            
        Returns:
            Suit name ('hearts', 'diamonds', 'clubs', 'spades') or None if detection fails
        """
        if card_image is None or card_image.size == 0:
            logger.warning("Empty card image provided for suit detection")
            return None
        
        # Convert BGR to HSV for better color detection
        hsv_image = cv2.cvtColor(card_image, cv2.COLOR_BGR2HSV)
        
        # Count pixels for each suit
        suit_pixels = {}
        
        # Hearts - red color (check both ranges since red wraps around)
        hearts_pixels = 0
        for range_pair in self.hearts_ranges:
            mask = cv2.inRange(hsv_image, np.array(range_pair[0]), np.array(range_pair[1]))
            hearts_pixels += cv2.countNonZero(mask)
        suit_pixels['hearts'] = hearts_pixels
        
        # Diamonds - blue
        diamonds_mask = cv2.inRange(hsv_image, np.array(self.diamonds_range[0]), np.array(self.diamonds_range[1]))
        suit_pixels['diamonds'] = cv2.countNonZero(diamonds_mask)
        
        # Clubs - green
        clubs_mask = cv2.inRange(hsv_image, np.array(self.clubs_range[0]), np.array(self.clubs_range[1]))
        suit_pixels['clubs'] = cv2.countNonZero(clubs_mask)
        
        # Spades - black/dark
        spades_mask = cv2.inRange(hsv_image, np.array(self.spades_range[0]), np.array(self.spades_range[1]))
        suit_pixels['spades'] = cv2.countNonZero(spades_mask)
        
        # Find suit with most pixels
        min_threshold = self.settings.get("parser.cards.min_pixel_threshold")
        best_suit = None
        max_pixels = 0
        
        for suit_name, pixel_count in suit_pixels.items():
            if pixel_count > max_pixels and pixel_count >= min_threshold:
                max_pixels = pixel_count
                best_suit = suit_name
        
        # Log pixel counts for debugging
        logger.debug(f"Suit pixel counts: {suit_pixels}, best: {best_suit} ({max_pixels} pixels)")
        
        return best_suit
    
    def detect_rank(self, card_image: np.ndarray) -> Optional[str]:
        """
        Detect card rank using OCR.
        
        Args:
            card_image: BGR image of a single card
            
        Returns:
            Rank string (A, K, Q, J, T, 9-2) or None if detection fails
        """
        if card_image is None or card_image.size == 0:
            logger.warning("Empty card image provided for rank detection")
            return None
        
        # Use OCR engine to extract text with confidence
        try:
            text, confidence, method = self.ocr_engine.extract_text(card_image)
            if not text:
                logger.debug("No text extracted from card image")
                return None
            
            # Parse and validate rank
            rank = self._normalize_rank(text)
            if self._validate_rank(rank):
                # Store confidence for later use in confidence calculation
                self._last_rank_confidence = confidence
                logger.debug(f"Detected rank: {rank} from text: '{text}' (confidence: {confidence:.3f})")
                return rank
            else:
                logger.debug(f"Invalid rank detected: '{rank}' from text: '{text}'")
                return None
                
        except Exception as e:
            logger.error(f"Error during rank detection: {e}")
            return None
    
    def _create_suit_mask(self, hsv_image: np.ndarray, suit_name: str) -> np.ndarray:
        """
        Create a mask for a specific suit color.
        
        Args:
            hsv_image: HSV image of the card
            suit_name: Name of the suit ('hearts', 'diamonds', 'clubs', 'spades')
            
        Returns:
            Binary mask for the suit color
        """
        if suit_name == 'hearts':
            # Hearts use multiple ranges
            mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
            for range_pair in self.hearts_ranges:
                range_mask = cv2.inRange(hsv_image, np.array(range_pair[0]), np.array(range_pair[1]))
                mask = cv2.bitwise_or(mask, range_mask)
            return mask
        elif suit_name == 'diamonds':
            return cv2.inRange(hsv_image, np.array(self.diamonds_range[0]), np.array(self.diamonds_range[1]))
        elif suit_name == 'clubs':
            return cv2.inRange(hsv_image, np.array(self.clubs_range[0]), np.array(self.clubs_range[1]))
        elif suit_name == 'spades':
            return cv2.inRange(hsv_image, np.array(self.spades_range[0]), np.array(self.spades_range[1]))
        else:
            logger.warning(f"Unknown suit name: {suit_name}")
            return np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    
    def _validate_rank(self, rank: str) -> bool:
        """
        Check if a rank string is valid.
        
        Args:
            rank: Rank string to validate
            
        Returns:
            True if rank is valid, False otherwise
        """
        return rank.upper() in self.VALID_RANKS
    
    def _normalize_rank(self, rank: str) -> str:
        """
        Normalize rank string (handle "10" -> "T", case conversion).
        
        Args:
            rank: Raw rank string from OCR
            
        Returns:
            Normalized rank string
        """
        if not rank:
            return ""
        
        # Strip whitespace and convert to uppercase
        rank = rank.strip().upper()
        
        # Handle "10" -> "T" conversion
        if rank == "10":
            return "T"
        
        # Handle common OCR mistakes
        replacements = {
            "0": "O",  # Zero might be confused with O
            "1": "I",  # One might be confused with I
        }
        
        for old, new in replacements.items():
            rank = rank.replace(old, new)
        
        return rank
    
    def get_suit_pixel_counts(self, card_image: np.ndarray) -> Dict[str, int]:
        """
        Get pixel counts for all suits (useful for debugging).
        
        Args:
            card_image: BGR image of a single card
            
        Returns:
            Dictionary mapping suit names to pixel counts
        """
        if card_image is None or card_image.size == 0:
            return {}
        
        hsv_image = cv2.cvtColor(card_image, cv2.COLOR_BGR2HSV)
        
        counts = {}
        
        # Hearts
        hearts_pixels = 0
        for range_pair in self.hearts_ranges:
            mask = cv2.inRange(hsv_image, np.array(range_pair[0]), np.array(range_pair[1]))
            hearts_pixels += cv2.countNonZero(mask)
        counts['hearts'] = hearts_pixels
        
        # Other suits
        for suit_name in ['diamonds', 'clubs', 'spades']:
            mask = self._create_suit_mask(hsv_image, suit_name)
            counts[suit_name] = cv2.countNonZero(mask)
        
        return counts
    
    def update_color_ranges(self, suit_name: str, hsv_ranges: List[List[List[int]]]) -> None:
        """
        Update HSV color ranges for a specific suit.
        
        Args:
            suit_name: Name of the suit to update
            hsv_ranges: List of HSV range pairs [[[h1,s1,v1], [h2,s2,v2]], ...]
        """
        if suit_name == 'hearts':
            self.settings.update("parser.cards.hearts_hsv_ranges", hsv_ranges)
            self.hearts_ranges = hsv_ranges
        elif suit_name == 'diamonds':
            self.settings.update("parser.cards.diamonds_hsv_range", hsv_ranges[0])
            self.diamonds_range = hsv_ranges[0]
        elif suit_name == 'clubs':
            self.settings.update("parser.cards.clubs_hsv_range", hsv_ranges[0])
            self.clubs_range = hsv_ranges[0]
        elif suit_name == 'spades':
            self.settings.update("parser.cards.spades_hsv_range", hsv_ranges[0])
            self.spades_range = hsv_ranges[0]
        else:
            logger.warning(f"Unknown suit name for range update: {suit_name}")
            return
        
        logger.info(f"Updated {suit_name} color ranges: {hsv_ranges}")
    
    def _calculate_card_confidence(self, rank: str, suit: str, card_image: np.ndarray) -> float:
        """
        Calculate overall confidence for card detection.
        
        Args:
            rank: Detected rank
            suit: Detected suit
            card_image: Original card image
            
        Returns:
            Combined confidence score (0.0-1.0)
        """
        # Get rank confidence from last detection
        rank_confidence = getattr(self, '_last_rank_confidence', 0.0)
        
        # Calculate suit confidence
        suit_confidence = self._calculate_suit_confidence(card_image, suit)
        
        # Combine confidences (average)
        combined_confidence = (rank_confidence + suit_confidence) / 2.0
        
        return combined_confidence
    
    def _calculate_suit_confidence(self, card_image: np.ndarray, detected_suit: str) -> float:
        """
        Calculate confidence for suit detection based on pixel ratios.
        
        Args:
            card_image: Card image
            detected_suit: Detected suit name
            
        Returns:
            Suit confidence score (0.0-1.0)
        """
        if card_image is None or card_image.size == 0:
            return 0.0
        
        # Get pixel counts for all suits
        suit_counts = self.get_suit_pixel_counts(card_image)
        
        if not suit_counts or detected_suit not in suit_counts:
            return 0.0
        
        # Get pixel count for detected suit
        detected_pixels = suit_counts[detected_suit]
        total_pixels = card_image.size
        
        if total_pixels == 0:
            return 0.0
        
        # Base confidence on pixel ratio
        pixel_ratio = detected_pixels / total_pixels
        base_confidence = min(pixel_ratio * 10, 1.0)  # Scale up small ratios
        
        # Boost confidence if there's clear separation from other suits
        other_pixels = sum(count for suit, count in suit_counts.items() if suit != detected_suit)
        if other_pixels > 0:
            separation_ratio = detected_pixels / other_pixels
            if separation_ratio > 2:  # Clear winner
                base_confidence = min(base_confidence * 1.2, 1.0)
        
        return base_confidence
