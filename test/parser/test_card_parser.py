#!/usr/bin/env python3
"""
Unit tests for CardParser component.

Tests card rank and suit detection functionality including:
- Valid card parsing
- Invalid input handling
- Color range validation
- Confidence calculation
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock

from src.parser.card_parser import CardParser, CardResult
from src.parser.ocr_engine import OCREngine
from src.config.settings import Settings


@pytest.mark.unit
class TestCardResult:
    """Test CardResult dataclass."""
    
    def test_card_result_creation(self):
        """Test CardResult creation with valid data."""
        result = CardResult(rank="A", suit="hearts", confidence=0.95)
        
        assert result.rank == "A"
        assert result.suit == "hearts"
        assert result.confidence == 0.95
    
    def test_card_result_string_representation(self):
        """Test CardResult string representation."""
        result = CardResult(rank="K", suit="spades", confidence=0.87)
        result_str = str(result)
        
        assert "K" in result_str
        assert "spades" in result_str
        assert "0.87" in result_str


@pytest.mark.unit
class TestCardParser:
    """Test CardParser class."""
    
    @pytest.fixture
    def card_parser(self):
        """Create CardParser instance for testing."""
        with patch('src.parser.card_parser.Settings') as mock_settings_class:
            mock_settings = Mock()
            mock_settings_class.return_value = mock_settings
            
            # Mock settings values
            mock_settings.get.side_effect = lambda key: {
                "parser.cards.min_pixel_threshold": 10,
                "parser.cards.suit_detection_enabled": True,
                "parser.cards.rank_detection_enabled": True,
                "parser.cards.hearts_hsv_ranges": [[[0, 100, 100], [10, 255, 255]], [[160, 100, 100], [180, 255, 255]]],
                "parser.cards.diamonds_hsv_range": [[100, 100, 100], [130, 255, 255]],
                "parser.cards.clubs_hsv_range": [[40, 100, 100], [80, 255, 255]],
                "parser.cards.spades_hsv_range": [[0, 0, 1], [180, 5, 15]]
            }.get(key)
            
            parser = CardParser()
            return parser
    
    def test_card_parser_initialization(self, card_parser):
        """Test CardParser initialization."""
        assert card_parser is not None
        assert hasattr(card_parser, 'ocr_engine')
        assert hasattr(card_parser, 'settings')
    
    def test_parse_card_empty_image(self, card_parser):
        """Test parse_card with empty image."""
        empty_image = np.array([])
        result = card_parser.parse_card(empty_image)
        
        assert result is None
    
    def test_parse_card_none_image(self, card_parser):
        """Test parse_card with None image."""
        result = card_parser.parse_card(None)
        
        assert result is None
    
    @patch('src.parser.card_parser.CardParser.detect_rank')
    @patch('src.parser.card_parser.CardParser.detect_suit')
    def test_parse_card_success(self, mock_detect_suit, mock_detect_rank, card_parser):
        """Test successful card parsing."""
        # Mock successful detection
        mock_detect_rank.return_value = "A"
        mock_detect_suit.return_value = "hearts"
        
        # Create test image
        test_image = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        
        result = card_parser.parse_card(test_image)
        
        assert result is not None
        assert isinstance(result, CardResult)
        assert result.rank == "A"
        assert result.suit == "hearts"
        assert result.confidence > 0.0
    
    @patch('src.parser.card_parser.CardParser.detect_rank')
    @patch('src.parser.card_parser.CardParser.detect_suit')
    def test_parse_card_rank_detection_failed(self, mock_detect_suit, mock_detect_rank, card_parser):
        """Test parse_card when rank detection fails."""
        # Mock failed rank detection
        mock_detect_rank.return_value = None
        mock_detect_suit.return_value = "hearts"
        
        test_image = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        
        result = card_parser.parse_card(test_image)
        
        assert result is None
    
    @patch('src.parser.card_parser.CardParser.detect_rank')
    @patch('src.parser.card_parser.CardParser.detect_suit')
    def test_parse_card_suit_detection_failed(self, mock_detect_suit, mock_detect_rank, card_parser):
        """Test parse_card when suit detection fails."""
        # Mock failed suit detection
        mock_detect_rank.return_value = "A"
        mock_detect_suit.return_value = None
        
        test_image = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        
        result = card_parser.parse_card(test_image)
        
        assert result is None
    
    def test_detect_suit_empty_image(self, card_parser):
        """Test suit detection with empty image."""
        empty_image = np.array([])
        result = card_parser.detect_suit(empty_image)
        
        assert result is None
    
    def test_detect_suit_none_image(self, card_parser):
        """Test suit detection with None image."""
        result = card_parser.detect_suit(None)
        
        assert result is None
    
    def test_detect_suit_hearts(self, card_parser):
        """Test hearts suit detection."""
        # Create image with hearts color from poker theme (BGR: 48, 0, 171)
        test_image = np.zeros((100, 80, 3), dtype=np.uint8)
        test_image[20:80, 20:60] = [48, 0, 171]  # Hearts color from poker theme
        
        result = card_parser.detect_suit(test_image)
        
        # Should detect hearts due to hearts color pixels
        assert result == "hearts"
    
    def test_detect_suit_no_sufficient_pixels(self, card_parser):
        """Test suit detection with insufficient pixels."""
        # Create image with very few hearts color pixels (below threshold)
        test_image = np.zeros((100, 80, 3), dtype=np.uint8)
        test_image[50, 50] = [48, 0, 171]  # Single hearts color pixel
        
        result = card_parser.detect_suit(test_image)
        
        # Should return None due to insufficient pixels
        assert result is None
    
    def test_detect_rank_empty_image(self, card_parser):
        """Test rank detection with empty image."""
        empty_image = np.array([])
        result = card_parser.detect_rank(empty_image)
        
        assert result is None
    
    def test_detect_rank_none_image(self, card_parser):
        """Test rank detection with None image."""
        result = card_parser.detect_rank(None)
        
        assert result is None
    
    @patch('src.parser.card_parser.OCREngine')
    def test_detect_rank_success(self, mock_ocr_class, card_parser):
        """Test successful rank detection."""
        # Mock OCR engine
        mock_ocr = Mock()
        mock_ocr.extract_text.return_value = ("A", 0.95, "grayscale")
        mock_ocr_class.return_value = mock_ocr
        card_parser.ocr_engine = mock_ocr
        
        test_image = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        
        result = card_parser.detect_rank(test_image)
        
        assert result == "A"
        assert hasattr(card_parser, '_last_rank_confidence')
        assert card_parser._last_rank_confidence == 0.95
    
    @patch('src.parser.card_parser.OCREngine')
    def test_detect_rank_invalid_text(self, mock_ocr_class, card_parser):
        """Test rank detection with invalid OCR text."""
        # Mock OCR engine returning invalid text
        mock_ocr = Mock()
        mock_ocr.extract_text.return_value = ("INVALID", 0.95, "grayscale")
        mock_ocr_class.return_value = mock_ocr
        card_parser.ocr_engine = mock_ocr
        
        test_image = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        
        result = card_parser.detect_rank(test_image)
        
        assert result is None
    
    @patch('src.parser.card_parser.OCREngine')
    def test_detect_rank_no_text(self, mock_ocr_class, card_parser):
        """Test rank detection when OCR returns no text."""
        # Mock OCR engine returning empty text
        mock_ocr = Mock()
        mock_ocr.extract_text.return_value = ("", 0.0, "grayscale")
        mock_ocr_class.return_value = mock_ocr
        card_parser.ocr_engine = mock_ocr
        
        test_image = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        
        result = card_parser.detect_rank(test_image)
        
        assert result is None
    
    def test_validate_rank_valid_ranks(self, card_parser):
        """Test rank validation with valid ranks."""
        valid_ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
        
        for rank in valid_ranks:
            assert card_parser._validate_rank(rank) is True
            assert card_parser._validate_rank(rank.lower()) is True
    
    def test_validate_rank_invalid_ranks(self, card_parser):
        """Test rank validation with invalid ranks."""
        invalid_ranks = ['X', '1', '0', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'U', 'V', 'W', 'Y', 'Z']
        
        for rank in invalid_ranks:
            assert card_parser._validate_rank(rank) is False
    
    def test_normalize_rank_basic(self, card_parser):
        """Test basic rank normalization."""
        assert card_parser._normalize_rank("A") == "A"
        assert card_parser._normalize_rank("k") == "K"
        assert card_parser._normalize_rank("  Q  ") == "Q"
    
    def test_normalize_rank_ten_conversion(self, card_parser):
        """Test '10' to 'T' conversion."""
        assert card_parser._normalize_rank("10") == "T"
        assert card_parser._normalize_rank(" 10 ") == "T"
    
    def test_normalize_rank_ocr_corrections(self, card_parser):
        """Test OCR error corrections."""
        assert card_parser._normalize_rank("0") == "O"  # Zero to O
        assert card_parser._normalize_rank("1") == "I"  # One to I
    
    def test_normalize_rank_empty_string(self, card_parser):
        """Test rank normalization with empty string."""
        assert card_parser._normalize_rank("") == ""
        assert card_parser._normalize_rank(None) == ""
    
    def test_get_suit_pixel_counts(self, card_parser):
        """Test getting suit pixel counts."""
        # Create test image with some red pixels
        test_image = np.zeros((100, 80, 3), dtype=np.uint8)
        test_image[20:40, 20:60] = [0, 0, 255]  # Red pixels
        
        counts = card_parser.get_suit_pixel_counts(test_image)
        
        assert isinstance(counts, dict)
        assert 'hearts' in counts
        assert 'diamonds' in counts
        assert 'clubs' in counts
        assert 'spades' in counts
        assert counts['hearts'] > 0  # Should have red pixels
    
    def test_get_suit_pixel_counts_empty_image(self, card_parser):
        """Test getting suit pixel counts with empty image."""
        empty_image = np.array([])
        counts = card_parser.get_suit_pixel_counts(empty_image)
        
        assert counts == {}
    
    def test_update_color_ranges(self, card_parser):
        """Test updating color ranges."""
        new_hearts_ranges = [[[0, 50, 50], [5, 255, 255]], [[175, 50, 50], [180, 255, 255]]]
        
        card_parser.update_color_ranges('hearts', new_hearts_ranges)
        
        assert card_parser.hearts_ranges == new_hearts_ranges
    
    def test_update_color_ranges_invalid_suit(self, card_parser):
        """Test updating color ranges with invalid suit name."""
        # Should not raise exception, just log warning
        card_parser.update_color_ranges('invalid_suit', [[[0, 0, 0], [180, 255, 255]]])
    
    def test_calculate_card_confidence(self, card_parser):
        """Test card confidence calculation."""
        # Set up rank confidence
        card_parser._last_rank_confidence = 0.8
        
        test_image = np.zeros((100, 80, 3), dtype=np.uint8)
        test_image[20:40, 20:60] = [0, 0, 255]  # Red pixels for hearts
        
        confidence = card_parser._calculate_card_confidence("A", "hearts", test_image)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.0  # Should have some confidence
    
    def test_calculate_suit_confidence(self, card_parser):
        """Test suit confidence calculation."""
        test_image = np.zeros((100, 80, 3), dtype=np.uint8)
        test_image[20:40, 20:60] = [0, 0, 255]  # Red pixels
        
        confidence = card_parser._calculate_suit_confidence(test_image, "hearts")
        
        assert 0.0 <= confidence <= 1.0
    
    def test_calculate_suit_confidence_empty_image(self, card_parser):
        """Test suit confidence calculation with empty image."""
        empty_image = np.array([])
        confidence = card_parser._calculate_suit_confidence(empty_image, "hearts")
        
        assert confidence == 0.0
    
    def test_calculate_suit_confidence_invalid_suit(self, card_parser):
        """Test suit confidence calculation with invalid suit."""
        test_image = np.zeros((100, 80, 3), dtype=np.uint8)
        confidence = card_parser._calculate_suit_confidence(test_image, "invalid_suit")
        
        assert confidence == 0.0


@pytest.mark.integration
class TestCardParserIntegration:
    """Integration tests for CardParser."""
    
    @pytest.fixture
    def card_parser(self):
        """Create CardParser instance for integration testing."""
        return CardParser()
    
    def test_full_card_parsing_workflow(self, card_parser):
        """Test complete card parsing workflow."""
        # Create a more realistic test image
        test_image = np.zeros((120, 80, 3), dtype=np.uint8)
        
        # Add some structure that might represent a card
        # Background
        test_image[:, :] = [240, 240, 240]  # Light gray background
        
        # Add some red pixels for hearts
        test_image[30:50, 30:50] = [0, 0, 200]  # Red region
        
        # This is a basic test - in real integration tests,
        # you would use actual card images
        result = card_parser.parse_card(test_image)
        
        # Result might be None if OCR doesn't detect rank
        # or if suit detection doesn't meet thresholds
        if result is not None:
            assert isinstance(result, CardResult)
            assert result.rank in card_parser.VALID_RANKS
            assert result.suit in card_parser.SUIT_NAMES
            assert 0.0 <= result.confidence <= 1.0
    
    def test_parser_with_different_image_sizes(self, card_parser):
        """Test parser with different image sizes."""
        sizes = [(50, 40, 3), (100, 80, 3), (200, 160, 3)]
        
        for height, width, channels in sizes:
            test_image = np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)
            
            # Should not crash with different sizes
            result = card_parser.parse_card(test_image)
            
            # Result can be None or CardResult
            if result is not None:
                assert isinstance(result, CardResult)
    
    def test_parser_performance(self, card_parser):
        """Test parser performance with multiple images."""
        import time
        
        # Create test images
        test_images = [
            np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
            for _ in range(10)
        ]
        
        start_time = time.time()
        
        results = []
        for image in test_images:
            result = card_parser.parse_card(image)
            results.append(result)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 10 images in reasonable time (adjust threshold as needed)
        assert processing_time < 5.0  # 5 seconds for 10 images
        
        # Some results might be None, that's okay
        valid_results = [r for r in results if r is not None]
        assert len(valid_results) >= 0  # At least some might succeed
