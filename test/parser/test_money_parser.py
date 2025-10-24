#!/usr/bin/env python3
"""
Unit tests for MoneyParser component.

Tests monetary amount parsing functionality including:
- Amount extraction and validation
- Currency symbol handling
- Abbreviation parsing (K, M)
- OCR error correction
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.parser.money_parser import MoneyParser, AmountResult
from src.parser.ocr_engine import OCREngine
from src.config.settings import Settings


@pytest.mark.unit
class TestAmountResult:
    """Test AmountResult dataclass."""
    
    def test_amount_result_creation(self):
        """Test AmountResult creation with valid data."""
        result = AmountResult(value=1000.0, confidence=0.95)
        
        assert result.value == 1000.0
        assert result.confidence == 0.95
    
    def test_amount_result_string_representation(self):
        """Test AmountResult string representation."""
        result = AmountResult(value=2500.0, confidence=0.87)
        result_str = str(result)
        
        assert "2500.0" in result_str
        assert "0.87" in result_str


@pytest.mark.unit
class TestMoneyParser:
    """Test MoneyParser class."""
    
    @pytest.fixture
    def money_parser(self):
        """Create MoneyParser instance for testing."""
        with patch('src.parser.money_parser.Settings') as mock_settings_class:
            mock_settings = Mock()
            mock_settings_class.return_value = mock_settings
            
            # Mock settings values
            mock_settings.get.side_effect = lambda key: {
                "parser.money.min_reasonable_amount": 0.01,
                "parser.money.max_reasonable_amount": 1000000000.0,
                "parser.money.currency_symbols": ['$', '€', '£', 'G', 'S'],
                "parser.money.abbreviation_multipliers": {'K': 1000, 'M': 1000000},
                "parser.money.ocr_error_corrections": {'111K': '1.11K', '1M1K': '1.11K'}
            }.get(key)
            
            parser = MoneyParser()
            return parser
    
    def test_money_parser_initialization(self, money_parser):
        """Test MoneyParser initialization."""
        assert money_parser is not None
        assert hasattr(money_parser, 'ocr_engine')
        assert hasattr(money_parser, 'settings')
        assert money_parser.min_reasonable_amount == 0.01
        assert money_parser.max_reasonable_amount == 1000000000.0
    
    def test_parse_amounts_empty_image(self, money_parser):
        """Test parse_amounts with empty image."""
        empty_image = np.array([])
        results = money_parser.parse_amounts(empty_image)
        
        assert results == []
    
    def test_parse_amounts_none_image(self, money_parser):
        """Test parse_amounts with None image."""
        results = money_parser.parse_amounts(None)
        
        assert results == []
    
    @patch('src.parser.money_parser.OCREngine')
    def test_parse_amounts_success(self, mock_ocr_class, money_parser):
        """Test successful amount parsing."""
        # Mock OCR engine
        mock_ocr = Mock()
        mock_ocr.extract_text.return_value = ("$1000", 0.95, "grayscale")
        mock_ocr_class.return_value = mock_ocr
        money_parser.ocr_engine = mock_ocr
        
        test_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        
        results = money_parser.parse_amounts(test_image)
        
        assert len(results) == 1
        assert isinstance(results[0], AmountResult)
        assert results[0].value == 1000.0
        assert results[0].confidence == 0.95
    
    @patch('src.parser.money_parser.OCREngine')
    def test_parse_amounts_multiple_amounts(self, mock_ocr_class, money_parser):
        """Test parsing multiple amounts from text."""
        # Mock OCR engine returning multiple amounts
        mock_ocr = Mock()
        mock_ocr.extract_text.return_value = ("Pot: $1000 Stack: $5000", 0.95, "grayscale")
        mock_ocr_class.return_value = mock_ocr
        money_parser.ocr_engine = mock_ocr
        
        test_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        
        results = money_parser.parse_amounts(test_image)
        
        assert len(results) == 2
        assert results[0].value == 1000.0
        assert results[1].value == 5000.0
    
    @patch('src.parser.money_parser.OCREngine')
    def test_parse_amounts_no_text(self, mock_ocr_class, money_parser):
        """Test parse_amounts when OCR returns no text."""
        # Mock OCR engine returning empty text
        mock_ocr = Mock()
        mock_ocr.extract_text.return_value = ("", 0.0, "grayscale")
        mock_ocr_class.return_value = mock_ocr
        money_parser.ocr_engine = mock_ocr
        
        test_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        
        results = money_parser.parse_amounts(test_image)
        
        assert results == []
    
    def test_parse_single_amount_basic(self, money_parser):
        """Test parsing single amount with basic values."""
        test_cases = [
            ("100", 0.95, 100.0),
            ("$100", 0.95, 100.0),
            ("100.50", 0.95, 100.50),
            ("1,000", 0.95, 1000.0),
        ]
        
        for text, ocr_confidence, expected_value in test_cases:
            result = money_parser._parse_single_amount(text, ocr_confidence)
            
            assert result is not None
            assert isinstance(result, AmountResult)
            assert result.value == expected_value
            assert result.confidence == ocr_confidence
    
    def test_parse_single_amount_abbreviations(self, money_parser):
        """Test parsing single amount with abbreviations."""
        test_cases = [
            ("1K", 0.95, 1000.0),
            ("1.5K", 0.95, 1500.0),
            ("2M", 0.95, 2000000.0),
            ("1.2M", 0.95, 1200000.0),
            ("$5K", 0.95, 5000.0),
            ("G10K", 0.95, 10000.0),
        ]
        
        for text, ocr_confidence, expected_value in test_cases:
            result = money_parser._parse_single_amount(text, ocr_confidence)
            
            assert result is not None
            assert isinstance(result, AmountResult)
            assert result.value == expected_value
    
    def test_parse_single_amount_special_cases(self, money_parser):
        """Test parsing special cases like all-in."""
        test_cases = [
            ("ALL-IN", 0.95, 0.0),
            ("ALLIN", 0.95, 0.0),
            ("AII-IN", 0.95, 0.0),
            ("DISCONNECTED", 0.95, 0.0),
        ]
        
        for text, ocr_confidence, expected_value in test_cases:
            result = money_parser._parse_single_amount(text, ocr_confidence)
            
            assert result is not None
            assert isinstance(result, AmountResult)
            assert result.value == expected_value
    
    def test_parse_single_amount_ocr_corrections(self, money_parser):
        """Test OCR error corrections."""
        test_cases = [
            ("111K", 0.95, 1110.0),  # Should be corrected to 1.11K
            ("1M1K", 0.95, 1110.0),  # Should be corrected to 1.11K
        ]
        
        for text, ocr_confidence, expected_value in test_cases:
            result = money_parser._parse_single_amount(text, ocr_confidence)
            
            assert result is not None
            assert isinstance(result, AmountResult)
            assert result.value == expected_value
    
    def test_parse_single_amount_invalid(self, money_parser):
        """Test parsing invalid amounts."""
        invalid_cases = [
            ("", 0.95),
            ("abc", 0.95),
            ("$", 0.95),
            ("K", 0.95),
            ("M", 0.95),
        ]
        
        for text, ocr_confidence in invalid_cases:
            result = money_parser._parse_single_amount(text, ocr_confidence)
            
            assert result is None
    
    def test_normalize_amount_text(self, money_parser):
        """Test amount text normalization."""
        test_cases = [
            ("111K", "1.11K"),  # OCR correction
            ("1M1K", "1.11K"),  # OCR correction
            ("100", "100"),     # No change
        ]
        
        for input_text, expected_output in test_cases:
            result = money_parser._normalize_amount_text(input_text)
            assert result == expected_output
    
    def test_remove_currency_symbols(self, money_parser):
        """Test currency symbol removal."""
        test_cases = [
            ("$100", "100"),
            ("€50", "50"),
            ("£25", "25"),
            ("G1000", "1000"),
            ("S500", "500"),
            ("$$100", "100"),  # Multiple symbols
        ]
        
        for input_text, expected_output in test_cases:
            result = money_parser._remove_currency_symbols(input_text)
            assert result == expected_output
    
    def test_parse_abbreviations(self, money_parser):
        """Test abbreviation parsing."""
        test_cases = [
            ("1K", 1000.0),
            ("1.5K", 1500.0),
            ("2M", 2000000.0),
            ("1.2M", 1200000.0),
            ("100", 100.0),  # No abbreviation
            ("1,000", 1000.0),  # With comma
        ]
        
        for input_text, expected_value in test_cases:
            result = money_parser._parse_abbreviations(input_text)
            assert result == expected_value
    
    def test_validate_amount_valid(self, money_parser):
        """Test amount validation with valid amounts."""
        valid_amounts = [0.01, 100.0, 1000.0, 1000000.0, 999999999.0]
        
        for amount in valid_amounts:
            assert money_parser._validate_amount(amount) is True
    
    def test_validate_amount_invalid(self, money_parser):
        """Test amount validation with invalid amounts."""
        invalid_amounts = [0.0, -100.0, 0.005, 2000000000.0]
        
        for amount in invalid_amounts:
            assert money_parser._validate_amount(amount) is False
    
    def test_calculate_confidence_valid(self, money_parser):
        """Test confidence calculation with valid amount."""
        confidence = money_parser._calculate_confidence(0.95, True)
        
        assert confidence == 0.95
    
    def test_calculate_confidence_invalid(self, money_parser):
        """Test confidence calculation with invalid amount."""
        confidence = money_parser._calculate_confidence(0.95, False)
        
        assert confidence == 0.0
    
    def test_parse_amounts_from_text(self, money_parser):
        """Test parsing multiple amounts from text."""
        test_text = "Pot: $1000 Stack: $5000 Bet: $500"
        ocr_confidence = 0.95
        
        results = money_parser._parse_amounts_from_text(test_text, ocr_confidence)
        
        assert len(results) == 3
        assert results[0].value == 1000.0
        assert results[1].value == 5000.0
        assert results[2].value == 500.0
    
    def test_parse_amounts_from_text_no_amounts(self, money_parser):
        """Test parsing text with no amounts."""
        test_text = "Player name and status"
        ocr_confidence = 0.95
        
        results = money_parser._parse_amounts_from_text(test_text, ocr_confidence)
        
        assert results == []
    
    def test_parse_amounts_from_text_mixed_content(self, money_parser):
        """Test parsing text with mixed content."""
        test_text = "Player1: $1000 Player2: ALL-IN Player3: $500"
        ocr_confidence = 0.95
        
        results = money_parser._parse_amounts_from_text(test_text, ocr_confidence)
        
        assert len(results) == 3
        assert results[0].value == 1000.0
        assert results[1].value == 0.0  # ALL-IN
        assert results[2].value == 500.0
    
    @patch('src.parser.money_parser.OCREngine')
    def test_extract_text_from_image_error(self, mock_ocr_class, money_parser):
        """Test error handling in text extraction."""
        # Mock OCR engine to raise exception
        mock_ocr = Mock()
        mock_ocr.extract_text.side_effect = Exception("OCR Error")
        mock_ocr_class.return_value = mock_ocr
        money_parser.ocr_engine = mock_ocr
        
        test_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        
        text, confidence = money_parser._extract_text_from_image(test_image)
        
        assert text == ""
        assert confidence == 0.0
    
    def test_parse_single_amount_exception_handling(self, money_parser):
        """Test exception handling in single amount parsing."""
        # Test with text that causes ValueError in float conversion
        result = money_parser._parse_single_amount("invalid_float", 0.95)
        
        assert result is None


@pytest.mark.integration
class TestMoneyParserIntegration:
    """Integration tests for MoneyParser."""
    
    @pytest.fixture
    def money_parser(self):
        """Create MoneyParser instance for integration testing."""
        return MoneyParser()
    
    def test_full_parsing_workflow(self, money_parser):
        """Test complete money parsing workflow."""
        # Create a test image
        test_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        
        # This is a basic test - in real integration tests,
        # you would use actual money region images
        results = money_parser.parse_amounts(test_image)
        
        # Results might be empty if OCR doesn't detect text
        # or if no valid amounts are found
        if results:
            for result in results:
                assert isinstance(result, AmountResult)
                assert result.value >= 0.0
                assert 0.0 <= result.confidence <= 1.0
    
    def test_parser_with_different_image_sizes(self, money_parser):
        """Test parser with different image sizes."""
        sizes = [(50, 100, 3), (100, 200, 3), (200, 400, 3)]
        
        for height, width, channels in sizes:
            test_image = np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)
            
            # Should not crash with different sizes
            results = money_parser.parse_amounts(test_image)
            
            # Results can be empty or contain AmountResult objects
            assert isinstance(results, list)
            for result in results:
                assert isinstance(result, AmountResult)
    
    def test_parser_performance(self, money_parser):
        """Test parser performance with multiple images."""
        import time
        
        # Create test images
        test_images = [
            np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
            for _ in range(10)
        ]
        
        start_time = time.time()
        
        all_results = []
        for image in test_images:
            results = money_parser.parse_amounts(image)
            all_results.extend(results)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 10 images in reasonable time
        assert processing_time < 10.0  # 10 seconds for 10 images
        
        # All results should be valid AmountResult objects
        for result in all_results:
            assert isinstance(result, AmountResult)
            assert result.value >= 0.0
            assert 0.0 <= result.confidence <= 1.0

