#!/usr/bin/env python3
"""
Unit tests for OCREngine component.

Tests OCR text extraction functionality including:
- Text extraction with preprocessing
- Confidence scoring
- Method selection
- Error handling
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.parser.ocr_engine import OCREngine
from src.parser.image_preprocessor import ImagePreprocessor
from src.config.settings import Settings


@pytest.mark.unit
class TestOCREngine:
    """Test OCREngine class."""
    
    @pytest.fixture
    def mock_easyocr(self):
        """Mock EasyOCR reader."""
        with patch('src.parser.ocr_engine.easyocr.Reader') as mock_reader_class:
            mock_reader = Mock()
            mock_reader_class.return_value = mock_reader
            
            # Mock readtext method
            mock_reader.readtext.return_value = [
                ([], "Test Text", 0.95)  # (bbox, text, confidence)
            ]
            
            yield mock_reader
    
    @pytest.fixture
    def ocr_engine(self, mock_easyocr):
        """Create OCREngine instance for testing."""
        with patch('src.parser.ocr_engine.Settings') as mock_settings_class:
            mock_settings = Mock()
            mock_settings_class.return_value = mock_settings
            
            # Mock settings values
            mock_settings.get.side_effect = lambda key: {
                "parser.ocr.languages": ["en"],
                "parser.ocr.gpu": True,
                "parser.ocr.min_confidence": 0.5,
                "parser.ocr.preprocessing_enabled": True,
                "parser.ocr.paragraph": False
            }.get(key)
            
            engine = OCREngine()
            return engine
    
    def test_ocr_engine_initialization(self, ocr_engine):
        """Test OCREngine initialization."""
        assert ocr_engine is not None
        assert hasattr(ocr_engine, 'reader')
        assert hasattr(ocr_engine, 'preprocessor')
        assert hasattr(ocr_engine, 'settings')
    
    def test_extract_text_empty_image(self, ocr_engine):
        """Test text extraction with empty image."""
        empty_image = np.array([])
        text, confidence, method = ocr_engine.extract_text(empty_image)
        
        assert text == ""
        assert confidence == 0.0
        assert method == "none"
    
    def test_extract_text_none_image(self, ocr_engine):
        """Test text extraction with None image."""
        text, confidence, method = ocr_engine.extract_text(None)
        
        assert text == ""
        assert confidence == 0.0
        assert method == "none"
    
    def test_extract_text_success(self, ocr_engine, mock_easyocr):
        """Test successful text extraction."""
        test_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        
        text, confidence, method = ocr_engine.extract_text(test_image)
        
        assert text == "Test Text"
        assert confidence == 0.95
        assert method in ["original", "grayscale", "threshold", "adaptive_threshold", "otsu_threshold", "denoise", "contrast", "sharpen"]
    
    def test_extract_text_no_results(self, ocr_engine, mock_easyocr):
        """Test text extraction when OCR returns no results."""
        # Mock OCR to return empty results
        mock_easyocr.readtext.return_value = []
        
        test_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        
        text, confidence, method = ocr_engine.extract_text(test_image)
        
        assert text == ""
        assert confidence == 0.0
        assert method != "none"
    
    def test_extract_text_paragraph_mode(self, ocr_engine, mock_easyocr):
        """Test text extraction in paragraph mode."""
        # Mock paragraph mode settings
        ocr_engine.settings.get.side_effect = lambda key: {
            "parser.ocr.languages": ["en"],
            "parser.ocr.gpu": True,
            "parser.ocr.min_confidence": 0.5,
            "parser.ocr.preprocessing_enabled": True,
            "parser.ocr.paragraph": True
        }.get(key)
        
        test_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        
        text, confidence, method = ocr_engine.extract_text(test_image)
        
        assert text == "Test Text"
        assert confidence == 0.95
        assert method != "none"
    
    def test_extract_text_word_mode(self, ocr_engine, mock_easyocr):
        """Test text extraction in word mode with multiple results."""
        # Mock OCR to return multiple word results
        mock_easyocr.readtext.return_value = [
            ([], "Hello", 0.9),
            ([], "World", 0.85)
        ]
        
        test_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        
        text, confidence, method = ocr_engine.extract_text(test_image)
        
        assert text == "Hello World"
        assert confidence == 0.875  # Average of 0.9 and 0.85
        assert method != "none"
    
    def test_extract_text_preprocessing_disabled(self, ocr_engine, mock_easyocr):
        """Test text extraction with preprocessing disabled."""
        # Mock preprocessing disabled
        ocr_engine.settings.get.side_effect = lambda key: {
            "parser.ocr.languages": ["en"],
            "parser.ocr.gpu": True,
            "parser.ocr.min_confidence": 0.5,
            "parser.ocr.preprocessing_enabled": False,
            "parser.ocr.paragraph": False
        }.get(key)
        
        test_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        
        text, confidence, method = ocr_engine.extract_text(test_image)
        
        assert text == "Test Text"
        assert confidence == 0.95
        assert method == "original"
    
    def test_extract_text_low_confidence(self, ocr_engine, mock_easyocr):
        """Test text extraction with low confidence results."""
        # Mock OCR to return low confidence
        mock_easyocr.readtext.return_value = [
            ([], "Low Confidence Text", 0.3)
        ]
        
        test_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        
        text, confidence, method = ocr_engine.extract_text(test_image)
        
        assert text == "Low Confidence Text"
        assert confidence == 0.3
        assert method != "none"
    
    def test_extract_text_simple(self, ocr_engine, mock_easyocr):
        """Test simple text extraction interface."""
        test_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        
        text = ocr_engine.extract_text_simple(test_image)
        
        assert text == "Test Text"
    
    def test_extract_text_exception_handling(self, ocr_engine, mock_easyocr):
        """Test exception handling during text extraction."""
        # Mock OCR to raise exception
        mock_easyocr.readtext.side_effect = Exception("OCR Error")
        
        test_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        
        text, confidence, method = ocr_engine.extract_text(test_image)
        
        assert text == ""
        assert confidence == 0.0
        assert method == "none"
    
    def test_run_ocr_word_mode(self, ocr_engine, mock_easyocr):
        """Test _run_ocr method in word mode."""
        # Mock multiple word results
        mock_easyocr.readtext.return_value = [
            ([], "First", 0.9),
            ([], "Second", 0.8),
            ([], "Third", 0.7)
        ]
        
        test_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        
        text, confidence = ocr_engine._run_ocr(test_image)
        
        assert text == "First Second Third"
        assert confidence == 0.8  # Average of 0.9, 0.8, 0.7
    
    def test_run_ocr_paragraph_mode(self, ocr_engine, mock_easyocr):
        """Test _run_ocr method in paragraph mode."""
        # Mock paragraph mode
        ocr_engine.settings.get.side_effect = lambda key: {
            "parser.ocr.paragraph": True
        }.get(key)
        
        test_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        
        text, confidence = ocr_engine._run_ocr(test_image)
        
        assert text == "Test Text"
        assert confidence == 0.95
    
    def test_run_ocr_empty_results(self, ocr_engine, mock_easyocr):
        """Test _run_ocr method with empty results."""
        mock_easyocr.readtext.return_value = []
        
        test_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        
        text, confidence = ocr_engine._run_ocr(test_image)
        
        assert text == ""
        assert confidence == 0.0
    
    def test_method_success_tracking(self, ocr_engine, mock_easyocr):
        """Test tracking of preprocessing method success."""
        test_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        
        # Run extraction multiple times
        for _ in range(3):
            ocr_engine.extract_text(test_image)
        
        # Check that method statistics are being tracked
        assert len(ocr_engine.method_success_count) > 0
        assert len(ocr_engine.method_total_confidence) > 0
    
    def test_get_best_preprocessing_methods(self, ocr_engine):
        """Test getting best preprocessing methods."""
        # Manually set some method statistics
        ocr_engine.method_success_count = {
            "grayscale": 5,
            "threshold": 3,
            "original": 2
        }
        ocr_engine.method_total_confidence = {
            "grayscale": 4.5,  # avg 0.9
            "threshold": 2.4,  # avg 0.8
            "original": 1.4    # avg 0.7
        }
        
        best_methods = ocr_engine.get_best_preprocessing_methods(3)
        
        assert len(best_methods) == 3
        assert best_methods[0][0] == "grayscale"  # Highest average confidence
        assert best_methods[0][1] == 0.9
        assert best_methods[1][0] == "threshold"
        assert best_methods[2][0] == "original"
    
    def test_get_best_preprocessing_methods_empty(self, ocr_engine):
        """Test getting best preprocessing methods when no data available."""
        best_methods = ocr_engine.get_best_preprocessing_methods(5)
        
        assert len(best_methods) == 0
    
    def test_log_preprocessing_stats(self, ocr_engine, caplog):
        """Test logging preprocessing statistics."""
        # Set up some method statistics
        ocr_engine.method_success_count = {
            "grayscale": 10,
            "threshold": 5
        }
        ocr_engine.method_total_confidence = {
            "grayscale": 9.0,  # avg 0.9
            "threshold": 4.0   # avg 0.8
        }
        
        ocr_engine.log_preprocessing_stats()
        
        # Check that stats were logged
        assert "OCR Preprocessing Statistics" in caplog.text
        assert "grayscale" in caplog.text
        assert "threshold" in caplog.text
    
    def test_log_preprocessing_stats_empty(self, ocr_engine, caplog):
        """Test logging preprocessing statistics when no data available."""
        ocr_engine.log_preprocessing_stats()
        
        assert "No preprocessing statistics available yet" in caplog.text
    
    @patch('src.parser.ocr_engine.easyocr.Reader')
    def test_reload_settings(self, mock_reader_class):
        """Test reloading OCR settings."""
        # Create initial engine
        mock_reader = Mock()
        mock_reader_class.return_value = mock_reader
        
        with patch('src.parser.ocr_engine.Settings') as mock_settings_class:
            mock_settings = Mock()
            mock_settings_class.return_value = mock_settings
            
            # Mock initial settings
            initial_settings = {
                "parser.ocr.languages": ["en"],
                "parser.ocr.gpu": True,
                "parser.ocr.min_confidence": 0.5,
                "parser.ocr.preprocessing_enabled": True,
                "parser.ocr.paragraph": False
            }
            
            # Mock reloaded settings
            reloaded_settings = {
                "parser.ocr.languages": ["en", "es"],
                "parser.ocr.gpu": False,
                "parser.ocr.min_confidence": 0.5,
                "parser.ocr.preprocessing_enabled": True,
                "parser.ocr.paragraph": False
            }
            
            # Track whether reload has been called
            reload_called = False
            def mock_get(key):
                nonlocal reload_called
                # After reload_settings() is called, return new values
                if reload_called:
                    return reloaded_settings.get(key)
                return initial_settings.get(key)
            
            mock_settings.get.side_effect = mock_get
            
            # Mock _load_from_file to simulate settings change
            def mock_load_from_file():
                nonlocal reload_called
                reload_called = True  # Simulate that reload happened
            
            mock_settings._load_from_file = mock_load_from_file
            
            engine = OCREngine()
            
            # Should reinitialize reader with new settings
            engine.reload_settings()
            
            # Verify reader was recreated
            assert mock_reader_class.call_count >= 2


@pytest.mark.integration
class TestOCREngineIntegration:
    """Integration tests for OCREngine."""
    
    @pytest.fixture
    def ocr_engine(self):
        """Create OCREngine instance for integration testing."""
        return OCREngine()
    
    def test_ocr_with_real_image_shapes(self, ocr_engine):
        """Test OCR with various real image shapes."""
        # Test different image dimensions
        test_shapes = [
            (100, 200, 3),  # Standard
            (50, 100, 3),   # Small
            (200, 400, 3),  # Large
            (100, 200, 1),  # Grayscale
        ]
        
        for height, width, channels in test_shapes:
            test_image = np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)
            
            # Should not crash with different shapes
            text, confidence, method = ocr_engine.extract_text(test_image)
            
            # Results can vary, but should be valid
            assert isinstance(text, str)
            assert 0.0 <= confidence <= 1.0
            assert isinstance(method, str)
    
    def test_ocr_performance(self, ocr_engine):
        """Test OCR performance with multiple images."""
        import time
        
        # Create test images
        test_images = [
            np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        
        start_time = time.time()
        
        results = []
        for image in test_images:
            text, confidence, method = ocr_engine.extract_text(image)
            results.append((text, confidence, method))
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process 5 images in reasonable time
        assert processing_time < 10.0  # 10 seconds for 5 images
        
        # All results should be valid
        for text, confidence, method in results:
            assert isinstance(text, str)
            assert 0.0 <= confidence <= 1.0
            assert isinstance(method, str)
    
    def test_ocr_with_edge_cases(self, ocr_engine):
        """Test OCR with edge case images."""
        # Very small image
        small_image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        text, confidence, method = ocr_engine.extract_text(small_image)
        assert isinstance(text, str)
        
        # Very large image
        large_image = np.random.randint(0, 255, (1000, 2000, 3), dtype=np.uint8)
        text, confidence, method = ocr_engine.extract_text(large_image)
        assert isinstance(text, str)
        
        # All black image
        black_image = np.zeros((100, 200, 3), dtype=np.uint8)
        text, confidence, method = ocr_engine.extract_text(black_image)
        assert isinstance(text, str)
        
        # All white image
        white_image = np.ones((100, 200, 3), dtype=np.uint8) * 255
        text, confidence, method = ocr_engine.extract_text(white_image)
        assert isinstance(text, str)
