#!/usr/bin/env python3
"""
Tests for screen capture module.

Tests screen capture functionality, error handling, and image format conversion.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.capture.screen_capture import ScreenCapture
from src.capture.monitor_config import MonitorInfo


class TestScreenCapture:
    """Test ScreenCapture class."""
    
    @pytest.fixture
    def mock_monitor_config(self):
        """Mock MonitorConfig."""
        mock_config = Mock()
        mock_config.get_monitor_bounds.return_value = {
            'left': 0,
            'top': 0,
            'width': 1920,
            'height': 1080
        }
        return mock_config
    
    @pytest.fixture
    def mock_mss(self):
        """Mock mss.mss() instance."""
        mock_sct = Mock()
        # Create a mock image array (BGR format)
        mock_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_sct.grab.return_value = mock_image
        return mock_sct
    
    @patch('src.capture.screen_capture.mss.mss')
    @patch('src.capture.screen_capture.MonitorConfig')
    def test_screen_capture_initialization(self, mock_monitor_config_class, 
                                        mock_mss_class, mock_monitor_config):
        """Test ScreenCapture initialization."""
        mock_monitor_config_class.return_value = mock_monitor_config
        mock_mss_class.return_value = Mock()
        
        capture = ScreenCapture()
        
        assert capture.monitor_config == mock_monitor_config
        assert capture.monitor_bounds == mock_monitor_config.get_monitor_bounds.return_value
        mock_monitor_config.get_monitor_bounds.assert_called_once()
    
    @patch('src.capture.screen_capture.mss.mss')
    @patch('src.capture.screen_capture.MonitorConfig')
    def test_capture_frame_success(self, mock_monitor_config_class, 
                                 mock_mss_class, mock_monitor_config, mock_mss):
        """Test successful frame capture."""
        mock_monitor_config_class.return_value = mock_monitor_config
        mock_mss_class.return_value = mock_mss
        
        capture = ScreenCapture()
        frame = capture.capture_frame()
        
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (1080, 1920, 3)
        assert frame.dtype == np.uint8
        mock_mss.grab.assert_called_once_with(mock_monitor_config.get_monitor_bounds.return_value)
    
    @patch('src.capture.screen_capture.mss.mss')
    @patch('src.capture.screen_capture.MonitorConfig')
    def test_capture_frame_bgra_to_bgr_conversion(self, mock_monitor_config_class, 
                                                mock_mss_class, mock_monitor_config):
        """Test BGRA to BGR conversion."""
        mock_monitor_config_class.return_value = mock_monitor_config
        mock_sct = Mock()
        # Create BGRA image (4 channels)
        mock_image = np.zeros((1080, 1920, 4), dtype=np.uint8)
        mock_sct.grab.return_value = mock_image
        mock_mss_class.return_value = mock_sct
        
        capture = ScreenCapture()
        frame = capture.capture_frame()
        
        assert frame.shape == (1080, 1920, 3)  # Should be BGR (3 channels)
        assert frame.dtype == np.uint8
    
    @patch('src.capture.screen_capture.mss.mss')
    @patch('src.capture.screen_capture.MonitorConfig')
    def test_capture_frame_error_handling(self, mock_monitor_config_class, 
                                        mock_mss_class, mock_monitor_config):
        """Test error handling during frame capture."""
        mock_monitor_config_class.return_value = mock_monitor_config
        mock_sct = Mock()
        mock_sct.grab.side_effect = Exception("Capture failed")
        mock_mss_class.return_value = mock_sct
        
        capture = ScreenCapture()
        
        with pytest.raises(RuntimeError, match="Screen capture failed"):
            capture.capture_frame()
    
    @patch('src.capture.screen_capture.mss.mss')
    @patch('src.capture.screen_capture.MonitorConfig')
    def test_close_method(self, mock_monitor_config_class, mock_mss_class, 
                         mock_monitor_config):
        """Test close method."""
        mock_monitor_config_class.return_value = mock_monitor_config
        mock_sct = Mock()
        mock_mss_class.return_value = mock_sct
        
        capture = ScreenCapture()
        capture.close()
        
        mock_sct.close.assert_called_once()
    
    @patch('src.capture.screen_capture.mss.mss')
    @patch('src.capture.screen_capture.MonitorConfig')
    def test_context_manager(self, mock_monitor_config_class, mock_mss_class, 
                           mock_monitor_config):
        """Test ScreenCapture as context manager."""
        mock_monitor_config_class.return_value = mock_monitor_config
        mock_sct = Mock()
        mock_mss_class.return_value = mock_sct
        
        with ScreenCapture() as capture:
            assert capture.monitor_config == mock_monitor_config
        
        # Should call close on exit
        mock_sct.close.assert_called_once()
    
    @patch('src.capture.screen_capture.mss.mss')
    @patch('src.capture.screen_capture.MonitorConfig')
    def test_different_monitor_bounds(self, mock_monitor_config_class, 
                                    mock_mss_class):
        """Test capture with different monitor bounds."""
        # Mock monitor config with different bounds
        mock_config = Mock()
        mock_config.get_monitor_bounds.return_value = {
            'left': 1920,
            'top': 0,
            'width': 2560,
            'height': 1440
        }
        mock_monitor_config_class.return_value = mock_config
        
        mock_sct = Mock()
        mock_image = np.zeros((1440, 2560, 3), dtype=np.uint8)
        mock_sct.grab.return_value = mock_image
        mock_mss_class.return_value = mock_sct
        
        capture = ScreenCapture()
        frame = capture.capture_frame()
        
        assert frame.shape == (1440, 2560, 3)
        mock_sct.grab.assert_called_once_with(mock_config.get_monitor_bounds.return_value)
    
    @patch('src.capture.screen_capture.mss.mss')
    @patch('src.capture.screen_capture.MonitorConfig')
    def test_capture_frame_logging(self, mock_monitor_config_class, 
                                 mock_mss_class, mock_monitor_config, mock_mss, caplog):
        """Test that capture operations are logged."""
        import logging
        
        # Set logging level to DEBUG to capture debug messages
        caplog.set_level(logging.DEBUG)
        
        mock_monitor_config_class.return_value = mock_monitor_config
        mock_mss_class.return_value = mock_mss
        
        capture = ScreenCapture()
        capture.capture_frame()
        
        # Check that debug logs are generated
        assert "Capturing frame from monitor bounds" in caplog.text
        assert "Frame captured successfully" in caplog.text


@pytest.mark.integration
class TestScreenCaptureIntegration:
    """Integration tests for ScreenCapture."""
    
    @pytest.mark.integration
    @patch('src.capture.screen_capture.mss.mss')
    def test_real_capture_simulation(self, mock_mss_class):
        """Test capture with realistic image data."""
        # Create a more realistic mock image
        mock_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        mock_sct = Mock()
        mock_sct.grab.return_value = mock_image
        mock_mss_class.return_value = mock_sct
        
        # Mock MonitorConfig to avoid actual monitor detection
        with patch('src.capture.screen_capture.MonitorConfig') as mock_config_class:
            mock_config = Mock()
            mock_config.get_monitor_bounds.return_value = {
                'left': 0, 'top': 0, 'width': 1920, 'height': 1080
            }
            mock_config_class.return_value = mock_config
            
            capture = ScreenCapture()
            frame = capture.capture_frame()
            
            # Verify image properties
            assert frame.shape == (1080, 1920, 3)
            assert frame.dtype == np.uint8
            assert np.all(frame >= 0) and np.all(frame <= 255)
    
    @pytest.mark.integration
    def test_capture_performance(self):
        """Test capture performance (timing)."""
        import time
        
        with patch('src.capture.screen_capture.mss.mss') as mock_mss_class:
            mock_sct = Mock()
            mock_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
            mock_sct.grab.return_value = mock_image
            mock_mss_class.return_value = mock_sct
            
            with patch('src.capture.screen_capture.MonitorConfig') as mock_config_class:
                mock_config = Mock()
                mock_config.get_monitor_bounds.return_value = {
                    'left': 0, 'top': 0, 'width': 1920, 'height': 1080
                }
                mock_config_class.return_value = mock_config
                
                capture = ScreenCapture()
                
                # Time multiple captures
                start_time = time.time()
                for _ in range(10):
                    capture.capture_frame()
                end_time = time.time()
                
                avg_time = (end_time - start_time) / 10
                # Should be fast (less than 100ms per capture)
                assert avg_time < 0.1


if __name__ == '__main__':
    pytest.main([__file__])
