#!/usr/bin/env python3
"""
Tests for monitor configuration module.

Tests monitor detection, validation, and configuration management.
"""

import pytest
import mss
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.capture.monitor_config import MonitorConfig, MonitorInfo


class TestMonitorInfo:
    """Test MonitorInfo dataclass."""
    
    @pytest.mark.unit
    def test_monitor_info_creation(self):
        """Test MonitorInfo creation with valid data."""
        monitor = MonitorInfo(
            index=1,
            width=1920,
            height=1080,
            left=0,
            top=0,
            name="Test Monitor"
        )
        
        assert monitor.index == 1
        assert monitor.width == 1920
        assert monitor.height == 1080
        assert monitor.left == 0
        assert monitor.top == 0
        assert monitor.name == "Test Monitor"
    
    @pytest.mark.unit
    def test_monitor_info_properties(self):
        """Test MonitorInfo computed properties."""
        monitor = MonitorInfo(
            index=2,
            width=2560,
            height=1440,
            left=1920,
            top=0,
            name="Secondary Monitor"
        )
        
        assert monitor.dimensions == (2560, 1440)
        assert monitor.position == (1920, 0)
        assert monitor.bounds == {
            'left': 1920,
            'top': 0,
            'width': 2560,
            'height': 1440
        }
    
    @pytest.mark.unit
    def test_monitor_info_string_representation(self):
        """Test MonitorInfo string representation."""
        monitor = MonitorInfo(
            index=1,
            width=1920,
            height=1080,
            left=0,
            top=0,
            name="Primary Monitor"
        )
        
        str_repr = str(monitor)
        assert "Monitor 1" in str_repr
        assert "Primary Monitor" in str_repr
        assert "1920x1080" in str_repr


@pytest.mark.unit
class TestMonitorConfig:
    """Test MonitorConfig class."""
    
    @pytest.fixture
    def mock_mss_monitors(self):
        """Mock mss monitors data."""
        return [
            {'left': 0, 'top': 0, 'width': 1920, 'height': 1080},  # All monitors
            {'left': 0, 'top': 0, 'width': 1920, 'height': 1080},  # Primary
            {'left': 1920, 'top': 0, 'width': 2560, 'height': 1440},  # Secondary
        ]
    
    @pytest.fixture
    def mock_settings(self):
        """Mock Settings class."""
        mock_settings = Mock()
        mock_settings.create = Mock()
        mock_settings.get = Mock(return_value=2)  # Default to monitor 2
        mock_settings.update = Mock()
        return mock_settings
    
    @patch('src.capture.monitor_config.Settings')
    @patch('src.capture.monitor_config.mss.mss')
    def test_monitor_config_initialization(self, mock_mss, mock_settings_class, 
                                         mock_mss_monitors, mock_settings):
        """Test MonitorConfig initialization."""
        mock_settings_class.return_value = mock_settings
        mock_sct = Mock()
        mock_sct.monitors = mock_mss_monitors
        mock_mss.return_value = mock_sct
        
        config = MonitorConfig()
        
        assert config.settings == mock_settings
        assert len(config.available_monitors) == 2  # Excludes "all monitors" entry
        assert config.poker_monitor.index == 2  # Should default to monitor 2
    
    @patch('src.capture.monitor_config.Settings')
    @patch('src.capture.monitor_config.mss.mss')
    def test_monitor_detection(self, mock_mss, mock_settings_class, 
                             mock_mss_monitors, mock_settings):
        """Test monitor detection functionality."""
        mock_settings_class.return_value = mock_settings
        mock_sct = Mock()
        mock_sct.monitors = mock_mss_monitors
        mock_mss.return_value = mock_sct
        
        config = MonitorConfig()
        monitors = config.available_monitors
        
        assert len(monitors) == 2
        assert monitors[0].index == 1
        assert monitors[0].name == "Primary"
        assert monitors[1].index == 2
        assert monitors[1].name == "Poker Monitor"
    
    @patch('src.capture.monitor_config.Settings')
    @patch('src.capture.monitor_config.mss.mss')
    def test_monitor_validation_success(self, mock_mss, mock_settings_class, 
                                      mock_mss_monitors, mock_settings):
        """Test successful monitor validation."""
        mock_settings_class.return_value = mock_settings
        mock_sct = Mock()
        mock_sct.monitors = mock_mss_monitors
        mock_mss.return_value = mock_sct
        
        config = MonitorConfig()
        
        # Should not raise any exception
        config._validate_monitor(config.poker_monitor)
    
    @patch('src.capture.monitor_config.Settings')
    @patch('src.capture.monitor_config.mss.mss')
    def test_monitor_validation_failure(self, mock_mss, mock_settings_class, 
                                      mock_settings):
        """Test monitor validation failure for low resolution."""
        mock_settings_class.return_value = mock_settings
        mock_settings.get.side_effect = lambda key: {
            "capture.monitor.index": 1,  # Use the low-res monitor
            "capture.monitor.auto_validate": True,
            "capture.monitor.fallback_to_primary": False  # Disable fallback
        }.get(key)
        
        mock_sct = Mock()
        mock_sct.monitors = [
            {'left': 0, 'top': 0, 'width': 1920, 'height': 1080},  # All monitors
            {'left': 0, 'top': 0, 'width': 600, 'height': 400},    # Low res monitor (below minimum)
        ]
        mock_mss.return_value = mock_sct
        
        with pytest.raises(ValueError, match="resolution too low"):
            MonitorConfig()
    
    @patch('src.capture.monitor_config.Settings')
    @patch('src.capture.monitor_config.mss.mss')
    def test_fallback_to_primary(self, mock_mss, mock_settings_class, 
                               mock_mss_monitors, mock_settings):
        """Test fallback to primary monitor when configured monitor not found."""
        mock_settings_class.return_value = mock_settings
        mock_settings.get.side_effect = lambda key: {
            "capture.monitor.index": 3,  # Non-existent monitor
            "capture.monitor.auto_validate": True,
            "capture.monitor.fallback_to_primary": True
        }.get(key)
        
        mock_sct = Mock()
        mock_sct.monitors = mock_mss_monitors
        mock_mss.return_value = mock_sct
        
        config = MonitorConfig()
        
        # Should fallback to primary monitor (index 1)
        assert config.poker_monitor.index == 1
        mock_settings.update.assert_called_with("capture.monitor.index", 1)
    
    @patch('src.capture.monitor_config.Settings')
    @patch('src.capture.monitor_config.mss.mss')
    def test_no_fallback_when_disabled(self, mock_mss, mock_settings_class, 
                                     mock_mss_monitors, mock_settings):
        """Test error when fallback is disabled and monitor not found."""
        mock_settings_class.return_value = mock_settings
        mock_settings.get.side_effect = lambda key: {
            "capture.monitor.index": 3,  # Non-existent monitor
            "capture.monitor.auto_validate": True,
            "capture.monitor.fallback_to_primary": False
        }.get(key)
        
        mock_sct = Mock()
        mock_sct.monitors = mock_mss_monitors
        mock_mss.return_value = mock_sct
        
        with pytest.raises(RuntimeError, match="Configured monitor 3 not found"):
            MonitorConfig()
    
    @patch('src.capture.monitor_config.Settings')
    @patch('src.capture.monitor_config.mss.mss')
    def test_set_poker_monitor(self, mock_mss, mock_settings_class, 
                             mock_mss_monitors, mock_settings):
        """Test setting poker monitor to different index."""
        mock_settings_class.return_value = mock_settings
        mock_sct = Mock()
        mock_sct.monitors = mock_mss_monitors
        mock_mss.return_value = mock_sct
        
        config = MonitorConfig()
        
        # Change to monitor 1
        config.set_poker_monitor(1)
        
        assert config.poker_monitor.index == 1
        mock_settings.update.assert_called_with("capture.monitor.index", 1)
    
    @patch('src.capture.monitor_config.Settings')
    @patch('src.capture.monitor_config.mss.mss')
    def test_set_invalid_monitor(self, mock_mss, mock_settings_class, 
                               mock_mss_monitors, mock_settings):
        """Test error when setting invalid monitor index."""
        mock_settings_class.return_value = mock_settings
        mock_sct = Mock()
        mock_sct.monitors = mock_mss_monitors
        mock_mss.return_value = mock_sct
        
        config = MonitorConfig()
        
        with pytest.raises(ValueError, match="Monitor 5 not found"):
            config.set_poker_monitor(5)
    
    @patch('src.capture.monitor_config.Settings')
    @patch('src.capture.monitor_config.mss.mss')
    def test_refresh_monitors(self, mock_mss, mock_settings_class, 
                            mock_mss_monitors, mock_settings):
        """Test monitor refresh functionality."""
        mock_settings_class.return_value = mock_settings
        mock_sct = Mock()
        mock_sct.monitors = mock_mss_monitors
        mock_mss.return_value = mock_sct
        
        config = MonitorConfig()
        original_count = len(config.available_monitors)
        
        # Refresh should work without error
        config.refresh_monitors()
        
        assert len(config.available_monitors) == original_count
    
    @patch('src.capture.monitor_config.Settings')
    @patch('src.capture.monitor_config.mss.mss')
    def test_context_manager(self, mock_mss, mock_settings_class, 
                           mock_mss_monitors, mock_settings):
        """Test MonitorConfig as context manager."""
        mock_settings_class.return_value = mock_settings
        mock_sct = Mock()
        mock_sct.monitors = mock_mss_monitors
        mock_sct.close = Mock()
        mock_mss.return_value = mock_sct
        
        with MonitorConfig() as config:
            assert config.poker_monitor is not None
        
        # Should call close on exit
        mock_sct.close.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__])
