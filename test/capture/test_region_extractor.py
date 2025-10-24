#!/usr/bin/env python3
"""
Tests for region extractor module.

Tests region extraction, image cropping, and integration with other capture components.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.capture.region_extractor import RegionExtractor
from src.capture.region_loader import RegionModel


class TestRegionExtractor:
    """Test RegionExtractor class."""
    
    @pytest.fixture
    def mock_screen_capture(self):
        """Mock ScreenCapture instance."""
        mock_capture = Mock()
        # Create a mock frame (1920x1080 BGR image)
        mock_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        mock_capture.capture_frame.return_value = mock_frame
        return mock_capture
    
    @pytest.fixture
    def mock_region_loader(self):
        """Mock RegionLoader instance."""
        mock_loader = Mock()
        mock_regions = {
            "player_1_name": RegionModel(x=100, y=200, width=150, height=30),
            "player_1_stack": RegionModel(x=100, y=250, width=100, height=25),
            "pot": RegionModel(x=400, y=300, width=120, height=40),
            "table_info": RegionModel(x=50, y=50, width=200, height=30)
        }
        mock_loader.load_regions.return_value = mock_regions
        return mock_loader
    
    @patch('src.capture.region_extractor.RegionLoader')
    @patch('src.capture.region_extractor.ScreenCapture')
    def test_region_extractor_initialization(self, mock_screen_capture_class, 
                                           mock_region_loader_class, 
                                           mock_screen_capture, mock_region_loader):
        """Test RegionExtractor initialization."""
        mock_screen_capture_class.return_value = mock_screen_capture
        mock_region_loader_class.return_value = mock_region_loader
        
        extractor = RegionExtractor()
        
        assert extractor.screen_capture == mock_screen_capture
        assert extractor.region_loader == mock_region_loader
        assert len(extractor.regions) == 4
        mock_region_loader.load_regions.assert_called_once()
    
    @patch('src.capture.region_extractor.RegionLoader')
    @patch('src.capture.region_extractor.ScreenCapture')
    def test_extract_all_regions_success(self, mock_screen_capture_class, 
                                       mock_region_loader_class, 
                                       mock_screen_capture, mock_region_loader):
        """Test successful extraction of all regions."""
        mock_screen_capture_class.return_value = mock_screen_capture
        mock_region_loader_class.return_value = mock_region_loader
        
        extractor = RegionExtractor()
        extracted_regions = extractor.extract_all_regions()
        
        # Verify all regions were extracted
        assert len(extracted_regions) == 4
        assert "player_1_name" in extracted_regions
        assert "player_1_stack" in extracted_regions
        assert "pot" in extracted_regions
        assert "table_info" in extracted_regions
        
        # Verify each extracted region is a numpy array
        for region_name, region_image in extracted_regions.items():
            assert isinstance(region_image, np.ndarray)
            assert region_image.dtype == np.uint8
        
        # Verify specific region dimensions
        player_name_region = extracted_regions["player_1_name"]
        assert player_name_region.shape == (30, 150, 3)  # height, width, channels
        
        pot_region = extracted_regions["pot"]
        assert pot_region.shape == (40, 120, 3)
    
    @patch('src.capture.region_extractor.RegionLoader')
    @patch('src.capture.region_extractor.ScreenCapture')
    def test_extract_all_regions_cropping(self, mock_screen_capture_class, 
                                        mock_region_loader_class):
        """Test that regions are correctly cropped from the frame."""
        # Create a frame with known pattern
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Add a distinctive pattern in a specific region
        frame[200:230, 100:250] = [255, 0, 0]  # Red rectangle for player_1_name region
        
        mock_capture = Mock()
        mock_capture.capture_frame.return_value = frame
        mock_screen_capture_class.return_value = mock_capture
        
        mock_loader = Mock()
        mock_regions = {
            "player_1_name": RegionModel(x=100, y=200, width=150, height=30)
        }
        mock_loader.load_regions.return_value = mock_regions
        mock_region_loader_class.return_value = mock_loader
        
        extractor = RegionExtractor()
        extracted_regions = extractor.extract_all_regions()
        
        # Verify the extracted region contains the red pattern
        player_name_region = extracted_regions["player_1_name"]
        assert np.all(player_name_region == [255, 0, 0])
    
    @patch('src.capture.region_extractor.RegionLoader')
    @patch('src.capture.region_extractor.ScreenCapture')
    def test_extract_all_regions_boundary_conditions(self, mock_screen_capture_class, 
                                                   mock_region_loader_class):
        """Test region extraction with boundary conditions."""
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        mock_capture = Mock()
        mock_capture.capture_frame.return_value = frame
        mock_screen_capture_class.return_value = mock_capture
        
        mock_loader = Mock()
        # Test regions at frame boundaries
        mock_regions = {
            "top_left": RegionModel(x=0, y=0, width=100, height=50),
            "top_right": RegionModel(x=1820, y=0, width=100, height=50),
            "bottom_left": RegionModel(x=0, y=1030, width=100, height=50),
            "bottom_right": RegionModel(x=1820, y=1030, width=100, height=50)
        }
        mock_loader.load_regions.return_value = mock_regions
        mock_region_loader_class.return_value = mock_loader
        
        extractor = RegionExtractor()
        extracted_regions = extractor.extract_all_regions()
        
        # Verify all boundary regions were extracted
        assert len(extracted_regions) == 4
        for region_name, region_image in extracted_regions.items():
            assert isinstance(region_image, np.ndarray)
            assert region_image.shape == (50, 100, 3)
    
    @patch('src.capture.region_extractor.RegionLoader')
    @patch('src.capture.region_extractor.ScreenCapture')
    def test_extract_all_regions_edge_case_coordinates(self, mock_screen_capture_class, 
                                                     mock_region_loader_class):
        """Test extraction with edge case coordinates."""
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        mock_capture = Mock()
        mock_capture.capture_frame.return_value = frame
        mock_screen_capture_class.return_value = mock_capture
        
        mock_loader = Mock()
        # Test edge case coordinates (at frame boundaries)
        mock_regions = {
            "edge_case": RegionModel(x=1910, y=1070, width=10, height=10)
        }
        mock_loader.load_regions.return_value = mock_regions
        mock_region_loader_class.return_value = mock_loader
        
        extractor = RegionExtractor()
        extracted_regions = extractor.extract_all_regions()
        
        # Should extract successfully even at boundaries
        assert len(extracted_regions) == 1
        assert "edge_case" in extracted_regions
        assert extracted_regions["edge_case"].shape == (10, 10, 3)
    
    @patch('src.capture.region_extractor.RegionLoader')
    @patch('src.capture.region_extractor.ScreenCapture')
    def test_context_manager(self, mock_screen_capture_class, 
                           mock_region_loader_class, mock_screen_capture, 
                           mock_region_loader):
        """Test RegionExtractor as context manager."""
        mock_screen_capture_class.return_value = mock_screen_capture
        mock_region_loader_class.return_value = mock_region_loader
        mock_screen_capture.close = Mock()
        
        with RegionExtractor() as extractor:
            assert extractor.screen_capture == mock_screen_capture
        
        # Should call close on exit
        mock_screen_capture.close.assert_called_once()
    
    @patch('src.capture.region_extractor.RegionLoader')
    @patch('src.capture.region_extractor.ScreenCapture')
    def test_empty_regions(self, mock_screen_capture_class, 
                         mock_region_loader_class, mock_screen_capture):
        """Test extraction with no regions defined."""
        mock_screen_capture_class.return_value = mock_screen_capture
        
        mock_loader = Mock()
        mock_loader.load_regions.return_value = {}
        mock_region_loader_class.return_value = mock_loader
        
        extractor = RegionExtractor()
        extracted_regions = extractor.extract_all_regions()
        
        assert len(extracted_regions) == 0
        assert isinstance(extracted_regions, dict)
    
    @patch('src.capture.region_extractor.RegionLoader')
    @patch('src.capture.region_extractor.ScreenCapture')
    def test_extract_all_regions_logging(self, mock_screen_capture_class, 
                                       mock_region_loader_class, 
                                       mock_screen_capture, mock_region_loader, 
                                       caplog):
        """Test that extraction operations are logged."""
        import logging
        
        # Set logging level to DEBUG to capture debug messages
        caplog.set_level(logging.DEBUG)
        
        mock_screen_capture_class.return_value = mock_screen_capture
        mock_region_loader_class.return_value = mock_region_loader
        
        extractor = RegionExtractor()
        extractor.extract_all_regions()
        
        # Check that debug logs are generated
        assert "Extracted 4 regions from new frame" in caplog.text


@pytest.mark.integration
class TestRegionExtractorIntegration:
    """Integration tests for RegionExtractor."""
    
    @pytest.mark.integration
    def test_full_extraction_pipeline(self):
        """Test the complete extraction pipeline."""
        # Create a realistic test frame
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        with patch('src.capture.region_extractor.ScreenCapture') as mock_capture_class:
            mock_capture = Mock()
            mock_capture.capture_frame.return_value = frame
            mock_capture_class.return_value = mock_capture
            
            with patch('src.capture.region_extractor.RegionLoader') as mock_loader_class:
                mock_loader = Mock()
                mock_regions = {
                    "test_region_1": RegionModel(x=100, y=200, width=150, height=30),
                    "test_region_2": RegionModel(x=300, y=400, width=200, height=50)
                }
                mock_loader.load_regions.return_value = mock_regions
                mock_loader_class.return_value = mock_loader
                
                extractor = RegionExtractor()
                extracted_regions = extractor.extract_all_regions()
                
                # Verify the complete pipeline worked
                assert len(extracted_regions) == 2
                assert all(isinstance(img, np.ndarray) for img in extracted_regions.values())
                assert all(img.dtype == np.uint8 for img in extracted_regions.values())
    
    @pytest.mark.integration
    def test_extraction_performance(self):
        """Test extraction performance with multiple regions."""
        import time
        
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        with patch('src.capture.region_extractor.ScreenCapture') as mock_capture_class:
            mock_capture = Mock()
            mock_capture.capture_frame.return_value = frame
            mock_capture_class.return_value = mock_capture
            
            with patch('src.capture.region_extractor.RegionLoader') as mock_loader_class:
                mock_loader = Mock()
                # Create many regions
                mock_regions = {}
                for i in range(50):
                    mock_regions[f"region_{i}"] = RegionModel(
                        x=i * 20, y=i * 15, width=100, height=50
                    )
                mock_loader.load_regions.return_value = mock_regions
                mock_loader_class.return_value = mock_loader
                
                extractor = RegionExtractor()
                
                # Time the extraction
                start_time = time.time()
                extracted_regions = extractor.extract_all_regions()
                end_time = time.time()
                
                assert len(extracted_regions) == 50
                # Should be fast (less than 100ms for 50 regions)
                assert (end_time - start_time) < 0.1


if __name__ == '__main__':
    pytest.main([__file__])
