#!/usr/bin/env python3
"""
Tests for region loader module.

Tests region loading, validation, and error handling.
"""

import pytest
import json
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.capture.region_loader import RegionLoader, RegionModel, load_regions_typed, save_regions_typed


@pytest.mark.unit
class TestRegionModel:
    """Test RegionModel validation."""
    
    def test_valid_region_model(self):
        """Test valid RegionModel creation."""
        region = RegionModel(x=100, y=200, width=300, height=400)
        
        assert region.x == 100
        assert region.y == 200
        assert region.width == 300
        assert region.height == 400
    
    def test_region_model_validation(self):
        """Test RegionModel field validation."""
        # Test with invalid types
        with pytest.raises(ValueError):
            RegionModel(x="invalid", y=200, width=300, height=400)
        
        with pytest.raises(ValueError):
            RegionModel(x=100, y=200, width=300, height="invalid")
    
    def test_region_model_dict_conversion(self):
        """Test RegionModel to dict conversion."""
        region = RegionModel(x=100, y=200, width=300, height=400)
        region_dict = region.model_dump()
        
        expected = {'x': 100, 'y': 200, 'width': 300, 'height': 400}
        assert region_dict == expected


@pytest.mark.unit
class TestRegionLoader:
    """Test RegionLoader class."""
    
    @pytest.fixture
    def sample_regions_data(self):
        """Sample regions data for testing."""
        return {
            "player_1_name": {"x": 100, "y": 200, "width": 150, "height": 30},
            "player_1_stack": {"x": 100, "y": 250, "width": 100, "height": 25},
            "pot": {"x": 400, "y": 300, "width": 120, "height": 40},
            "table_info": {"x": 50, "y": 50, "width": 200, "height": 30}
        }
    
    @pytest.fixture
    def mock_regions_file(self, sample_regions_data, tmp_path):
        """Create a temporary regions file."""
        regions_file = tmp_path / "regions.json"
        with open(regions_file, 'w') as f:
            json.dump(sample_regions_data, f)
        return regions_file
    
    def test_load_regions_typed_success(self, sample_regions_data, mock_regions_file):
        """Test successful loading of typed regions."""
        with patch('src.capture.region_loader.REGIONS_FILE_PATH', mock_regions_file):
            regions = load_regions_typed()
            
            assert len(regions) == 4
            assert "player_1_name" in regions
            assert "pot" in regions
            
            # Check that all regions are RegionModel instances
            for region in regions.values():
                assert isinstance(region, RegionModel)
            
            # Check specific region
            player_name_region = regions["player_1_name"]
            assert player_name_region.x == 100
            assert player_name_region.y == 200
            assert player_name_region.width == 150
            assert player_name_region.height == 30
    
    def test_load_regions_typed_file_not_found(self):
        """Test error when regions file doesn't exist."""
        non_existent_file = Path("/non/existent/regions.json")
        
        with patch('src.capture.region_loader.REGIONS_FILE_PATH', non_existent_file):
            with pytest.raises(FileNotFoundError):
                load_regions_typed()
    
    def test_load_regions_typed_validation_error(self, tmp_path):
        """Test error when regions data is invalid."""
        invalid_regions_file = tmp_path / "invalid_regions.json"
        invalid_data = {
            "invalid_region": {"x": "not_a_number", "y": 200, "width": 300, "height": 400}
        }
        
        with open(invalid_regions_file, 'w') as f:
            json.dump(invalid_data, f)
        
        with patch('src.capture.region_loader.REGIONS_FILE_PATH', invalid_regions_file):
            with pytest.raises(ValueError):
                load_regions_typed()
    
    def test_save_regions_typed_success(self, sample_regions_data, tmp_path):
        """Test successful saving of typed regions."""
        regions_file = tmp_path / "regions.json"
        
        # Convert sample data to RegionModel objects
        regions = {
            name: RegionModel(**data) 
            for name, data in sample_regions_data.items()
        }
        
        with patch('src.capture.region_loader.REGIONS_FILE_PATH', regions_file):
            save_regions_typed(regions)
            
            # Verify file was created and contains correct data
            assert regions_file.exists()
            
            with open(regions_file, 'r') as f:
                saved_data = json.load(f)
            
            assert saved_data == sample_regions_data
    
    def test_save_regions_typed_io_error(self):
        """Test error handling during save operation."""
        # Mock a file that can't be written to
        mock_file_path = Mock()
        mock_file_path.parent.mkdir = Mock()
        mock_file_path.exists = Mock(return_value=False)
        
        with patch('src.capture.region_loader.REGIONS_FILE_PATH', mock_file_path):
            with patch('builtins.open', mock_open()) as mock_file:
                mock_file.side_effect = IOError("Permission denied")
                
                regions = {"test": RegionModel(x=0, y=0, width=100, height=100)}
                
                with pytest.raises(IOError):
                    save_regions_typed(regions)
    
    def test_region_loader_integration(self, sample_regions_data, mock_regions_file):
        """Test RegionLoader integration."""
        with patch('src.capture.region_loader.REGIONS_FILE_PATH', mock_regions_file):
            loader = RegionLoader()
            regions = loader.load_regions()
            
            assert len(regions) == 4
            assert all(isinstance(region, RegionModel) for region in regions.values())
    
    def test_empty_regions_file(self, tmp_path):
        """Test loading empty regions file."""
        empty_regions_file = tmp_path / "empty_regions.json"
        with open(empty_regions_file, 'w') as f:
            json.dump({}, f)
        
        with patch('src.capture.region_loader.REGIONS_FILE_PATH', empty_regions_file):
            regions = load_regions_typed()
            assert len(regions) == 0
    
    def test_malformed_json(self, tmp_path):
        """Test error handling for malformed JSON."""
        malformed_file = tmp_path / "malformed.json"
        with open(malformed_file, 'w') as f:
            f.write("{ invalid json }")
        
        with patch('src.capture.region_loader.REGIONS_FILE_PATH', malformed_file):
            with pytest.raises(json.JSONDecodeError):
                load_regions_typed()
    
    def test_region_model_edge_cases(self):
        """Test RegionModel with edge case values."""
        # Test with zero dimensions
        region = RegionModel(x=0, y=0, width=0, height=0)
        assert region.x == 0
        assert region.y == 0
        assert region.width == 0
        assert region.height == 0
        
        # Test with large values
        region = RegionModel(x=9999, y=9999, width=9999, height=9999)
        assert region.x == 9999
        assert region.y == 9999
        assert region.width == 9999
        assert region.height == 9999
    
    def test_region_model_negative_values(self):
        """Test RegionModel with negative values."""
        # Pydantic should handle negative values (they might be valid for some use cases)
        region = RegionModel(x=-100, y=-200, width=300, height=400)
        assert region.x == -100
        assert region.y == -200
        assert region.width == 300
        assert region.height == 400


class TestRegionLoaderPerformance:
    """Performance tests for RegionLoader."""
    
    @pytest.mark.performance
    def test_large_regions_file_performance(self, tmp_path):
        """Test performance with large regions file."""
        # Create a large regions file
        large_regions_data = {}
        for i in range(1000):
            large_regions_data[f"region_{i}"] = {
                "x": i * 10,
                "y": i * 10,
                "width": 100,
                "height": 50
            }
        
        large_regions_file = tmp_path / "large_regions.json"
        with open(large_regions_file, 'w') as f:
            json.dump(large_regions_data, f)
        
        with patch('src.capture.region_loader.REGIONS_FILE_PATH', large_regions_file):
            import time
            start_time = time.time()
            regions = load_regions_typed()
            end_time = time.time()
            
            assert len(regions) == 1000
            # Should load quickly (less than 1 second)
            assert (end_time - start_time) < 1.0


if __name__ == '__main__':
    pytest.main([__file__])
