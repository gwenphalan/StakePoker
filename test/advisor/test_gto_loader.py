#!/usr/bin/env python3
"""
Unit tests for GTOChartLoader.

Tests loading and querying GTO preflop charts with various scenarios.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open
from src.advisor.gto_loader import GTOChartLoader


class TestGTOChartLoader:
    """Test cases for GTOChartLoader."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for test charts
        self.temp_dir = tempfile.mkdtemp()
        self.chart_dir = Path(self.temp_dir)
        
        # Create test metadata
        self.metadata = {
            "format": "poker_range",
            "solver": "PioSolver",
            "stack_depth": "100bb",
            "source": "test_data"
        }
        
        # Create test position data
        self.test_position_data = {
            "opening": {
                "hands": ["AA", "KK", "QQ", "JJ", "TT", "AKs", "AKo", "AQs", "AQo"],
                "frequencies": {
                    "AA": 1.0,
                    "KK": 1.0,
                    "QQ": 1.0,
                    "JJ": 0.8,
                    "TT": 0.6,
                    "AKs": 1.0,
                    "AKo": 0.9,
                    "AQs": 0.7,
                    "AQo": 0.5
                },
                "bet_size": 2.5
            },
            "3bet": {
                "hands": ["AA", "KK", "QQ", "AKs", "AKo"],
                "frequencies": {
                    "AA": 1.0,
                    "KK": 1.0,
                    "QQ": 0.8,
                    "AKs": 0.7,
                    "AKo": 0.5
                },
                "bet_size": 11.0
            },
            "calling": {
                "hands": ["JJ", "TT", "99", "88", "AQs", "AQo", "KQs", "KQo"],
                "frequencies": {
                    "JJ": 0.2,
                    "TT": 0.4,
                    "99": 0.6,
                    "88": 0.5,
                    "AQs": 0.3,
                    "AQo": 0.2,
                    "KQs": 0.4,
                    "KQo": 0.3
                }
            },
            "blind_defense": {
                "hands": ["22+", "A2s+", "K2s+", "Q2s+", "J2s+", "T2s+", "92s+", "82s+", "72s+", "62s+", "52s+", "42s+", "32s"],
                "frequencies": {}
            }
        }
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_charts(self):
        """Create test chart files."""
        # Create metadata file
        metadata_path = self.chart_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f)
        
        # Create position files
        positions = ["UTG", "MP", "CO", "BTN", "SB", "BB"]
        for position in positions:
            # 8-max file
            file_path = self.chart_dir / f"{position}.json"
            with open(file_path, 'w') as f:
                json.dump(self.test_position_data, f)
            
            # 6-max file (except BB)
            if position != "BB":
                file_path = self.chart_dir / f"{position}_6max.json"
                with open(file_path, 'w') as f:
                    json.dump(self.test_position_data, f)
    
    def test_init_with_valid_directory(self):
        """Test initialization with valid chart directory."""
        self._create_test_charts()
        
        loader = GTOChartLoader(str(self.chart_dir))
        
        assert loader.chart_dir == self.chart_dir
        assert loader.metadata == self.metadata
        assert len(loader._position_cache) == 0  # Cache starts empty
    
    def test_init_with_nonexistent_directory(self):
        """Test initialization with nonexistent directory."""
        nonexistent_dir = self.chart_dir / "nonexistent"
        
        with pytest.raises(FileNotFoundError):
            GTOChartLoader(str(nonexistent_dir))
    
    def test_init_with_default_directory(self):
        """Test initialization with default directory from settings."""
        self._create_test_charts()
        
        with patch('src.advisor.gto_loader.Settings') as mock_settings:
            mock_instance = mock_settings.return_value
            mock_instance.create.return_value = None
            mock_instance.get.return_value = str(self.chart_dir)
            
            loader = GTOChartLoader()
            
            assert loader.chart_dir == self.chart_dir
            mock_instance.create.assert_called_with("advisor.gto.chart_path", default="data/gto_charts")
            mock_instance.get.assert_called_with("advisor.gto.chart_path")
    
    def test_get_position_file_6max(self):
        """Test getting position file for 6-max table."""
        self._create_test_charts()
        loader = GTOChartLoader(str(self.chart_dir))
        
        # Test 6-max file selection
        file_path = loader._get_position_file("UTG", 6)
        assert file_path.name == "UTG_6max.json"
        
        # Test BB fallback to 8-max
        file_path = loader._get_position_file("BB", 6)
        assert file_path.name == "BB.json"
    
    def test_get_position_file_8max(self):
        """Test getting position file for 8-max table."""
        self._create_test_charts()
        loader = GTOChartLoader(str(self.chart_dir))
        
        file_path = loader._get_position_file("UTG", 8)
        assert file_path.name == "UTG.json"
        
        file_path = loader._get_position_file("BB", 8)
        assert file_path.name == "BB.json"
    
    def test_load_position_data_with_caching(self):
        """Test loading position data with caching."""
        self._create_test_charts()
        loader = GTOChartLoader(str(self.chart_dir))
        
        # First load
        data1 = loader._load_position_data("UTG", 6)
        assert data1 == self.test_position_data
        assert "UTG_6" in loader._position_cache
        
        # Second load should use cache
        data2 = loader._load_position_data("UTG", 6)
        assert data1 is data2  # Same object from cache
    
    def test_load_position_data_nonexistent_file(self):
        """Test loading nonexistent position file."""
        self._create_test_charts()
        loader = GTOChartLoader(str(self.chart_dir))
        
        # Remove a file
        (self.chart_dir / "MP.json").unlink()
        
        data = loader._load_position_data("MP", 8)
        assert data == {}
    
    def test_load_position_data_bb_6max_fallback(self):
        """Test BB 6-max fallback to 8-max data."""
        self._create_test_charts()
        loader = GTOChartLoader(str(self.chart_dir))
        
        # BB_6max.json doesn't exist, should fallback to BB.json
        data = loader._load_position_data("BB", 6)
        assert data == self.test_position_data
        assert "BB_6" in loader._position_cache
    
    def test_should_open(self):
        """Test should_open method."""
        self._create_test_charts()
        loader = GTOChartLoader(str(self.chart_dir))
        
        # Test hands in opening range
        assert loader.should_open("UTG", "AA", 6) is True
        assert loader.should_open("UTG", "AKs", 6) is True
        
        # Test hands not in opening range
        assert loader.should_open("UTG", "72o", 6) is False
        
        # Test with invalid position
        assert loader.should_open("INVALID", "AA", 6) is False
    
    def test_should_3bet(self):
        """Test should_3bet method."""
        self._create_test_charts()
        loader = GTOChartLoader(str(self.chart_dir))
        
        assert loader.should_3bet("UTG", "AA", 6) is True
        assert loader.should_3bet("UTG", "AKs", 6) is True
        assert loader.should_3bet("UTG", "72o", 6) is False
    
    def test_should_call(self):
        """Test should_call method."""
        self._create_test_charts()
        loader = GTOChartLoader(str(self.chart_dir))
        
        assert loader.should_call("UTG", "JJ", 6) is True
        assert loader.should_call("UTG", "AQs", 6) is True
        assert loader.should_call("UTG", "72o", 6) is False
    
    def test_should_defend_blind(self):
        """Test should_defend_blind method."""
        self._create_test_charts()
        loader = GTOChartLoader(str(self.chart_dir))
        
        # Test SB/BB positions
        assert loader.should_defend_blind("SB", "22", 6) is True
        assert loader.should_defend_blind("BB", "A2s", 6) is True
        
        # Test non-blind positions
        assert loader.should_defend_blind("UTG", "22", 6) is False
    
    def test_get_opening_frequency(self):
        """Test get_opening_frequency method."""
        self._create_test_charts()
        loader = GTOChartLoader(str(self.chart_dir))
        
        assert loader.get_opening_frequency("UTG", "AA", 6) == 1.0
        assert loader.get_opening_frequency("UTG", "JJ", 6) == 0.8
        assert loader.get_opening_frequency("UTG", "72o", 6) == 0.0
    
    def test_get_3bet_frequency(self):
        """Test get_3bet_frequency method."""
        self._create_test_charts()
        loader = GTOChartLoader(str(self.chart_dir))
        
        assert loader.get_3bet_frequency("UTG", "AA", 6) == 1.0
        assert loader.get_3bet_frequency("UTG", "QQ", 6) == 0.8
        assert loader.get_3bet_frequency("UTG", "72o", 6) == 0.0
    
    def test_get_opening_range(self):
        """Test get_opening_range method."""
        self._create_test_charts()
        loader = GTOChartLoader(str(self.chart_dir))
        
        range_list = loader.get_opening_range("UTG", 6)
        assert isinstance(range_list, list)
        assert "AA" in range_list
        assert "AKs" in range_list
        assert "72o" not in range_list
    
    def test_get_3bet_range(self):
        """Test get_3bet_range method."""
        self._create_test_charts()
        loader = GTOChartLoader(str(self.chart_dir))
        
        range_list = loader.get_3bet_range("UTG", 6)
        assert isinstance(range_list, list)
        assert "AA" in range_list
        assert "AKs" in range_list
        assert "JJ" not in range_list  # JJ not in 3bet range
    
    def test_get_bet_size(self):
        """Test get_bet_size method."""
        self._create_test_charts()
        loader = GTOChartLoader(str(self.chart_dir))
        
        assert loader.get_bet_size("UTG", "opening", 6) == 2.5
        assert loader.get_bet_size("UTG", "3bet", 6) == 11.0
        
        # Test default bet sizes
        assert loader.get_bet_size("UTG", "nonexistent", 6) == 2.5
    
    def test_get_preflop_decision_opening(self):
        """Test get_preflop_decision for opening context."""
        self._create_test_charts()
        loader = GTOChartLoader(str(self.chart_dir))
        
        action, bet_size, reasoning = loader.get_preflop_decision("UTG", "AA", 6, "opening")
        assert action == "raise"
        assert bet_size == 2.5
        assert "GTO opening range" in reasoning
        
        action, bet_size, reasoning = loader.get_preflop_decision("UTG", "72o", 6, "opening")
        assert action == "fold"
        assert bet_size == 0.0
        assert "Not in GTO opening range" in reasoning
    
    def test_get_preflop_decision_vs_open(self):
        """Test get_preflop_decision for vs_open context."""
        self._create_test_charts()
        loader = GTOChartLoader(str(self.chart_dir))
        
        # Test 3bet
        action, bet_size, reasoning = loader.get_preflop_decision("UTG", "AA", 6, "vs_open")
        assert action == "3bet"
        assert bet_size == 11.0
        assert "GTO 3-bet range" in reasoning
        
        # Test call
        action, bet_size, reasoning = loader.get_preflop_decision("UTG", "JJ", 6, "vs_open")
        assert action == "call"
        assert bet_size == 0.0
        assert "GTO calling range" in reasoning
        
        # Test fold
        action, bet_size, reasoning = loader.get_preflop_decision("UTG", "72o", 6, "vs_open")
        assert action == "fold"
        assert bet_size == 0.0
        assert "Not in GTO range" in reasoning
    
    def test_get_preflop_decision_vs_3bet(self):
        """Test get_preflop_decision for vs_3bet context."""
        self._create_test_charts()
        loader = GTOChartLoader(str(self.chart_dir))
        
        action, bet_size, reasoning = loader.get_preflop_decision("UTG", "AA", 6, "vs_3bet")
        assert action == "4bet"
        assert bet_size == 24.0  # Default 4bet size
        assert "GTO 4-bet range" in reasoning
    
    def test_get_preflop_decision_blind_defense(self):
        """Test get_preflop_decision for blind_defense context."""
        self._create_test_charts()
        loader = GTOChartLoader(str(self.chart_dir))
        
        action, bet_size, reasoning = loader.get_preflop_decision("BB", "22", 6, "blind_defense")
        assert action == "call"
        assert bet_size == 0.0
        assert "GTO blind defense" in reasoning
    
    def test_get_preflop_decision_unknown_context(self):
        """Test get_preflop_decision for unknown context."""
        self._create_test_charts()
        loader = GTOChartLoader(str(self.chart_dir))
        
        action, bet_size, reasoning = loader.get_preflop_decision("UTG", "AA", 6, "unknown")
        assert action == "fold"
        assert bet_size == 0.0
        assert "Unknown action context" in reasoning
    
    def test_normalize_hand(self):
        """Test _normalize_hand method."""
        self._create_test_charts()
        loader = GTOChartLoader(str(self.chart_dir))
        
        # Test pocket pairs
        assert loader._normalize_hand("AA") == "AA"
        assert loader._normalize_hand("AhAd") == "AA"
        
        # Test suited hands
        assert loader._normalize_hand("AKs") == "AKs"
        assert loader._normalize_hand("AhKh") == "AKs"
        assert loader._normalize_hand("A♠K♠") == "AKs"
        
        # Test offsuit hands
        assert loader._normalize_hand("AKo") == "AKo"
        assert loader._normalize_hand("AhKd") == "AKo"
        
        # Test unclear hands (default to offsuit)
        assert loader._normalize_hand("AK") == "AKo"
    
    def test_get_available_positions(self):
        """Test get_available_positions method."""
        self._create_test_charts()
        loader = GTOChartLoader(str(self.chart_dir))
        
        positions = loader.get_available_positions()
        expected = ["UTG", "MP", "CO", "BTN", "SB", "BB"]
        assert positions == expected
    
    def test_get_available_positions_6max(self):
        """Test get_available_positions_6max method."""
        self._create_test_charts()
        loader = GTOChartLoader(str(self.chart_dir))
        
        positions = loader.get_available_positions_6max()
        expected = ["UTG", "MP", "CO", "BTN", "SB"]
        assert positions == expected
    
    def test_get_available_positions_8max(self):
        """Test get_available_positions_8max method."""
        self._create_test_charts()
        loader = GTOChartLoader(str(self.chart_dir))
        
        positions = loader.get_available_positions_8max()
        expected = ["UTG", "MP", "CO", "BTN", "SB", "BB"]
        assert positions == expected
    
    def test_get_chart_info(self):
        """Test get_chart_info method."""
        self._create_test_charts()
        loader = GTOChartLoader(str(self.chart_dir))
        
        info = loader.get_chart_info()
        
        assert info['format'] == "poker_range"
        assert info['solver'] == "PioSolver"
        assert info['stack_depth'] == "100bb"
        assert info['source'] == "test_data"
        assert info['chart_dir'] == str(self.chart_dir)
        assert 'positions' in info
        assert 'notes' in info
        assert 'bb_6max_fallback' in info['notes']
    
    def test_error_handling_invalid_json(self):
        """Test error handling for invalid JSON files."""
        self._create_test_charts()
        
        # Corrupt a JSON file
        corrupted_file = self.chart_dir / "UTG.json"
        with open(corrupted_file, 'w') as f:
            f.write("invalid json content")
        
        loader = GTOChartLoader(str(self.chart_dir))
        
        # Should return empty dict for corrupted file
        data = loader._load_position_data("UTG", 8)
        assert data == {}
    
    def test_error_handling_missing_metadata(self):
        """Test initialization with missing metadata file."""
        # Create charts without metadata
        for position in ["UTG", "MP", "CO", "BTN", "SB", "BB"]:
            file_path = self.chart_dir / f"{position}.json"
            with open(file_path, 'w') as f:
                json.dump(self.test_position_data, f)
        
        loader = GTOChartLoader(str(self.chart_dir))
        
        # Should have empty metadata
        assert loader.metadata == {}
        
        # Chart info should still work
        info = loader.get_chart_info()
        assert info['format'] == 'Unknown'
        assert info['solver'] == 'Unknown'
