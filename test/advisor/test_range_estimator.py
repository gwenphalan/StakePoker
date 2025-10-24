#!/usr/bin/env python3
"""
Unit tests for RangeEstimator.

Tests opponent hand range estimation based on position, actions, and game state.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.advisor.range_estimator import RangeEstimator
from src.models.game_state import GameState
from src.models.player import Player
from src.models.card import Card
from src.models.table_info import TableInfo
from test.advisor.test_config import AdvisorTestFixtures


class TestRangeEstimator:
    """Test cases for RangeEstimator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('src.advisor.range_estimator.Settings') as mock_settings, \
             patch('src.advisor.range_estimator.GTOChartLoader') as mock_gto_loader:
            
            mock_settings_instance = mock_settings.return_value
            mock_settings_instance.create.return_value = None
            mock_settings_instance.get.side_effect = lambda key: {
                "advisor.ranges.tight_adjustment": 0.8,
                "advisor.ranges.loose_adjustment": 1.2,
                "advisor.ranges.default_fold_equity": 0.3
            }.get(key, 0.8)
            
            self.mock_gto_loader = mock_gto_loader.return_value
            self.estimator = RangeEstimator()
    
    def test_init(self):
        """Test RangeEstimator initialization."""
        assert self.estimator.tight_adjustment == 0.8
        assert self.estimator.loose_adjustment == 1.2
        assert self.estimator.gto_loader == self.mock_gto_loader
    
    def test_estimate_ranges_no_hero(self):
        """Test range estimation with no hero."""
        from test.advisor.test_config import AdvisorTestFixtures
        
        game_state = AdvisorTestFixtures.create_test_game_state()
        # Make hero inactive instead of removing to maintain minimum player count
        for player in game_state.players:
            if player.is_hero:
                player.is_active = False
        
        ranges = self.estimator.estimate_ranges(game_state)
        assert ranges == []
    
    def test_estimate_ranges_no_active_opponents(self):
        """Test range estimation with no active opponents."""
        from test.advisor.test_config import AdvisorTestFixtures
        
        game_state = AdvisorTestFixtures.create_test_game_state()
        # Make all opponents inactive
        for player in game_state.players:
            if not player.is_hero:
                player.is_active = False
        
        ranges = self.estimator.estimate_ranges(game_state)
        assert ranges == []
    
    def test_estimate_ranges_with_opponents(self):
        """Test range estimation with active opponents."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.is_active = True
        
        opponent1 = Player(seat_number=2, is_hero=False, stack=1000.0)
        opponent1.is_active = True
        opponent1.position = "UTG"
        
        opponent2 = Player(seat_number=3, is_hero=False, stack=1000.0)
        opponent2.is_active = True
        opponent2.position = "CO"
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        game_state.players = [hero, opponent1, opponent2]
        
        # Mock GTO loader responses
        self.mock_gto_loader.get_opening_range.return_value = ["AA", "KK", "QQ"]
        
        ranges = self.estimator.estimate_ranges(game_state)
        
        assert isinstance(ranges, list)
        assert len(ranges) > 0
        assert "AA" in ranges
        assert "KK" in ranges
        assert "QQ" in ranges
    
    def test_estimate_player_range_public_method(self):
        """Test public estimate_player_range method."""
        from test.advisor.test_config import AdvisorTestFixtures
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        player = game_state.players[0]  # Use the first player from the fixture
        
        # Mock GTO loader response
        self.mock_gto_loader.get_opening_range.return_value = ["AA", "KK", "QQ"]
        
        ranges = self.estimator.estimate_player_range(player, game_state)
        
        assert isinstance(ranges, list)
        assert "AA" in ranges
    
    def test_estimate_player_range_no_position(self):
        """Test range estimation for player with no position."""
        player = Player(seat_number=1, is_hero=False, stack=1000.0)
        player.is_active = True
        player.position = None
        
        opponent = Player(seat_number=2, is_hero=False, stack=1000.0)

        
        

        
        game_state = GameState(

        
            players=[player, opponent],

        
            pot=0.0,

        
            phase="preflop",

        
            button_position=1,

        
            table_info=TableInfo(bb=25.0, sb=12.5),

        
            community_cards=[]

        
        )
        
        ranges = self.estimator._estimate_player_range(player, game_state)
        
        # Should return default range
        assert isinstance(ranges, list)
        assert "AA" in ranges  # Default range includes premium hands
    
    def test_get_gto_range_opening_context(self):
        """Test getting GTO range for opening context."""
        player = Player(seat_number=1, is_hero=False, stack=1000.0)
        player.is_active = True
        player.position = "UTG"
        player.current_bet = 0
        
        opponent = Player(seat_number=2, is_hero=False, stack=1000.0)

        
        

        
        game_state = GameState(

        
            players=[player, opponent],

        
            pot=0.0,

        
            phase="preflop",

        
            button_position=1,

        
            table_info=TableInfo(bb=25.0, sb=12.5),

        
            community_cards=[]

        
        )
        
        self.mock_gto_loader.get_opening_range.return_value = ["AA", "KK", "QQ"]
        
        ranges = self.estimator._get_gto_range(player, game_state)
        
        assert ranges == ["AA", "KK", "QQ"]
        self.mock_gto_loader.get_opening_range.assert_called_once()
    
    def test_get_gto_range_calling_context(self):
        """Test getting GTO range for calling context."""
        player = Player(seat_number=1, is_hero=False, stack=1000.0)
        player.is_active = True
        player.position = "UTG"
        player.current_bet = 50  # Small bet
        
        opponent = Player(seat_number=2, is_hero=False, stack=1000.0)

        
        

        
        game_state = GameState(

        
            players=[player, opponent],

        
            pot=0.0,

        
            phase="preflop",

        
            button_position=1,

        
            table_info=TableInfo(bb=25.0, sb=12.5),

        
            community_cards=[]

        
        )
        
        self.mock_gto_loader.get_calling_range.return_value = ["JJ", "TT", "99"]
        
        ranges = self.estimator._get_gto_range(player, game_state)
        
        assert ranges == ["JJ", "TT", "99"]
        self.mock_gto_loader.get_calling_range.assert_called_once()
    
    def test_get_gto_range_3betting_context(self):
        """Test getting GTO range for 3betting context."""
        player = Player(seat_number=1, is_hero=False, stack=1000.0)
        player.is_active = True
        player.position = "UTG"
        player.current_bet = 200  # Large bet
        
        opponent = Player(seat_number=2, is_hero=False, stack=1000.0)

        
        

        
        game_state = GameState(

        
            players=[player, opponent],

        
            pot=0.0,

        
            phase="preflop",

        
            button_position=1,

        
            table_info=TableInfo(bb=25.0, sb=12.5),

        
            community_cards=[]

        
        )
        
        self.mock_gto_loader.get_3bet_range.return_value = ["AA", "KK", "AKs"]
        
        ranges = self.estimator._get_gto_range(player, game_state)
        
        assert ranges == ["AA", "KK", "AKs"]
        self.mock_gto_loader.get_3bet_range.assert_called_once()
    
    def test_get_gto_range_blind_defense_context(self):
        """Test getting GTO range for blind defense context."""
        player = Player(seat_number=1, is_hero=False, stack=1000.0)
        player.is_active = True
        player.position = "BB"
        player.current_bet = 0
        
        # Create another player who raised
        raiser = Player(seat_number=2, is_hero=False, stack=1000.0)
        raiser.is_active = True
        raiser.current_bet = 100
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        game_state.players = [player, raiser]
        
        self.mock_gto_loader.get_blind_defense_range.return_value = ["22+", "A2s+"]
        
        ranges = self.estimator._get_gto_range(player, game_state)
        
        assert ranges == ["22+", "A2s+"]
        self.mock_gto_loader.get_blind_defense_range.assert_called_once()
    
    def test_get_action_context_opening(self):
        """Test action context detection for opening."""
        player = Player(seat_number=1, is_hero=False, stack=1000.0)
        player.is_active = True
        player.current_bet = 0
        
        opponent = Player(seat_number=2, is_hero=False, stack=1000.0)

        
        

        
        game_state = GameState(

        
            players=[player, opponent],

        
            pot=0.0,

        
            phase="preflop",

        
            button_position=1,

        
            table_info=TableInfo(bb=25.0, sb=12.5),

        
            community_cards=[]

        
        )
        
        context = self.estimator._get_action_context(player, game_state)
        assert context == "opening"
    
    def test_get_action_context_3betting(self):
        """Test action context detection for 3betting."""
        player = Player(seat_number=1, is_hero=False, stack=1000.0)
        player.is_active = True
        player.current_bet = 200  # Large bet
        
        opponent = Player(seat_number=2, is_hero=False, stack=1000.0)

        
        

        
        game_state = GameState(

        
            players=[player, opponent],

        
            pot=0.0,

        
            phase="preflop",

        
            button_position=1,

        
            table_info=TableInfo(bb=25.0, sb=12.5),

        
            community_cards=[]

        
        )
        
        context = self.estimator._get_action_context(player, game_state)
        assert context == "3betting"
    
    def test_get_action_context_calling(self):
        """Test action context detection for calling."""
        player = Player(seat_number=1, is_hero=False, stack=1000.0)
        player.is_active = True
        player.current_bet = 50  # Small bet
        
        opponent = Player(seat_number=2, is_hero=False, stack=1000.0)

        
        

        
        game_state = GameState(

        
            players=[player, opponent],

        
            pot=0.0,

        
            phase="preflop",

        
            button_position=1,

        
            table_info=TableInfo(bb=25.0, sb=12.5),

        
            community_cards=[]

        
        )
        
        context = self.estimator._get_action_context(player, game_state)
        assert context == "calling"
    
    def test_adjust_for_actions_folded_player(self):
        """Test range adjustment for folded player."""
        player = Player(seat_number=1, is_hero=False, stack=1000.0)
        player.is_active = False  # Folded
        
        base_range = ["AA", "KK", "QQ", "JJ", "72o"]
        
        adjusted_range = self.estimator._adjust_for_actions(base_range, player, GameState())
        
        # Premium hands should be removed from folded player's range
        assert "AA" not in adjusted_range
        assert "KK" not in adjusted_range
        assert "72o" in adjusted_range  # Weak hands remain
    
    def test_adjust_for_actions_large_raise(self):
        """Test range adjustment for large raise."""
        player = Player(seat_number=1, is_hero=False, stack=1000.0)
        player.is_active = True
        player.current_bet = 200  # Large raise
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        
        base_range = ["AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55"]
        
        adjusted_range = self.estimator._adjust_for_actions(base_range, player, game_state)
        
        # Range should be tightened
        assert len(adjusted_range) < len(base_range)
        assert "AA" in adjusted_range  # Premium hands should remain
    
    def test_adjust_for_tendencies(self):
        """Test range adjustment for player tendencies."""
        player = Player(seat_number=1, is_hero=False, stack=1000.0)
        base_range = ["AA", "KK", "QQ"]
        
        # Currently just returns range as-is (placeholder)
        adjusted_range = self.estimator._adjust_for_tendencies(base_range, player)
        
        assert adjusted_range == base_range
    
    def test_tighten_range(self):
        """Test range tightening."""
        base_range = ["AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55"]
        
        tightened_range = self.estimator._tighten_range(base_range, 0.5)
        
        assert len(tightened_range) == 5  # Half the hands
        assert "AA" in tightened_range  # Strongest hands should remain
        assert "55" not in tightened_range  # Weakest hands should be removed
    
    def test_tighten_range_factor_one(self):
        """Test range tightening with factor 1.0."""
        base_range = ["AA", "KK", "QQ"]
        
        tightened_range = self.estimator._tighten_range(base_range, 1.0)
        
        assert tightened_range == base_range
    
    def test_get_hand_strength_order(self):
        """Test hand strength ordering."""
        strength_map = self.estimator._get_hand_strength_order()
        
        assert isinstance(strength_map, dict)
        assert "AA" in strength_map
        assert "KK" in strength_map
        assert "72o" in strength_map
        
        # AA should be stronger than KK
        assert strength_map["AA"] > strength_map["KK"]
        
        # Premium hands should be stronger than weak hands
        assert strength_map["AA"] > strength_map["72o"]
    
    def test_get_default_range(self):
        """Test default range when GTO range unavailable."""
        default_range = self.estimator._get_default_range()
        
        assert isinstance(default_range, list)
        assert "AA" in default_range
        assert "KK" in default_range
        assert "72o" not in default_range  # Should be conservative
    
    def test_is_facing_raise(self):
        """Test detection of facing raise."""
        player = Player(seat_number=1, is_hero=False, stack=1000.0)
        
        raiser = Player(seat_number=2, is_hero=False, stack=1000.0)
        raiser.current_bet = 100
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        game_state.players = [player, raiser]
        
        assert self.estimator._is_facing_raise(player, game_state) is True
    
    def test_is_facing_raise_no_raise(self):
        """Test detection when not facing raise."""
        player = Player(seat_number=1, is_hero=False, stack=1000.0)
        
        other_player = Player(seat_number=2, is_hero=False, stack=1000.0)
        other_player.current_bet = 25  # Just big blind
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        game_state.players = [player, other_player]
        
        assert self.estimator._is_facing_raise(player, game_state) is False
    
    def test_is_facing_open(self):
        """Test detection of facing open raise."""
        player = Player(seat_number=1, is_hero=False, stack=1000.0)
        
        raiser = Player(seat_number=2, is_hero=False, stack=1000.0)
        raiser.current_bet = 100
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        game_state.players = [player, raiser]
        
        assert self.estimator._is_facing_open(player, game_state) is True
    
    def test_error_handling_gto_loader_exception(self):
        """Test error handling when GTO loader raises exception."""
        player = Player(seat_number=1, is_hero=False, stack=1000.0)
        player.is_active = True
        player.position = "UTG"
        
        opponent = Player(seat_number=2, is_hero=False, stack=1000.0)

        
        

        
        game_state = GameState(

        
            players=[player, opponent],

        
            pot=0.0,

        
            phase="preflop",

        
            button_position=1,

        
            table_info=TableInfo(bb=25.0, sb=12.5),

        
            community_cards=[]

        
        )
        
        # Mock GTO loader to raise exception
        self.mock_gto_loader.get_opening_range.side_effect = Exception("GTO error")
        
        ranges = self.estimator._get_gto_range(player, game_state)
        
        # Should return default range on error
        assert isinstance(ranges, list)
        assert "AA" in ranges
    
    def test_range_estimation_with_multiple_opponents(self):
        """Test range estimation with multiple opponents."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.is_active = True
        
        opponent1 = Player(seat_number=2, is_hero=False, stack=1000.0)
        opponent1.is_active = True
        opponent1.position = "UTG"
        
        opponent2 = Player(seat_number=3, is_hero=False, stack=1000.0)
        opponent2.is_active = True
        opponent2.position = "CO"
        
        opponent3 = Player(seat_number=4, is_hero=False, stack=1000.0)
        opponent3.is_active = False  # Folded
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        game_state.players = [hero, opponent1, opponent2, opponent3]
        
        # Mock different ranges for different opponents
        self.mock_gto_loader.get_opening_range.side_effect = [
            ["AA", "KK", "QQ"],  # UTG range
            ["AA", "KK", "QQ", "JJ", "TT"]  # CO range
        ]
        
        ranges = self.estimator.estimate_ranges(game_state)
        
        # Should combine ranges from active opponents
        assert isinstance(ranges, list)
        assert len(ranges) > 0
        assert "AA" in ranges
        assert "KK" in ranges
        assert "QQ" in ranges
    
    def test_player_count_calculation(self):
        """Test player count calculation for GTO range selection."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.is_active = True
        
        opponent1 = Player(seat_number=2, is_hero=False, stack=1000.0)
        opponent1.is_active = True
        
        opponent2 = Player(seat_number=3, is_hero=False, stack=1000.0)
        opponent2.is_active = False  # Folded
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        game_state.players = [hero, opponent1, opponent2]
        
        player = Player(seat_number=2, is_hero=False, stack=1000.0)
        player.is_active = True
        player.position = "UTG"
        
        self.mock_gto_loader.get_opening_range.return_value = ["AA", "KK"]
        
        ranges = self.estimator._get_gto_range(player, game_state)
        
        # Should call with correct player count (2 active players)
        self.mock_gto_loader.get_opening_range.assert_called_with("UTG", 2)
    
    def test_range_adjustment_edge_cases(self):
        """Test range adjustment edge cases."""
        player = Player(seat_number=1, is_hero=False, stack=1000.0)
        player.is_active = True
        player.current_bet = 0
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        
        # Test with empty base range
        empty_range = []
        adjusted_range = self.estimator._adjust_for_actions(empty_range, player, game_state)
        assert adjusted_range == []
        
        # Test with single hand
        single_range = ["AA"]
        adjusted_range = self.estimator._adjust_for_actions(single_range, player, game_state)
        assert adjusted_range == ["AA"]
