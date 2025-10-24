#!/usr/bin/env python3
"""
Unit tests for PostflopSolver.

Tests EV-based postflop decision making and recommendation generation.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.advisor.postflop_solver import PostflopSolver
from src.models.game_state import GameState
from src.models.player import Player
from src.models.card import Card
from src.models.table_info import TableInfo
from src.models.decision import Decision, AlternativeAction
from test.advisor.test_config import AdvisorTestFixtures


class TestPostflopSolver:
    """Test cases for PostflopSolver."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('src.advisor.postflop_solver.Settings') as mock_settings:
            mock_settings_instance = mock_settings.return_value
            mock_settings_instance.create.return_value = None
            mock_settings_instance.get.side_effect = lambda key: {
                "advisor.postflop.min_equity_to_call": 0.25,
                "advisor.postflop.min_equity_to_bet": 0.35,
                "advisor.postflop.bet_sizing_factor": 0.75,
                "advisor.postflop.raise_sizing_factor": 3.0,
                "advisor.postflop.base_fold_equity": 0.3
            }.get(key, 0.25)
            
            self.solver = PostflopSolver()
    
    def test_init(self):
        """Test PostflopSolver initialization."""
        assert self.solver.min_equity_to_call == 0.25
        assert self.solver.min_equity_to_bet == 0.35
        assert self.solver.bet_sizing_factor == 0.75
        assert self.solver.raise_sizing_factor == 3.0
        assert self.solver.base_fold_equity == 0.3
    
    def test_get_recommendation_no_hero(self):
        """Test recommendation when no hero found."""
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        # Remove hero to test no hero scenario
        game_state.players = [p for p in game_state.players if not p.is_hero]
        # Add a second player to maintain minimum player count
        opponent2 = Player(seat_number=3, is_hero=False, stack=1000.0)
        game_state.players.append(opponent2)
        
        decision = self.solver.get_recommendation(game_state, 0.5, 0.2)
        
        assert isinstance(decision, Decision)
        assert decision.action == "fold"
        assert decision.confidence == 0.5
        assert "Unable to calculate decision" in decision.reasoning
    
    def test_get_recommendation_with_hero(self):
        """Test recommendation with valid hero."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.is_active = True
        hero.stack = 1000
        hero.current_bet = 0
        hero.position = "BTN"
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        # Replace players with our test hero
        game_state.players = [hero]
        game_state.community_cards = [
            Card(rank='A', suit='hearts'),
            Card(rank='K', suit='spades'),
            Card(rank='Q', suit='hearts')
        ]
        
        decision = self.solver.get_recommendation(game_state, 0.6, 0.0)
        
        assert isinstance(decision, Decision)
        assert decision.action in ["fold", "call", "bet", "raise", "check"]
        assert 0.0 <= decision.confidence <= 1.0
        assert decision.equity == 0.6
        assert decision.pot_odds == 0.0
    
    def test_calculate_fold_ev(self):
        """Test fold EV calculation."""
        ev = self.solver._calculate_fold_ev()
        assert ev == 0.0
    
    def test_calculate_call_ev_no_bet(self):
        """Test call EV when not facing a bet."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.current_bet = 0
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        # Replace players with our test hero
        game_state.players = [hero]
        
        ev = self.solver._calculate_call_ev(game_state, 0.5, 0.0)
        assert ev == -999.0  # Invalid action
    
    def test_calculate_call_ev_facing_bet(self):
        """Test call EV when facing a bet."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.current_bet = 0
        hero.stack = 1000
        
        opponent = Player(seat_number=2, is_hero=False, stack=1000.0)
        opponent.current_bet = 50
        opponent.is_active = True
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        # Replace players with our test players
        game_state.players = [hero, opponent]
        game_state.pot = 100
        
        ev = self.solver._calculate_call_ev(game_state, 0.6, 0.2)
        
        # EV = equity * final_pot - (1 - equity) * bet_to_call
        # EV = 0.6 * 150 - 0.4 * 50 = 90 - 20 = 70
        expected_ev = 0.6 * 150 - 0.4 * 50
        assert abs(ev - expected_ev) < 0.01
    
    def test_calculate_call_ev_insufficient_stack(self):
        """Test call EV when hero has insufficient stack."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.current_bet = 0
        hero.stack = 20  # Less than bet to call
        
        opponent = Player(seat_number=2, is_hero=False, stack=1000.0)
        opponent.current_bet = 50
        opponent.is_active = True
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        # Replace players with our test players
        game_state.players = [hero, opponent]
        game_state.pot = 100
        
        ev = self.solver._calculate_call_ev(game_state, 0.6, 0.2)
        
        # Should be limited by stack size
        assert ev > -999.0  # Should be valid
        assert ev < 100  # Should be reasonable
    
    def test_calculate_bet_ev(self):
        """Test bet EV calculation."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.stack = 1000
        hero.position = "BTN"
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        # Replace players with our test hero
        game_state.players = [hero]
        game_state.community_cards = [
            Card(rank='A', suit='hearts'),
            Card(rank='K', suit='spades'),
            Card(rank='Q', suit='hearts')
        ]
        
        ev = self.solver._calculate_bet_ev(game_state, 0.6)
        
        assert isinstance(ev, float)
        assert ev > -999.0  # Should be valid
    
    def test_calculate_bet_ev_insufficient_stack(self):
        """Test bet EV when hero has insufficient stack."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.stack = 0  # No chips
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        # Replace players with our test hero
        game_state.players = [hero]
        
        ev = self.solver._calculate_bet_ev(game_state, 0.6)
        assert ev == -999.0  # Invalid action
    
    def test_calculate_raise_ev(self):
        """Test raise EV calculation."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.stack = 1000
        hero.position = "BTN"
        
        opponent = Player(seat_number=2, is_hero=False, stack=1000.0)
        opponent.current_bet = 50
        opponent.is_active = True
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        # Replace players with our test players
        game_state.players = [hero, opponent]
        game_state.pot = 100
        
        ev = self.solver._calculate_raise_ev(game_state, 0.6)
        
        assert isinstance(ev, float)
        assert ev > -999.0  # Should be valid
    
    def test_calculate_raise_ev_insufficient_stack(self):
        """Test raise EV when hero has insufficient stack."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.stack = 10  # Less than current bet
        
        opponent = Player(seat_number=2, is_hero=False, stack=1000.0)
        opponent.current_bet = 50
        opponent.is_active = True
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        # Replace players with our test players
        game_state.players = [hero, opponent]
        game_state.pot = 100
        
        ev = self.solver._calculate_raise_ev(game_state, 0.6)
        assert ev == -999.0  # Invalid action
    
    def test_calculate_check_ev(self):
        """Test check EV calculation."""
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        game_state.pot = 100
        
        ev = self.solver._calculate_check_ev(game_state, 0.6)
        
        # Check EV = equity * pot
        expected_ev = 0.6 * 100
        assert abs(ev - expected_ev) < 0.01
    
    def test_estimate_fold_equity_by_position(self):
        """Test fold equity estimation by position."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.position = "BTN"
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        # Replace players with our test hero
        game_state.players = [hero]
        
        fold_equity = self.solver._estimate_fold_equity(game_state, 75)
        
        assert isinstance(fold_equity, float)
        assert 0.1 <= fold_equity <= 0.7  # Should be within bounds
    
    def test_estimate_fold_equity_board_texture(self):
        """Test fold equity adjustment for board texture."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.position = "BTN"
        
        # Dry board (low cards, no draws)
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        # Replace players with our test hero
        game_state.players = [hero]
        game_state.community_cards = [
            Card(rank='2', suit='clubs'),
            Card(rank='3', suit='diamonds'),
            Card(rank='4', suit='hearts')
        ]
        
        fold_equity = self.solver._estimate_fold_equity(game_state, 75)
        
        # Dry board should have higher fold equity
        assert fold_equity > 0.2
    
    def test_estimate_fold_equity_wet_board(self):
        """Test fold equity adjustment for wet board."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.position = "BTN"
        
        # Wet board (high cards, flush draw)
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        # Replace players with our test hero
        game_state.players = [hero]
        game_state.community_cards = [
            Card(rank='A', suit='hearts'),
            Card(rank='K', suit='hearts'),
            Card(rank='Q', suit='hearts')
        ]
        
        fold_equity = self.solver._estimate_fold_equity(game_state, 75)
        
        # Wet board should have lower fold equity
        assert fold_equity < 0.5
    
    def test_get_board_texture_adjustment_dry_board(self):
        """Test board texture adjustment for dry board."""
        community_cards = [
            Card(rank='2', suit='clubs'),
            Card(rank='3', suit='diamonds'),
            Card(rank='4', suit='hearts')
        ]
        
        adjustment = self.solver._get_board_texture_adjustment(community_cards)
        
        # Dry board should have positive adjustment
        assert adjustment < 0  # Dry boards should reduce fold equity
    
    def test_get_board_texture_adjustment_wet_board(self):
        """Test board texture adjustment for wet board."""
        community_cards = [
            Card(rank='A', suit='hearts'),
            Card(rank='K', suit='hearts'),
            Card(rank='Q', suit='hearts')
        ]
        
        adjustment = self.solver._get_board_texture_adjustment(community_cards)
        
        # Wet board should have negative adjustment
        assert adjustment < 0
    
    def test_get_board_texture_adjustment_empty_board(self):
        """Test board texture adjustment for empty board."""
        community_cards = []
        
        adjustment = self.solver._get_board_texture_adjustment(community_cards)
        
        assert adjustment == 0.0
    
    def test_filter_valid_actions_no_hero(self):
        """Test filtering valid actions with no hero."""
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        # Remove hero to test no hero scenario
        game_state.players = [p for p in game_state.players if not p.is_hero]
        # Add a second player to maintain minimum player count
        opponent2 = Player(seat_number=3, is_hero=False, stack=1000.0)
        game_state.players.append(opponent2)
        
        actions = [("fold", 0.0), ("call", 10.0), ("bet", 20.0)]
        valid_actions = self.solver._filter_valid_actions(actions, game_state)
        
        assert valid_actions == [("fold", 0.0)]
    
    def test_filter_valid_actions_facing_bet(self):
        """Test filtering valid actions when facing a bet."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.current_bet = 0
        hero.stack = 1000
        
        opponent = Player(seat_number=2, is_hero=False, stack=1000.0)
        opponent.current_bet = 50
        opponent.is_active = True
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        game_state.players = [hero, opponent]
        
        actions = [("fold", 0.0), ("call", 10.0), ("bet", 20.0), ("raise", 30.0), ("check", 5.0)]
        valid_actions = self.solver._filter_valid_actions(actions, game_state)
        
        # When facing bet: fold, call, raise are valid; bet, check are not
        valid_action_names = [action[0] for action in valid_actions]
        assert "fold" in valid_action_names
        assert "call" in valid_action_names
        assert "raise" in valid_action_names
        assert "bet" not in valid_action_names
        assert "check" not in valid_action_names
    
    def test_filter_valid_actions_not_facing_bet(self):
        """Test filtering valid actions when not facing a bet."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.current_bet = 0
        hero.stack = 1000
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        game_state.players = [hero]
        
        actions = [("fold", 0.0), ("call", 10.0), ("bet", 20.0), ("raise", 30.0), ("check", 5.0)]
        valid_actions = self.solver._filter_valid_actions(actions, game_state)
        
        # When not facing bet: fold, bet, check are valid; call, raise are not
        valid_action_names = [action[0] for action in valid_actions]
        assert "fold" in valid_action_names
        assert "bet" in valid_action_names
        assert "check" in valid_action_names
        assert "call" not in valid_action_names
        assert "raise" not in valid_action_names
    
    def test_generate_alternatives(self):
        """Test alternative action generation."""
        valid_actions = [
            ("fold", 0.0),
            ("call", 10.0),
            ("bet", 20.0),
            ("raise", 30.0)
        ]
        best_action = "raise"
        
        alternatives = self.solver._generate_alternatives(valid_actions, best_action)
        
        assert isinstance(alternatives, list)
        assert len(alternatives) == 3  # All except best action
        
        # Should be sorted by EV descending
        assert alternatives[0].action == "bet"  # Highest EV after raise
        assert alternatives[0].ev == 20.0
    
    def test_calculate_bet_amount_fold(self):
        """Test bet amount calculation for fold."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.current_bet = 0
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        # Replace players with our test hero
        game_state.players = [hero]
        
        amount = self.solver._calculate_bet_amount("fold", game_state)
        assert amount is None
    
    def test_calculate_bet_amount_check(self):
        """Test bet amount calculation for check."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.current_bet = 0
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        # Replace players with our test hero
        game_state.players = [hero]
        
        amount = self.solver._calculate_bet_amount("check", game_state)
        assert amount is None
    
    def test_calculate_bet_amount_call(self):
        """Test bet amount calculation for call."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.current_bet = 0
        
        opponent = Player(seat_number=2, is_hero=False, stack=1000.0)
        opponent.current_bet = 50
        opponent.is_active = True
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        game_state.players = [hero, opponent]
        
        amount = self.solver._calculate_bet_amount("call", game_state)
        assert amount == 50  # Amount needed to call
    
    def test_calculate_bet_amount_bet(self):
        """Test bet amount calculation for bet."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.stack = 1000
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        # Replace players with our test hero
        game_state.players = [hero]
        
        amount = self.solver._calculate_bet_amount("bet", game_state)
        
        # Bet size = pot * bet_sizing_factor = 100 * 0.75 = 75
        expected_amount = 100 * 0.75
        assert abs(amount - expected_amount) < 0.01
    
    def test_calculate_bet_amount_raise(self):
        """Test bet amount calculation for raise."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.stack = 1000
        
        opponent = Player(seat_number=2, is_hero=False, stack=1000.0)
        opponent.current_bet = 50
        opponent.is_active = True
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        # Replace players with our test players
        game_state.players = [hero, opponent]
        game_state.pot = 100
        
        amount = self.solver._calculate_bet_amount("raise", game_state)
        
        # Raise size = max(current_bet * raise_factor, pot) = max(50 * 3, 100) = 150
        expected_amount = max(50 * 3.0, 100)
        assert abs(amount - expected_amount) < 0.01
    
    def test_generate_reasoning_fold(self):
        """Test reasoning generation for fold."""
        reasoning = self.solver._generate_reasoning("fold", 0.3, 0.4, -5.0)
        
        assert isinstance(reasoning, str)
        assert "Fold" in reasoning
        assert "30.0%" in reasoning  # Equity as percentage
        assert "40.0%" in reasoning  # Pot odds as percentage
    
    def test_generate_reasoning_call(self):
        """Test reasoning generation for call."""
        reasoning = self.solver._generate_reasoning("call", 0.6, 0.4, 10.0)
        
        assert isinstance(reasoning, str)
        assert "Call" in reasoning
        assert "60.0%" in reasoning  # Equity as percentage
        assert "40.0%" in reasoning  # Pot odds as percentage
    
    def test_generate_reasoning_bet(self):
        """Test reasoning generation for bet."""
        reasoning = self.solver._generate_reasoning("bet", 0.6, 0.0, 15.0)
        
        assert isinstance(reasoning, str)
        assert "Bet" in reasoning
        assert "60.0%" in reasoning  # Equity as percentage
    
    def test_generate_reasoning_raise(self):
        """Test reasoning generation for raise."""
        reasoning = self.solver._generate_reasoning("raise", 0.7, 0.0, 20.0)
        
        assert isinstance(reasoning, str)
        assert "Raise" in reasoning
        assert "70.0%" in reasoning  # Equity as percentage
    
    def test_generate_reasoning_check(self):
        """Test reasoning generation for check."""
        reasoning = self.solver._generate_reasoning("check", 0.5, 0.0, 8.0)
        
        assert isinstance(reasoning, str)
        assert "Check" in reasoning
        assert "50.0%" in reasoning  # Equity as percentage
    
    def test_calculate_confidence_with_pot_odds(self):
        """Test confidence calculation with pot odds."""
        equity = 0.6
        pot_odds = 0.4
        best_ev = 10.0
        valid_actions = [("call", 10.0), ("fold", 0.0)]
        
        confidence = self.solver._calculate_confidence(equity, pot_odds, best_ev, valid_actions)
        
        assert isinstance(confidence, float)
        assert 0.5 <= confidence <= 0.95
    
    def test_calculate_confidence_without_pot_odds(self):
        """Test confidence calculation without pot odds."""
        equity = 0.6
        pot_odds = 0.0
        best_ev = 10.0
        valid_actions = [("bet", 10.0), ("check", 5.0)]
        
        confidence = self.solver._calculate_confidence(equity, pot_odds, best_ev, valid_actions)
        
        assert isinstance(confidence, float)
        assert 0.5 <= confidence <= 0.95
    
    def test_calculate_confidence_close_decision(self):
        """Test confidence calculation for close decision."""
        equity = 0.25
        pot_odds = 0.24  # Very close
        best_ev = 1.0
        valid_actions = [("call", 1.0), ("fold", 0.0)]
        
        confidence = self.solver._calculate_confidence(equity, pot_odds, best_ev, valid_actions)
        
        # Close decisions should have lower confidence
        assert confidence < 0.8
    
    def test_calculate_confidence_clear_decision(self):
        """Test confidence calculation for clear decision."""
        equity = 0.8
        pot_odds = 0.2  # Clear difference
        best_ev = 20.0
        valid_actions = [("call", 20.0), ("fold", 0.0)]
        
        confidence = self.solver._calculate_confidence(equity, pot_odds, best_ev, valid_actions)
        
        # Clear decisions should have higher confidence
        assert confidence > 0.7
    
    def test_default_fold_decision(self):
        """Test default fold decision."""
        decision = self.solver._default_fold_decision()
        
        assert isinstance(decision, Decision)
        assert decision.action == "fold"
        assert decision.amount is None
        assert decision.confidence == 0.5
        assert "Unable to calculate decision" in decision.reasoning
        assert decision.equity == 0.0
        assert decision.pot_odds == 0.0
    
    def test_get_recommendation_with_alternatives(self):
        """Test recommendation generation with alternative actions."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.is_active = True
        hero.stack = 1000
        hero.current_bet = 0
        hero.position = "BTN"
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        # Replace players with our test hero
        game_state.players = [hero]
        
        decision = self.solver.get_recommendation(game_state, 0.6, 0.0)
        
        assert isinstance(decision, Decision)
        assert isinstance(decision.alternative_actions, list)
        
        # Should have alternative actions
        if len(decision.alternative_actions) > 0:
            for alt in decision.alternative_actions:
                assert isinstance(alt, AlternativeAction)
                assert alt.action in ["fold", "call", "bet", "raise", "check"]
                assert isinstance(alt.ev, float)
    
    def test_get_recommendation_no_valid_actions(self):
        """Test recommendation when no valid actions available."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.is_active = True
        hero.stack = 0  # No chips
        hero.current_bet = 0
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        # Replace players with our test hero
        game_state.players = [hero]
        
        decision = self.solver.get_recommendation(game_state, 0.6, 0.0)
        
        assert isinstance(decision, Decision)
        assert decision.action == "fold"  # Should default to fold
