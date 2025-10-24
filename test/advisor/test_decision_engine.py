#!/usr/bin/env python3
"""
Unit tests for DecisionEngine.

Tests the central decision engine that coordinates all advisor components.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.advisor.decision_engine import DecisionEngine
from src.models.game_state import GameState
from src.models.player import Player
from src.models.card import Card
from src.models.table_info import TableInfo
from src.models.decision import Decision
from test.advisor.test_config import AdvisorTestFixtures


class TestDecisionEngine:
    """Test cases for DecisionEngine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('src.advisor.decision_engine.Settings') as mock_settings, \
             patch('src.advisor.decision_engine.GTOChartLoader') as mock_gto_loader, \
             patch('src.advisor.decision_engine.EquityCalculator') as mock_equity_calc, \
             patch('src.advisor.decision_engine.RangeEstimator') as mock_range_est, \
             patch('src.advisor.decision_engine.PostflopSolver') as mock_postflop_solver:
            
            mock_settings_instance = mock_settings.return_value
            mock_settings_instance.create.return_value = None
            mock_settings_instance.get.side_effect = lambda key: {
                "advisor.decision.min_confidence": 0.7,
                "advisor.decision.equity_buffer": 0.05,
                "advisor.decision.preflop_base_confidence": 0.9
            }.get(key, 0.7)
            
            self.mock_gto_loader = mock_gto_loader.return_value
            self.mock_equity_calc = mock_equity_calc.return_value
            self.mock_range_est = mock_range_est.return_value
            self.mock_postflop_solver = mock_postflop_solver.return_value
            
            self.engine = DecisionEngine()
    
    def test_init(self):
        """Test DecisionEngine initialization."""
        assert self.engine.min_confidence == 0.7
        assert self.engine.equity_buffer == 0.05
        assert self.engine.preflop_base_confidence == 0.9
        assert self.engine.gto_loader == self.mock_gto_loader
        assert self.engine.equity_calculator == self.mock_equity_calc
        assert self.engine.range_estimator == self.mock_range_est
        assert self.engine.postflop_solver == self.mock_postflop_solver
    
    def test_get_recommendation_no_hero(self):
        """Test recommendation when no hero found."""
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        # Remove hero to test no hero scenario
        game_state.players = [p for p in game_state.players if not p.is_hero]
        # Add a second player to maintain minimum player count
        opponent2 = Player(seat_number=3, is_hero=False, stack=1000.0)
        game_state.players.append(opponent2)
        
        decision = self.engine.get_recommendation(game_state)
        
        assert decision is None
    
    def test_get_recommendation_not_hero_turn(self):
        """Test recommendation when not hero's turn."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.timer_state = None  # Not hero's turn
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        game_state.players = [hero]
        
        decision = self.engine.get_recommendation(game_state)
        
        assert decision is None
    
    def test_get_recommendation_invalid_hole_cards(self):
        """Test recommendation with invalid hole cards."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.timer_state = "active"
        hero.hole_cards = [Card(rank='A', suit='hearts')]  # Only 1 card
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        game_state.players = [hero]
        
        decision = self.engine.get_recommendation(game_state)
        
        assert decision is None
    
    def test_get_recommendation_preflop(self):
        """Test preflop recommendation."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.timer_state = "active"
        hero.hole_cards = [
            Card(rank='A', suit='hearts'),
            Card(rank='K', suit='spades')
        ]
        hero.position = "UTG"
        hero.stack = 1000
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        game_state.players = [hero]
        game_state.phase = "preflop"
        game_state.table_info = TableInfo(bb=25, sb=12.5)
        
        # Mock GTO loader response
        self.mock_gto_loader.get_preflop_decision.return_value = ("raise", 2.5, "GTO opening range")
        
        decision = self.engine.get_recommendation(game_state)
        
        assert isinstance(decision, Decision)
        assert decision.action == "raise"
        assert decision.amount == 62.5  # 2.5 * 25 BB
        assert decision.confidence > 0.5
        assert "GTO opening range" in decision.reasoning
    
    def test_get_recommendation_postflop(self):
        """Test postflop recommendation."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.timer_state = "active"
        hero.hole_cards = [
            Card(rank='A', suit='hearts'),
            Card(rank='K', suit='spades')
        ]
        hero.position = "BTN"
        hero.stack = 1000
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        game_state.players = [hero]
        game_state.phase = "flop"
        game_state.community_cards = [
            Card(rank='A', suit='clubs'),
            Card(rank='K', suit='hearts'),
            Card(rank='Q', suit='diamonds')
        ]
        
        # Mock advisor components
        self.mock_range_est.estimate_ranges.return_value = ["AA", "KK", "QQ"]
        self.mock_equity_calc.calculate_equity_vs_range.return_value = 0.8
        self.mock_postflop_solver.get_recommendation.return_value = Decision(
            action="bet",
            amount=75.0,
            confidence=0.85,
            reasoning="Strong equity",
            equity=0.8,
            pot_odds=0.0
        )
        
        decision = self.engine.get_recommendation(game_state)
        
        assert isinstance(decision, Decision)
        assert decision.action == "bet"
        assert decision.amount == 75.0
        assert decision.confidence == 0.85
        assert decision.equity == 0.8
    
    def test_preflop_decision_no_hero(self):
        """Test preflop decision with no hero."""
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        # Remove hero to test no hero scenario
        game_state.players = [p for p in game_state.players if not p.is_hero]
        # Add a second player to maintain minimum player count
        opponent2 = Player(seat_number=3, is_hero=False, stack=1000.0)
        game_state.players.append(opponent2)
        
        decision = self.engine._preflop_decision(game_state)
        
        assert isinstance(decision, Decision)
        assert decision.action == "fold"
        assert "Hero not found" in decision.reasoning
    
    def test_preflop_decision_invalid_hand_format(self):
        """Test preflop decision with invalid hand format."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.hole_cards = [
            Card(rank='INVALID', suit='hearts'),
            Card(rank='K', suit='spades')
        ]
        hero.position = "UTG"
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        game_state.players = [hero]
        game_state.table_info = TableInfo(bb=25, sb=12.5)
        
        decision = self.engine._preflop_decision(game_state)
        
        assert isinstance(decision, Decision)
        assert decision.action == "fold"
        assert "Invalid hand format" in decision.reasoning
    
    def test_preflop_decision_no_position(self):
        """Test preflop decision with no position."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.hole_cards = [
            Card(rank='A', suit='hearts'),
            Card(rank='K', suit='spades')
        ]
        hero.position = None
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        game_state.players = [hero]
        game_state.table_info = TableInfo(bb=25, sb=12.5)
        
        decision = self.engine._preflop_decision(game_state)
        
        assert isinstance(decision, Decision)
        assert decision.action == "fold"
        assert "Position unknown" in decision.reasoning
    
    def test_preflop_decision_success(self):
        """Test successful preflop decision."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.hole_cards = [
            Card(rank='A', suit='hearts'),
            Card(rank='K', suit='spades')
        ]
        hero.position = "UTG"
        hero.stack = 1000
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        game_state.players = [hero]
        game_state.table_info = TableInfo(bb=25, sb=12.5)
        
        # Mock GTO loader response
        self.mock_gto_loader.get_preflop_decision.return_value = ("raise", 2.5, "GTO opening range")
        
        decision = self.engine._preflop_decision(game_state)
        
        assert isinstance(decision, Decision)
        assert decision.action == "raise"
        assert decision.amount == 62.5  # 2.5 * 25 BB
        assert decision.confidence > 0.5
        assert decision.equity is None  # Preflop equity not calculated
        assert decision.pot_odds is None
    
    def test_preflop_decision_gto_error(self):
        """Test preflop decision when GTO loader raises error."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.hole_cards = [
            Card(rank='A', suit='hearts'),
            Card(rank='K', suit='spades')
        ]
        hero.position = "UTG"
        hero.stack = 1000
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        game_state.players = [hero]
        game_state.table_info = TableInfo(bb=25, sb=12.5)
        
        # Mock GTO loader to raise exception
        self.mock_gto_loader.get_preflop_decision.side_effect = Exception("GTO error")
        
        decision = self.engine._preflop_decision(game_state)
        
        assert isinstance(decision, Decision)
        assert decision.action == "fold"
        assert "Error: GTO error" in decision.reasoning
    
    def test_postflop_decision_no_hero(self):
        """Test postflop decision with no hero."""
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        # Remove hero to test no hero scenario
        game_state.players = [p for p in game_state.players if not p.is_hero]
        # Add a second player to maintain minimum player count
        opponent2 = Player(seat_number=3, is_hero=False, stack=1000.0)
        game_state.players.append(opponent2)
        
        decision = self.engine._postflop_decision(game_state)
        
        assert isinstance(decision, Decision)
        assert decision.action == "fold"
        assert "Hero not found" in decision.reasoning
    
    def test_postflop_decision_success(self):
        """Test successful postflop decision."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.hole_cards = [
            Card(rank='A', suit='hearts'),
            Card(rank='K', suit='spades')
        ]
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        game_state.players = [hero]
        game_state.community_cards = [
            Card(rank='A', suit='clubs'),
            Card(rank='K', suit='hearts'),
            Card(rank='Q', suit='diamonds')
        ]
        
        # Mock advisor components
        self.mock_range_est.estimate_ranges.return_value = ["AA", "KK", "QQ"]
        self.mock_equity_calc.calculate_equity_vs_range.return_value = 0.8
        self.mock_postflop_solver.get_recommendation.return_value = Decision(
            action="bet",
            amount=75.0,
            confidence=0.85,
            reasoning="Strong equity",
            equity=0.8,
            pot_odds=0.0
        )
        
        decision = self.engine._postflop_decision(game_state)
        
        assert isinstance(decision, Decision)
        assert decision.action == "bet"
        assert decision.amount == 75.0
        assert decision.confidence == 0.85
        assert decision.equity == 0.8
    
    def test_postflop_decision_no_opponent_ranges(self):
        """Test postflop decision when no opponent ranges estimated."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.hole_cards = [
            Card(rank='A', suit='hearts'),
            Card(rank='K', suit='spades')
        ]
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        game_state.players = [hero]
        game_state.community_cards = [
            Card(rank='A', suit='clubs'),
            Card(rank='K', suit='hearts'),
            Card(rank='Q', suit='diamonds')
        ]
        
        # Mock range estimator to return empty ranges
        self.mock_range_est.estimate_ranges.return_value = []
        self.mock_equity_calc.calculate_equity_vs_range.return_value = 0.5
        self.mock_postflop_solver.get_recommendation.return_value = Decision(
            action="check",
            amount=None,
            confidence=0.7,
            reasoning="Default range",
            equity=0.5,
            pot_odds=0.0
        )
        
        decision = self.engine._postflop_decision(game_state)
        
        assert isinstance(decision, Decision)
        assert decision.action == "check"
        assert decision.equity == 0.5
    
    def test_postflop_decision_error(self):
        """Test postflop decision when error occurs."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.hole_cards = [
            Card(rank='A', suit='hearts'),
            Card(rank='K', suit='spades')
        ]
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        game_state.players = [hero]
        game_state.community_cards = [
            Card(rank='A', suit='clubs'),
            Card(rank='K', suit='hearts'),
            Card(rank='Q', suit='diamonds')
        ]
        
        # Mock range estimator to raise exception
        self.mock_range_est.estimate_ranges.side_effect = Exception("Range estimation error")
        
        decision = self.engine._postflop_decision(game_state)
        
        assert isinstance(decision, Decision)
        assert decision.action == "fold"
        assert "Error: Range estimation error" in decision.reasoning
    
    def test_get_preflop_context_opening(self):
        """Test preflop context detection for opening."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.position = "UTG"
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        game_state.players = [hero]
        game_state.table_info = TableInfo(bb=25, sb=12.5)
        
        context = self.engine._get_preflop_context(game_state)
        assert context == "opening"
    
    def test_get_preflop_context_vs_open(self):
        """Test preflop context detection for vs_open."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.position = "CO"
        
        opponent = Player(seat_number=2, is_hero=False, stack=1000.0)
        opponent.current_bet = 50  # Raised
        opponent.is_active = True
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        game_state.players = [hero, opponent]
        game_state.table_info = TableInfo(bb=25, sb=12.5)
        
        context = self.engine._get_preflop_context(game_state)
        assert context == "vs_open"
    
    def test_get_preflop_context_vs_3bet(self):
        """Test preflop context detection for vs_3bet."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.position = "BTN"
        
        opponent1 = Player(seat_number=2, is_hero=False, stack=1000.0)
        opponent1.current_bet = 50
        opponent1.is_active = True
        
        opponent2 = Player(seat_number=3, is_hero=False, stack=1000.0)
        opponent2.current_bet = 150  # 3-bet
        opponent2.is_active = True
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        game_state.players = [hero, opponent1, opponent2]
        game_state.table_info = TableInfo(bb=25, sb=12.5)
        
        context = self.engine._get_preflop_context(game_state)
        assert context == "vs_3bet"
    
    def test_get_preflop_context_blind_defense(self):
        """Test preflop context detection for blind defense."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.position = "BB"
        
        opponent = Player(seat_number=2, is_hero=False, stack=1000.0)
        opponent.current_bet = 50  # Raised
        opponent.is_active = True
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        game_state.players = [hero, opponent]
        game_state.table_info = TableInfo(bb=25, sb=12.5)
        
        context = self.engine._get_preflop_context(game_state)
        assert context == "blind_defense"
    
    def test_calculate_preflop_confidence_fold(self):
        """Test preflop confidence calculation for fold."""
        confidence = self.engine._calculate_preflop_confidence("72o", "UTG", "fold", "opening")
        assert confidence == 0.95  # Folding decisions are very confident
    
    def test_calculate_preflop_confidence_raise(self):
        """Test preflop confidence calculation for raise."""
        confidence = self.engine._calculate_preflop_confidence("AA", "UTG", "raise", "opening")
        assert confidence > 0.8  # Should be high confidence
        assert confidence <= 0.95
    
    def test_calculate_preflop_confidence_call(self):
        """Test preflop confidence calculation for call."""
        confidence = self.engine._calculate_preflop_confidence("JJ", "CO", "call", "vs_open")
        assert confidence > 0.7  # Should be moderate confidence
        assert confidence <= 0.9
    
    def test_calculate_preflop_confidence_premium_hand(self):
        """Test preflop confidence calculation for premium hand."""
        confidence = self.engine._calculate_preflop_confidence("AA", "UTG", "raise", "opening")
        premium_confidence = self.engine._calculate_preflop_confidence("72o", "UTG", "fold", "opening")
        
        # Premium hands should have higher confidence
        assert confidence > premium_confidence
    
    def test_calculate_preflop_confidence_complex_context(self):
        """Test preflop confidence calculation for complex context."""
        simple_confidence = self.engine._calculate_preflop_confidence("AA", "UTG", "raise", "opening")
        complex_confidence = self.engine._calculate_preflop_confidence("AA", "UTG", "raise", "vs_3bet")
        
        # Complex contexts should have slightly lower confidence
        assert complex_confidence < simple_confidence
    
    def test_calculate_pot_odds_no_bet(self):
        """Test pot odds calculation when not facing a bet."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.current_bet = 0
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        game_state.players = [hero]
        game_state.pot = 100
        
        pot_odds = self.engine._calculate_pot_odds(game_state)
        assert pot_odds == 0.0
    
    def test_calculate_pot_odds_facing_bet(self):
        """Test pot odds calculation when facing a bet."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.current_bet = 0
        
        opponent = Player(seat_number=2, is_hero=False, stack=1000.0)
        opponent.current_bet = 50
        opponent.is_active = True
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        game_state.players = [hero, opponent]
        game_state.pot = 100
        
        pot_odds = self.engine._calculate_pot_odds(game_state)
        
        # Pot odds = bet_to_call / (pot + bet_to_call) = 50 / (100 + 50) = 1/3
        expected_pot_odds = 50 / (100 + 50)
        assert abs(pot_odds - expected_pot_odds) < 0.01
    
    def test_format_hand_pocket_pair(self):
        """Test hand formatting for pocket pair."""
        cards = [
            Card(rank='A', suit='hearts'),
            Card(rank='A', suit='spades')
        ]
        
        hand = self.engine._format_hand(cards)
        assert hand == "AA"
    
    def test_format_hand_suited(self):
        """Test hand formatting for suited cards."""
        cards = [
            Card(rank='A', suit='hearts'),
            Card(rank='K', suit='hearts')
        ]
        
        hand = self.engine._format_hand(cards)
        assert hand == "AKs"
    
    def test_format_hand_offsuit(self):
        """Test hand formatting for offsuit cards."""
        cards = [
            Card(rank='A', suit='hearts'),
            Card(rank='K', suit='spades')
        ]
        
        hand = self.engine._format_hand(cards)
        assert hand == "AKo"
    
    def test_format_hand_invalid_count(self):
        """Test hand formatting with invalid card count."""
        cards = [Card(rank='A', suit='hearts')]  # Only 1 card
        
        hand = self.engine._format_hand(cards)
        assert hand == ""
    
    def test_format_hand_invalid_ranks(self):
        """Test hand formatting with invalid ranks."""
        cards = [
            Card(rank='INVALID', suit='hearts'),
            Card(rank='K', suit='spades')
        ]
        
        hand = self.engine._format_hand(cards)
        assert hand == ""
    
    def test_get_default_opponent_range(self):
        """Test default opponent range."""
        default_range = self.engine._get_default_opponent_range()
        
        assert isinstance(default_range, list)
        assert "AA" in default_range
        assert "KK" in default_range
        assert "72o" not in default_range  # Should be conservative
    
    def test_default_fold_decision(self):
        """Test default fold decision."""
        decision = self.engine._default_fold_decision("Test reason")
        
        assert isinstance(decision, Decision)
        assert decision.action == "fold"
        assert decision.amount is None
        assert decision.confidence == 0.5
        assert "Default fold: Test reason" in decision.reasoning
        assert decision.equity == 0.0
        assert decision.pot_odds == 0.0
    
    def test_bet_amount_calculation_with_stack_limit(self):
        """Test bet amount calculation with stack limit."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.stack = 50  # Less than bet amount
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        game_state.players = [hero]
        game_state.table_info = TableInfo(bb=25, sb=12.5)
        
        # Mock GTO loader to return large bet size
        self.mock_gto_loader.get_preflop_decision.return_value = ("raise", 10.0, "GTO range")
        
        decision = self.engine._preflop_decision(game_state)
        
        # Bet amount should be limited by stack
        assert decision.amount == 50  # Limited by stack
        assert decision.amount <= hero.stack
    
    def test_player_count_calculation(self):
        """Test player count calculation for GTO decisions."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.timer_state = "active"
        hero.hole_cards = [
            Card(rank='A', suit='hearts'),
            Card(rank='K', suit='spades')
        ]
        hero.position = "UTG"
        hero.stack = 1000
        
        opponent1 = Player(seat_number=2, is_hero=False, stack=1000.0)
        opponent1.is_active = True
        
        opponent2 = Player(seat_number=3, is_hero=False, stack=1000.0)
        opponent2.is_active = False  # Folded
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        game_state.players = [hero, opponent1, opponent2]
        game_state.phase = "preflop"
        game_state.table_info = TableInfo(bb=25, sb=12.5)
        
        # Mock GTO loader
        self.mock_gto_loader.get_preflop_decision.return_value = ("raise", 2.5, "GTO range")
        
        decision = self.engine.get_recommendation(game_state)
        
        # Should call GTO loader with correct player count (2 active players)
        self.mock_gto_loader.get_preflop_decision.assert_called_once()
        call_args = self.mock_gto_loader.get_preflop_decision.call_args
        assert call_args[0][2] == 2  # Player count should be 2
    
    def test_equity_calculation_integration(self):
        """Test equity calculation integration in postflop decision."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.timer_state = "active"
        hero.hole_cards = [
            Card(rank='A', suit='hearts'),
            Card(rank='K', suit='spades')
        ]
        
        game_state = AdvisorTestFixtures.create_minimal_game_state()
        game_state.players = [hero]
        game_state.phase = "flop"
        game_state.community_cards = [
            Card(rank='A', suit='clubs'),
            Card(rank='K', suit='hearts'),
            Card(rank='Q', suit='diamonds')
        ]
        
        # Mock components
        self.mock_range_est.estimate_ranges.return_value = ["AA", "KK", "QQ"]
        self.mock_equity_calc.calculate_equity_vs_range.return_value = 0.8
        self.mock_postflop_solver.get_recommendation.return_value = Decision(
            action="bet",
            amount=75.0,
            confidence=0.85,
            reasoning="Strong equity",
            equity=0.8,
            pot_odds=0.0
        )
        
        decision = self.engine.get_recommendation(game_state)
        
        # Verify equity calculator was called with correct parameters
        self.mock_equity_calc.calculate_equity_vs_range.assert_called_once()
        call_args = self.mock_equity_calc.calculate_equity_vs_range.call_args
        
        assert call_args[0][0] == hero.hole_cards  # Hero cards
        assert call_args[0][1] == game_state.community_cards  # Community cards
        assert call_args[0][2] == ["AA", "KK", "QQ"]  # Opponent range
