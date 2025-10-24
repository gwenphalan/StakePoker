#!/usr/bin/env python3
"""
Configuration and utilities for advisor module tests.

Provides common test fixtures, mocks, and helper functions.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, List, Any

try:
    from src.models.card import Card
    from src.models.player import Player
    from src.models.game_state import GameState
    from src.models.table_info import TableInfo
    from src.models.decision import Decision, AlternativeAction
except ImportError:
    # For testing purposes, create minimal mock classes
    class Card:
        def __init__(self, rank: str, suit: str):
            self.rank = rank
            self.suit = suit
    
    class Player:
        def __init__(self, seat_number: int, is_hero: bool = False, stack: float = 1000.0):
            self.seat_number = seat_number
            self.is_hero = is_hero
            self.is_active = True
            self.timer_state = None
            self.position = None
            self.stack = stack
            self.current_bet = 0.0
            self.hole_cards = []
            self.is_dealer = False
    
    class GameState:
        def __init__(self):
            self.players = []
            self.pot = 0.0
            self.phase = "preflop"
            self.community_cards = []
            self.button_position = 1
            self.active_player = None
            self.hand_id = None
            self.table_info = None
    
    class TableInfo:
        def __init__(self, bb: float, sb: float):
            self.bb = bb
            self.sb = sb
    
    class Decision:
        def __init__(self, action: str, amount: float = None, confidence: float = 0.0, 
                     reasoning: str = "", equity: float = 0.0, pot_odds: float = 0.0,
                     alternative_actions: list = None):
            self.action = action
            self.amount = amount
            self.confidence = confidence
            self.reasoning = reasoning
            self.equity = equity
            self.pot_odds = pot_odds
            self.alternative_actions = alternative_actions or []
    
    class AlternativeAction:
        def __init__(self, action: str, amount: float = None, ev: float = 0.0):
            self.action = action
            self.amount = amount
            self.ev = ev


class AdvisorTestFixtures:
    """Common test fixtures for advisor module tests."""
    
    @staticmethod
    def create_test_cards() -> List[Card]:
        """Create common test cards."""
        return [
            Card(rank='A', suit='hearts'),
            Card(rank='K', suit='spades'),
            Card(rank='Q', suit='hearts'),
            Card(rank='J', suit='spades'),
            Card(rank='T', suit='hearts'),
            Card(rank='9', suit='spades'),
            Card(rank='8', suit='hearts'),
            Card(rank='7', suit='spades'),
            Card(rank='6', suit='hearts'),
            Card(rank='5', suit='spades'),
            Card(rank='4', suit='hearts'),
            Card(rank='3', suit='spades'),
            Card(rank='2', suit='hearts')
        ]
    
    @staticmethod
    def create_test_hero() -> Player:
        """Create a test hero player."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        hero.is_active = True
        hero.timer_state = "purple"
        hero.position = "BTN"
        hero.hole_cards = [
            Card(rank='A', suit='hearts'),
            Card(rank='K', suit='spades')
        ]
        return hero
    
    @staticmethod
    def create_test_opponent(seat_number: int = 2) -> Player:
        """Create a test opponent player."""
        opponent = Player(seat_number=seat_number, is_hero=False, stack=1000.0)
        opponent.is_active = True
        opponent.position = "UTG"
        return opponent
    
    @staticmethod
    def create_minimal_game_state() -> GameState:
        """Create a minimal valid game state with required fields."""
        hero = Player(seat_number=1, is_hero=True, stack=1000.0)
        opponent = Player(seat_number=2, is_hero=False, stack=1000.0)
        
        return GameState(
            players=[hero, opponent],
            pot=0.0,
            phase="preflop",
            button_position=1,
            table_info=TableInfo(bb=25.0, sb=12.5),
            community_cards=[]
        )
    
    @staticmethod
    def create_test_game_state() -> GameState:
        """Create a test game state."""
        hero = AdvisorTestFixtures.create_test_hero()
        opponent = AdvisorTestFixtures.create_test_opponent()
        
        return GameState(
            players=[hero, opponent],
            pot=100.0,
            phase="preflop",
            button_position=1,
            table_info=TableInfo(bb=25.0, sb=12.5),
            community_cards=[]
        )
    
    @staticmethod
    def create_test_postflop_game_state() -> GameState:
        """Create a test postflop game state."""
        hero = AdvisorTestFixtures.create_test_hero()
        opponent = AdvisorTestFixtures.create_test_opponent()
        
        return GameState(
            players=[hero, opponent],
            pot=200.0,
            phase="flop",
            button_position=1,
            table_info=TableInfo(bb=25.0, sb=12.5),
            community_cards=[
                Card(rank='A', suit='clubs'),
                Card(rank='K', suit='hearts'),
                Card(rank='Q', suit='diamonds')
            ]
        )
    
    @staticmethod
    def create_test_gto_chart_data() -> Dict[str, Any]:
        """Create test GTO chart data."""
        return {
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
    
    @staticmethod
    def create_test_gto_metadata() -> Dict[str, Any]:
        """Create test GTO chart metadata."""
        return {
            "format": "poker_range",
            "solver": "PioSolver",
            "stack_depth": "100bb",
            "source": "test_data"
        }
    
    @staticmethod
    def create_temp_gto_charts() -> Path:
        """Create temporary GTO chart files for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create metadata
        metadata_path = temp_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(AdvisorTestFixtures.create_test_gto_metadata(), f)
        
        # Create position files
        positions = ["UTG", "MP", "CO", "BTN", "SB", "BB"]
        chart_data = AdvisorTestFixtures.create_test_gto_chart_data()
        
        for position in positions:
            # 8-max file
            file_path = temp_dir / f"{position}.json"
            with open(file_path, 'w') as f:
                json.dump(chart_data, f)
            
            # 6-max file (except BB)
            if position != "BB":
                file_path = temp_dir / f"{position}_6max.json"
                with open(file_path, 'w') as f:
                    json.dump(chart_data, f)
        
        return temp_dir
    
    @staticmethod
    def create_test_decision() -> Decision:
        """Create a test decision."""
        return Decision(
            action="bet",
            amount=75.0,
            confidence=0.85,
            reasoning="Test decision",
            equity=0.6,
            pot_odds=0.2,
            alternative_actions=[
                AlternativeAction(action="call", amount=50.0, ev=10.0),
                AlternativeAction(action="fold", amount=None, ev=0.0)
            ]
        )


class MockAdvisorComponents:
    """Mock advisor components for testing."""
    
    @staticmethod
    def mock_gto_loader():
        """Create a mock GTO loader."""
        mock_loader = MagicMock()
        mock_loader.get_preflop_decision.return_value = ("raise", 2.5, "GTO opening range")
        mock_loader.get_opening_range.return_value = ["AA", "KK", "QQ", "JJ", "TT"]
        mock_loader.get_3bet_range.return_value = ["AA", "KK", "QQ", "AKs", "AKo"]
        mock_loader.get_calling_range.return_value = ["JJ", "TT", "99", "88"]
        mock_loader.get_blind_defense_range.return_value = ["22+", "A2s+", "K2s+"]
        return mock_loader
    
    @staticmethod
    def mock_equity_calculator():
        """Create a mock equity calculator."""
        mock_calc = MagicMock()
        mock_calc.calculate_equity_vs_range.return_value = 0.6
        mock_calc.calculate_equity_vs_specific_hand.return_value = 0.7
        mock_calc.calculate_pot_odds.return_value = 0.25
        mock_calc.calculate_implied_odds.return_value = 0.2
        return mock_calc
    
    @staticmethod
    def mock_range_estimator():
        """Create a mock range estimator."""
        mock_estimator = MagicMock()
        mock_estimator.estimate_ranges.return_value = ["AA", "KK", "QQ", "JJ", "TT"]
        mock_estimator.estimate_player_range.return_value = ["AA", "KK", "QQ"]
        return mock_estimator
    
    @staticmethod
    def mock_postflop_solver():
        """Create a mock postflop solver."""
        mock_solver = MagicMock()
        mock_solver.get_recommendation.return_value = AdvisorTestFixtures.create_test_decision()
        return mock_solver


class TestHelpers:
    """Helper functions for advisor tests."""
    
    @staticmethod
    def assert_valid_decision(decision: Decision) -> None:
        """Assert that a decision is valid."""
        assert isinstance(decision, Decision)
        assert decision.action in ["fold", "call", "bet", "raise", "check"]
        assert 0.0 <= decision.confidence <= 1.0
        assert isinstance(decision.reasoning, str)
        assert len(decision.reasoning) > 0
        assert 0.0 <= decision.equity <= 1.0
        assert 0.0 <= decision.pot_odds <= 1.0
    
    @staticmethod
    def assert_valid_hand_notation(hand: str) -> None:
        """Assert that a hand notation is valid."""
        assert isinstance(hand, str)
        assert len(hand) >= 2
        assert len(hand) <= 3
        
        # Check for valid ranks
        valid_ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        for char in hand:
            if char in valid_ranks:
                continue
            elif char in ['s', 'o']:
                continue
            else:
                assert False, f"Invalid character in hand notation: {char}"
    
    @staticmethod
    def assert_valid_range(range_list: List[str]) -> None:
        """Assert that a range list is valid."""
        assert isinstance(range_list, list)
        for hand in range_list:
            TestHelpers.assert_valid_hand_notation(hand)
    
    @staticmethod
    def assert_valid_game_state(game_state: GameState) -> None:
        """Assert that a game state is valid."""
        assert isinstance(game_state, GameState)
        assert isinstance(game_state.players, list)
        assert len(game_state.players) > 0
        assert isinstance(game_state.pot, (int, float))
        assert game_state.pot >= 0
        assert isinstance(game_state.phase, str)
        assert game_state.phase in ["preflop", "flop", "turn", "river"]
        assert isinstance(game_state.community_cards, list)
        assert len(game_state.community_cards) <= 5


# Pytest fixtures
@pytest.fixture
def test_cards():
    """Fixture for test cards."""
    return AdvisorTestFixtures.create_test_cards()


@pytest.fixture
def test_hero():
    """Fixture for test hero player."""
    return AdvisorTestFixtures.create_test_hero()


@pytest.fixture
def test_opponent():
    """Fixture for test opponent player."""
    return AdvisorTestFixtures.create_test_opponent()


@pytest.fixture
def test_game_state():
    """Fixture for test game state."""
    return AdvisorTestFixtures.create_test_game_state()


@pytest.fixture
def test_postflop_game_state():
    """Fixture for test postflop game state."""
    return AdvisorTestFixtures.create_test_postflop_game_state()


@pytest.fixture
def temp_gto_charts():
    """Fixture for temporary GTO charts."""
    temp_dir = AdvisorTestFixtures.create_temp_gto_charts()
    yield temp_dir
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_gto_loader():
    """Fixture for mock GTO loader."""
    return MockAdvisorComponents.mock_gto_loader()


@pytest.fixture
def mock_equity_calculator():
    """Fixture for mock equity calculator."""
    return MockAdvisorComponents.mock_equity_calculator()


@pytest.fixture
def mock_range_estimator():
    """Fixture for mock range estimator."""
    return MockAdvisorComponents.mock_range_estimator()


@pytest.fixture
def mock_postflop_solver():
    """Fixture for mock postflop solver."""
    return MockAdvisorComponents.mock_postflop_solver()


@pytest.fixture
def test_decision():
    """Fixture for test decision."""
    return AdvisorTestFixtures.create_test_decision()
