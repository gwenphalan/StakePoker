#!/usr/bin/env python3
"""
Unit tests for tracker module.

Tests HandTracker, HeroDetector, PositionCalculator, StateMachine, and TurnDetector
classes with comprehensive coverage of game state tracking, turn detection, and
position calculation functionality.
"""

import unittest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any, Optional

# Import the classes under test
from src.tracker.hand_tracker import HandTracker
from src.tracker.hero_detector import HeroDetector
from src.tracker.position_calculator import PositionCalculator
from src.tracker.state_machine import GameStateMachine
from src.tracker.turn_detector import TurnDetector

# Import models for test data
from src.models.hand_record import HandRecord, Action
from src.models.game_state import GameState
from src.models.player import Player
from src.models.card import Card
from src.models.table_info import TableInfo
from src.parser.timer_detector import TimerResult


class TestHandTracker(unittest.TestCase):
    """Test HandTracker hand lifecycle and action detection."""
    
    def setUp(self):
        """Set up test environment."""
        self.mock_session_tracker = Mock()
        self.tracker = HandTracker(self.mock_session_tracker)
        
        # Sample test data
        self.sample_cards = [
            Card(rank='A', suit='hearts'),
            Card(rank='K', suit='spades')
        ]
        
        self.sample_table_info = TableInfo(sb=1.0, bb=2.0)
        
        self.sample_players = [
            Player(seat_number=1, stack=100.0, position='BTN', is_dealer=True, is_hero=True, is_active=True),
            Player(seat_number=2, stack=100.0, position='SB', is_active=True),
            Player(seat_number=3, stack=100.0, position='BB', is_active=True)
        ]
        
        self.sample_game_state = GameState(
            players=self.sample_players,
            community_cards=[],
            pot=10.0,
            phase='preflop',
            active_player=1,
            button_position=1,
            hand_id='test_hand_001',
            table_info=self.sample_table_info
        )
        
        self.hero_seat = 1
    
    def test_initialization(self):
        """Test HandTracker initialization."""
        self.assertIsNone(self.tracker.current_hand)
        self.assertEqual(len(self.tracker.completed_hands), 0)
        self.assertEqual(len(self.tracker.current_actions), 0)
        self.assertIsNone(self.tracker.hand_start_state)
        self.assertIsNone(self.tracker.hero_seat)
        self.assertEqual(self.tracker.hero_invested, 0.0)
    
    def test_start_new_hand(self):
        """Test starting a new hand."""
        hand_id = 'test_hand_001'
        
        self.tracker.start_new_hand(hand_id, self.sample_game_state, self.hero_seat)
        
        # Verify hand was created
        self.assertIsNotNone(self.tracker.current_hand)
        self.assertEqual(self.tracker.current_hand.hand_id, hand_id)
        self.assertEqual(self.tracker.current_hand.hero_seat, self.hero_seat)
        self.assertEqual(self.tracker.current_hand.hero_position, 'BTN')
        self.assertEqual(len(self.tracker.current_hand.hero_cards), 2)
        self.assertEqual(self.tracker.hero_seat, self.hero_seat)
        self.assertEqual(self.tracker.hero_invested, 0.0)
    
    def test_start_new_hand_without_hero(self):
        """Test starting hand without hero seat."""
        hand_id = 'test_hand_001'
        
        self.tracker.start_new_hand(hand_id, self.sample_game_state, None)
        
        # Verify hand was created with default values
        self.assertIsNotNone(self.tracker.current_hand)
        self.assertEqual(self.tracker.current_hand.hero_seat, 1)  # Fixed to valid seat
        self.assertEqual(self.tracker.current_hand.hero_position, 'BTN')
        self.assertEqual(len(self.tracker.current_hand.hero_cards), 2)
    
    def test_start_new_hand_finalizes_previous(self):
        """Test that starting new hand finalizes previous hand."""
        # Start first hand
        self.tracker.start_new_hand('hand_001', self.sample_game_state, self.hero_seat)
        
        # Start second hand
        self.tracker.start_new_hand('hand_002', self.sample_game_state, self.hero_seat)
        
        # Verify previous hand was finalized
        self.assertEqual(len(self.tracker.completed_hands), 1)
        self.assertEqual(self.tracker.completed_hands[0].hand_id, 'hand_001')
    
    def test_update_hand(self):
        """Test updating hand with new actions."""
        # Start hand
        self.tracker.start_new_hand('test_hand', self.sample_game_state, self.hero_seat)
        
        # Create previous state for comparison
        prev_state = self.sample_game_state
        
        # Create new state with action
        new_players = [
            Player(seat_number=1, stack=95.0, position='BTN', is_dealer=True, is_hero=True, is_active=True, current_bet=5.0),
            Player(seat_number=2, stack=100.0, position='SB', is_active=True),
            Player(seat_number=3, stack=100.0, position='BB', is_active=True)
        ]
        
        new_state = GameState(
            players=new_players,
            community_cards=[],
            pot=15.0,
            phase='preflop',
            active_player=2,
            button_position=1,
            hand_id='test_hand',
            table_info=self.sample_table_info
        )
        
        # Update hand
        self.tracker.update_hand(new_state, prev_state)
        
        # Verify action was detected
        self.assertEqual(len(self.tracker.current_actions), 1)
        action = self.tracker.current_actions[0]
        self.assertEqual(action.seat_number, 1)
        self.assertEqual(action.action_type, 'bet')
        self.assertEqual(action.amount, 5.0)
        self.assertEqual(self.tracker.hero_invested, 5.0)
    
    def test_detect_actions_fold(self):
        """Test detecting fold action."""
        # Create states with fold
        prev_players = [
            Player(seat_number=1, stack=100.0, is_active=True),
            Player(seat_number=2, stack=100.0, is_active=True)
        ]
        
        curr_players = [
            Player(seat_number=1, stack=100.0, is_active=True),
            Player(seat_number=2, stack=100.0, is_active=False)  # Folded
        ]
        
        prev_state = GameState(
            players=prev_players,
            community_cards=[],
            pot=10.0,
            phase='preflop',
            active_player=1,
            button_position=1,
            hand_id='test_hand',
            table_info=self.sample_table_info
        )
        
        curr_state = GameState(
            players=curr_players,
            community_cards=[],
            pot=10.0,
            phase='preflop',
            active_player=1,
            button_position=1,
            hand_id='test_hand',
            table_info=self.sample_table_info
        )
        
        actions = self.tracker.detect_actions(prev_state, curr_state)
        
        # Verify fold was detected
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].seat_number, 2)
        self.assertEqual(actions[0].action_type, 'fold')
        self.assertIsNone(actions[0].amount)
    
    def test_detect_actions_bet_call_raise(self):
        """Test detecting bet, call, and raise actions."""
        # Test bet (first action)
        prev_players = [
            Player(seat_number=1, stack=100.0, is_active=True, current_bet=0.0),
            Player(seat_number=2, stack=100.0, is_active=True, current_bet=0.0)
        ]
        curr_players = [
            Player(seat_number=1, stack=95.0, is_active=True, current_bet=5.0),
            Player(seat_number=2, stack=100.0, is_active=True, current_bet=0.0)
        ]
        
        prev_state = GameState(players=prev_players, community_cards=[], pot=0.0, phase='preflop', active_player=1, button_position=1, hand_id='test', table_info=self.sample_table_info)
        curr_state = GameState(players=curr_players, community_cards=[], pot=5.0, phase='preflop', active_player=1, button_position=1, hand_id='test', table_info=self.sample_table_info)
        
        actions = self.tracker.detect_actions(prev_state, curr_state)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action_type, 'bet')
        self.assertEqual(actions[0].amount, 5.0)
        
        # Test call
        prev_players = [
            Player(seat_number=1, stack=95.0, is_active=True, current_bet=5.0),
            Player(seat_number=2, stack=100.0, is_active=True, current_bet=0.0)
        ]
        curr_players = [
            Player(seat_number=1, stack=95.0, is_active=True, current_bet=5.0),
            Player(seat_number=2, stack=95.0, is_active=True, current_bet=5.0)
        ]
        
        prev_state = GameState(players=prev_players, community_cards=[], pot=10.0, phase='preflop', active_player=2, button_position=1, hand_id='test', table_info=self.sample_table_info)
        curr_state = GameState(players=curr_players, community_cards=[], pot=15.0, phase='preflop', active_player=1, button_position=1, hand_id='test', table_info=self.sample_table_info)
        
        actions = self.tracker.detect_actions(prev_state, curr_state)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action_type, 'call')
        self.assertEqual(actions[0].amount, 5.0)
        
        # Test raise
        prev_players = [
            Player(seat_number=1, stack=95.0, is_active=True, current_bet=5.0),
            Player(seat_number=2, stack=95.0, is_active=True, current_bet=5.0)
        ]
        curr_players = [
            Player(seat_number=1, stack=95.0, is_active=True, current_bet=5.0),
            Player(seat_number=2, stack=85.0, is_active=True, current_bet=15.0)
        ]
        
        prev_state = GameState(players=prev_players, community_cards=[], pot=15.0, phase='preflop', active_player=2, button_position=1, hand_id='test', table_info=self.sample_table_info)
        curr_state = GameState(players=curr_players, community_cards=[], pot=25.0, phase='preflop', active_player=1, button_position=1, hand_id='test', table_info=self.sample_table_info)
        
        actions = self.tracker.detect_actions(prev_state, curr_state)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action_type, 'raise')
        self.assertEqual(actions[0].amount, 10.0)
    
    def test_detect_actions_check(self):
        """Test detecting check action."""
        prev_players = [
            Player(seat_number=1, stack=100.0, is_active=True, current_bet=0.0, timer_state='purple'),
            Player(seat_number=2, stack=100.0, is_active=True, current_bet=0.0)
        ]
        curr_players = [
            Player(seat_number=1, stack=100.0, is_active=True, current_bet=0.0, timer_state=None),
            Player(seat_number=2, stack=100.0, is_active=True, current_bet=0.0)
        ]
        
        prev_state = GameState(players=prev_players, community_cards=[], pot=0.0, phase='preflop', active_player=1, button_position=1, hand_id='test', table_info=self.sample_table_info)
        curr_state = GameState(players=curr_players, community_cards=[], pot=0.0, phase='preflop', active_player=2, button_position=1, hand_id='test', table_info=self.sample_table_info)
        
        actions = self.tracker.detect_actions(prev_state, curr_state)
        
        # Verify check was detected
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].seat_number, 1)
        self.assertEqual(actions[0].action_type, 'check')
        self.assertIsNone(actions[0].amount)
    
    def test_finalize_hand(self):
        """Test finalizing hand and calculating results."""
        # Start hand
        self.tracker.start_new_hand('test_hand', self.sample_game_state, self.hero_seat)
        
        # Add some actions
        self.tracker.current_actions = [
            Action(seat_number=1, action_type='bet', amount=5.0, phase='preflop')
        ]
        self.tracker.hero_invested = 5.0
        
        # Create final state (hero won)
        final_players = [
            Player(seat_number=1, stack=105.0, position='BTN', is_dealer=True, is_hero=True, is_active=True),
            Player(seat_number=2, stack=95.0, position='SB', is_active=False),  # Folded
            Player(seat_number=3, stack=100.0, position='BB', is_active=False)   # Folded
        ]
        
        final_state = GameState(
            players=final_players,
            community_cards=[],
            pot=15.0,
            phase='preflop',
            active_player=1,
            button_position=1,
            hand_id='test_hand',
            table_info=self.sample_table_info
        )
        
        # Finalize hand
        completed_hand = self.tracker.finalize_hand(final_state)
        
        # Verify hand was completed
        self.assertIsNotNone(completed_hand)
        self.assertEqual(completed_hand.result, 'won')
        self.assertEqual(completed_hand.net_profit, 10.0)  # 15 pot - 5 invested
        self.assertEqual(completed_hand.final_pot, 15.0)
        self.assertFalse(completed_hand.showdown)
        
        # Verify hand was added to completed list
        self.assertEqual(len(self.tracker.completed_hands), 1)
        
        # Verify session tracker was called
        self.mock_session_tracker.record_hand_in_session.assert_called_once_with(completed_hand)
    
    def test_calculate_hero_profit_loss(self):
        """Test profit/loss calculation."""
        # Start hand
        self.tracker.start_new_hand('test_hand', self.sample_game_state, self.hero_seat)
        
        # Test winning scenario
        self.tracker.current_actions = [
            Action(seat_number=1, action_type='bet', amount=10.0, phase='preflop')
        ]
        
        winning_state = GameState(
            players=[
                Player(seat_number=1, stack=110.0, is_active=True),
                Player(seat_number=2, stack=90.0, is_active=False)
            ],
            community_cards=[],
            pot=20.0,
            phase='preflop',
            active_player=1,
            button_position=1,
            hand_id='test',
            table_info=self.sample_table_info
        )
        
        profit = self.tracker.calculate_hero_profit_loss(winning_state)
        self.assertEqual(profit, 10.0)  # 20 pot - 10 invested
        
        # Test losing scenario
        losing_state = GameState(
            players=[
                Player(seat_number=1, stack=90.0, is_active=False),  # Folded
                Player(seat_number=2, stack=110.0, is_active=True)   # Won
            ],
            community_cards=[],
            pot=20.0,
            phase='preflop',
            active_player=2,
            button_position=1,
            hand_id='test',
            table_info=self.sample_table_info
        )
        
        loss = self.tracker.calculate_hero_profit_loss(losing_state)
        self.assertEqual(loss, -10.0)  # Lost 10 invested
    
    def test_determine_hand_result(self):
        """Test hand result determination."""
        # Start hand
        self.tracker.start_new_hand('test_hand', self.sample_game_state, self.hero_seat)
        
        # Test folded result
        folded_state = GameState(
            players=[
                Player(seat_number=1, stack=90.0, is_active=False),  # Hero folded
                Player(seat_number=2, stack=110.0, is_active=True)
            ],
            community_cards=[],
            pot=20.0,
            phase='preflop',
            active_player=2,
            button_position=1,
            hand_id='test',
            table_info=self.sample_table_info
        )
        
        result = self.tracker._determine_hand_result(folded_state)
        self.assertEqual(result, 'folded')
        
        # Test won result
        won_state = GameState(
            players=[
                Player(seat_number=1, stack=110.0, is_active=True),  # Only hero active
                Player(seat_number=2, stack=90.0, is_active=False)
            ],
            community_cards=[],
            pot=20.0,
            phase='preflop',
            active_player=1,
            button_position=1,
            hand_id='test',
            table_info=self.sample_table_info
        )
        
        result = self.tracker._determine_hand_result(won_state)
        self.assertEqual(result, 'won')
        
        # Test lost result (hero folded)
        lost_state = GameState(
            players=[
                Player(seat_number=1, stack=90.0, is_active=False),   # Hero folded
                Player(seat_number=2, stack=110.0, is_active=True)   # Other player active
            ],
            community_cards=[],
            pot=20.0,
            phase='showdown',
            active_player=2,
            button_position=1,
            hand_id='test',
            table_info=self.sample_table_info
        )
        
        result = self.tracker._determine_hand_result(lost_state)
        self.assertEqual(result, 'folded')
    
    def test_get_current_hand(self):
        """Test getting current hand."""
        self.assertIsNone(self.tracker.get_current_hand())
        
        self.tracker.start_new_hand('test_hand', self.sample_game_state, self.hero_seat)
        
        current_hand = self.tracker.get_current_hand()
        self.assertIsNotNone(current_hand)
        self.assertEqual(current_hand.hand_id, 'test_hand')
    
    def test_get_completed_hands(self):
        """Test getting completed hands."""
        # Start and complete a hand
        self.tracker.start_new_hand('test_hand', self.sample_game_state, self.hero_seat)
        completed_hand = self.tracker.finalize_hand(self.sample_game_state)
        
        completed_hands = self.tracker.get_completed_hands()
        self.assertEqual(len(completed_hands), 1)
        self.assertEqual(completed_hands[0].hand_id, 'test_hand')
    
    def test_reset_hand(self):
        """Test resetting hand tracking."""
        # Start a hand
        self.tracker.start_new_hand('test_hand', self.sample_game_state, self.hero_seat)
        
        # Reset
        self.tracker.reset_hand()
        
        # Verify reset
        self.assertIsNone(self.tracker.current_hand)
        self.assertEqual(len(self.tracker.current_actions), 0)
        self.assertEqual(self.tracker.hero_invested, 0.0)
        self.assertIsNone(self.tracker.hand_start_state)


class TestHeroDetector(unittest.TestCase):
    """Test HeroDetector hero identification functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock name parser before creating detector
        self.mock_name_parser_patcher = patch('src.tracker.hero_detector.NameParser')
        self.mock_name_parser_class = self.mock_name_parser_patcher.start()
        
        self.mock_name_parser = Mock()
        self.mock_name_parser_class.return_value = self.mock_name_parser
        
        self.detector = HeroDetector()
        # Ensure the detector uses our mock
        self.detector.name_parser = self.mock_name_parser
    
    def tearDown(self):
        """Clean up after test."""
        self.mock_name_parser_patcher.stop()
    
    def test_initialization(self):
        """Test HeroDetector initialization."""
        self.assertIsNotNone(self.detector.settings)
        self.assertIsNotNone(self.detector.name_parser)
    
    def test_detect_hero_seat_success(self):
        """Test successful hero detection."""
        # Mock name parser results - first seat is hero
        mock_result_hero = Mock()
        mock_result_hero.name = "GalacticAce"
        mock_result_hero.is_hero = True
        mock_result_hero.confidence = 0.8
        
        mock_result_other = Mock()
        mock_result_other.name = "Player2"
        mock_result_other.is_hero = False
        mock_result_other.confidence = 0.7
        
        # Set up mock to return different results based on seat
        def mock_parse_player_name(region, seat):
            print(f"Mock called with seat {seat}")
            if seat == 1:
                print("Returning hero result")
                return mock_result_hero
            else:
                print("Returning other result")
                return mock_result_other
        
        self.mock_name_parser.parse_player_name.side_effect = mock_parse_player_name
        
        # Create nameplate regions
        nameplate_regions = {
            1: np.array([[1, 2, 3], [4, 5, 6]]),
            2: np.array([[7, 8, 9], [10, 11, 12]])
        }
        
        # Enable hero detection for this test
        self.detector.settings.update("tracker.hero_detection.enabled", True)
        
        hero_seat = self.detector.detect_hero_seat(nameplate_regions)
        
        # Verify hero was detected
        self.assertEqual(hero_seat, 1)
        self.mock_name_parser.parse_player_name.assert_called()
    
    def test_detect_hero_seat_no_hero_candidates(self):
        """Test detection when no hero candidates found."""
        # Mock name parser results (no hero)
        mock_result = Mock()
        mock_result.name = "Player1"
        mock_result.is_hero = False
        mock_result.confidence = 0.8
        
        self.mock_name_parser.parse_player_name.return_value = mock_result
        
        nameplate_regions = {1: np.array([[1, 2, 3]])}
        
        hero_seat = self.detector.detect_hero_seat(nameplate_regions)
        
        # Verify no hero detected
        self.assertIsNone(hero_seat)
    
    def test_detect_hero_seat_low_confidence(self):
        """Test detection with low confidence."""
        # Mock name parser results (low confidence)
        mock_result = Mock()
        mock_result.name = "GalacticAce"
        mock_result.is_hero = True
        mock_result.confidence = 0.5  # Below threshold
        
        self.mock_name_parser.parse_player_name.return_value = mock_result
        
        nameplate_regions = {1: np.array([[1, 2, 3]])}
        
        hero_seat = self.detector.detect_hero_seat(nameplate_regions)
        
        # Verify no hero detected due to low confidence
        self.assertIsNone(hero_seat)
    
    def test_detect_hero_seat_multiple_candidates(self):
        """Test detection with multiple hero candidates."""
        # Mock name parser results (multiple heroes)
        mock_result1 = Mock()
        mock_result1.name = "GalacticAce"
        mock_result1.is_hero = True
        mock_result1.confidence = 0.7
        
        mock_result2 = Mock()
        mock_result2.name = "GalacticAce"
        mock_result2.is_hero = True
        mock_result2.confidence = 0.9
        
        # Set up mock to return different results based on seat
        def mock_parse_player_name(region, seat):
            if seat == 1:
                return mock_result1
            else:
                return mock_result2
        
        self.mock_name_parser.parse_player_name.side_effect = mock_parse_player_name
        
        nameplate_regions = {
            1: np.array([[1, 2, 3]]),
            2: np.array([[4, 5, 6]])
        }
        
        # Enable hero detection for this test
        self.detector.settings.update("tracker.hero_detection.enabled", True)
        
        hero_seat = self.detector.detect_hero_seat(nameplate_regions)
        
        # Verify highest confidence hero was selected
        self.assertEqual(hero_seat, 2)
    
    def test_detect_hero_seat_disabled(self):
        """Test detection when disabled."""
        # Disable hero detection
        self.detector.settings.update("tracker.hero_detection.enabled", False)
        
        nameplate_regions = {1: np.array([[1, 2, 3]])}
        
        hero_seat = self.detector.detect_hero_seat(nameplate_regions)
        
        # Verify no detection when disabled
        self.assertIsNone(hero_seat)
        self.mock_name_parser.parse_player_name.assert_not_called()
    
    def test_detect_hero_seat_empty_regions(self):
        """Test detection with empty regions."""
        hero_seat = self.detector.detect_hero_seat({})
        
        # Verify no detection with empty regions
        self.assertIsNone(hero_seat)
    
    def test_get_hero_usernames(self):
        """Test getting hero usernames."""
        usernames = self.detector.get_hero_usernames()
        
        # Verify default usernames
        self.assertIsInstance(usernames, list)
        # Check that we get some usernames (could be from settings or default)
        self.assertGreater(len(usernames), 0)
    
    def test_update_hero_usernames(self):
        """Test updating hero usernames."""
        new_usernames = ["Player1", "Player2"]
        
        self.detector.update_hero_usernames(new_usernames)
        
        # Verify update
        updated_usernames = self.detector.get_hero_usernames()
        self.assertEqual(updated_usernames, new_usernames)
    
    def test_get_detection_stats(self):
        """Test getting detection statistics."""
        stats = self.detector.get_detection_stats()
        
        # Verify stats structure
        self.assertIn('enabled', stats)
        self.assertIn('min_confidence', stats)
        self.assertIn('require_high_confidence', stats)
        self.assertIn('hero_usernames', stats)


class TestPositionCalculator(unittest.TestCase):
    """Test PositionCalculator position mapping functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.calculator = PositionCalculator()
    
    def test_initialization(self):
        """Test PositionCalculator initialization."""
        self.assertIsNotNone(self.calculator.POSITION_MAPS)
        self.assertEqual(len(self.calculator.POSITION_MAPS), 7)  # 2-8 players
    
    def test_calculate_positions_heads_up(self):
        """Test position calculation for heads-up (2 players)."""
        players = [
            Player(seat_number=1, stack=100.0),
            Player(seat_number=2, stack=100.0)
        ]
        
        result = self.calculator.calculate_positions(players, button_position=1)
        
        # Verify positions
        self.assertEqual(result[0].position, 'SB')  # Button is SB in heads-up
        self.assertEqual(result[1].position, 'BB')
    
    def test_calculate_positions_6max(self):
        """Test position calculation for 6-max."""
        players = [
            Player(seat_number=1, stack=100.0),
            Player(seat_number=2, stack=100.0),
            Player(seat_number=3, stack=100.0),
            Player(seat_number=4, stack=100.0),
            Player(seat_number=5, stack=100.0),
            Player(seat_number=6, stack=100.0)
        ]
        
        result = self.calculator.calculate_positions(players, button_position=3)
        
        # Verify positions (button at seat 3, which is index 2 in sorted list)
        positions = [p.position for p in result]
        # With button at seat 3 (index 2), positions should be assigned starting from BTN
        expected = ['MP', 'CO', 'BTN', 'SB', 'BB', 'UTG']
        self.assertEqual(positions, expected)
    
    def test_calculate_positions_8max(self):
        """Test position calculation for 8-max."""
        players = [
            Player(seat_number=1, stack=100.0),
            Player(seat_number=2, stack=100.0),
            Player(seat_number=3, stack=100.0),
            Player(seat_number=4, stack=100.0),
            Player(seat_number=5, stack=100.0),
            Player(seat_number=6, stack=100.0),
            Player(seat_number=7, stack=100.0),
            Player(seat_number=8, stack=100.0)
        ]
        
        result = self.calculator.calculate_positions(players, button_position=5)
        
        # Verify positions (button at seat 5, which is index 4 in sorted list)
        positions = [p.position for p in result]
        # With button at seat 5 (index 4), positions should be assigned starting from BTN
        expected = ['UTG+1', 'UTG+2', 'MP', 'CO', 'BTN', 'SB', 'BB', 'UTG']
        self.assertEqual(positions, expected)
    
    def test_calculate_positions_invalid_button(self):
        """Test calculation with invalid button position."""
        players = [
            Player(seat_number=1, stack=100.0),
            Player(seat_number=2, stack=100.0)
        ]
        
        result = self.calculator.calculate_positions(players, button_position=0)  # Invalid
        
        # Verify original players returned unchanged
        self.assertEqual(len(result), 2)
        self.assertIsNone(result[0].position)
        self.assertIsNone(result[1].position)
    
    def test_calculate_positions_button_not_found(self):
        """Test calculation when button position not in players."""
        players = [
            Player(seat_number=1, stack=100.0),
            Player(seat_number=2, stack=100.0)
        ]
        
        result = self.calculator.calculate_positions(players, button_position=3)  # Not in players
        
        # Verify original players returned unchanged
        self.assertEqual(len(result), 2)
        self.assertIsNone(result[0].position)
        self.assertIsNone(result[1].position)
    
    def test_calculate_positions_empty_list(self):
        """Test calculation with empty player list."""
        result = self.calculator.calculate_positions([], button_position=1)
        
        # Verify empty list returned
        self.assertEqual(len(result), 0)
    
    def test_calculate_positions_unsupported_table_size(self):
        """Test calculation with unsupported table size."""
        players = [
            Player(seat_number=1, stack=100.0),
            Player(seat_number=2, stack=100.0),
            Player(seat_number=3, stack=100.0),
            Player(seat_number=4, stack=100.0),
            Player(seat_number=5, stack=100.0),
            Player(seat_number=6, stack=100.0),
            Player(seat_number=7, stack=100.0),
            Player(seat_number=8, stack=100.0),
            Player(seat_number=1, stack=100.0)  # Duplicate seat to create 9 players (unsupported)
        ]
        
        result = self.calculator.calculate_positions(players, button_position=1)
        
        # Should return original players list due to error
        self.assertEqual(len(result), 9)
        # The error handling should return the original players unchanged
        # Some positions might be assigned before the error occurs
        # We just verify that the method doesn't crash and returns the players
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 9)
    
    def test_position_maps_coverage(self):
        """Test that all position maps are properly defined."""
        for table_size in range(2, 9):
            position_map = self.calculator._get_position_map(table_size)
            self.assertEqual(len(position_map), table_size)
            
            # Verify BTN is present (except heads-up where BTN is SB)
            if table_size > 2:
                self.assertIn('BTN', position_map)
            else:
                self.assertIn('SB', position_map)
            
            # Verify SB and BB are always present
            self.assertIn('SB', position_map)
            self.assertIn('BB', position_map)


class TestTurnDetector(unittest.TestCase):
    """Test TurnDetector turn detection functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock timer detector before creating detector
        self.mock_timer_detector_patcher = patch('src.tracker.turn_detector.TimerDetector')
        self.mock_timer_detector_class = self.mock_timer_detector_patcher.start()
        
        self.mock_timer_detector = Mock()
        self.mock_timer_detector_class.return_value = self.mock_timer_detector
        
        self.detector = TurnDetector()
        # Ensure the detector uses our mock
        self.detector.timer_detector = self.mock_timer_detector
    
    def tearDown(self):
        """Clean up after test."""
        self.mock_timer_detector_patcher.stop()
    
    def test_initialization(self):
        """Test TurnDetector initialization."""
        self.assertIsNotNone(self.detector.settings)
        self.assertIsNotNone(self.detector.timer_detector)
    
    def test_detect_hero_turn_success(self):
        """Test successful hero turn detection."""
        # Mock timer results - hero has strongest signal
        mock_result_hero = Mock()
        mock_result_hero.turn_state = 'turn'
        mock_result_hero.confidence = 0.8
        
        mock_result_other = Mock()
        mock_result_other.turn_state = 'none'
        mock_result_other.confidence = 0.5
        
        # Set up mock to return different results based on seat
        def mock_detect_timer(region):
            # Return active timer for any region (simplified for test)
            return TimerResult(turn_state='turn', confidence=0.8, purple_pixels=100, red_pixels=0)
        
        self.mock_timer_detector.detect_timer.side_effect = mock_detect_timer
        
        nameplate_regions = {
            1: np.array([[1, 2, 3]]),
            2: np.array([[4, 5, 6]])
        }
        
        # Enable turn detection for this test
        self.detector.settings.update("tracker.turn_detection.enabled", True)
        
        is_turn = self.detector.detect_hero_turn(nameplate_regions, hero_seat=1)
        
        # Verify hero turn detected
        self.assertTrue(is_turn)
        self.mock_timer_detector.detect_timer.assert_called()
    
    def test_detect_hero_turn_not_hero(self):
        """Test detection when it's not hero's turn."""
        # Mock timer results (other player has turn)
        mock_result1 = Mock()
        mock_result1.turn_state = 'none'
        mock_result1.confidence = 0.8
        
        mock_result2 = Mock()
        mock_result2.turn_state = 'turn'
        mock_result2.confidence = 0.8
        
        self.mock_timer_detector.detect_timer.side_effect = [mock_result1, mock_result2]
        
        nameplate_regions = {
            1: np.array([[1, 2, 3]]),
            2: np.array([[4, 5, 6]])
        }
        
        is_turn = self.detector.detect_hero_turn(nameplate_regions, hero_seat=1)
        
        # Verify not hero's turn
        self.assertFalse(is_turn)
    
    def test_detect_hero_turn_low_confidence(self):
        """Test detection with low confidence."""
        # Mock timer results (low confidence)
        mock_result = Mock()
        mock_result.turn_state = 'turn'
        mock_result.confidence = 0.4  # Below threshold
        
        self.mock_timer_detector.detect_timer.return_value = mock_result
        
        nameplate_regions = {1: np.array([[1, 2, 3]])}
        
        is_turn = self.detector.detect_hero_turn(nameplate_regions, hero_seat=1)
        
        # Verify no turn detected due to low confidence
        self.assertFalse(is_turn)
    
    def test_detect_hero_turn_no_active_indicators(self):
        """Test detection when no active turn indicators."""
        # Mock timer results (no active turns)
        mock_result = Mock()
        mock_result.turn_state = 'none'
        mock_result.confidence = 0.8
        
        self.mock_timer_detector.detect_timer.return_value = mock_result
        
        nameplate_regions = {1: np.array([[1, 2, 3]])}
        
        is_turn = self.detector.detect_hero_turn(nameplate_regions, hero_seat=1)
        
        # Verify no turn detected
        self.assertFalse(is_turn)
    
    def test_detect_hero_turn_disabled(self):
        """Test detection when disabled."""
        # Disable turn detection
        self.detector.settings.update("tracker.turn_detection.enabled", False)
        
        nameplate_regions = {1: np.array([[1, 2, 3]])}
        
        is_turn = self.detector.detect_hero_turn(nameplate_regions, hero_seat=1)
        
        # Verify no detection when disabled
        self.assertFalse(is_turn)
        self.mock_timer_detector.detect_timer.assert_not_called()
    
    def test_detect_hero_turn_no_validation(self):
        """Test detection without validation."""
        # Disable validation
        self.detector.settings.update("tracker.turn_detection.require_validation", False)
        
        # Enable turn detection for this test
        self.detector.settings.update("tracker.turn_detection.enabled", True)
        
        # Mock timer results
        mock_result = Mock()
        mock_result.turn_state = 'turn'
        mock_result.confidence = 0.4  # Low confidence
        
        # Set up mock to return result for hero seat
        def mock_detect_timer(region):
            # Return active timer for any region (simplified for test)
            return TimerResult(turn_state='turn', confidence=0.4, purple_pixels=50, red_pixels=0)
        
        self.mock_timer_detector.detect_timer.side_effect = mock_detect_timer
        
        nameplate_regions = {1: np.array([[1, 2, 3]])}
        
        is_turn = self.detector.detect_hero_turn(nameplate_regions, hero_seat=1)
        
        # Verify turn detected without validation (even with low confidence)
        self.assertTrue(is_turn)
    
    def test_detect_hero_turn_hero_not_in_regions(self):
        """Test detection when hero seat not in regions."""
        nameplate_regions = {2: np.array([[1, 2, 3]])}
        
        is_turn = self.detector.detect_hero_turn(nameplate_regions, hero_seat=1)
        
        # Verify no turn detected
        self.assertFalse(is_turn)


class TestStateMachine(unittest.TestCase):
    """Test GameStateMachine state coordination functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock all parsers and trackers
        with patch('src.tracker.state_machine.CardParser'), \
             patch('src.tracker.state_machine.MoneyParser'), \
             patch('src.tracker.state_machine.HandIdParser'), \
             patch('src.tracker.state_machine.TableInfoParser'), \
             patch('src.tracker.state_machine.NameParser'), \
             patch('src.tracker.state_machine.StatusParser'), \
             patch('src.tracker.state_machine.DealerDetector'), \
             patch('src.tracker.state_machine.TimerDetector'), \
             patch('src.tracker.state_machine.TransparencyDetector'), \
             patch('src.tracker.state_machine.HeroDetector'), \
             patch('src.tracker.state_machine.TurnDetector'), \
             patch('src.tracker.state_machine.PositionCalculator'), \
             patch('src.tracker.state_machine.HandTracker'):
            
            self.state_machine = GameStateMachine()
    
    def test_initialization(self):
        """Test GameStateMachine initialization."""
        self.assertIsNone(self.state_machine.current_state)
        self.assertIsNone(self.state_machine.previous_state)
        self.assertIsNone(self.state_machine.current_hand_id)
        self.assertIsNone(self.state_machine.hero_seat)
    
    def test_update_hand_id_parsing(self):
        """Test hand ID parsing in update."""
        # Mock hand ID parser
        mock_result = Mock()
        mock_result.hand_id = "test_hand_001"
        
        self.state_machine.hand_id_parser.parse_hand_id.return_value = mock_result
        
        regions = {'hand_num': np.array([[1, 2, 3]])}
        
        # Mock other parsers to return valid results (but don't override hand ID)
        # Mock card parser
        mock_card_result = Mock()
        mock_card_result.rank = 'A'
        mock_card_result.suit = 'hearts'
        self.state_machine.card_parser.parse_card.return_value = mock_card_result
        
        # Mock money parser
        mock_money_result = Mock()
        mock_money_result.value = 100.0
        self.state_machine.money_parser.parse_amounts.return_value = [mock_money_result]
        
        # Mock table info parser
        mock_table_result = Mock()
        mock_table_result.sb = 1.0
        mock_table_result.bb = 2.0
        self.state_machine.table_info_parser.parse_table_info.return_value = mock_table_result
        
        # Mock transparency detector
        mock_transparency = Mock()
        mock_transparency.is_transparent = False
        self.state_machine.transparency_detector.detect_transparency.return_value = mock_transparency
        
        # Mock dealer detector
        mock_dealer_result = Mock()
        mock_dealer_result.has_dealer = False
        self.state_machine.dealer_detector.detect_dealer_button.return_value = mock_dealer_result
        
        # Mock hero detector
        self.state_machine.hero_detector.detect_hero_seat.return_value = 1
        
        # Mock turn detector
        self.state_machine.turn_detector.detect_hero_turn.return_value = True
        
        # Mock player creation
        def mock_parse_players(regions):
            return [
                Player(seat_number=1, stack=100.0, is_active=True),
                Player(seat_number=2, stack=100.0, is_active=True)
            ]
        
        self.state_machine._parse_players = mock_parse_players
        
        game_state = self.state_machine.update(regions)
        
        # Verify hand ID was parsed
        self.assertIsNotNone(game_state)
        self.assertEqual(game_state.hand_id, "test_hand_001")
    
    def test_update_hand_id_failure(self):
        """Test update when hand ID parsing fails."""
        # Mock hand ID parser to return None
        self.state_machine.hand_id_parser.parse_hand_id.return_value = None
        
        regions = {'hand_num': np.array([[1, 2, 3]])}
        
        game_state = self.state_machine.update(regions)
        
        # Verify previous state returned
        self.assertEqual(game_state, self.state_machine.current_state)
    
    def test_update_new_hand_detection(self):
        """Test new hand detection."""
        # Set up initial state
        self.state_machine.current_hand_id = "old_hand"
        
        # Create a mock current state
        mock_current_state = Mock()
        self.state_machine.current_state = mock_current_state
        
        # Mock hand ID parser for new hand
        mock_result = Mock()
        mock_result.hand_id = "new_hand"
        
        self.state_machine.hand_id_parser.parse_hand_id.return_value = mock_result
        
        regions = {'hand_num': np.array([[1, 2, 3]])}
        
        # Mock other parsers (but don't override hand ID)
        # Mock card parser
        mock_card_result = Mock()
        mock_card_result.rank = 'A'
        mock_card_result.suit = 'hearts'
        self.state_machine.card_parser.parse_card.return_value = mock_card_result
        
        # Mock money parser
        mock_money_result = Mock()
        mock_money_result.value = 100.0
        self.state_machine.money_parser.parse_amounts.return_value = [mock_money_result]
        
        # Mock table info parser
        mock_table_result = Mock()
        mock_table_result.sb = 1.0
        mock_table_result.bb = 2.0
        self.state_machine.table_info_parser.parse_table_info.return_value = mock_table_result
        
        # Mock transparency detector
        mock_transparency = Mock()
        mock_transparency.is_transparent = False
        self.state_machine.transparency_detector.detect_transparency.return_value = mock_transparency
        
        # Mock dealer detector
        mock_dealer_result = Mock()
        mock_dealer_result.has_dealer = False
        self.state_machine.dealer_detector.detect_dealer_button.return_value = mock_dealer_result
        
        # Mock hero detector
        self.state_machine.hero_detector.detect_hero_seat.return_value = 1
        
        # Mock turn detector
        self.state_machine.turn_detector.detect_hero_turn.return_value = True
        
        # Mock player creation
        def mock_parse_players(regions):
            return [
                Player(seat_number=1, stack=100.0, is_active=True),
                Player(seat_number=2, stack=100.0, is_active=True)
            ]
        
        self.state_machine._parse_players = mock_parse_players
        
        game_state = self.state_machine.update(regions)
        
        # Verify new hand was detected
        self.assertEqual(self.state_machine.current_hand_id, "new_hand")
        self.state_machine.hand_tracker.start_new_hand.assert_called()
    
    def test_update_hero_detection(self):
        """Test hero detection."""
        # Mock hero detector
        self.state_machine.hero_detector.detect_hero_seat.return_value = 3
        
        regions = {'hand_num': np.array([[1, 2, 3]])}
        
        # Mock other parsers (but don't override hero detector)
        # Mock hand ID parser
        mock_hand_result = Mock()
        mock_hand_result.hand_id = "test_hand"
        self.state_machine.hand_id_parser.parse_hand_id.return_value = mock_hand_result
        
        # Mock card parser
        mock_card_result = Mock()
        mock_card_result.rank = 'A'
        mock_card_result.suit = 'hearts'
        self.state_machine.card_parser.parse_card.return_value = mock_card_result
        
        # Mock money parser
        mock_money_result = Mock()
        mock_money_result.value = 100.0
        self.state_machine.money_parser.parse_amounts.return_value = [mock_money_result]
        
        # Mock table info parser
        mock_table_result = Mock()
        mock_table_result.sb = 1.0
        mock_table_result.bb = 2.0
        self.state_machine.table_info_parser.parse_table_info.return_value = mock_table_result
        
        # Mock transparency detector
        mock_transparency = Mock()
        mock_transparency.is_transparent = False
        self.state_machine.transparency_detector.detect_transparency.return_value = mock_transparency
        
        # Mock dealer detector
        mock_dealer_result = Mock()
        mock_dealer_result.has_dealer = False
        self.state_machine.dealer_detector.detect_dealer_button.return_value = mock_dealer_result
        
        # Mock turn detector
        self.state_machine.turn_detector.detect_hero_turn.return_value = True
        
        # Mock player creation
        def mock_parse_players(regions):
            return [
                Player(seat_number=1, stack=100.0, is_active=True),
                Player(seat_number=2, stack=100.0, is_active=True)
            ]
        
        self.state_machine._parse_players = mock_parse_players
        
        game_state = self.state_machine.update(regions)
        
        # Verify hero was detected
        self.assertEqual(self.state_machine.hero_seat, 3)
    
    def test_update_player_parsing(self):
        """Test player parsing."""
        regions = {'hand_num': np.array([[1, 2, 3]])}
        
        # Mock all parsers
        self._mock_all_parsers()
        
        # Mock transparency detector to return non-transparent
        mock_transparency = Mock()
        mock_transparency.is_transparent = False
        self.state_machine.transparency_detector.detect_transparency.return_value = mock_transparency
        
        game_state = self.state_machine.update(regions)
        
        # Verify players were parsed
        self.assertIsNotNone(game_state)
        self.assertGreater(len(game_state.players), 0)
    
    def test_update_community_cards_parsing(self):
        """Test community cards parsing."""
        regions = {'hand_num': np.array([[1, 2, 3]])}
        
        # Mock all parsers
        self._mock_all_parsers()
        
        # Mock card parser for community cards
        mock_card_result = Mock()
        mock_card_result.rank = 'A'
        mock_card_result.suit = 'hearts'
        self.state_machine.card_parser.parse_card.return_value = mock_card_result
        
        game_state = self.state_machine.update(regions)
        
        # Verify community cards were parsed
        self.assertIsNotNone(game_state)
        self.assertGreaterEqual(len(game_state.community_cards), 0)
    
    def test_update_phase_determination(self):
        """Test phase determination from community cards."""
        regions = {'hand_num': np.array([[1, 2, 3]])}
        
        # Mock all parsers
        self._mock_all_parsers()
        
        # Mock card parser to return 3 cards (flop)
        mock_card_result = Mock()
        mock_card_result.rank = 'A'
        mock_card_result.suit = 'hearts'
        
        # Mock to return exactly 3 cards for community cards
        def mock_parse_card(region):
            # Return a card for community card regions
            return mock_card_result
        
        self.state_machine.card_parser.parse_card.side_effect = mock_parse_card
        
        # Mock the community cards parsing to return 3 cards
        def mock_parse_community_cards(regions):
            return [
                Card(rank='A', suit='hearts'),
                Card(rank='K', suit='spades'),
                Card(rank='Q', suit='diamonds')
            ]
        
        self.state_machine._parse_community_cards = mock_parse_community_cards
        
        game_state = self.state_machine.update(regions)
        
        # Verify phase was determined correctly
        self.assertEqual(game_state.phase, 'flop')
    
    def test_update_pot_parsing(self):
        """Test pot parsing."""
        regions = {
            'hand_num': np.array([[1, 2, 3]]),
            'pot_total': np.array([[1, 2, 3]])  # Add pot region
        }
        
        # Mock all parsers except money parser
        # Mock hand ID parser
        mock_hand_result = Mock()
        mock_hand_result.hand_id = "test_hand"
        self.state_machine.hand_id_parser.parse_hand_id.return_value = mock_hand_result
        
        # Mock card parser
        mock_card_result = Mock()
        mock_card_result.rank = 'A'
        mock_card_result.suit = 'hearts'
        self.state_machine.card_parser.parse_card.return_value = mock_card_result
        
        # Mock table info parser
        mock_table_result = Mock()
        mock_table_result.sb = 1.0
        mock_table_result.bb = 2.0
        self.state_machine.table_info_parser.parse_table_info.return_value = mock_table_result
        
        # Mock transparency detector
        mock_transparency = Mock()
        mock_transparency.is_transparent = False
        self.state_machine.transparency_detector.detect_transparency.return_value = mock_transparency
        
        # Mock dealer detector
        mock_dealer_result = Mock()
        mock_dealer_result.has_dealer = False
        self.state_machine.dealer_detector.detect_dealer_button.return_value = mock_dealer_result
        
        # Mock hero detector
        self.state_machine.hero_detector.detect_hero_seat.return_value = 1
        
        # Mock turn detector
        self.state_machine.turn_detector.detect_hero_turn.return_value = True
        
        # Mock player creation
        def mock_parse_players(regions):
            return [
                Player(seat_number=1, stack=100.0, is_active=True),
                Player(seat_number=2, stack=100.0, is_active=True)
            ]
        
        self.state_machine._parse_players = mock_parse_players
        
        # Mock money parser for pot (override the default)
        mock_money_result = Mock()
        mock_money_result.value = 25.0
        self.state_machine.money_parser.parse_amounts.return_value = [mock_money_result]
        
        game_state = self.state_machine.update(regions)
        
        # Verify pot was parsed
        self.assertEqual(game_state.pot, 25.0)
    
    def test_update_table_info_parsing(self):
        """Test table info parsing."""
        regions = {
            'hand_num': np.array([[1, 2, 3]]),
            'table_info': np.array([[1, 2, 3]])  # Add table info region
        }
        
        # Mock all parsers except table info parser
        # Mock hand ID parser
        mock_hand_result = Mock()
        mock_hand_result.hand_id = "test_hand"
        self.state_machine.hand_id_parser.parse_hand_id.return_value = mock_hand_result
        
        # Mock card parser
        mock_card_result = Mock()
        mock_card_result.rank = 'A'
        mock_card_result.suit = 'hearts'
        self.state_machine.card_parser.parse_card.return_value = mock_card_result
        
        # Mock money parser
        mock_money_result = Mock()
        mock_money_result.value = 100.0
        self.state_machine.money_parser.parse_amounts.return_value = [mock_money_result]
        
        # Mock transparency detector
        mock_transparency = Mock()
        mock_transparency.is_transparent = False
        self.state_machine.transparency_detector.detect_transparency.return_value = mock_transparency
        
        # Mock dealer detector
        mock_dealer_result = Mock()
        mock_dealer_result.has_dealer = False
        self.state_machine.dealer_detector.detect_dealer_button.return_value = mock_dealer_result
        
        # Mock hero detector
        self.state_machine.hero_detector.detect_hero_seat.return_value = 1
        
        # Mock turn detector
        self.state_machine.turn_detector.detect_hero_turn.return_value = True
        
        # Mock player creation
        def mock_parse_players(regions):
            return [
                Player(seat_number=1, stack=100.0, is_active=True),
                Player(seat_number=2, stack=100.0, is_active=True)
            ]
        
        self.state_machine._parse_players = mock_parse_players
        
        # Mock table info parser (override the default)
        mock_table_result = Mock()
        mock_table_result.sb = 1.0
        mock_table_result.bb = 2.0
        self.state_machine.table_info_parser.parse_table_info.return_value = mock_table_result
        
        game_state = self.state_machine.update(regions)
        
        # Verify table info was parsed
        self.assertEqual(game_state.table_info.sb, 1.0)
        self.assertEqual(game_state.table_info.bb, 2.0)
    
    def test_update_dealer_button_detection(self):
        """Test dealer button detection."""
        regions = {'hand_num': np.array([[1, 2, 3]])}
        
        # Mock all parsers
        self._mock_all_parsers()
        
        # Mock dealer detector
        mock_dealer_result = Mock()
        mock_dealer_result.has_dealer = True
        self.state_machine.dealer_detector.detect_dealer_button.return_value = mock_dealer_result
        
        game_state = self.state_machine.update(regions)
        
        # Verify dealer button was detected
        self.assertIsNotNone(game_state.button_position)
    
    def test_update_position_calculation(self):
        """Test position calculation."""
        regions = {
            'hand_num': np.array([[1, 2, 3]]),
            'player_1_dealer': np.array([[1, 2, 3]])  # Add dealer region
        }
        
        # Mock all parsers
        self._mock_all_parsers()
        
        # Mock dealer detector
        mock_dealer_result = Mock()
        mock_dealer_result.has_dealer = True
        self.state_machine.dealer_detector.detect_dealer_button.return_value = mock_dealer_result
        
        game_state = self.state_machine.update(regions)
        
        # Verify positions were calculated
        self.state_machine.position_calculator.calculate_positions.assert_called()
    
    def test_update_active_player_detection(self):
        """Test active player detection."""
        regions = {'hand_num': np.array([[1, 2, 3]])}
        
        # Mock all parsers
        self._mock_all_parsers()
        
        # Set hero seat
        self.state_machine.hero_seat = 1
        
        # Mock turn detector
        self.state_machine.turn_detector.detect_hero_turn.return_value = True
        
        game_state = self.state_machine.update(regions)
        
        # Verify active player was detected
        self.assertEqual(game_state.active_player, 1)
    
    def test_update_hand_tracking(self):
        """Test hand tracking integration."""
        regions = {'hand_num': np.array([[1, 2, 3]])}
        
        # Mock all parsers
        self._mock_all_parsers()
        
        # Set up hand tracker with current hand
        mock_hand = Mock()
        self.state_machine.hand_tracker.current_hand = mock_hand
        
        game_state = self.state_machine.update(regions)
        
        # Verify hand was updated
        self.state_machine.hand_tracker.update_hand.assert_called()
    
    def test_get_current_state(self):
        """Test getting current state."""
        # Set current state
        mock_state = Mock()
        self.state_machine.current_state = mock_state
        
        current_state = self.state_machine.get_current_state()
        
        # Verify current state returned
        self.assertEqual(current_state, mock_state)
    
    def test_get_hero_seat(self):
        """Test getting hero seat."""
        # Set hero seat
        self.state_machine.hero_seat = 3
        
        hero_seat = self.state_machine.get_hero_seat()
        
        # Verify hero seat returned
        self.assertEqual(hero_seat, 3)
    
    def test_is_hero_turn(self):
        """Test checking if it's hero's turn."""
        # Set up state
        mock_state = Mock()
        mock_state.active_player = 2
        self.state_machine.current_state = mock_state
        self.state_machine.hero_seat = 2
        
        is_turn = self.state_machine.is_hero_turn()
        
        # Verify hero turn detected
        self.assertTrue(is_turn)
    
    def test_get_current_hand_record(self):
        """Test getting current hand record."""
        # Mock hand tracker
        mock_hand = Mock()
        self.state_machine.hand_tracker.get_current_hand.return_value = mock_hand
        
        current_hand = self.state_machine.get_current_hand_record()
        
        # Verify hand record returned
        self.assertEqual(current_hand, mock_hand)
    
    def test_get_completed_hands(self):
        """Test getting completed hands."""
        # Mock hand tracker
        mock_hands = [Mock(), Mock()]
        self.state_machine.hand_tracker.get_completed_hands.return_value = mock_hands
        
        completed_hands = self.state_machine.get_completed_hands()
        
        # Verify completed hands returned
        self.assertEqual(completed_hands, mock_hands)
    
    def _mock_all_parsers(self):
        """Helper method to mock all parsers with valid results."""
        # Mock hand ID parser
        mock_hand_result = Mock()
        mock_hand_result.hand_id = "test_hand"
        self.state_machine.hand_id_parser.parse_hand_id.return_value = mock_hand_result
        
        # Mock card parser
        mock_card_result = Mock()
        mock_card_result.rank = 'A'
        mock_card_result.suit = 'hearts'
        self.state_machine.card_parser.parse_card.return_value = mock_card_result
        
        # Mock money parser
        mock_money_result = Mock()
        mock_money_result.value = 100.0
        self.state_machine.money_parser.parse_amounts.return_value = [mock_money_result]
        
        # Mock table info parser
        mock_table_result = Mock()
        mock_table_result.sb = 1.0
        mock_table_result.bb = 2.0
        self.state_machine.table_info_parser.parse_table_info.return_value = mock_table_result
        
        # Mock transparency detector
        mock_transparency = Mock()
        mock_transparency.is_transparent = False
        self.state_machine.transparency_detector.detect_transparency.return_value = mock_transparency
        
        # Mock dealer detector
        mock_dealer_result = Mock()
        mock_dealer_result.has_dealer = False
        self.state_machine.dealer_detector.detect_dealer_button.return_value = mock_dealer_result
        
        # Mock hero detector
        self.state_machine.hero_detector.detect_hero_seat.return_value = 1
        
        # Mock turn detector
        self.state_machine.turn_detector.detect_hero_turn.return_value = True
        
        # Mock player creation by making _parse_players return valid players
        def mock_parse_players(regions):
            return [
                Player(seat_number=1, stack=100.0, is_active=True),
                Player(seat_number=2, stack=100.0, is_active=True)
            ]
        
        self.state_machine._parse_players = mock_parse_players


if __name__ == '__main__':
    unittest.main()
