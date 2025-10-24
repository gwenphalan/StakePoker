#!/usr/bin/env python3
"""
Unit tests for the models module in src/models/.

Tests all Pydantic models including validation, constraints, 
cross-field validation, and edge cases.
"""

import unittest
from datetime import datetime
from typing import List
from pydantic import ValidationError

# Add src to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from models.card import Card
from models.player import Player
from models.table_info import TableInfo
from models.game_state import GameState
from models.decision import Decision, AlternativeAction
from models.hand_record import HandRecord, Action
from models.regions import RegionConfig


class TestCard(unittest.TestCase):
    """Test Card model validation and functionality."""
    
    def test_valid_card_creation(self):
        """Test creating valid cards."""
        # Test all valid ranks
        valid_ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
        valid_suits = ['hearts', 'diamonds', 'clubs', 'spades']
        
        for rank in valid_ranks:
            for suit in valid_suits:
                card = Card(rank=rank, suit=suit)
                self.assertEqual(card.rank, rank)
                self.assertEqual(card.suit, suit)
    
    def test_invalid_rank_validation(self):
        """Test validation of invalid ranks."""
        invalid_ranks = ['1', '0', 'B', 'X', 'ace', 'king', '']
        
        for rank in invalid_ranks:
            with self.assertRaises(ValidationError) as context:
                Card(rank=rank, suit='hearts')
            self.assertIn('Invalid rank', str(context.exception))
    
    def test_invalid_suit_validation(self):
        """Test validation of invalid suits."""
        invalid_suits = ['heart', 'diamond', 'club', 'spade', 'red', 'black', '']
        
        for suit in invalid_suits:
            with self.assertRaises(ValidationError) as context:
                Card(rank='A', suit=suit)
            self.assertIn('Invalid suit', str(context.exception))
    
    def test_string_representation(self):
        """Test string representation of cards."""
        test_cases = [
            ('A', 'hearts', 'Ah'),
            ('K', 'diamonds', 'Kd'),
            ('Q', 'clubs', 'Qc'),
            ('J', 'spades', 'Js'),
            ('T', 'hearts', 'Th'),
            ('9', 'diamonds', '9d'),
            ('2', 'clubs', '2c')
        ]
        
        for rank, suit, expected in test_cases:
            card = Card(rank=rank, suit=suit)
            self.assertEqual(str(card), expected)
    
    def test_card_equality(self):
        """Test card equality comparison."""
        card1 = Card(rank='A', suit='hearts')
        card2 = Card(rank='A', suit='hearts')
        card3 = Card(rank='K', suit='hearts')
        
        self.assertEqual(card1, card2)
        self.assertNotEqual(card1, card3)
    
    def test_card_serialization(self):
        """Test card serialization to dict."""
        card = Card(rank='A', suit='hearts')
        card_dict = card.model_dump()
        
        self.assertEqual(card_dict['rank'], 'A')
        self.assertEqual(card_dict['suit'], 'hearts')


class TestPlayer(unittest.TestCase):
    """Test Player model validation and functionality."""
    
    def test_valid_player_creation(self):
        """Test creating valid players."""
        player = Player(
            seat_number=1,
            position='BTN',
            stack=100.0,
            hole_cards=[],
            timer_state='purple',
            is_dealer=True,
            current_bet=10.0,
            is_hero=False,
            is_active=True
        )
        
        self.assertEqual(player.seat_number, 1)
        self.assertEqual(player.position, 'BTN')
        self.assertEqual(player.stack, 100.0)
        self.assertEqual(player.hole_cards, [])
        self.assertEqual(player.timer_state, 'purple')
        self.assertTrue(player.is_dealer)
        self.assertEqual(player.current_bet, 10.0)
        self.assertFalse(player.is_hero)
        self.assertTrue(player.is_active)
    
    def test_seat_number_validation(self):
        """Test seat number validation."""
        # Valid seat numbers
        for seat in range(1, 9):
            player = Player(seat_number=seat, stack=100.0)
            self.assertEqual(player.seat_number, seat)
        
        # Invalid seat numbers
        invalid_seats = [0, -1, 9, 10]
        for seat in invalid_seats:
            with self.assertRaises(ValidationError):
                Player(seat_number=seat, stack=100.0)
    
    def test_stack_validation(self):
        """Test stack validation."""
        # Valid stacks
        valid_stacks = [0.0, 1.0, 100.0, 1000.0]
        for stack in valid_stacks:
            player = Player(seat_number=1, stack=stack)
            self.assertEqual(player.stack, stack)
        
        # Invalid stacks
        with self.assertRaises(ValidationError):
            Player(seat_number=1, stack=-1.0)
    
    def test_hole_cards_validation(self):
        """Test hole cards validation."""
        # Valid hole cards
        card1 = Card(rank='A', suit='hearts')
        card2 = Card(rank='K', suit='spades')
        
        player = Player(seat_number=1, stack=100.0, hole_cards=[card1.model_dump(), card2.model_dump()])
        self.assertEqual(len(player.hole_cards), 2)
        
        # Too many hole cards
        card3 = Card(rank='Q', suit='diamonds')
        with self.assertRaises(ValidationError):
            Player(seat_number=1, stack=100.0, hole_cards=[card1.model_dump(), card2.model_dump(), card3.model_dump()])
    
    def test_timer_state_validation(self):
        """Test timer state validation."""
        # Valid timer states
        valid_states = [None, 'purple', 'red']
        for state in valid_states:
            player = Player(seat_number=1, stack=100.0, timer_state=state)
            self.assertEqual(player.timer_state, state)
        
        # Invalid timer state
        with self.assertRaises(ValidationError):
            Player(seat_number=1, stack=100.0, timer_state='blue')
    
    def test_position_validation(self):
        """Test position validation."""
        valid_positions = ['BTN', 'SB', 'BB', 'UTG', 'UTG+1', 'UTG+2', 'MP', 'MP+1', 'CO']
        
        for position in valid_positions:
            player = Player(seat_number=1, stack=100.0, position=position)
            self.assertEqual(player.position, position)
        
        # Invalid position
        with self.assertRaises(ValidationError):
            Player(seat_number=1, stack=100.0, position='INVALID')
    
    def test_current_bet_validation(self):
        """Test current bet validation."""
        # Valid bets
        valid_bets = [0.0, 1.0, 10.0, 100.0]
        for bet in valid_bets:
            player = Player(seat_number=1, stack=100.0, current_bet=bet)
            self.assertEqual(player.current_bet, bet)
        
        # Invalid bet
        with self.assertRaises(ValidationError):
            Player(seat_number=1, stack=100.0, current_bet=-1.0)


class TestTableInfo(unittest.TestCase):
    """Test TableInfo model validation and functionality."""
    
    def test_valid_table_info_creation(self):
        """Test creating valid table info."""
        table_info = TableInfo(sb=1.0, bb=2.0)
        
        self.assertEqual(table_info.sb, 1.0)
        self.assertEqual(table_info.bb, 2.0)
    
    def test_blind_validation(self):
        """Test blind validation."""
        # Valid blinds
        valid_cases = [
            (0.5, 1.0),
            (1.0, 2.0),
            (2.0, 5.0),
            (5.0, 10.0)
        ]
        
        for sb, bb in valid_cases:
            table_info = TableInfo(sb=sb, bb=bb)
            self.assertEqual(table_info.sb, sb)
            self.assertEqual(table_info.bb, bb)
        
        # Invalid blinds (bb <= sb)
        invalid_cases = [
            (1.0, 1.0),  # bb equals sb
            (2.0, 1.0),  # bb less than sb
            (0.0, 1.0),  # sb is zero
            (-1.0, 1.0)  # sb is negative
        ]
        
        for sb, bb in invalid_cases:
            with self.assertRaises(ValidationError):
                TableInfo(sb=sb, bb=bb)
    
    def test_normalize_to_bb(self):
        """Test normalization to big blinds."""
        table_info = TableInfo(sb=1.0, bb=2.0)
        
        test_cases = [
            (0.0, 0.0),
            (2.0, 1.0),
            (4.0, 2.0),
            (10.0, 5.0),
            (1.0, 0.5)
        ]
        
        for amount, expected_bb in test_cases:
            result = table_info.normalize_to_bb(amount)
            self.assertEqual(result, expected_bb)


class TestDecision(unittest.TestCase):
    """Test Decision and AlternativeAction models."""
    
    def test_valid_alternative_action_creation(self):
        """Test creating valid alternative actions."""
        action = AlternativeAction(action='call', amount=10.0, ev=5.0)
        
        self.assertEqual(action.action, 'call')
        self.assertEqual(action.amount, 10.0)
        self.assertEqual(action.ev, 5.0)
    
    def test_alternative_action_validation(self):
        """Test alternative action validation."""
        # Valid actions
        valid_actions = ['fold', 'call', 'raise', 'check', 'bet']
        for action_type in valid_actions:
            action = AlternativeAction(action=action_type, ev=0.0)
            self.assertEqual(action.action, action_type)
        
        # Invalid action
        with self.assertRaises(ValidationError):
            AlternativeAction(action='invalid', ev=0.0)
        
        # Invalid amount
        with self.assertRaises(ValidationError):
            AlternativeAction(action='call', amount=-1.0, ev=0.0)
    
    def test_valid_decision_creation(self):
        """Test creating valid decisions."""
        decision = Decision(
            action='call',
            amount=10.0,
            confidence=0.8,
            reasoning='Good pot odds',
            equity=0.6,
            pot_odds=0.3
        )
        
        self.assertEqual(decision.action, 'call')
        self.assertEqual(decision.amount, 10.0)
        self.assertEqual(decision.confidence, 0.8)
        self.assertEqual(decision.reasoning, 'Good pot odds')
        self.assertEqual(decision.equity, 0.6)
        self.assertEqual(decision.pot_odds, 0.3)
    
    def test_decision_validation(self):
        """Test decision validation."""
        # Valid actions
        valid_actions = ['fold', 'call', 'raise', 'check', 'bet']
        for action_type in valid_actions:
            decision = Decision(action=action_type, confidence=0.5, reasoning='test')
            self.assertEqual(decision.action, action_type)
        
        # Invalid action
        with self.assertRaises(ValidationError):
            Decision(action='invalid', confidence=0.5, reasoning='test')
        
        # Invalid confidence
        with self.assertRaises(ValidationError):
            Decision(action='call', confidence=1.5, reasoning='test')
        
        with self.assertRaises(ValidationError):
            Decision(action='call', confidence=-0.1, reasoning='test')
        
        # Invalid equity
        with self.assertRaises(ValidationError):
            Decision(action='call', confidence=0.5, reasoning='test', equity=1.5)
        
        # Invalid pot odds
        with self.assertRaises(ValidationError):
            Decision(action='call', confidence=0.5, reasoning='test', pot_odds=-0.1)
    
    def test_decision_with_alternatives(self):
        """Test decision with alternative actions."""
        alternatives = [
            AlternativeAction(action='fold', ev=0.0),
            AlternativeAction(action='call', amount=10.0, ev=5.0),
            AlternativeAction(action='raise', amount=30.0, ev=8.0)
        ]
        
        decision = Decision(
            action='call',
            confidence=0.7,
            reasoning='Call is optimal',
            alternative_actions=alternatives
        )
        
        self.assertEqual(len(decision.alternative_actions), 3)
        self.assertEqual(decision.alternative_actions[0].action, 'fold')
        self.assertEqual(decision.alternative_actions[1].action, 'call')
        self.assertEqual(decision.alternative_actions[2].action, 'raise')


class TestHandRecord(unittest.TestCase):
    """Test HandRecord and Action models."""
    
    def test_valid_action_creation(self):
        """Test creating valid actions."""
        action = Action(
            seat_number=1,
            action_type='call',
            amount=10.0,
            phase='preflop'
        )
        
        self.assertEqual(action.seat_number, 1)
        self.assertEqual(action.action_type, 'call')
        self.assertEqual(action.amount, 10.0)
        self.assertEqual(action.phase, 'preflop')
    
    def test_action_validation(self):
        """Test action validation."""
        # Valid seat numbers
        for seat in range(1, 9):
            action = Action(seat_number=seat, action_type='fold', phase='preflop')
            self.assertEqual(action.seat_number, seat)
        
        # Invalid seat number
        with self.assertRaises(ValidationError):
            Action(seat_number=0, action_type='fold', phase='preflop')
        
        # Invalid action type
        with self.assertRaises(ValidationError):
            Action(seat_number=1, action_type='invalid', phase='preflop')
        
        # Invalid phase
        with self.assertRaises(ValidationError):
            Action(seat_number=1, action_type='fold', phase='invalid')
        
        # Invalid amount
        with self.assertRaises(ValidationError):
            Action(seat_number=1, action_type='call', amount=-1.0, phase='preflop')
    
    def test_valid_hand_record_creation(self):
        """Test creating valid hand records."""
        timestamp = datetime.now()
        table_info = TableInfo(sb=1.0, bb=2.0)
        hero_cards = [Card(rank='A', suit='hearts'), Card(rank='K', suit='spades')]
        
        hand_record = HandRecord(
            hand_id='test_hand_123',
            timestamp=timestamp,
            table_info=table_info.model_dump(),
            hero_position='BTN',
            hero_seat=1,
            hero_cards=[card.model_dump() for card in hero_cards],
            result='won',
            net_profit=50.0,
            final_pot=100.0,
            showdown=True
        )
        
        self.assertEqual(hand_record.hand_id, 'test_hand_123')
        self.assertEqual(hand_record.timestamp, timestamp)
        self.assertEqual(hand_record.hero_position, 'BTN')
        self.assertEqual(hand_record.hero_seat, 1)
        self.assertEqual(len(hand_record.hero_cards), 2)
        self.assertEqual(hand_record.result, 'won')
        self.assertEqual(hand_record.net_profit, 50.0)
        self.assertEqual(hand_record.final_pot, 100.0)
        self.assertTrue(hand_record.showdown)
    
    def test_hand_record_validation(self):
        """Test hand record validation."""
        timestamp = datetime.now()
        table_info = TableInfo(sb=1.0, bb=2.0)
        hero_cards = [Card(rank='A', suit='hearts'), Card(rank='K', suit='spades')]
        
        # Valid hero positions
        valid_positions = ['BTN', 'SB', 'BB', 'UTG', 'UTG+1', 'UTG+2', 'MP', 'MP+1', 'CO']
        for position in valid_positions:
            hand_record = HandRecord(
                hand_id='test',
                timestamp=timestamp,
                table_info=table_info.model_dump(),
                hero_position=position,
                hero_seat=1,
                hero_cards=[card.model_dump() for card in hero_cards],
                net_profit=0.0,
                final_pot=10.0
            )
            self.assertEqual(hand_record.hero_position, position)
        
        # Invalid hero position
        with self.assertRaises(ValidationError):
            HandRecord(
                hand_id='test',
                timestamp=timestamp,
                table_info=table_info.model_dump(),
                hero_position='INVALID',
                hero_seat=1,
                hero_cards=[card.model_dump() for card in hero_cards],
                net_profit=0.0,
                final_pot=10.0
            )
        
        # Invalid hero seat
        with self.assertRaises(ValidationError):
            HandRecord(
                hand_id='test',
                timestamp=timestamp,
                table_info=table_info.model_dump(),
                hero_position='BTN',
                hero_seat=0,
                hero_cards=[card.model_dump() for card in hero_cards],
                net_profit=0.0,
                final_pot=10.0
            )
        
        # Invalid result
        with self.assertRaises(ValidationError):
            HandRecord(
                hand_id='test',
                timestamp=timestamp,
                table_info=table_info.model_dump(),
                hero_position='BTN',
                hero_seat=1,
                hero_cards=[card.model_dump() for card in hero_cards],
                result='invalid',
                net_profit=0.0,
                final_pot=10.0
            )
        
        # Invalid final pot
        with self.assertRaises(ValidationError):
            HandRecord(
                hand_id='test',
                timestamp=timestamp,
                table_info=table_info.model_dump(),
                hero_position='BTN',
                hero_seat=1,
                hero_cards=[card.model_dump() for card in hero_cards],
                net_profit=0.0,
                final_pot=-10.0
            )
    
    def test_hand_record_with_actions(self):
        """Test hand record with action sequence."""
        timestamp = datetime.now()
        table_info = TableInfo(sb=1.0, bb=2.0)
        hero_cards = [Card(rank='A', suit='hearts'), Card(rank='K', suit='spades')]
        
        actions = [
            Action(seat_number=1, action_type='call', amount=2.0, phase='preflop'),
            Action(seat_number=2, action_type='raise', amount=6.0, phase='preflop'),
            Action(seat_number=1, action_type='call', amount=4.0, phase='preflop')
        ]
        
        hand_record = HandRecord(
            hand_id='test_hand_with_actions',
            timestamp=timestamp,
            table_info=table_info.model_dump(),
            hero_position='BTN',
            hero_seat=1,
            hero_cards=[card.model_dump() for card in hero_cards],
            actions=[action.model_dump() for action in actions],
            net_profit=20.0,
            final_pot=40.0
        )
        
        self.assertEqual(len(hand_record.actions), 3)
        self.assertEqual(hand_record.actions[0].action_type, 'call')
        self.assertEqual(hand_record.actions[1].action_type, 'raise')
        self.assertEqual(hand_record.actions[2].action_type, 'call')


class TestRegionConfig(unittest.TestCase):
    """Test RegionConfig model validation and functionality."""
    
    def test_valid_region_creation(self):
        """Test creating valid regions."""
        region = RegionConfig(
            name='test_region',
            x=100,
            y=200,
            width=300,
            height=400
        )
        
        self.assertEqual(region.name, 'test_region')
        self.assertEqual(region.x, 100)
        self.assertEqual(region.y, 200)
        self.assertEqual(region.width, 300)
        self.assertEqual(region.height, 400)
    
    def test_region_validation(self):
        """Test region validation."""
        # Valid coordinates
        region = RegionConfig(name='test', x=0, y=0, width=100, height=100)
        self.assertEqual(region.x, 0)
        self.assertEqual(region.y, 0)
        
        # Invalid negative coordinates
        with self.assertRaises(ValidationError):
            RegionConfig(name='test', x=-1, y=0, width=100, height=100)
        
        with self.assertRaises(ValidationError):
            RegionConfig(name='test', x=0, y=-1, width=100, height=100)
        
        # Invalid dimensions
        with self.assertRaises(ValidationError):
            RegionConfig(name='test', x=0, y=0, width=0, height=100)
        
        with self.assertRaises(ValidationError):
            RegionConfig(name='test', x=0, y=0, width=100, height=0)
        
        with self.assertRaises(ValidationError):
            RegionConfig(name='test', x=0, y=0, width=-100, height=100)
    
    def test_to_tuple_conversion(self):
        """Test conversion to tuple for mss."""
        region = RegionConfig(
            name='test_region',
            x=100,
            y=200,
            width=300,
            height=400
        )
        
        tuple_result = region.to_tuple()
        expected = (100, 200, 300, 400)
        
        self.assertEqual(tuple_result, expected)


class TestGameState(unittest.TestCase):
    """Test GameState model complex validation and cross-field validation."""
    
    def setUp(self):
        """Set up test data."""
        self.table_info = TableInfo(sb=1.0, bb=2.0)
        self.hero_cards = [Card(rank='A', suit='hearts'), Card(rank='K', suit='spades')]
        
        self.players = [
            Player(seat_number=1, position='BTN', stack=100.0, is_dealer=True, is_hero=True),
            Player(seat_number=2, position='SB', stack=100.0),
            Player(seat_number=3, position='BB', stack=100.0)
        ]
        
        # Convert to dictionaries for Pydantic v2 compatibility
        self.players_dict = [player.model_dump() for player in self.players]
        self.table_info_dict = self.table_info.model_dump()
    
    def test_valid_game_state_creation(self):
        """Test creating valid game state."""
        game_state = GameState(
            players=self.players_dict,
            community_cards=[],
            pot=10.0,
            phase='preflop',
            active_player=1,
            button_position=1,
            hand_id='test_hand',
            table_info=self.table_info_dict
        )
        
        self.assertEqual(len(game_state.players), 3)
        self.assertEqual(len(game_state.community_cards), 0)
        self.assertEqual(game_state.pot, 10.0)
        self.assertEqual(game_state.phase, 'preflop')
        self.assertEqual(game_state.active_player, 1)
        self.assertEqual(game_state.button_position, 1)
        self.assertEqual(game_state.hand_id, 'test_hand')
    
    def test_phase_validation(self):
        """Test phase validation."""
        valid_phases = ['preflop', 'flop', 'turn', 'river', 'showdown']
        
        for phase in valid_phases:
            # Create appropriate community cards for each phase
            if phase == 'preflop':
                community_cards = []
            elif phase == 'flop':
                community_cards = [Card(rank='A', suit='hearts'), Card(rank='K', suit='spades'), Card(rank='Q', suit='diamonds')]
                community_cards = [card.model_dump() for card in community_cards]
            elif phase == 'turn':
                community_cards = [Card(rank='A', suit='hearts'), Card(rank='K', suit='spades'), Card(rank='Q', suit='diamonds'), Card(rank='J', suit='clubs')]
                community_cards = [card.model_dump() for card in community_cards]
            elif phase == 'river':
                community_cards = [Card(rank='A', suit='hearts'), Card(rank='K', suit='spades'), Card(rank='Q', suit='diamonds'), Card(rank='J', suit='clubs'), Card(rank='T', suit='hearts')]
                community_cards = [card.model_dump() for card in community_cards]
            else:  # showdown
                community_cards = [Card(rank='A', suit='hearts'), Card(rank='K', suit='spades'), Card(rank='Q', suit='diamonds'), Card(rank='J', suit='clubs'), Card(rank='T', suit='hearts')]
                community_cards = [card.model_dump() for card in community_cards]
            
            game_state = GameState(
                players=self.players_dict,
                community_cards=community_cards,
                pot=10.0,
                phase=phase,
                button_position=1,
                table_info=self.table_info_dict
            )
            self.assertEqual(game_state.phase, phase)
        
        # Invalid phase
        with self.assertRaises(ValidationError):
            GameState(
                players=self.players_dict,
                pot=10.0,
                phase='invalid',
                button_position=1,
                table_info=self.table_info_dict
            )
    
    def test_community_cards_by_phase_validation(self):
        """Test community cards validation by phase."""
        flop_cards = [Card(rank='A', suit='hearts'), Card(rank='K', suit='spades'), Card(rank='Q', suit='diamonds')]
        turn_cards = flop_cards + [Card(rank='J', suit='clubs')]
        river_cards = turn_cards + [Card(rank='T', suit='hearts')]
        
        # Convert to dictionaries
        flop_cards_dict = [card.model_dump() for card in flop_cards]
        turn_cards_dict = [card.model_dump() for card in turn_cards]
        river_cards_dict = [card.model_dump() for card in river_cards]
        
        # Preflop - no community cards
        game_state = GameState(
            players=self.players_dict,
            community_cards=[],
            pot=10.0,
            phase='preflop',
            button_position=1,
            table_info=self.table_info_dict
        )
        self.assertEqual(len(game_state.community_cards), 0)
        
        # Preflop with community cards - should fail
        with self.assertRaises(ValidationError):
            GameState(
                players=self.players_dict,
                community_cards=flop_cards_dict,
                pot=10.0,
                phase='preflop',
                button_position=1,
                table_info=self.table_info_dict
            )
        
        # Flop - exactly 3 cards
        game_state = GameState(
            players=self.players_dict,
            community_cards=flop_cards_dict,
            pot=10.0,
            phase='flop',
            button_position=1,
            table_info=self.table_info_dict
        )
        self.assertEqual(len(game_state.community_cards), 3)
        
        # Flop with wrong number of cards - should fail
        with self.assertRaises(ValidationError):
            GameState(
                players=self.players_dict,
                community_cards=flop_cards_dict[:2],  # Only 2 cards
                pot=10.0,
                phase='flop',
                button_position=1,
                table_info=self.table_info_dict
            )
        
        # Turn - exactly 4 cards
        game_state = GameState(
            players=self.players_dict,
            community_cards=turn_cards_dict,
            pot=10.0,
            phase='turn',
            button_position=1,
            table_info=self.table_info_dict
        )
        self.assertEqual(len(game_state.community_cards), 4)
        
        # River - exactly 5 cards
        game_state = GameState(
            players=self.players_dict,
            community_cards=river_cards_dict,
            pot=10.0,
            phase='river',
            button_position=1,
            table_info=self.table_info_dict
        )
        self.assertEqual(len(game_state.community_cards), 5)
    
    def test_player_validation(self):
        """Test player validation."""
        # Valid players
        game_state = GameState(
            players=self.players_dict,
            pot=10.0,
            phase='preflop',
            button_position=1,
            table_info=self.table_info_dict
        )
        self.assertEqual(len(game_state.players), 3)
        
        # Too few players
        with self.assertRaises(ValidationError):
            GameState(
                players=[self.players_dict[0]],  # Only 1 player
                pot=10.0,
                phase='preflop',
                button_position=1,
                table_info=self.table_info_dict
            )
        
        # Test with exactly 8 players (should be valid)
        eight_players = self.players_dict + [
            Player(seat_number=4, stack=100.0).model_dump(),
            Player(seat_number=5, stack=100.0).model_dump(),
            Player(seat_number=6, stack=100.0).model_dump(),
            Player(seat_number=7, stack=100.0).model_dump(),
            Player(seat_number=8, stack=100.0).model_dump()
        ]
        
        # This should work (8 players is the maximum)
        game_state = GameState(
            players=eight_players,
            pot=10.0,
            phase='preflop',
            button_position=1,
            table_info=self.table_info_dict
        )
        self.assertEqual(len(game_state.players), 8)
    
    def test_unique_seat_numbers_validation(self):
        """Test unique seat numbers validation."""
        # Duplicate seat numbers
        duplicate_players = [
            Player(seat_number=1, stack=100.0).model_dump(),
            Player(seat_number=1, stack=100.0).model_dump()  # Duplicate seat
        ]
        
        with self.assertRaises(ValidationError) as context:
            GameState(
                players=duplicate_players,
                pot=10.0,
                phase='preflop',
                button_position=1,
                table_info=self.table_info_dict
            )
        self.assertIn('unique seat numbers', str(context.exception))
    
    def test_single_hero_validation(self):
        """Test single hero validation."""
        # Multiple heroes
        multiple_heroes = [
            Player(seat_number=1, stack=100.0, is_hero=True).model_dump(),
            Player(seat_number=2, stack=100.0, is_hero=True).model_dump()  # Second hero
        ]
        
        with self.assertRaises(ValidationError) as context:
            GameState(
                players=multiple_heroes,
                pot=10.0,
                phase='preflop',
                button_position=1,
                table_info=self.table_info_dict
            )
        self.assertIn('more than one hero', str(context.exception))
    
    def test_single_dealer_validation(self):
        """Test single dealer validation."""
        # Multiple dealers
        multiple_dealers = [
            Player(seat_number=1, stack=100.0, is_dealer=True).model_dump(),
            Player(seat_number=2, stack=100.0, is_dealer=True).model_dump()  # Second dealer
        ]
        
        with self.assertRaises(ValidationError) as context:
            GameState(
                players=multiple_dealers,
                pot=10.0,
                phase='preflop',
                button_position=1,
                table_info=self.table_info_dict
            )
        self.assertIn('more than one dealer', str(context.exception))
    
    def test_get_hero_method(self):
        """Test get_hero method."""
        game_state = GameState(
            players=self.players_dict,
            pot=10.0,
            phase='preflop',
            button_position=1,
            table_info=self.table_info_dict
        )
        
        hero = game_state.get_hero()
        self.assertIsNotNone(hero)
        self.assertTrue(hero.is_hero)
        self.assertEqual(hero.seat_number, 1)
        
        # No hero
        no_hero_players = [
            Player(seat_number=1, stack=100.0, is_hero=False).model_dump(),
            Player(seat_number=2, stack=100.0, is_hero=False).model_dump()
        ]
        
        game_state_no_hero = GameState(
            players=no_hero_players,
            pot=10.0,
            phase='preflop',
            button_position=1,
            table_info=self.table_info_dict
        )
        
        hero = game_state_no_hero.get_hero()
        self.assertIsNone(hero)
    
    def test_get_player_by_seat_method(self):
        """Test get_player_by_seat method."""
        game_state = GameState(
            players=self.players_dict,
            pot=10.0,
            phase='preflop',
            button_position=1,
            table_info=self.table_info_dict
        )
        
        # Valid seat
        player = game_state.get_player_by_seat(1)
        self.assertIsNotNone(player)
        self.assertEqual(player.seat_number, 1)
        
        # Invalid seat
        player = game_state.get_player_by_seat(99)
        self.assertIsNone(player)
    
    def test_pot_validation(self):
        """Test pot validation."""
        # Valid pot
        game_state = GameState(
            players=self.players_dict,
            pot=0.0,
            phase='preflop',
            button_position=1,
            table_info=self.table_info_dict
        )
        self.assertEqual(game_state.pot, 0.0)
        
        # Invalid pot
        with self.assertRaises(ValidationError):
            GameState(
                players=self.players_dict,
                pot=-10.0,
                phase='preflop',
                button_position=1,
                table_info=self.table_info_dict
            )
    
    def test_active_player_validation(self):
        """Test active player validation."""
        # Valid active player
        game_state = GameState(
            players=self.players_dict,
            pot=10.0,
            phase='preflop',
            active_player=1,
            button_position=1,
            table_info=self.table_info_dict
        )
        self.assertEqual(game_state.active_player, 1)
        
        # Invalid active player
        with self.assertRaises(ValidationError):
            GameState(
                players=self.players_dict,
                pot=10.0,
                phase='preflop',
                active_player=0,  # Invalid seat
                button_position=1,
                table_info=self.table_info_dict
            )
    
    def test_button_position_validation(self):
        """Test button position validation."""
        # Valid button position
        game_state = GameState(
            players=self.players_dict,
            pot=10.0,
            phase='preflop',
            button_position=1,
            table_info=self.table_info_dict
        )
        self.assertEqual(game_state.button_position, 1)
        
        # Invalid button position
        with self.assertRaises(ValidationError):
            GameState(
                players=self.players_dict,
                pot=10.0,
                phase='preflop',
                button_position=0,  # Invalid seat
                table_info=self.table_info_dict
            )


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
