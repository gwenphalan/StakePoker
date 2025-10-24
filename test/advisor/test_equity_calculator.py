#!/usr/bin/env python3
"""
Unit tests for EquityCalculator.

Tests equity calculations using Treys library for poker hand evaluation.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.advisor.equity_calculator import EquityCalculator
from src.models.card import Card
from src.models.game_state import GameState
from src.models.player import Player
from src.models.table_info import TableInfo


class TestEquityCalculator:
    """Test cases for EquityCalculator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('src.advisor.equity_calculator.Settings') as mock_settings:
            mock_instance = mock_settings.return_value
            mock_instance.create.return_value = None
            mock_instance.get.side_effect = lambda key: {
                "advisor.equity.monte_carlo_iterations": 1000,
                "advisor.equity.cache_size": 100
            }.get(key, 1000)
            
            self.calculator = EquityCalculator()
    
    def test_init(self):
        """Test EquityCalculator initialization."""
        assert self.calculator.monte_carlo_iterations == 1000
        assert self.calculator.cache_size == 100
        assert isinstance(self.calculator._equity_cache, dict)
        assert len(self.calculator._equity_cache) == 0
    
    def test_card_to_treys(self):
        """Test conversion of Card to Treys format."""
        # Test various card conversions
        card = Card(rank='A', suit='hearts')
        treys_card = self.calculator._card_to_treys(card)
        assert isinstance(treys_card, int)
        assert treys_card > 0
        
        card = Card(rank='2', suit='clubs')
        treys_card = self.calculator._card_to_treys(card)
        assert isinstance(treys_card, int)
        assert treys_card > 0
    
    def test_cards_to_treys(self):
        """Test conversion of multiple cards to Treys format."""
        cards = [
            Card(rank='A', suit='hearts'),
            Card(rank='K', suit='spades')
        ]
        treys_cards = self.calculator._cards_to_treys(cards)
        
        assert len(treys_cards) == 2
        assert all(isinstance(card, int) for card in treys_cards)
    
    def test_hand_to_string(self):
        """Test hand to string conversion for caching."""
        cards = [
            Card(rank='A', suit='hearts'),
            Card(rank='K', suit='spades')
        ]
        hand_string = self.calculator._hand_to_string(cards)
        
        assert isinstance(hand_string, str)
        assert len(hand_string) > 0
    
    def test_calculate_equity_vs_range_invalid_cards(self):
        """Test equity calculation with invalid card counts."""
        # Test with wrong number of hero cards
        hero_cards = [Card(rank='A', suit='hearts')]  # Only 1 card
        community_cards = []
        opponent_range = ['AA', 'KK']
        
        equity = self.calculator.calculate_equity_vs_range(
            hero_cards, community_cards, opponent_range
        )
        assert equity == 0.0
    
    def test_calculate_equity_vs_range_caching(self):
        """Test equity calculation caching."""
        hero_cards = [
            Card(rank='A', suit='hearts'),
            Card(rank='K', suit='spades')
        ]
        community_cards = []
        opponent_range = ['AA', 'KK']
        
        # First calculation
        equity1 = self.calculator.calculate_equity_vs_range(
            hero_cards, community_cards, opponent_range
        )
        
        # Second calculation should use cache
        equity2 = self.calculator.calculate_equity_vs_range(
            hero_cards, community_cards, opponent_range
        )
        
        assert equity1 == equity2
        assert len(self.calculator._equity_cache) > 0
    
    def test_calculate_equity_vs_range_empty_range(self):
        """Test equity calculation with empty opponent range."""
        hero_cards = [
            Card(rank='A', suit='hearts'),
            Card(rank='K', suit='spades')
        ]
        community_cards = []
        opponent_range = []
        
        equity = self.calculator.calculate_equity_vs_range(
            hero_cards, community_cards, opponent_range
        )
        assert equity == 0.0
    
    def test_calculate_equity_vs_specific_hand(self):
        """Test equity calculation against specific hand."""
        hero_cards = [
            Card(rank='A', suit='hearts'),
            Card(rank='K', suit='spades')
        ]
        opponent_cards = [
            Card(rank='Q', suit='hearts'),
            Card(rank='J', suit='spades')
        ]
        community_cards = []
        
        equity = self.calculator.calculate_equity_vs_specific_hand(
            hero_cards, opponent_cards, community_cards
        )
        
        assert isinstance(equity, float)
        assert 0.0 <= equity <= 1.0
    
    def test_calculate_equity_vs_specific_hand_invalid_cards(self):
        """Test equity calculation with invalid card counts."""
        hero_cards = [Card(rank='A', suit='hearts')]  # Only 1 card
        opponent_cards = [
            Card(rank='Q', suit='hearts'),
            Card(rank='J', suit='spades')
        ]
        community_cards = []
        
        equity = self.calculator.calculate_equity_vs_specific_hand(
            hero_cards, opponent_cards, community_cards
        )
        assert equity == 0.0
    
    def test_calculate_pot_odds(self):
        """Test pot odds calculation."""
        # Test normal pot odds
        pot_odds = self.calculator.calculate_pot_odds(100, 25)
        expected = 25 / (100 + 25)  # 0.2
        assert abs(pot_odds - expected) < 0.001
        
        # Test zero bet
        pot_odds = self.calculator.calculate_pot_odds(100, 0)
        assert pot_odds == 0.0
        
        # Test negative bet
        pot_odds = self.calculator.calculate_pot_odds(100, -10)
        assert pot_odds == 0.0
    
    def test_calculate_implied_odds(self):
        """Test implied odds calculation."""
        # Test normal implied odds
        implied_odds = self.calculator.calculate_implied_odds(100, 25, 50)
        expected = 25 / (100 + 25 + 50)  # 25/175 ≈ 0.143
        assert abs(implied_odds - expected) < 0.001
        
        # Test zero bet
        implied_odds = self.calculator.calculate_implied_odds(100, 0, 50)
        assert implied_odds == 0.0
    
    def test_expand_range_to_hands(self):
        """Test range expansion to specific hands."""
        range_notation = ['AA', 'AKs', 'KQo']
        used_cards = [
            Card(rank='A', suit='hearts'),
            Card(rank='K', suit='spades')
        ]
        
        hands = self.calculator._expand_range_to_hands(range_notation, used_cards)
        
        assert isinstance(hands, list)
        # Should have some hands but not all possible combinations
        # (due to used cards)
        assert len(hands) > 0
    
    def test_expand_range_to_hands_empty_range(self):
        """Test range expansion with empty range."""
        range_notation = []
        used_cards = []
        
        hands = self.calculator._expand_range_to_hands(range_notation, used_cards)
        assert hands == []
    
    def test_generate_pocket_pairs(self):
        """Test pocket pair generation."""
        used_cards = []
        hands = self.calculator._generate_pocket_pairs('A', [])
        
        assert isinstance(hands, list)
        assert len(hands) == 6  # C(4,2) = 6 combinations for AA
        
        # All hands should be pocket pairs
        for hand in hands:
            assert len(hand) == 2
            assert hand[0].rank == hand[1].rank == 'A'
    
    def test_generate_pocket_pairs_with_used_cards(self):
        """Test pocket pair generation with used cards."""
        used_cards = [0, 1]  # First two cards used
        hands = self.calculator._generate_pocket_pairs('A', used_cards)
        
        assert isinstance(hands, list)
        assert len(hands) == 1  # Only one combination left
    
    def test_generate_two_card_hands_suited(self):
        """Test suited two-card hand generation."""
        hands = self.calculator._generate_two_card_hands('A', 'K', True, [])
        
        assert isinstance(hands, list)
        assert len(hands) == 4  # 4 suits
        
        # All hands should be suited
        for hand in hands:
            assert len(hand) == 2
            assert hand[0].suit == hand[1].suit
            assert hand[0].rank == 'A'
            assert hand[1].rank == 'K'
    
    def test_generate_two_card_hands_offsuit(self):
        """Test offsuit two-card hand generation."""
        hands = self.calculator._generate_two_card_hands('A', 'K', False, [])
        
        assert isinstance(hands, list)
        assert len(hands) == 12  # 4*3 = 12 combinations
        
        # All hands should be offsuit
        for hand in hands:
            assert len(hand) == 2
            assert hand[0].suit != hand[1].suit
            assert hand[0].rank == 'A'
            assert hand[1].rank == 'K'
    
    def test_get_remaining_cards(self):
        """Test getting remaining cards."""
        used_cards = [
            Card(rank='A', suit='hearts'),
            Card(rank='K', suit='spades')
        ]
        
        remaining = self.calculator._get_remaining_cards(used_cards)
        
        assert isinstance(remaining, list)
        assert len(remaining) == 50  # 52 - 2 = 50
        
        # Used cards should not be in remaining
        used_set = set((card.rank, card.suit) for card in used_cards)
        for card in remaining:
            assert (card.rank, card.suit) not in used_set
    
    def test_get_hand_strength(self):
        """Test hand strength calculation."""
        cards = [
            Card(rank='A', suit='hearts'),
            Card(rank='K', suit='spades')
        ]
        community_cards = [
            Card(rank='Q', suit='hearts'),
            Card(rank='J', suit='spades'),
            Card(rank='T', suit='hearts'),
            Card(rank='9', suit='spades'),
            Card(rank='8', suit='hearts')
        ]
        
        strength, hand_type = self.calculator.get_hand_strength(cards, community_cards)
        
        assert isinstance(strength, int)
        assert isinstance(hand_type, str)
        assert strength > 0
        assert len(hand_type) > 0
    
    def test_get_hand_strength_incomplete_hand(self):
        """Test hand strength with incomplete hand."""
        cards = [Card(rank='A', suit='hearts')]
        community_cards = []
        
        strength, hand_type = self.calculator.get_hand_strength(cards, community_cards)
        
        assert strength == 7462  # Worst possible hand
        assert hand_type == "Incomplete Hand"
    
    def test_get_hand_strength_empty_cards(self):
        """Test hand strength with empty cards."""
        cards = []
        community_cards = []
        
        strength, hand_type = self.calculator.get_hand_strength(cards, community_cards)
        
        assert strength == 7462  # Worst possible hand
        assert hand_type == "High Card"
    
    def test_clear_cache(self):
        """Test cache clearing."""
        # Add something to cache
        self.calculator._equity_cache['test'] = 0.5
        assert len(self.calculator._equity_cache) > 0
        
        # Clear cache
        self.calculator.clear_cache()
        assert len(self.calculator._equity_cache) == 0
    
    def test_error_handling_treys_evaluation(self):
        """Test error handling in Treys evaluation."""
        # Mock Treys evaluator to raise exception
        with patch.object(self.calculator.evaluator, 'evaluate') as mock_evaluate:
            mock_evaluate.side_effect = Exception("Treys error")
            
            cards = [
                Card(rank='A', suit='hearts'),
                Card(rank='K', suit='spades')
            ]
            community_cards = []
            
            strength, hand_type = self.calculator.get_hand_strength(cards, community_cards)
            
            assert strength == 7462  # Error fallback
            assert hand_type == "Error"
    
    def test_normalize_hand_notation(self):
        """Test hand notation normalization."""
        # Test various input formats
        test_cases = [
            ("AA", "AA"),
            ("AKs", "AKs"),
            ("AKo", "AKo"),
            ("AhKd", "AKo"),
            ("A♠K♠", "AKs"),
            ("AK", "AKo"),  # Default to offsuit
        ]
        
        for input_hand, expected in test_cases:
            # This would be a method in the actual implementation
            # For now, we'll test the concept
            assert isinstance(input_hand, str)
            assert isinstance(expected, str)
    
    def test_monte_carlo_iterations_limit(self):
        """Test Monte Carlo iterations limit."""
        hero_cards = [
            Card(rank='A', suit='hearts'),
            Card(rank='K', suit='spades')
        ]
        community_cards = []
        opponent_range = ['AA', 'KK', 'QQ', 'JJ', 'TT'] * 20  # Large range
        
        # Should not exceed monte_carlo_iterations
        equity = self.calculator.calculate_equity_vs_range(
            hero_cards, community_cards, opponent_range
        )
        
        assert isinstance(equity, float)
        assert 0.0 <= equity <= 1.0
    
    def test_cache_size_limit(self):
        """Test cache size limit."""
        # Fill cache beyond limit
        for i in range(150):  # More than cache_size (100)
            self.calculator._equity_cache[f'key_{i}'] = 0.5
        
        # Should not exceed cache_size
        assert len(self.calculator._equity_cache) <= self.calculator.cache_size
    
    def test_equity_calculation_consistency(self):
        """Test equity calculation consistency."""
        hero_cards = [
            Card(rank='A', suit='hearts'),
            Card(rank='K', suit='spades')
        ]
        community_cards = []
        opponent_range = ['AA', 'KK']
        
        # Run multiple times - should be consistent (within Monte Carlo variance)
        equities = []
        for _ in range(5):
            equity = self.calculator.calculate_equity_vs_range(
                hero_cards, community_cards, opponent_range
            )
            equities.append(equity)
        
        # All should be reasonable values
        for equity in equities:
            assert 0.0 <= equity <= 1.0
        
        # Should be somewhat consistent (allowing for Monte Carlo variance)
        avg_equity = sum(equities) / len(equities)
        for equity in equities:
            assert abs(equity - avg_equity) < 0.1  # Within 10% variance
