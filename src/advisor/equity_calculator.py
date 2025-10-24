#!/usr/bin/env python3
"""
Equity calculator using Treys library for poker hand evaluation.

Provides equity calculation against opponent ranges, Monte Carlo simulations,
and integration with the StakePoker decision engine.
"""

import logging
import random
from typing import List, Dict, Optional, Tuple
from treys import Evaluator, Card as TreysCard
from src.models.card import Card
from src.models.game_state import GameState
from src.models.player import Player
from src.config.settings import Settings

logger = logging.getLogger(__name__)


class EquityCalculator:
    """Equity calculator using Treys for hand evaluation and Monte Carlo simulation."""
    
    def __init__(self):
        """Initialize equity calculator with Treys evaluator."""
        self.evaluator = Evaluator()
        self.settings = Settings()
        
        # Create settings with defaults
        self.settings.create("advisor.equity.monte_carlo_iterations", default=10000)
        self.settings.create("advisor.equity.cache_size", default=1000)
        
        # Configuration
        self.monte_carlo_iterations = self.settings.get("advisor.equity.monte_carlo_iterations")
        self.cache_size = self.settings.get("advisor.equity.cache_size")
        
        # Cache for expensive calculations
        self._equity_cache: Dict[str, float] = {}
        
        logger.info("Initialized equity calculator with Treys")
    
    def _card_to_treys(self, card: Card) -> int:
        """
        Convert StakePoker Card to Treys card integer.
        
        Treys uses Card.new() which takes rank and suit strings.
        """
        # Map our rank notation to Treys notation
        rank_map = {
            '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8',
            '9': '9', 'T': 'T', 'J': 'J', 'Q': 'Q', 'K': 'K', 'A': 'A'
        }
        
        # Map our suit notation to Treys notation (first letter)
        suit_map = {
            'clubs': 'c',
            'diamonds': 'd',
            'hearts': 'h',
            'spades': 's'
        }
        
        rank_str = rank_map[card.rank]
        suit_str = suit_map[card.suit]
        
        # Use Treys Card.new() to create the card integer
        return TreysCard.new(rank_str + suit_str)
    
    def _cards_to_treys(self, cards: List[Card]) -> List[int]:
        """Convert list of StakePoker Cards to Treys card integers."""
        return [self._card_to_treys(card) for card in cards]
    
    def _hand_to_string(self, cards: List[Card]) -> str:
        """Convert cards to string notation for caching."""
        return ''.join(str(card) for card in sorted(cards, key=lambda c: (c.rank, c.suit)))
    
    def calculate_equity_vs_range(self, hero_cards: List[Card], 
                                 community_cards: List[Card],
                                 opponent_range: List[str]) -> float:
        """
        Calculate hero's equity against a range of opponent hands.
        
        Args:
            hero_cards: Hero's hole cards
            community_cards: Community cards on board
            opponent_range: List of hand notations (e.g., ['AA', 'KK', 'AKs'])
            
        Returns:
            Equity as float between 0.0 and 1.0
        """
        if len(hero_cards) != 2:
            logger.warning(f"Invalid hero cards count: {len(hero_cards)}")
            return 0.0
        
        # Create cache key
        cache_key = f"{self._hand_to_string(hero_cards)}|{self._hand_to_string(community_cards)}|{sorted(opponent_range)}"
        
        if cache_key in self._equity_cache:
            return self._equity_cache[cache_key]
        
        # Convert to Treys format
        hero_treys = self._cards_to_treys(hero_cards)
        community_treys = self._cards_to_treys(community_cards)
        
        # Generate all possible opponent hands from range
        opponent_hands = self._expand_range_to_hands(opponent_range, hero_cards + community_cards)
        
        if not opponent_hands:
            logger.warning("No valid opponent hands in range")
            return 0.0
        
        # Monte Carlo simulation
        wins = 0
        ties = 0
        total_games = 0
        
        # Sample random opponent hands for efficiency
        sample_size = min(len(opponent_hands), self.monte_carlo_iterations)
        sampled_hands = random.sample(opponent_hands, sample_size)
        
        for opponent_hand in sampled_hands:
            # Generate remaining community cards if needed
            remaining_cards = self._get_remaining_cards(hero_cards + community_cards + opponent_hand)
            
            if len(community_cards) < 5:
                # Complete the board randomly
                board = community_cards + random.sample(remaining_cards, 5 - len(community_cards))
            else:
                board = community_cards
            
            # Convert board to Treys format
            board_treys = self._cards_to_treys(board)
            opponent_treys = self._cards_to_treys(opponent_hand)
            
            # Evaluate hands - Treys evaluator takes (hand, board) where hand is 2 cards and board is 5 cards
            try:
                hero_strength = self.evaluator.evaluate(board_treys, hero_treys)
                opponent_strength = self.evaluator.evaluate(board_treys, opponent_treys)
                
                if hero_strength < opponent_strength:  # Lower is better in Treys
                    wins += 1
                elif hero_strength == opponent_strength:
                    ties += 1
                
                total_games += 1
            except Exception as e:
                logger.debug(f"Error evaluating hand: {e}")
                continue
        
        # Calculate equity
        equity = (wins + ties * 0.5) / total_games if total_games > 0 else 0.0
        
        # Cache result
        if len(self._equity_cache) < self.cache_size:
            self._equity_cache[cache_key] = equity
        
        logger.debug(f"Equity calculation: {equity:.3f} vs {len(opponent_hands)} hands")
        return equity
    
    def calculate_equity_vs_specific_hand(self, hero_cards: List[Card],
                                         opponent_cards: List[Card],
                                         community_cards: List[Card]) -> float:
        """
        Calculate equity against a specific opponent hand.
        
        Args:
            hero_cards: Hero's hole cards
            opponent_cards: Opponent's hole cards
            community_cards: Community cards on board
            
        Returns:
            Equity as float between 0.0 and 1.0
        """
        if len(hero_cards) != 2 or len(opponent_cards) != 2:
            logger.warning("Invalid card counts for equity calculation")
            return 0.0
        
        # Convert to Treys format
        hero_treys = self._cards_to_treys(hero_cards)
        opponent_treys = self._cards_to_treys(opponent_cards)
        
        # Generate all possible remaining community cards
        used_cards = hero_cards + opponent_cards + community_cards
        remaining_cards = self._get_remaining_cards(used_cards)
        
        wins = 0
        ties = 0
        total_games = 0
        
        # Monte Carlo simulation for remaining cards
        iterations = min(len(remaining_cards), self.monte_carlo_iterations)
        
        for _ in range(iterations):
            # Complete the board randomly
            if len(community_cards) < 5:
                board_cards = community_cards + random.sample(remaining_cards, 5 - len(community_cards))
            else:
                board_cards = community_cards
            
            board_treys = self._cards_to_treys(board_cards)
            
            # Evaluate hands - Treys evaluator takes (board, hand)
            try:
                hero_strength = self.evaluator.evaluate(board_treys, hero_treys)
                opponent_strength = self.evaluator.evaluate(board_treys, opponent_treys)
                
                if hero_strength < opponent_strength:  # Lower is better in Treys
                    wins += 1
                elif hero_strength == opponent_strength:
                    ties += 1
                
                total_games += 1
            except Exception as e:
                logger.debug(f"Error evaluating hand: {e}")
                continue
        
        equity = (wins + ties * 0.5) / total_games if total_games > 0 else 0.0
        logger.debug(f"Equity vs specific hand: {equity:.3f}")
        return equity
    
    def calculate_pot_odds(self, pot_size: float, bet_to_call: float) -> float:
        """
        Calculate pot odds.
        
        Args:
            pot_size: Current pot size
            bet_to_call: Amount needed to call
            
        Returns:
            Pot odds as decimal (e.g., 0.25 for 25%)
        """
        if bet_to_call <= 0:
            return 0.0
        
        return bet_to_call / (pot_size + bet_to_call)
    
    def calculate_implied_odds(self, pot_size: float, bet_to_call: float,
                              future_bets: float) -> float:
        """
        Calculate implied odds including future betting.
        
        Args:
            pot_size: Current pot size
            bet_to_call: Amount needed to call
            future_bets: Expected future bets
            
        Returns:
            Implied odds as decimal
        """
        if bet_to_call <= 0:
            return 0.0
        
        return bet_to_call / (pot_size + bet_to_call + future_bets)
    
    def _expand_range_to_hands(self, range_notation: List[str], 
                              used_cards: List[Card]) -> List[List[Card]]:
        """
        Expand range notation to specific hands.
        
        Args:
            range_notation: List of hand notations (e.g., ['AA', 'AKs', 'KQo'])
            used_cards: Cards already in use
            
        Returns:
            List of specific two-card hands
        """
        hands = []
        used_treys = self._cards_to_treys(used_cards)
        
        for notation in range_notation:
            try:
                # Parse hand notation (simplified - you may want to use a proper parser)
                if len(notation) == 2:  # Pocket pair like 'AA'
                    rank = notation[0]
                    hands.extend(self._generate_pocket_pairs(rank, used_treys))
                elif len(notation) == 3:  # Suited/offsuit like 'AKs' or 'AKo'
                    rank1, rank2, suited = notation[0], notation[1], notation[2] == 's'
                    hands.extend(self._generate_two_card_hands(rank1, rank2, suited, used_treys))
                else:
                    logger.warning(f"Unsupported hand notation: {notation}")
            except Exception as e:
                logger.warning(f"Error parsing hand notation {notation}: {e}")
        
        return hands
    
    def _generate_pocket_pairs(self, rank: str, used_cards: List[int]) -> List[List[Card]]:
        """Generate all pocket pairs for a rank."""
        hands = []
        rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6,
                   '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
        suit_map = {0: 'clubs', 1: 'diamonds', 2: 'hearts', 3: 'spades'}
        
        if rank not in rank_map:
            return hands
        
        base_card = rank_map[rank] * 4
        
        for i in range(4):
            for j in range(i + 1, 4):
                card1_treys = base_card + i
                card2_treys = base_card + j
                
                if card1_treys not in used_cards and card2_treys not in used_cards:
                    card1 = Card(rank=rank, suit=suit_map[i])
                    card2 = Card(rank=rank, suit=suit_map[j])
                    hands.append([card1, card2])
        
        return hands
    
    def _generate_two_card_hands(self, rank1: str, rank2: str, suited: bool, 
                                used_cards: List[int]) -> List[List[Card]]:
        """Generate two-card hands (suited or offsuit)."""
        hands = []
        rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6,
                   '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
        suit_map = {0: 'clubs', 1: 'diamonds', 2: 'hearts', 3: 'spades'}
        
        if rank1 not in rank_map or rank2 not in rank_map:
            return hands
        
        base1 = rank_map[rank1] * 4
        base2 = rank_map[rank2] * 4
        
        if suited:
            for suit in range(4):
                card1_treys = base1 + suit
                card2_treys = base2 + suit
                
                if card1_treys not in used_cards and card2_treys not in used_cards:
                    card1 = Card(rank=rank1, suit=suit_map[suit])
                    card2 = Card(rank=rank2, suit=suit_map[suit])
                    hands.append([card1, card2])
        else:
            for suit1 in range(4):
                for suit2 in range(4):
                    if suit1 != suit2:
                        card1_treys = base1 + suit1
                        card2_treys = base2 + suit2
                        
                        if card1_treys not in used_cards and card2_treys not in used_cards:
                            card1 = Card(rank=rank1, suit=suit_map[suit1])
                            card2 = Card(rank=rank2, suit=suit_map[suit2])
                            hands.append([card1, card2])
        
        return hands
    
    def _get_remaining_cards(self, used_cards: List[Card]) -> List[Card]:
        """Get all cards not in the used_cards list."""
        all_cards = []
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suits = ['clubs', 'diamonds', 'hearts', 'spades']
        
        for rank in ranks:
            for suit in suits:
                all_cards.append(Card(rank=rank, suit=suit))
        
        used_set = set((card.rank, card.suit) for card in used_cards)
        remaining = [card for card in all_cards if (card.rank, card.suit) not in used_set]
        
        return remaining
    
    def get_hand_strength(self, cards: List[Card], community_cards: List[Card]) -> Tuple[int, str]:
        """
        Get hand strength and hand type.
        
        Args:
            cards: Player's cards (2 for hole cards, 5+ for full hand)
            community_cards: Community cards
            
        Returns:
            Tuple of (strength, hand_type) where strength is Treys integer
        """
        if len(cards) < 2:
            return 7462, "High Card"  # Worst possible hand
        
        all_cards = cards + community_cards
        if len(all_cards) < 7:
            return 7462, "Incomplete Hand"
        
        # Convert to Treys format
        hand_treys = self._cards_to_treys(cards[:2])  # 2 hole cards
        board_treys = self._cards_to_treys(community_cards[:5])  # 5 community cards
        
        try:
            strength = self.evaluator.evaluate(board_treys, hand_treys)
            hand_type = self.evaluator.class_to_string(self.evaluator.get_rank_class(strength))
            return strength, hand_type
        except Exception as e:
            logger.error(f"Error evaluating hand strength: {e}")
            return 7462, "Error"
    
    def clear_cache(self):
        """Clear the equity calculation cache."""
        self._equity_cache.clear()
        logger.info("Equity calculation cache cleared")
