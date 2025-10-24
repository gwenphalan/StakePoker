#!/usr/bin/env python3
"""
Decision Engine for GTO-based poker recommendations.

Central coordinator that integrates GTO charts, equity calculations, range estimation,
and postflop solving to provide optimal recommendations for hero's actions.
"""

import logging
from typing import Optional, List, Tuple
from src.models.game_state import GameState
from src.models.decision import Decision
from src.models.card import Card
from src.advisor.gto_loader import GTOChartLoader
from src.advisor.equity_calculator import EquityCalculator
from src.advisor.range_estimator import RangeEstimator
from src.advisor.postflop_solver import PostflopSolver
from src.config.settings import Settings

logger = logging.getLogger(__name__)


class DecisionEngine:
    """Central decision engine for GTO poker recommendations."""
    
    def __init__(self):
        """Initialize decision engine with all advisor components."""
        self.settings = Settings()
        
        # Initialize advisor components
        self.gto_loader = GTOChartLoader()
        self.equity_calculator = EquityCalculator()
        self.range_estimator = RangeEstimator()
        self.postflop_solver = PostflopSolver()
        
        # Create configuration settings
        self.settings.create("advisor.decision.min_confidence", default=0.7)
        self.settings.create("advisor.decision.equity_buffer", default=0.05)
        self.settings.create("advisor.decision.preflop_base_confidence", default=0.9)
        
        # Load configuration
        self.min_confidence = self.settings.get("advisor.decision.min_confidence")
        self.equity_buffer = self.settings.get("advisor.decision.equity_buffer")
        self.preflop_base_confidence = self.settings.get("advisor.decision.preflop_base_confidence")
        
        logger.info("Initialized decision engine with all components")
    
    def get_recommendation(self, game_state: GameState) -> Optional[Decision]:
        """
        Get GTO recommendation for current game state.
        
        Args:
            game_state: Current game state with all player and board information
            
        Returns:
            Decision recommendation or None if not hero's turn or insufficient data
        """
        hero = game_state.get_hero()
        
        # Check if it's hero's turn
        if not hero:
            logger.debug("No hero found in game state")
            return None
        
        if not hero.timer_state:
            logger.debug("Not hero's turn (no timer state)")
            return None
        
        # Check if hero has hole cards
        if len(hero.hole_cards) != 2:
            logger.warning(f"Hero has {len(hero.hole_cards)} hole cards, need 2")
            return None
        
        # Route to appropriate decision method based on phase
        if game_state.phase == 'preflop':
            return self._preflop_decision(game_state)
        else:
            return self._postflop_decision(game_state)
    
    def _preflop_decision(self, game_state: GameState) -> Decision:
        """
        Get preflop decision using GTO charts.
        
        Args:
            game_state: Current game state
            
        Returns:
            Decision based on GTO preflop ranges
        """
        hero = game_state.get_hero()
        if not hero:
            return self._default_fold_decision("Hero not found")
        
        # Format hero's hand
        hero_hand = self._format_hand(hero.hole_cards)
        if not hero_hand:
            logger.error("Failed to format hero's hand")
            return self._default_fold_decision("Invalid hand format")
        
        # Get hero's position
        position = hero.position
        if not position:
            logger.error("Hero has no position")
            return self._default_fold_decision("Position unknown")
        
        # Count active players
        player_count = len([p for p in game_state.players if p.is_active])
        
        # Determine action context (opening, vs_open, vs_3bet, etc.)
        action_context = self._get_preflop_context(game_state)
        
        logger.info(f"Preflop decision: {hero_hand} from {position}, context: {action_context}, {player_count} players")
        
        try:
            # Get GTO recommendation
            action, bet_size_bb, reasoning = self.gto_loader.get_preflop_decision(
                position, hero_hand, player_count, action_context
            )
            
            # Convert bet size from BBs to chips
            bet_amount = None
            if bet_size_bb > 0:
                bet_amount = bet_size_bb * game_state.table_info.bb
                # Ensure we don't bet more than our stack
                bet_amount = min(bet_amount, hero.stack)
            
            # Calculate confidence
            confidence = self._calculate_preflop_confidence(hero_hand, position, action, action_context)
            
            logger.info(f"GTO recommendation: {action} {bet_amount if bet_amount else ''} (confidence: {confidence:.2f})")
            
            return Decision(
                action=action,
                amount=bet_amount,
                confidence=confidence,
                reasoning=reasoning,
                equity=None,  # Preflop equity not calculated
                pot_odds=None
            )
            
        except Exception as e:
            logger.error(f"Error getting preflop decision: {e}")
            return self._default_fold_decision(f"Error: {e}")
    
    def _postflop_decision(self, game_state: GameState) -> Decision:
        """
        Get postflop decision using equity calculations and solver.
        
        Args:
            game_state: Current game state
            
        Returns:
            Decision based on equity vs pot odds and EV calculations
        """
        hero = game_state.get_hero()
        if not hero:
            return self._default_fold_decision("Hero not found")
        
        logger.info(f"Postflop decision: {game_state.phase}, board: {[str(c) for c in game_state.community_cards]}")
        
        try:
            # Estimate opponent ranges
            opponent_ranges = self.range_estimator.estimate_ranges(game_state)
            
            if not opponent_ranges:
                logger.warning("No opponent ranges estimated, using default range")
                opponent_ranges = self._get_default_opponent_range()
            
            logger.debug(f"Estimated opponent range: {len(opponent_ranges)} hands")
            
            # Calculate hero's equity vs opponent ranges
            equity = self.equity_calculator.calculate_equity_vs_range(
                hero.hole_cards,
                game_state.community_cards,
                opponent_ranges
            )
            
            logger.info(f"Hero equity: {equity:.1%}")
            
            # Calculate pot odds (if facing a bet)
            pot_odds = self._calculate_pot_odds(game_state)
            
            if pot_odds > 0:
                logger.info(f"Pot odds: {pot_odds:.1%}")
            
            # Get postflop solver recommendation
            decision = self.postflop_solver.get_recommendation(
                game_state, equity, pot_odds
            )
            
            logger.info(f"Postflop recommendation: {decision.action} (confidence: {decision.confidence:.2f})")
            
            return decision
            
        except Exception as e:
            logger.error(f"Error getting postflop decision: {e}", exc_info=True)
            return self._default_fold_decision(f"Error: {e}")
    
    def _get_preflop_context(self, game_state: GameState) -> str:
        """
        Determine preflop action context based on betting.
        
        Args:
            game_state: Current game state
            
        Returns:
            Action context string (opening, vs_open, vs_3bet, vs_4bet, blind_defense)
        """
        hero = game_state.get_hero()
        if not hero:
            return "opening"
        
        # Count how many players have bet/raised
        bb = game_state.table_info.bb
        players_with_bets = [
            p for p in game_state.players 
            if p.current_bet > bb and not p.is_hero and p.is_active
        ]
        
        bet_count = len(players_with_bets)
        
        # Check if hero is in blinds
        in_blinds = hero.position in ['SB', 'BB']
        
        # Determine context
        if bet_count == 0:
            # No one has raised yet
            return "opening"
        elif bet_count == 1:
            # Facing one raise
            if in_blinds:
                return "blind_defense"
            else:
                return "vs_open"
        elif bet_count == 2:
            # Facing a 3-bet
            return "vs_3bet"
        elif bet_count >= 3:
            # Facing a 4-bet or more
            return "vs_4bet"
        
        return "opening"
    
    def _calculate_preflop_confidence(self, hand: str, position: str, 
                                     action: str, context: str) -> float:
        """
        Calculate confidence for preflop decisions.
        
        Args:
            hand: Hand notation (e.g., "AKs", "99")
            position: Hero's position
            action: Recommended action
            context: Action context
            
        Returns:
            Confidence score (0.5 to 0.95)
        """
        base_confidence = self.preflop_base_confidence
        
        # Adjust based on action type
        if action == "fold":
            # Folding decisions are very confident
            base_confidence = 0.95
        elif action in ["raise", "3bet", "4bet", "5bet"]:
            # Aggressive actions slightly less confident
            base_confidence = 0.85
        elif action == "call":
            # Calling decisions moderate confidence
            base_confidence = 0.80
        
        # Adjust based on hand strength
        premium_hands = ['AA', 'KK', 'QQ', 'JJ', 'AKs', 'AKo']
        if hand in premium_hands:
            base_confidence += 0.05
        
        # Adjust based on context
        if context in ['vs_3bet', 'vs_4bet']:
            # More complex situations = slightly lower confidence
            base_confidence -= 0.05
        
        # Clamp between 0.5 and 0.95
        return max(0.5, min(0.95, base_confidence))
    
    def _calculate_pot_odds(self, game_state: GameState) -> float:
        """
        Calculate current pot odds hero is getting.
        
        Args:
            game_state: Current game state
            
        Returns:
            Pot odds as decimal (0.0 if not facing a bet)
        """
        hero = game_state.get_hero()
        if not hero:
            return 0.0
        
        # Find the current bet to call
        max_bet = max((p.current_bet for p in game_state.players if p.is_active), default=0)
        bet_to_call = max_bet - hero.current_bet
        
        if bet_to_call <= 0:
            return 0.0  # Not facing a bet
        
        # Calculate pot odds
        return self.equity_calculator.calculate_pot_odds(game_state.pot, bet_to_call)
    
    def _format_hand(self, cards: List[Card]) -> str:
        """
        Format hole cards to hand notation.
        
        Args:
            cards: List of 2 cards
            
        Returns:
            Hand notation (e.g., "AKs", "99", "AKo") or empty string if invalid
        """
        if len(cards) != 2:
            logger.error(f"Invalid card count: {len(cards)}")
            return ""
        
        card1, card2 = cards
        rank1, rank2 = card1.rank, card2.rank
        
        # Rank order for sorting
        rank_order = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        
        # Sort ranks (higher rank first)
        try:
            if rank_order.index(rank1) < rank_order.index(rank2):
                rank1, rank2 = rank2, rank1
        except ValueError:
            logger.error(f"Invalid ranks: {rank1}, {rank2}")
            return ""
        
        # Determine if suited, pair, or offsuit
        if rank1 == rank2:
            # Pocket pair
            return f"{rank1}{rank2}"
        elif card1.suit == card2.suit:
            # Suited
            return f"{rank1}{rank2}s"
        else:
            # Offsuit
            return f"{rank1}{rank2}o"
    
    def _get_default_opponent_range(self) -> List[str]:
        """
        Get a default opponent range when estimation fails.
        
        Returns:
            Conservative default range
        """
        return [
            'AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', '55',
            'AKs', 'AKo', 'AQs', 'AQo', 'AJs', 'AJo', 'ATs', 'ATo',
            'KQs', 'KQo', 'KJs', 'KJo', 'KTs', 'QJs', 'QJo', 'QTs', 'JTs'
        ]
    
    def _default_fold_decision(self, reason: str) -> Decision:
        """
        Return a default fold decision with given reason.
        
        Args:
            reason: Reason for folding
            
        Returns:
            Decision to fold
        """
        return Decision(
            action="fold",
            amount=None,
            confidence=0.5,
            reasoning=f"Default fold: {reason}",
            equity=0.0,
            pot_odds=0.0
        )

