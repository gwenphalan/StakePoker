#!/usr/bin/env python3
"""
Postflop Solver for EV-based postflop decisions.

Calculates expected value for different actions and provides optimal
recommendations based on equity, pot odds, and implied odds.
"""

import logging
from typing import List, Optional, Tuple
from src.models.game_state import GameState
from src.models.decision import Decision, AlternativeAction
from src.models.player import Player
from src.models.card import Card
from src.config.settings import Settings

logger = logging.getLogger(__name__)


class PostflopSolver:
    """Postflop solver for EV-based decision making."""
    
    def __init__(self):
        """Initialize postflop solver with configuration."""
        self.settings = Settings()
        
        # Create settings with defaults
        self.settings.create("advisor.postflop.min_equity_to_call", default=0.25)
        self.settings.create("advisor.postflop.min_equity_to_bet", default=0.35)
        self.settings.create("advisor.postflop.bet_sizing_factor", default=0.75)
        self.settings.create("advisor.postflop.raise_sizing_factor", default=3.0)
        self.settings.create("advisor.postflop.base_fold_equity", default=0.3)
        
        # Load configuration
        self.min_equity_to_call = self.settings.get("advisor.postflop.min_equity_to_call")
        self.min_equity_to_bet = self.settings.get("advisor.postflop.min_equity_to_bet")
        self.bet_sizing_factor = self.settings.get("advisor.postflop.bet_sizing_factor")
        self.raise_sizing_factor = self.settings.get("advisor.postflop.raise_sizing_factor")
        self.base_fold_equity = self.settings.get("advisor.postflop.base_fold_equity")
        
        logger.info("Initialized postflop solver")
    
    def get_recommendation(self, game_state: GameState, equity: float, 
                          pot_odds: float) -> Decision:
        """
        Get postflop recommendation based on equity and pot odds.
        
        Args:
            game_state: Current game state
            equity: Hero's equity vs opponent ranges
            pot_odds: Current pot odds (if facing a bet)
            
        Returns:
            Decision recommendation with action, amount, and reasoning
        """
        hero = game_state.get_hero()
        if not hero:
            logger.warning("No hero found in game state")
            return self._default_fold_decision()
        
        # Calculate EV for different actions
        fold_ev = self._calculate_fold_ev()
        call_ev = self._calculate_call_ev(game_state, equity, pot_odds)
        bet_ev = self._calculate_bet_ev(game_state, equity)
        raise_ev = self._calculate_raise_ev(game_state, equity)
        check_ev = self._calculate_check_ev(game_state, equity)
        
        # Collect all possible actions with their EVs
        actions = [
            ("fold", fold_ev),
            ("call", call_ev),
            ("bet", bet_ev),
            ("raise", raise_ev),
            ("check", check_ev)
        ]
        
        # Filter out invalid actions
        valid_actions = self._filter_valid_actions(actions, game_state)
        
        if not valid_actions:
            logger.warning("No valid actions found, defaulting to fold")
            return self._default_fold_decision()
        
        # Sort by EV and get best action
        best_action, best_ev = max(valid_actions, key=lambda x: x[1])
        
        # Generate alternatives (other valid actions)
        alternatives = self._generate_alternatives(valid_actions, best_action)
        
        # Calculate bet amount for the action
        amount = self._calculate_bet_amount(best_action, game_state)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(best_action, equity, pot_odds, best_ev)
        
        # Calculate confidence
        confidence = self._calculate_confidence(equity, pot_odds, best_ev, valid_actions)
        
        return Decision(
            action=best_action,
            amount=amount,
            confidence=confidence,
            reasoning=reasoning,
            equity=equity,
            pot_odds=pot_odds,
            alternative_actions=alternatives
        )
    
    def _calculate_fold_ev(self) -> float:
        """Calculate EV of folding (always 0)."""
        return 0.0
    
    def _calculate_call_ev(self, game_state: GameState, equity: float, 
                          pot_odds: float) -> float:
        """
        Calculate EV of calling.
        
        EV = (equity * final_pot) - (1 - equity) * bet_to_call
        """
        hero = game_state.get_hero()
        if not hero:
            return -999.0  # Invalid
        
        # Find bet to call
        max_bet = max((p.current_bet for p in game_state.players if p.is_active), default=0)
        bet_to_call = max_bet - hero.current_bet
        
        if bet_to_call <= 0:
            return -999.0  # Nothing to call, invalid action
        
        # Can't call more than our stack
        bet_to_call = min(bet_to_call, hero.stack)
        
        # Calculate final pot size
        final_pot = game_state.pot + bet_to_call
        
        # EV = equity * pot - (1 - equity) * cost
        call_ev = (equity * final_pot) - ((1 - equity) * bet_to_call)
        
        logger.debug(f"Call EV: {call_ev:.2f} (equity: {equity:.2%}, pot: {final_pot:.2f}, cost: {bet_to_call:.2f})")
        return call_ev
    
    def _calculate_bet_ev(self, game_state: GameState, equity: float) -> float:
        """
        Calculate EV of betting.
        
        EV = fold_equity * pot + (1 - fold_equity) * [equity * (pot + bet) - (1 - equity) * bet]
        """
        hero = game_state.get_hero()
        if not hero:
            return -999.0
        
        # Calculate bet size
        bet_size = game_state.pot * self.bet_sizing_factor
        bet_size = min(bet_size, hero.stack)  # Can't bet more than stack
        
        if bet_size <= 0:
            return -999.0
        
        # Estimate fold equity
        fold_equity = self._estimate_fold_equity(game_state, bet_size)
        
        # EV calculation
        # If they fold: we win the pot
        # If they call: we need equity
        fold_component = fold_equity * game_state.pot
        call_component = (1 - fold_equity) * (
            (equity * (game_state.pot + bet_size * 2)) - ((1 - equity) * bet_size)
        )
        
        bet_ev = fold_component + call_component
        
        logger.debug(f"Bet EV: {bet_ev:.2f} (fold_eq: {fold_equity:.2%}, equity: {equity:.2%})")
        return bet_ev
    
    def _calculate_raise_ev(self, game_state: GameState, equity: float) -> float:
        """
        Calculate EV of raising.
        
        Similar to betting but with higher fold equity and larger size.
        """
        hero = game_state.get_hero()
        if not hero:
            return -999.0
        
        # Calculate raise size (3x current bet or pot)
        max_bet = max((p.current_bet for p in game_state.players if p.is_active), default=0)
        raise_size = max(max_bet * self.raise_sizing_factor, game_state.pot)
        raise_size = min(raise_size, hero.stack)  # Can't raise more than stack
        
        if raise_size <= max_bet:
            return -999.0  # Can't raise to less than current bet
        
        # Higher fold equity for raises
        fold_equity = self._estimate_fold_equity(game_state, raise_size) * 1.3
        fold_equity = min(fold_equity, 0.8)  # Cap at 80%
        
        # EV calculation
        fold_component = fold_equity * game_state.pot
        call_component = (1 - fold_equity) * (
            (equity * (game_state.pot + raise_size * 2)) - ((1 - equity) * raise_size)
        )
        
        raise_ev = fold_component + call_component
        
        logger.debug(f"Raise EV: {raise_ev:.2f} (fold_eq: {fold_equity:.2%}, equity: {equity:.2%})")
        return raise_ev
    
    def _calculate_check_ev(self, game_state: GameState, equity: float) -> float:
        """
        Calculate EV of checking.
        
        Simplified: equity * pot (we get to see next card for free)
        """
        # Checking allows us to see the next card without investing more
        # EV is roughly our equity in the current pot
        check_ev = equity * game_state.pot
        
        logger.debug(f"Check EV: {check_ev:.2f} (equity: {equity:.2%})")
        return check_ev
    
    def _estimate_fold_equity(self, game_state: GameState, bet_size: float) -> float:
        """
        Estimate fold equity based on board texture, position, and bet size.
        
        Args:
            game_state: Current game state
            bet_size: Size of the bet/raise
            
        Returns:
            Estimated fold equity (0.0 to 1.0)
        """
        hero = game_state.get_hero()
        if not hero:
            return 0.0
        
        # Base fold equity by position
        position_fold_equity = {
            'BTN': 0.4,
            'CO': 0.35,
            'MP': 0.30,
            'UTG': 0.25,
            'SB': 0.30,
            'BB': 0.20
        }
        
        base_equity = position_fold_equity.get(hero.position, 0.25)
        
        # Adjust for bet size (larger bets = more fold equity)
        if game_state.pot > 0:
            bet_to_pot_ratio = bet_size / game_state.pot
            size_adjustment = min(0.15, bet_to_pot_ratio * 0.1)
        else:
            size_adjustment = 0.0
        
        # Adjust for board texture
        board_adjustment = self._get_board_texture_adjustment(game_state.community_cards)
        
        # Adjust for number of opponents (more opponents = less fold equity)
        opponent_count = len([p for p in game_state.players if p.is_active and not p.is_hero])
        opponent_adjustment = -0.05 * (opponent_count - 1)
        
        # Calculate total fold equity
        total_fold_equity = base_equity + size_adjustment + board_adjustment + opponent_adjustment
        
        # Clamp between 0.1 and 0.7
        return max(0.1, min(0.7, total_fold_equity))
    
    def _get_board_texture_adjustment(self, community_cards: List[Card]) -> float:
        """
        Get fold equity adjustment based on board texture.
        
        Args:
            community_cards: Community cards on board
            
        Returns:
            Adjustment to fold equity (-0.2 to +0.2)
        """
        if len(community_cards) == 0:
            return 0.0
        
        # Count high cards (T, J, Q, K, A)
        high_cards = ['T', 'J', 'Q', 'K', 'A']
        high_card_count = sum(1 for card in community_cards if card.rank in high_cards)
        
        # Check for potential flush draw (3+ of same suit)
        suits = [card.suit for card in community_cards]
        max_suit_count = max((suits.count(suit) for suit in set(suits)), default=0)
        
        # Check for potential straight draw (connected cards)
        rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                      '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        ranks = sorted([rank_values[card.rank] for card in community_cards])
        is_connected = False
        if len(ranks) >= 2:
            for i in range(len(ranks) - 1):
                if ranks[i+1] - ranks[i] <= 2:
                    is_connected = True
                    break
        
        adjustment = 0.0
        
        # Dry board (low cards, no draws) = higher fold equity
        if high_card_count <= 1 and max_suit_count < 3 and not is_connected:
            adjustment = 0.1
        
        # Wet board (high cards, flush/straight draws) = lower fold equity
        elif high_card_count >= 2 or max_suit_count >= 3 or is_connected:
            adjustment = -0.1
        
        logger.debug(f"Board texture adjustment: {adjustment:.2f} (high: {high_card_count}, flush: {max_suit_count >= 3}, connected: {is_connected})")
        return adjustment
    
    def _filter_valid_actions(self, actions: List[Tuple[str, float]], 
                            game_state: GameState) -> List[Tuple[str, float]]:
        """
        Filter out invalid actions based on game state.
        
        Args:
            actions: List of (action, ev) tuples
            game_state: Current game state
            
        Returns:
            List of valid (action, ev) tuples
        """
        valid_actions = []
        hero = game_state.get_hero()
        
        if not hero:
            return [("fold", 0.0)]
        
        # Find current max bet
        max_bet = max((p.current_bet for p in game_state.players if p.is_active), default=0)
        facing_bet = max_bet > hero.current_bet
        
        for action, ev in actions:
            if action == "fold":
                # Can always fold
                valid_actions.append((action, ev))
            
            elif action == "call":
                # Can only call if facing a bet
                if facing_bet and hero.stack > 0:
                    valid_actions.append((action, ev))
            
            elif action == "bet":
                # Can only bet if not facing a bet and have chips
                if not facing_bet and hero.stack > 0:
                    valid_actions.append((action, ev))
            
            elif action == "raise":
                # Can only raise if facing a bet and have chips
                if facing_bet and hero.stack > max_bet:
                    valid_actions.append((action, ev))
            
            elif action == "check":
                # Can only check if not facing a bet
                if not facing_bet:
                    valid_actions.append((action, ev))
        
        return valid_actions if valid_actions else [("fold", 0.0)]
    
    def _generate_alternatives(self, valid_actions: List[Tuple[str, float]], 
                             best_action: str) -> List[AlternativeAction]:
        """
        Generate alternative actions with their EVs.
        
        Args:
            valid_actions: List of valid (action, ev) tuples
            best_action: The best action (to exclude from alternatives)
            
        Returns:
            List of AlternativeAction objects
        """
        alternatives = []
        
        for action, ev in valid_actions:
            if action != best_action:
                amount = None
                if action in ["bet", "raise"]:
                    # Would need game_state to calculate exact amount
                    # For now, just mark as None
                    amount = None
                
                alternatives.append(AlternativeAction(
                    action=action,
                    amount=amount,
                    ev=ev
                ))
        
        # Sort by EV descending
        alternatives.sort(key=lambda x: x.ev, reverse=True)
        
        return alternatives
    
    def _calculate_bet_amount(self, action: str, game_state: GameState) -> Optional[float]:
        """
        Calculate bet amount for the given action.
        
        Args:
            action: Action type
            game_state: Current game state
            
        Returns:
            Bet amount in chips, or None for fold/call/check
        """
        if action in ["fold", "check"]:
            return None
        
        elif action == "call":
            # Call amount is determined by existing bet
            hero = game_state.get_hero()
            if not hero:
                return None
            max_bet = max((p.current_bet for p in game_state.players if p.is_active), default=0)
            call_amount = max_bet - hero.current_bet
            return max(0, call_amount)
        
        elif action == "bet":
            # Bet sizing based on pot
            bet_size = game_state.pot * self.bet_sizing_factor
            hero = game_state.get_hero()
            if hero:
                bet_size = min(bet_size, hero.stack)
            return bet_size
        
        elif action == "raise":
            # Raise sizing (3x current bet or pot)
            max_bet = max((p.current_bet for p in game_state.players if p.is_active), default=0)
            raise_size = max(max_bet * self.raise_sizing_factor, game_state.pot)
            hero = game_state.get_hero()
            if hero:
                raise_size = min(raise_size, hero.stack)
            return raise_size
        
        return None
    
    def _generate_reasoning(self, action: str, equity: float, pot_odds: float, 
                          ev: float) -> str:
        """
        Generate human-readable reasoning for the decision.
        
        Args:
            action: Recommended action
            equity: Hero's equity
            pot_odds: Pot odds (if applicable)
            ev: Expected value of the action
            
        Returns:
            Reasoning string
        """
        if action == "fold":
            if pot_odds > 0:
                return f"Fold: Equity {equity:.1%} < pot odds {pot_odds:.1%} (EV: {ev:.2f})"
            else:
                return f"Fold: Insufficient equity {equity:.1%} (EV: {ev:.2f})"
        
        elif action == "call":
            return f"Call: Equity {equity:.1%} > pot odds {pot_odds:.1%} (EV: +{ev:.2f})"
        
        elif action == "bet":
            return f"Bet: Strong equity {equity:.1%}, positive EV (+{ev:.2f})"
        
        elif action == "raise":
            return f"Raise: Excellent equity {equity:.1%}, high fold equity (EV: +{ev:.2f})"
        
        elif action == "check":
            return f"Check: Equity {equity:.1%}, see next card for free (EV: {ev:.2f})"
        
        return f"{action.capitalize()}: EV = {ev:.2f}"
    
    def _calculate_confidence(self, equity: float, pot_odds: float, best_ev: float,
                            valid_actions: List[Tuple[str, float]]) -> float:
        """
        Calculate confidence in the decision.
        
        Args:
            equity: Hero's equity
            pot_odds: Pot odds
            best_ev: EV of best action
            valid_actions: All valid actions with EVs
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Base confidence on equity vs pot odds difference
        if pot_odds > 0:
            equity_diff = abs(equity - pot_odds)
            
            if equity_diff > 0.2:
                base_confidence = 0.9  # Clear decision
            elif equity_diff > 0.1:
                base_confidence = 0.8  # Moderate decision
            elif equity_diff > 0.05:
                base_confidence = 0.7  # Close decision
            else:
                base_confidence = 0.6  # Very close decision
        else:
            # No pot odds (betting/checking situation)
            if equity > 0.6:
                base_confidence = 0.85
            elif equity > 0.4:
                base_confidence = 0.75
            else:
                base_confidence = 0.65
        
        # Adjust based on EV difference between best and second-best action
        if len(valid_actions) >= 2:
            sorted_actions = sorted(valid_actions, key=lambda x: x[1], reverse=True)
            ev_diff = sorted_actions[0][1] - sorted_actions[1][1]
            
            if ev_diff > 5.0:
                base_confidence += 0.05
            elif ev_diff < 1.0:
                base_confidence -= 0.05
        
        # Clamp between 0.5 and 0.95
        return max(0.5, min(0.95, base_confidence))
    
    def _default_fold_decision(self) -> Decision:
        """Return default fold decision when hero not found."""
        return Decision(
            action="fold",
            amount=None,
            confidence=0.5,
            reasoning="Unable to calculate decision, defaulting to fold",
            equity=0.0,
            pot_odds=0.0
        )

