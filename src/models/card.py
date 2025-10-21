#!/usr/bin/env python3
"""
Card model for representing playing cards with validation.

Represents a single playing card with rank and suit validation.
Used throughout the poker advisor for card representation and validation.
"""

from pydantic import BaseModel, Field, field_validator


class Card(BaseModel):
    """Represents a playing card with rank and suit."""
    rank: str = Field(..., description="Card rank (A, K, Q, J, T, 9-2)")
    suit: str = Field(..., description="Card suit (hearts, diamonds, clubs, spades)")
    
    @field_validator('rank')
    @classmethod
    def validate_rank(cls, v):
        valid_ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
        if v not in valid_ranks:
            raise ValueError(f'Invalid rank: {v}. Must be one of {valid_ranks}')
        return v
    
    @field_validator('suit')
    @classmethod
    def validate_suit(cls, v):
        valid_suits = ['hearts', 'diamonds', 'clubs', 'spades']
        if v not in valid_suits:
            raise ValueError(f'Invalid suit: {v}. Must be one of {valid_suits}')
        return v
    
    def __str__(self) -> str:
        """String representation (e.g., 'Ah' for Ace of hearts)."""
        suit_symbols = {'hearts': 'h', 'diamonds': 'd', 'clubs': 'c', 'spades': 's'}
        return f"{self.rank}{suit_symbols[self.suit]}"
    
    class Config:
        validate_assignment = True
        extra = "forbid"
