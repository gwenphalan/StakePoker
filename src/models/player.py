#!/usr/bin/env python3
"""
Player model for representing players at the poker table.

Represents player state at the table without name field.
Name is only needed for hero detection, not stored in model.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from src.models.card import Card


class Player(BaseModel):
    """Represents a player at the poker table."""
    seat_number: int = Field(..., ge=1, le=8, description="Physical seat position (1-8)")
    position: Optional[str] = Field(None, description="Relative position (BTN/SB/BB/UTG/MP/CO)")
    stack: float = Field(..., ge=0, description="Chip stack size")
    hole_cards: List[Card] = Field(default_factory=list, description="Player's hole cards")
    timer_state: Optional[str] = Field(None, description="Timer state (purple/red/None)")
    is_dealer: bool = Field(False, description="Has dealer button")
    current_bet: float = Field(0, ge=0, description="Current bet amount")
    is_hero: bool = Field(False, description="Is the hero player")
    is_active: bool = Field(True, description="Still in hand (not folded)")
    
    @field_validator('hole_cards')
    @classmethod
    def validate_hole_cards(cls, v):
        if len(v) > 2:
            raise ValueError('Cannot have more than 2 hole cards')
        return v
    
    @field_validator('timer_state')
    @classmethod
    def validate_timer_state(cls, v):
        if v is not None and v not in ['purple', 'red']:
            raise ValueError("Timer state must be 'purple', 'red', or None")
        return v
    
    @field_validator('position')
    @classmethod
    def validate_position(cls, v):
        if v is not None:
            valid_positions = ['BTN', 'SB', 'BB', 'UTG', 'UTG+1', 'UTG+2', 'MP', 'MP+1', 'CO']
            if v not in valid_positions:
                raise ValueError(f'Invalid position: {v}. Must be one of {valid_positions}')
        return v
    
    class Config:
        validate_assignment = True
        extra = "forbid"
