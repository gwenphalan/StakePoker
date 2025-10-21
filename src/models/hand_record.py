#!/usr/bin/env python3
"""
HandRecord model for complete hand history tracking.

Complete hand history record with renamed net_profit field.
Includes action sequence, hero information, and hand outcome.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import datetime
from src.models.card import Card
from src.models.table_info import TableInfo


class Action(BaseModel):
    """Single action in a hand."""
    seat_number: int = Field(..., ge=1, le=8, description="Player seat number")
    action_type: str = Field(..., description="Action type (fold/call/raise/check/bet)")
    amount: Optional[float] = Field(None, ge=0, description="Bet/raise amount if applicable")
    phase: str = Field(..., description="Game phase when action occurred")
    
    @field_validator('action_type')
    @classmethod
    def validate_action_type(cls, v):
        valid_actions = ['fold', 'call', 'raise', 'check', 'bet']
        if v not in valid_actions:
            raise ValueError(f'Invalid action type: {v}. Must be one of {valid_actions}')
        return v
    
    @field_validator('phase')
    @classmethod
    def validate_phase(cls, v):
        valid_phases = ['preflop', 'flop', 'turn', 'river']
        if v not in valid_phases:
            raise ValueError(f'Invalid phase: {v}. Must be one of {valid_phases}')
        return v
    
    class Config:
        validate_assignment = True
        extra = "forbid"


class HandRecord(BaseModel):
    """Complete hand history record."""
    hand_id: str = Field(..., description="Unique hand identifier")
    timestamp: datetime = Field(..., description="When hand occurred")
    table_info: TableInfo = Field(..., description="Table configuration")
    hero_position: str = Field(..., description="Hero's position (BTN/SB/BB/etc)")
    hero_seat: int = Field(..., ge=1, le=8, description="Hero's seat number")
    hero_cards: List[Card] = Field(..., min_items=2, max_items=2, description="Hero's hole cards")
    actions: List[Action] = Field(default_factory=list, description="Sequence of all actions")
    result: Optional[str] = Field(None, description="Hand result (won/lost/folded)")
    net_profit: float = Field(..., description="Net profit/loss for hand")
    final_pot: float = Field(..., ge=0, description="Pot size at end")
    showdown: bool = Field(False, description="Went to showdown")
    
    @field_validator('result')
    @classmethod
    def validate_result(cls, v):
        if v is not None:
            valid_results = ['won', 'lost', 'folded']
            if v not in valid_results:
                raise ValueError(f'Invalid result: {v}. Must be one of {valid_results}')
        return v
    
    @field_validator('hero_position')
    @classmethod
    def validate_hero_position(cls, v):
        valid_positions = ['BTN', 'SB', 'BB', 'UTG', 'UTG+1', 'UTG+2', 'MP', 'MP+1', 'CO']
        if v not in valid_positions:
            raise ValueError(f'Invalid position: {v}. Must be one of {valid_positions}')
        return v
    
    class Config:
        validate_assignment = True
        extra = "forbid"
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
