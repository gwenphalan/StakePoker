#!/usr/bin/env python3
"""
Decision model for GTO advisor recommendations.

Represents GTO advisor recommendation with alternative actions.
Includes confidence scores, equity, pot odds, and reasoning.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List


class AlternativeAction(BaseModel):
    """Alternative action with expected value."""
    action: str = Field(..., description="Action type (fold/call/raise/check/bet)")
    amount: Optional[float] = Field(None, ge=0, description="Bet/raise amount if applicable")
    ev: float = Field(..., description="Expected value")
    
    @field_validator('action')
    @classmethod
    def validate_action(cls, v):
        valid_actions = ['fold', 'call', 'raise', 'check', 'bet']
        if v not in valid_actions:
            raise ValueError(f'Invalid action: {v}. Must be one of {valid_actions}')
        return v
    
    class Config:
        validate_assignment = True
        extra = "forbid"


class Decision(BaseModel):
    """GTO advisor recommendation."""
    action: str = Field(..., description="Recommended action (fold/call/raise/check/bet)")
    amount: Optional[float] = Field(None, ge=0, description="Bet/raise amount if applicable")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in recommendation")
    reasoning: str = Field(..., description="Explanation of decision")
    equity: Optional[float] = Field(None, ge=0.0, le=1.0, description="Hero's equity vs range")
    pot_odds: Optional[float] = Field(None, ge=0.0, description="Pot odds being offered")
    alternative_actions: Optional[List[AlternativeAction]] = Field(None, description="Alternative actions")
    
    @field_validator('action')
    @classmethod
    def validate_action(cls, v):
        valid_actions = ['fold', 'call', 'raise', 'check', 'bet']
        if v not in valid_actions:
            raise ValueError(f'Invalid action: {v}. Must be one of {valid_actions}')
        return v
    
    class Config:
        validate_assignment = True
        extra = "forbid"
