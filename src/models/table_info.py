#!/usr/bin/env python3
"""
TableInfo model for table configuration.

Essential table configuration for gameplay decisions.
Contains only SB and BB - simplified per user requirements.
"""

from pydantic import BaseModel, Field, model_validator


class TableInfo(BaseModel):
    """Table configuration with stakes."""
    sb: float = Field(..., gt=0, description="Small blind amount")
    bb: float = Field(..., gt=0, description="Big blind amount")
    
    @model_validator(mode='after')
    def validate_bb_greater_than_sb(self):
        if self.bb <= self.sb:
            raise ValueError('Big blind must be greater than small blind')
        return self
    
    def normalize_to_bb(self, amount: float) -> float:
        """Convert chip amount to big blinds."""
        return amount / self.bb
    
    class Config:
        validate_assignment = True
        extra = "forbid"
