#!/usr/bin/env python3
"""
GameState model for complete poker game state.

Complete snapshot of current poker game state with comprehensive validation.
Includes cross-field validation for community cards vs phase, unique seats, etc.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional
from src.models.card import Card
from src.models.player import Player
from src.models.table_info import TableInfo


class GameState(BaseModel):
    """Complete poker game state snapshot."""
    players: List[Player] = Field(..., min_items=2, max_items=8, description="Players at table")
    community_cards: List[Card] = Field(default_factory=list, description="Community cards")
    pot: float = Field(..., ge=0, description="Total pot size")
    phase: str = Field(..., description="Game phase (preflop/flop/turn/river/showdown)")
    active_player: Optional[int] = Field(None, ge=1, le=8, description="Active player seat number")
    button_position: int = Field(..., ge=1, le=8, description="Dealer button seat")
    hand_id: Optional[str] = Field(None, description="Unique hand identifier")
    table_info: TableInfo = Field(..., description="Table configuration")
    
    @field_validator('phase')
    @classmethod
    def validate_phase(cls, v):
        valid_phases = ['preflop', 'flop', 'turn', 'river', 'showdown']
        if v not in valid_phases:
            raise ValueError(f'Invalid phase: {v}. Must be one of {valid_phases}')
        return v
    
    @field_validator('community_cards')
    @classmethod
    def validate_community_cards(cls, v):
        if len(v) > 5:
            raise ValueError('Cannot have more than 5 community cards')
        return v
    
    @model_validator(mode='after')
    def validate_community_cards_by_phase(self):
        phase = self.phase
        card_count = len(self.community_cards)
        
        if phase == 'preflop' and card_count > 0:
            raise ValueError('Preflop cannot have community cards')
        elif phase == 'flop' and card_count != 3:
            raise ValueError('Flop must have exactly 3 community cards')
        elif phase == 'turn' and card_count != 4:
            raise ValueError('Turn must have exactly 4 community cards')
        elif phase == 'river' and card_count != 5:
            raise ValueError('River must have exactly 5 community cards')
        
        return self
    
    @field_validator('players')
    @classmethod
    def validate_unique_seats(cls, v):
        seat_numbers = [p.seat_number for p in v]
        if len(seat_numbers) != len(set(seat_numbers)):
            raise ValueError('Players must have unique seat numbers')
        return v
    
    @field_validator('players')
    @classmethod
    def validate_single_hero(cls, v):
        hero_count = sum(1 for p in v if p.is_hero)
        if hero_count > 1:
            raise ValueError('Cannot have more than one hero player')
        return v
    
    @field_validator('players')
    @classmethod
    def validate_single_dealer(cls, v):
        dealer_count = sum(1 for p in v if p.is_dealer)
        if dealer_count > 1:
            raise ValueError('Cannot have more than one dealer')
        return v
    
    def get_hero(self) -> Optional[Player]:
        """Get the hero player."""
        for player in self.players:
            if player.is_hero:
                return player
        return None
    
    def get_player_by_seat(self, seat_number: int) -> Optional[Player]:
        """Get player by seat number."""
        for player in self.players:
            if player.seat_number == seat_number:
                return player
        return None
    
    class Config:
        validate_assignment = True
        extra = "forbid"
