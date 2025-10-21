#!/usr/bin/env python3
"""
RegionConfig model for screen region definitions.

Screen region definition for capture without monitor field.
Always captures from Monitor 2, so monitor field not needed.
"""

from pydantic import BaseModel, Field, field_validator


class RegionConfig(BaseModel):
    """Screen region definition for capture."""
    name: str = Field(..., description="Region identifier")
    x: int = Field(..., ge=0, description="Left coordinate")
    y: int = Field(..., ge=0, description="Top coordinate")
    width: int = Field(..., gt=0, description="Region width")
    height: int = Field(..., gt=0, description="Region height")
    
    @field_validator('width', 'height')
    @classmethod
    def validate_positive_dimensions(cls, v):
        if v <= 0:
            raise ValueError('Width and height must be positive')
        return v
    
    def to_tuple(self) -> tuple:
        """Convert to (x, y, width, height) tuple for mss."""
        return (self.x, self.y, self.width, self.height)
    
    class Config:
        validate_assignment = True
        extra = "forbid"
