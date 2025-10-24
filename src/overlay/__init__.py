#!/usr/bin/env python3
"""
Overlay module for educational poker advisor display.

Provides transparent, click-through overlay window that displays GTO
recommendations with detailed explanations to help users learn poker strategy.
"""

from src.overlay.window import EducationalOverlay, create_overlay
from src.overlay.renderer import OverlayRenderer
from src.overlay.position_manager import PositionManager

__all__ = [
    'EducationalOverlay',
    'create_overlay',
    'OverlayRenderer',
    'PositionManager',
]

