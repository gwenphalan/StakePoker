#!/usr/bin/env python3
"""
Position manager for overlay window.

Manages overlay window positioning and sizing based on the configured
poker monitor from the capture module.
"""

import logging
from typing import Tuple
from src.capture.monitor_config import MonitorConfig

logger = logging.getLogger(__name__)


class PositionManager:
    """
    Manages overlay window positioning and sizing.
    
    Integrates with MonitorConfig to position the overlay correctly
    on the poker monitor (typically Monitor 2).
    """
    
    def __init__(self):
        """Initialize position manager with monitor configuration."""
        self.monitor_config = MonitorConfig()
        self.monitor_info = self.monitor_config.get_monitor_info()
        
        logger.info(f"PositionManager initialized for {self.monitor_info}")
    
    def get_overlay_geometry(self) -> Tuple[int, int, int, int]:
        """
        Get overlay window geometry (x, y, width, height).
        
        Returns overlay should cover the entire poker monitor.
        
        Returns:
            Tuple of (x, y, width, height) for overlay window
        """
        return (
            self.monitor_info.left,
            self.monitor_info.top,
            self.monitor_info.width,
            self.monitor_info.height
        )
    
    def get_panel_position(self, panel_width: int = 380, 
                          panel_height: int = 400,
                          position: str = "top-right") -> Tuple[int, int]:
        """
        Get position for the educational panel within the overlay.
        
        Args:
            panel_width: Width of the panel in pixels
            panel_height: Height of the panel in pixels
            position: Panel position ("top-right", "top-left", "bottom-right", "bottom-left")
        
        Returns:
            Tuple of (x, y) coordinates for panel (relative to overlay, not screen)
        """
        margin = 20  # Margin from edges
        
        if position == "top-right":
            x = self.monitor_info.width - panel_width - margin
            y = margin
        elif position == "top-left":
            x = margin
            y = margin
        elif position == "bottom-right":
            x = self.monitor_info.width - panel_width - margin
            y = self.monitor_info.height - panel_height - margin
        elif position == "bottom-left":
            x = margin
            y = self.monitor_info.height - panel_height - margin
        else:
            # Default to top-right
            logger.warning(f"Unknown position '{position}', defaulting to top-right")
            x = self.monitor_info.width - panel_width - margin
            y = margin
        
        return (x, y)
    
    def get_monitor_dimensions(self) -> Tuple[int, int]:
        """
        Get poker monitor dimensions.
        
        Returns:
            Tuple of (width, height)
        """
        return self.monitor_info.dimensions
    
    def get_monitor_position(self) -> Tuple[int, int]:
        """
        Get poker monitor position on screen.
        
        Returns:
            Tuple of (left, top)
        """
        return self.monitor_info.position
    
    def is_point_in_monitor(self, x: int, y: int) -> bool:
        """
        Check if a point is within the poker monitor bounds.
        
        Args:
            x: X coordinate (screen coordinates)
            y: Y coordinate (screen coordinates)
        
        Returns:
            True if point is within monitor bounds
        """
        left, top = self.monitor_info.position
        width, height = self.monitor_info.dimensions
        
        return (left <= x < left + width and 
                top <= y < top + height)
    
    def refresh_monitor_config(self) -> None:
        """
        Refresh monitor configuration.
        
        Useful if monitors are added/removed or resolution changes.
        """
        logger.info("Refreshing monitor configuration...")
        self.monitor_config.refresh_monitors()
        self.monitor_info = self.monitor_config.get_monitor_info()
        logger.info(f"Monitor configuration refreshed: {self.monitor_info}")


if __name__ == "__main__":
    # Setup logging for testing
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test position manager
    print("=== Position Manager Test ===\n")
    
    pm = PositionManager()
    
    print(f"Overlay Geometry: {pm.get_overlay_geometry()}")
    print(f"Monitor Dimensions: {pm.get_monitor_dimensions()}")
    print(f"Monitor Position: {pm.get_monitor_position()}")
    print(f"\nPanel Positions:")
    print(f"  Top-Right: {pm.get_panel_position(position='top-right')}")
    print(f"  Top-Left: {pm.get_panel_position(position='top-left')}")
    print(f"  Bottom-Right: {pm.get_panel_position(position='bottom-right')}")
    print(f"  Bottom-Left: {pm.get_panel_position(position='bottom-left')}")

