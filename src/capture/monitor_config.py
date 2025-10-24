
#!/usr/bin/env python3
"""
Monitor configuration and validation for screen capture.

Detects available monitors, validates the configured monitor exists,
and provides monitor information needed by the capture module.

Uses the Settings system to store and retrieve monitor preferences.
"""

import mss
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

# Add project root to path for direct execution
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class MonitorInfo:
    """Information about a monitor."""
    index: int
    width: int
    height: int
    left: int
    top: int
    name: str
    
    @property
    def dimensions(self) -> Tuple[int, int]:
        """Get monitor dimensions as (width, height)."""
        return (self.width, self.height)
    
    @property
    def position(self) -> Tuple[int, int]:
        """Get monitor position as (left, top)."""
        return (self.left, self.top)
    
    @property
    def bounds(self) -> Dict[str, int]:
        """Get monitor bounds as dict for mss."""
        return {
            'left': self.left,
            'top': self.top,
            'width': self.width,
            'height': self.height
        }
    
    def __str__(self) -> str:
        """String representation of monitor info."""
        return f"Monitor {self.index} ({self.name}): {self.width}x{self.height} at ({self.left}, {self.top})"


class MonitorConfig:
    """
    Monitor configuration manager.
    
    Handles detection, validation, and configuration of the monitor
    to use for poker table capture.
    
    Default behavior:
    - Uses Monitor 2 (index=2) as the poker monitor
    - Falls back to primary monitor if Monitor 2 doesn't exist
    - Stores configuration in settings system
    """
    
    def __init__(self):
        """Initialize monitor configuration."""
        self.settings = Settings()
        self.sct = mss.mss()
        
        # Initialize default settings
        self._init_settings()
        
        # Detect and validate monitors
        self.available_monitors = self._detect_monitors()
        self.poker_monitor = self._get_poker_monitor()
        
        logger.info(f"Monitor configuration initialized: {self.poker_monitor}")
    
    def _init_settings(self) -> None:
        """Initialize monitor-related settings with defaults."""
        # Monitor index to use for capture (2 = poker monitor)
        self.settings.create("capture.monitor.index", default=2)
        
        # Whether to auto-detect and validate monitor on startup
        self.settings.create("capture.monitor.auto_validate", default=True)
        
        # Fallback to primary monitor if configured monitor not found
        self.settings.create("capture.monitor.fallback_to_primary", default=True)
        
        logger.debug("Monitor settings initialized")
    
    def _detect_monitors(self) -> List[MonitorInfo]:
        """
        Detect all available monitors.
        
        Returns:
            List of MonitorInfo objects for all detected monitors
        """
        monitors = []
        
        for i, monitor in enumerate(self.sct.monitors):
            # Skip the "all monitors" entry (index 0)
            if i == 0:
                continue
            
            # Determine monitor name
            if i == 1:
                name = "Primary"
            elif i == 2:
                name = "Poker Monitor"
            else:
                name = f"Monitor {i}"
            
            monitor_info = MonitorInfo(
                index=i,
                width=monitor['width'],
                height=monitor['height'],
                left=monitor['left'],
                top=monitor['top'],
                name=name
            )
            
            monitors.append(monitor_info)
            logger.debug(f"Detected: {monitor_info}")
        
        logger.info(f"Detected {len(monitors)} monitor(s)")
        return monitors
    
    def _get_poker_monitor(self) -> MonitorInfo:
        """
        Get the configured poker monitor.
        
        Returns:
            MonitorInfo for the poker monitor
            
        Raises:
            RuntimeError: If no monitors are available
        """
        if not self.available_monitors:
            raise RuntimeError("No monitors detected")
        
        # Get configured monitor index
        configured_index = self.settings.get("capture.monitor.index")
        auto_validate = self.settings.get("capture.monitor.auto_validate")
        fallback_enabled = self.settings.get("capture.monitor.fallback_to_primary")
        
        # Try to find the configured monitor
        poker_monitor = None
        for monitor in self.available_monitors:
            if monitor.index == configured_index:
                poker_monitor = monitor
                break
        
        # Handle monitor not found
        if poker_monitor is None:
            if fallback_enabled:
                logger.warning(
                    f"Configured monitor {configured_index} not found. "
                    f"Falling back to primary monitor."
                )
                poker_monitor = self.available_monitors[0]  # Primary monitor
                
                # Update settings to reflect fallback
                self.settings.update("capture.monitor.index", poker_monitor.index)
            else:
                available_indices = [m.index for m in self.available_monitors]
                raise RuntimeError(
                    f"Configured monitor {configured_index} not found. "
                    f"Available monitors: {available_indices}"
                )
        
        # Validate monitor if enabled
        if auto_validate:
            self._validate_monitor(poker_monitor)
        
        return poker_monitor
    
    def _validate_monitor(self, monitor: MonitorInfo) -> None:
        """
        Validate that a monitor is suitable for poker capture.
        
        Args:
            monitor: Monitor to validate
            
        Raises:
            ValueError: If monitor doesn't meet requirements
        """
        # Check minimum resolution (poker table needs reasonable space)
        min_width = 800
        min_height = 600
        
        if monitor.width < min_width or monitor.height < min_height:
            raise ValueError(
                f"Monitor {monitor.index} resolution too low: "
                f"{monitor.width}x{monitor.height}. "
                f"Minimum required: {min_width}x{min_height}"
            )
        
        logger.debug(f"Monitor {monitor.index} validated successfully")
    
    def get_monitor_info(self) -> MonitorInfo:
        """
        Get information about the current poker monitor.
        
        Returns:
            MonitorInfo for the configured poker monitor
        """
        return self.poker_monitor
    
    def get_monitor_bounds(self) -> Dict[str, int]:
        """
        Get monitor bounds for mss capture.
        
        Returns:
            Dictionary with 'left', 'top', 'width', 'height' keys
        """
        return self.poker_monitor.bounds
    
    def list_available_monitors(self) -> List[MonitorInfo]:
        """
        Get list of all available monitors.
        
        Returns:
            List of MonitorInfo objects
        """
        return self.available_monitors.copy()
    
    def set_poker_monitor(self, monitor_index: int) -> None:
        """
        Set which monitor to use for poker capture.
        
        Args:
            monitor_index: Index of monitor to use (1-based)
            
        Raises:
            ValueError: If monitor index is invalid
        """
        # Validate monitor exists
        monitor_exists = any(m.index == monitor_index for m in self.available_monitors)
        
        if not monitor_exists:
            available_indices = [m.index for m in self.available_monitors]
            raise ValueError(
                f"Monitor {monitor_index} not found. "
                f"Available monitors: {available_indices}"
            )
        
        # Update settings
        self.settings.update("capture.monitor.index", monitor_index)
        
        # Update current poker monitor
        for monitor in self.available_monitors:
            if monitor.index == monitor_index:
                self.poker_monitor = monitor
                break
        
        logger.info(f"Poker monitor changed to: {self.poker_monitor}")
    
    def get_monitor_by_index(self, index: int) -> Optional[MonitorInfo]:
        """
        Get monitor information by index.
        
        Args:
            index: Monitor index (1-based)
            
        Returns:
            MonitorInfo if found, None otherwise
        """
        for monitor in self.available_monitors:
            if monitor.index == index:
                return monitor
        return None
    
    def is_monitor_available(self, index: int) -> bool:
        """
        Check if a monitor index is available.
        
        Args:
            index: Monitor index to check
            
        Returns:
            True if monitor exists, False otherwise
        """
        return any(m.index == index for m in self.available_monitors)
    
    def refresh_monitors(self) -> None:
        """
        Refresh the list of available monitors.
        
        Useful if monitors are added/removed while application is running.
        """
        logger.info("Refreshing monitor list...")
        
        # Re-detect monitors
        self.available_monitors = self._detect_monitors()
        
        # Re-validate poker monitor
        try:
            self.poker_monitor = self._get_poker_monitor()
            logger.info(f"Monitor refresh complete: {self.poker_monitor}")
        except RuntimeError as e:
            logger.error(f"Failed to refresh monitors: {e}")
            raise
    
    def get_monitor_count(self) -> int:
        """
        Get the number of available monitors.
        
        Returns:
            Number of monitors (excluding "all monitors" entry)
        """
        return len(self.available_monitors)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup mss."""
        if hasattr(self, 'sct'):
            self.sct.close()
    
    def __str__(self) -> str:
        """String representation."""
        return (
            f"MonitorConfig(poker_monitor={self.poker_monitor.index}, "
            f"available={self.get_monitor_count()})"
        )
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"MonitorConfig(poker_monitor={self.poker_monitor}, "
            f"available_monitors={len(self.available_monitors)})"
        )


# Convenience function for quick access
def get_poker_monitor() -> MonitorInfo:
    """
    Quick access function to get poker monitor info.
    
    Returns:
        MonitorInfo for the configured poker monitor
    """
    config = MonitorConfig()
    return config.get_monitor_info()


if __name__ == "__main__":
    # Now import Settings
    from config.settings import Settings
    
    # Setup logging for testing
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test monitor configuration
    print("=== Monitor Configuration Test ===\n")
    
    with MonitorConfig() as config:
        print(f"Configuration: {config}\n")
        
        print("Available Monitors:")
        for monitor in config.list_available_monitors():
            print(f"  {monitor}")
        
        print(f"\nPoker Monitor: {config.get_monitor_info()}")
        print(f"Monitor Bounds: {config.get_monitor_bounds()}")
        print(f"Monitor Dimensions: {config.poker_monitor.dimensions}")
        print(f"Monitor Position: {config.poker_monitor.position}")
