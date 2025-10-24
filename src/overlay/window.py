#!/usr/bin/env python3
"""
Educational overlay window for poker advisor.

Displays GTO recommendations with detailed explanations to help user learn
optimal poker strategy over time. Uses PyQt6 with Windows extended styles
for click-through, transparency, and screenshot exclusion.
"""

import sys
import logging
from typing import Optional
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter
import win32gui
import win32con

from src.models.decision import Decision
from src.models.game_state import GameState
from src.overlay.renderer import OverlayRenderer
from src.overlay.position_manager import PositionManager
from src.config.settings import Settings

logger = logging.getLogger(__name__)


class EducationalOverlay(QWidget):
    """
    Transparent educational overlay that teaches GTO poker strategy.
    
    Features:
    - Click-through (doesn't interfere with game)
    - Transparent background
    - Excluded from screenshots
    - Always on top
    - Primary recommendation with confidence
    - Detailed reasoning explanation
    - Alternative actions with EV comparison
    - Equity vs pot odds visualization
    - Contextual learning tips
    """
    
    def __init__(self):
        """Initialize the educational overlay."""
        super().__init__()
        
        self.settings = Settings()
        self.position_manager = PositionManager()
        self.renderer = OverlayRenderer()
        
        self.current_decision: Optional[Decision] = None
        self.current_game_state: Optional[GameState] = None
        
        # Create settings
        self.settings.create("overlay.panel.position", default="top-left")
        self.settings.create("overlay.panel.width", default=380)
        self.settings.create("overlay.show_details", default=True)
        self.settings.create("overlay.update_interval_ms", default=100)
        
        self.init_ui()
        self.setup_windows_flags()
        
        logger.info("EducationalOverlay initialized")
    
    def init_ui(self):
        """Initialize the UI with transparent background."""
        # Set window flags for overlay behavior
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint |      # Always on top
            Qt.WindowType.FramelessWindowHint |       # No window frame
            Qt.WindowType.Tool |                      # Tool window (no taskbar)
            Qt.WindowType.WindowTransparentForInput   # Click-through
        )
        
        # Set transparent background
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        
        # Position and size based on poker monitor
        x, y, width, height = self.position_manager.get_overlay_geometry()
        self.setGeometry(x, y, width, height)
        
        logger.debug(f"Overlay geometry set: x={x}, y={y}, w={width}, h={height}")
    
    def setup_windows_flags(self):
        """Apply Windows-specific extended styles."""
        try:
            hwnd = int(self.winId())
            
            # Get current extended style
            ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
            
            # Add extended styles:
            # WS_EX_LAYERED: Required for transparency and screenshot exclusion
            # WS_EX_TRANSPARENT: Click-through behavior
            # WS_EX_NOACTIVATE: Don't activate when clicked
            ex_style |= (
                win32con.WS_EX_LAYERED |
                win32con.WS_EX_TRANSPARENT |
                win32con.WS_EX_NOACTIVATE
            )
            
            # Apply the extended style
            win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, ex_style)
            
            # Make window exclude from screen capture (Windows 10 2004+)
            try:
                from ctypes import windll, c_int
                DWMWA_EXCLUDED_FROM_PEEK = 12
                windll.dwmapi.DwmSetWindowAttribute(
                    hwnd,
                    DWMWA_EXCLUDED_FROM_PEEK,
                    c_int(1),
                    4
                )
                logger.info("Screenshot exclusion enabled")
            except Exception as e:
                logger.warning(f"Could not set screenshot exclusion: {e}")
            
            logger.info("Windows extended styles applied successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup Windows flags: {e}")
    
    def update_decision(self, decision: Decision, game_state: GameState):
        """
        Update the overlay with new decision and game state.
        
        Args:
            decision: Decision to display
            game_state: Current game state
        """
        self.current_decision = decision
        self.current_game_state = game_state
        self.update()  # Trigger repaint
        
        logger.debug(f"Decision updated: {decision.action} (confidence: {decision.confidence:.2f})")
    
    def clear_decision(self):
        """Clear the current decision (when not hero's turn)."""
        if self.current_decision is not None:
            self.current_decision = None
            self.current_game_state = None
            self.update()
            logger.debug("Decision cleared")
    
    def paintEvent(self, event):
        """Draw the educational overlay content."""
        if not self.current_decision or not self.current_game_state:
            return  # Nothing to draw
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Get panel position from settings
        panel_position = self.settings.get("overlay.panel.position")
        panel_width = self.settings.get("overlay.panel.width")
        
        # Calculate panel position (returns coordinates relative to overlay)
        panel_x, panel_y = self.position_manager.get_panel_position(
            panel_width=panel_width,
            position=panel_position
        )
        
        # Draw the educational panel
        self.renderer.draw_educational_panel(
            painter,
            self.current_decision,
            self.current_game_state,
            panel_x,
            panel_y,
            panel_width
        )
    
    def set_panel_position(self, position: str):
        """
        Set panel position.
        
        Args:
            position: Panel position ("top-right", "top-left", "bottom-right", "bottom-left")
        """
        self.settings.update("overlay.panel.position", position)
        self.update()
        logger.info(f"Panel position changed to: {position}")
    
    def toggle_details(self):
        """Toggle detailed view on/off."""
        current = self.settings.get("overlay.show_details")
        self.settings.update("overlay.show_details", not current)
        self.update()
        logger.info(f"Details toggled: {not current}")
    
    def refresh_monitor_config(self):
        """Refresh monitor configuration and reposition overlay."""
        self.position_manager.refresh_monitor_config()
        x, y, width, height = self.position_manager.get_overlay_geometry()
        self.setGeometry(x, y, width, height)
        logger.info("Overlay repositioned after monitor config refresh")


def create_overlay() -> tuple:
    """
    Create and show the educational overlay window.
    
    Returns:
        Tuple of (QApplication, EducationalOverlay)
    """
    app = QApplication(sys.argv)
    overlay = EducationalOverlay()
    overlay.show()
    return app, overlay


if __name__ == '__main__':
    # Setup logging for testing
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=== Educational Overlay Test ===\n")
    
    # Create test decision
    from src.models.decision import Decision, AlternativeAction
    from src.models.table_info import TableInfo
    from src.models.player import Player
    
    test_decision = Decision(
        action="raise",
        amount=15.0,
        confidence=0.85,
        reasoning="You have top pair with good kicker. Villain's range includes many worse pairs and draws. Betting for value and protection.",
        equity=0.68,
        pot_odds=0.35,
        alternative_actions=[
            AlternativeAction(action="call", amount=None, ev=2.5),
            AlternativeAction(action="fold", amount=None, ev=-5.0),
        ]
    )
    
    # Mock game state
    test_game_state = GameState(
        players=[
            Player(
                seat_number=1,
                position="BTN",
                stack=100.0,
                is_hero=True,
                is_active=True
            )
        ],
        community_cards=[],
        pot=30.0,
        phase="flop",
        button_position=1,
        table_info=TableInfo(
            sb=0.10,
            bb=0.25
        )
    )
    
    # Create overlay
    app, overlay = create_overlay()
    
    # Update with test decision
    overlay.update_decision(test_decision, test_game_state)
    
    print("Overlay displayed. Press Ctrl+C to exit.")
    
    # Run application
    sys.exit(app.exec())

