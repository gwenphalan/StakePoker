#!/usr/bin/env python3
"""
Renderer for educational overlay content.

Handles all drawing logic for the educational poker advisor overlay,
including recommendations, reasoning, equity analysis, and learning tips.
"""

import logging
from typing import Optional, List, Tuple
from PyQt6.QtGui import QPainter, QColor, QFont, QPen, QLinearGradient
from PyQt6.QtCore import Qt, QRect
from src.models.decision import Decision, AlternativeAction
from src.models.game_state import GameState

logger = logging.getLogger(__name__)


class OverlayRenderer:
    """
    Renders educational overlay content.
    
    Provides methods to draw various components of the educational overlay
    including recommendations, reasoning, equity analysis, and learning tips.
    """
    
    # Color scheme
    PANEL_BG_START = QColor(20, 20, 30, 230)
    PANEL_BG_END = QColor(30, 30, 40, 230)
    PANEL_BORDER = QColor(100, 100, 120, 200)
    HEADER_LINE = QColor(80, 120, 200, 150)
    
    # Action colors
    ACTION_COLORS = {
        'fold': QColor(220, 80, 80),      # Red
        'call': QColor(100, 180, 255),    # Blue
        'check': QColor(100, 200, 100),   # Green
        'raise': QColor(255, 180, 50),    # Orange
        'bet': QColor(255, 180, 50),      # Orange
    }
    
    # Text colors
    TEXT_PRIMARY = QColor(220, 220, 230)
    TEXT_SECONDARY = QColor(200, 200, 210)
    TEXT_ACCENT_BLUE = QColor(150, 180, 255)
    TEXT_ACCENT_GREEN = QColor(150, 255, 180)
    TEXT_ACCENT_YELLOW = QColor(255, 200, 150)
    TEXT_ACCENT_GOLD = QColor(255, 220, 150)
    
    # Confidence colors
    CONFIDENCE_HIGH = QColor(100, 220, 100)   # Green
    CONFIDENCE_MED = QColor(255, 200, 80)     # Yellow
    CONFIDENCE_LOW = QColor(220, 100, 100)    # Red
    
    def __init__(self):
        """Initialize the renderer."""
        logger.info("OverlayRenderer initialized")
    
    def draw_educational_panel(self, painter: QPainter, decision: Decision,
                               game_state: GameState, panel_x: int, panel_y: int,
                               panel_width: int = 380) -> None:
        """
        Draw the complete educational panel.
        
        Args:
            painter: QPainter instance
            decision: Decision to display
            game_state: Current game state
            panel_x: Panel X position
            panel_y: Panel Y position
            panel_width: Panel width in pixels
        """
        # Draw content first to track actual height
        y_offset = panel_y + 20
        
        # Store all drawing commands
        draw_commands = []
        
        # 1. Primary recommendation
        y_start = y_offset
        y_offset += 80  # Fixed height
        draw_commands.append(('primary', y_start))
        
        # 2. Reasoning explanation
        y_start = y_offset
        y_offset += 5  # spacing
        reasoning_lines = self._wrap_text(decision.reasoning, panel_width - 30, None)
        y_offset += 12 + 30 + (len(reasoning_lines) * 16) + 8
        draw_commands.append(('reasoning', y_start))
        
        # 3. Equity and pot odds (if available)
        if decision.equity is not None:
            y_start = y_offset
            y_offset += 10 + 12 + 28 + 18 + 14 + 8
            draw_commands.append(('equity', y_start))
        
        # 4. Alternative actions with EV comparison
        if decision.alternative_actions:
            y_start = y_offset
            y_offset += 10 + 12 + 32 + ((len(decision.alternative_actions) - 1) * 30)
            draw_commands.append(('alternatives', y_start))
        
        # 5. Learning tip at bottom
        tip = self._get_learning_tip(decision, game_state)
        if tip:
            y_start = y_offset
            tip_lines = self._wrap_text(tip, panel_width - 80, None)
            y_offset += 15 + 12 + 12 + (len(tip_lines) * 16)
            draw_commands.append(('tip', y_start))
        
        # Calculate final panel height
        panel_height = y_offset - panel_y + 15
        
        # Draw panel background
        self._draw_panel_background(painter, panel_x, panel_y, panel_width, panel_height)
        
        # Now execute all drawing commands
        for cmd_type, y_pos in draw_commands:
            if cmd_type == 'primary':
                self._draw_primary_recommendation(painter, decision, panel_x, y_pos, panel_width)
            elif cmd_type == 'reasoning':
                self._draw_reasoning_section(painter, decision, panel_x, y_pos, panel_width)
            elif cmd_type == 'equity':
                self._draw_equity_section(painter, decision, panel_x, y_pos, panel_width)
            elif cmd_type == 'alternatives':
                self._draw_alternatives_section(painter, decision, panel_x, y_pos, panel_width)
            elif cmd_type == 'tip':
                self._draw_learning_tip(painter, decision, game_state, panel_x, y_pos, panel_width)
    
    def _draw_panel_background(self, painter: QPainter, x: int, y: int,
                               width: int, height: int) -> None:
        """Draw the panel background with gradient."""
        # Create gradient background
        gradient = QLinearGradient(x, y, x, y + height)
        gradient.setColorAt(0, self.PANEL_BG_START)
        gradient.setColorAt(1, self.PANEL_BG_END)
        
        painter.setBrush(gradient)
        painter.setPen(QPen(self.PANEL_BORDER, 2))
        painter.drawRoundedRect(x, y, width, height, 12, 12)
        
        # Draw subtle header line
        painter.setPen(QPen(self.HEADER_LINE, 2))
        painter.drawLine(x + 15, y + 75, x + width - 15, y + 75)
    
    def _draw_primary_recommendation(self, painter: QPainter, decision: Decision,
                                     x: int, y: int, width: int) -> int:
        """Draw the primary recommendation prominently."""
        # Get action color
        action_color = self.ACTION_COLORS.get(
            decision.action.lower(), 
            QColor(200, 200, 200)
        )
        
        # Draw action text (large)
        font = QFont("Segoe UI", 32, QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(action_color)
        
        action_text = decision.action.upper()
        if decision.amount:
            action_text += f" ${decision.amount:.2f}"
        
        painter.drawText(
            QRect(x + 15, y, width - 30, 50),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            action_text
        )
        
        # Draw confidence bar
        conf_y = y + 50
        self._draw_confidence_bar(
            painter, x + 15, conf_y, width - 30, 15, decision.confidence
        )
        
        return y + 80
    
    def _draw_confidence_bar(self, painter: QPainter, x: int, y: int,
                            width: int, height: int, confidence: float) -> None:
        """Draw a confidence level bar."""
        # Background
        painter.setBrush(QColor(50, 50, 60, 200))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(x, y, width, height, 7, 7)
        
        # Confidence fill (color based on level)
        conf_width = int(width * confidence)
        if confidence >= 0.8:
            conf_color = self.CONFIDENCE_HIGH
        elif confidence >= 0.6:
            conf_color = self.CONFIDENCE_MED
        else:
            conf_color = self.CONFIDENCE_LOW
        
        painter.setBrush(conf_color)
        painter.drawRoundedRect(x, y, conf_width, height, 7, 7)
        
        # Confidence text
        font = QFont("Segoe UI", 9)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(
            QRect(x, y - 15, width, 15),
            Qt.AlignmentFlag.AlignRight,
            f"Confidence: {confidence:.0%}"
        )
    
    def _draw_reasoning_section(self, painter: QPainter, decision: Decision,
                                x: int, y: int, width: int) -> int:
        """Draw the reasoning explanation section."""
        # Add spacing
        y += 5
        
        # Section title
        font = QFont("Segoe UI", 10, QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(self.TEXT_ACCENT_BLUE)
        painter.drawText(x + 15, y + 12, "WHY THIS PLAY:")
        
        # Reasoning text (word-wrapped)
        font = QFont("Segoe UI", 9)
        painter.setFont(font)
        painter.setPen(self.TEXT_PRIMARY)
        
        reasoning = decision.reasoning
        wrapped_lines = self._wrap_text(reasoning, width - 30, painter.fontMetrics())
        
        text_y = y + 30
        for line in wrapped_lines:
            painter.drawText(x + 15, text_y, line)
            text_y += 16
        
        return text_y + 8
    
    def _draw_equity_section(self, painter: QPainter, decision: Decision,
                            x: int, y: int, width: int) -> int:
        """Draw equity vs pot odds comparison."""
        equity = decision.equity
        pot_odds = decision.pot_odds
        
        if equity is None:
            return y
        
        # Add spacing
        y += 10
        
        # Section title
        font = QFont("Segoe UI", 10, QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(self.TEXT_ACCENT_GREEN)
        painter.drawText(x + 15, y + 12, "EQUITY ANALYSIS:")
        
        # Draw equity bar
        bar_y = y + 28
        bar_height = 18
        self._draw_comparison_bars(
            painter, x + 15, bar_y, width - 30, bar_height,
            equity, pot_odds if pot_odds else 0
        )
        
        # Explanation text
        font = QFont("Segoe UI", 8)
        painter.setFont(font)
        painter.setPen(self.TEXT_SECONDARY)
        
        if pot_odds and pot_odds > 0:
            if equity > pot_odds:
                explanation = f"Your equity ({equity:.1%}) > pot odds ({pot_odds:.1%}) = PROFITABLE"
                painter.setPen(self.CONFIDENCE_HIGH)
            else:
                explanation = f"Your equity ({equity:.1%}) < pot odds ({pot_odds:.1%}) = UNPROFITABLE"
                painter.setPen(self.CONFIDENCE_LOW)
        else:
            explanation = f"Your equity: {equity:.1%} vs opponent's range"
        
        painter.drawText(x + 15, bar_y + bar_height + 14, explanation)
        
        return bar_y + bar_height + 22
    
    def _draw_comparison_bars(self, painter: QPainter, x: int, y: int,
                             width: int, height: int, equity: float,
                             pot_odds: float) -> None:
        """Draw equity vs pot odds comparison bars."""
        # Equity bar background
        painter.setBrush(QColor(50, 50, 60, 200))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(x, y, width, height, 5, 5)
        
        # Equity fill
        equity_width = int(width * equity)
        painter.setBrush(QColor(100, 200, 255))
        painter.drawRoundedRect(x, y, equity_width, height, 5, 5)
        
        # Pot odds marker (if applicable)
        if pot_odds > 0:
            marker_x = x + int(width * pot_odds)
            painter.setPen(QPen(QColor(255, 100, 100), 3))
            painter.drawLine(marker_x, y, marker_x, y + height)
            
            # Label
            font = QFont("Segoe UI", 8)
            painter.setFont(font)
            painter.setPen(QColor(255, 100, 100))
            painter.drawText(marker_x - 20, y - 5, "Need")
    
    def _draw_alternatives_section(self, painter: QPainter, decision: Decision,
                                   x: int, y: int, width: int) -> int:
        """Draw alternative actions with EV comparison."""
        alternatives = decision.alternative_actions
        if not alternatives:
            return y
        
        # Add spacing
        y += 10
        
        # Section title
        font = QFont("Segoe UI", 10, QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(self.TEXT_ACCENT_YELLOW)
        painter.drawText(x + 15, y + 12, "ALTERNATIVE PLAYS:")
        
        # Draw each alternative
        alt_y = y + 32
        font = QFont("Segoe UI", 9)
        painter.setFont(font)
        
        for alt in alternatives:
            # Action name
            painter.setPen(self.TEXT_SECONDARY)
            action_text = alt.action.upper()
            if alt.amount:
                action_text += f" ${alt.amount:.2f}"
            painter.drawText(x + 20, alt_y, action_text)
            
            # EV value (right-aligned)
            ev_color = self.CONFIDENCE_HIGH if alt.ev >= 0 else self.CONFIDENCE_LOW
            painter.setPen(ev_color)
            ev_text = f"EV: {alt.ev:+.2f}"
            
            # Right align the EV text
            text_width = painter.fontMetrics().horizontalAdvance(ev_text)
            painter.drawText(x + width - text_width - 20, alt_y, ev_text)
            
            alt_y += 30
        
        return alt_y
    
    def _draw_learning_tip(self, painter: QPainter, decision: Decision,
                          game_state: GameState, x: int, y: int, width: int) -> None:
        """Draw a learning tip at the bottom of the panel."""
        # Get contextual learning tip
        tip = self._get_learning_tip(decision, game_state)
        
        if not tip:
            return
        
        # Add spacing before tip section
        y += 15
        
        # Draw subtle separator
        painter.setPen(QPen(QColor(80, 80, 100, 150), 1))
        painter.drawLine(x + 15, y, x + width - 15, y)
        
        # Move down after separator
        y += 12
        
        # Draw lightbulb icon (simple circle with better styling)
        icon_x = x + 20
        icon_y = y + 2  # Align with first line of text
        icon_size = 14
        
        # Icon background
        painter.setBrush(QColor(255, 220, 100, 220))
        painter.setPen(QPen(QColor(200, 160, 50, 150), 1))
        painter.drawEllipse(icon_x, icon_y, icon_size, icon_size)
        
        # Draw "i" inside icon
        font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(QColor(40, 40, 50))
        painter.drawText(
            QRect(icon_x, icon_y, icon_size, icon_size),
            Qt.AlignmentFlag.AlignCenter,
            "i"
        )
        
        # Draw tip text (aligned with icon)
        font = QFont("Segoe UI", 9, QFont.Weight.Normal, True)  # Italic
        painter.setFont(font)
        painter.setPen(self.TEXT_ACCENT_GOLD)
        
        # Calculate text area
        text_x = icon_x + icon_size + 10
        text_width = width - (text_x - x) - 20
        
        wrapped_lines = self._wrap_text(tip, text_width, painter.fontMetrics())
        tip_y = y + 12  # Vertically center with icon
        
        for line in wrapped_lines:
            painter.drawText(text_x, tip_y, line)
            tip_y += 16
    
    def _get_learning_tip(self, decision: Decision, game_state: GameState) -> str:
        """Get a contextual learning tip based on current situation."""
        # Preflop tips
        if game_state.phase == 'preflop':
            if decision.action == 'fold':
                return "Tight is right preflop. Most hands lose money long-term."
            elif decision.action in ['raise', 'bet']:
                return "Raising builds the pot with strong hands and applies pressure."
            elif decision.action == 'call':
                return "Calling can be correct with speculative hands that play well postflop."
        
        # Postflop tips
        else:
            if decision.equity and decision.pot_odds:
                if decision.equity > decision.pot_odds:
                    return "When equity > pot odds, calling is mathematically profitable."
                else:
                    return "Fold when your equity doesn't justify the pot odds you're getting."
            
            if decision.action == 'fold':
                return "Good folds save money. Not every hand can be a winner."
            elif decision.action in ['raise', 'bet']:
                return "Betting for value or as a bluff puts opponents to tough decisions."
        
        return "Study GTO principles to improve your long-term win rate."
    
    def _wrap_text(self, text: str, max_width: int, font_metrics) -> List[str]:
        """Wrap text to fit within max_width."""
        words = text.split()
        lines = []
        current_line = []
        
        # If no font_metrics, estimate based on character count
        if font_metrics is None:
            chars_per_line = max(1, max_width // 7)  # Rough estimate: 7px per char
            current_chars = 0
            for word in words:
                word_len = len(word) + 1  # +1 for space
                if current_chars + word_len > chars_per_line and current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_chars = word_len
                else:
                    current_line.append(word)
                    current_chars += word_len
            if current_line:
                lines.append(' '.join(current_line))
            return lines
        
        # Use actual font metrics
        for word in words:
            test_line = ' '.join(current_line + [word])
            if font_metrics.horizontalAdvance(test_line) <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines


if __name__ == "__main__":
    # Setup logging for testing
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("OverlayRenderer initialized successfully")
    renderer = OverlayRenderer()
    print(f"Renderer ready with {len(renderer.ACTION_COLORS)} action colors")

