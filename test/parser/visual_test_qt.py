#!/usr/bin/env python3
"""
PyQt6-based visual testing interface for parser components.

Provides a modern, professional GUI for validating parser results with:
- Large, clear image display
- Easy-to-read result information
- Keyboard shortcuts
- Progress tracking
- Better user experience than OpenCV windows
"""

import sys
import numpy as np
from pathlib import Path
import qtawesome as qta
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QProgressBar, QTextEdit, QGroupBox, QFrame
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QKeySequence, QShortcut, QFont, QPalette

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


class VisualTestWindow(QMainWindow):
    """Main window for visual testing with PyQt6."""
    
    def __init__(self):
        super().__init__()
        self.result = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Parser Visual Test")
        self.setMinimumSize(800, 600)  # Allow smaller minimum size
        self.resize(1200, 800)  # Set default size
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Progress bar at top
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Test %v of %m")
        main_layout.addWidget(self.progress_bar)
        
        # Unified content area - single panel with all information
        content_group = QGroupBox("Parser Test Result")
        content_layout = QVBoxLayout()
        content_layout.setSpacing(20)
        
        # Parser type - Centered heading with icon
        parser_header_layout = QHBoxLayout()
        parser_header_layout.setSpacing(10)
        
        # Add stretch to center the icon and label
        parser_header_layout.addStretch()
        
        # Dynamic parser icon
        self.parser_icon_label = QLabel()
        self.parser_icon_label.setFixedSize(32, 32)
        parser_header_layout.addWidget(self.parser_icon_label)
        
        # Parser name as large heading - centered
        self.parser_label = QLabel()
        self.parser_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        self.parser_label.setStyleSheet("QLabel { color: #4CAF50; }")
        self.parser_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        parser_header_layout.addWidget(self.parser_label)
        
        # Add stretch to center the icon and label
        parser_header_layout.addStretch()
        
        content_layout.addLayout(parser_header_layout)
        
        # RESULT - Prominent centered box
        self.result_label = QLabel()
        self.result_label.setFont(QFont("Arial", 28, QFont.Weight.Bold))
        self.result_label.setWordWrap(True)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setMinimumHeight(100)
        self.result_label.setStyleSheet("""
            QLabel { 
                color: white; 
                padding: 25px; 
                background-color: #2d2d2d; 
                border-radius: 12px; 
                border: 3px solid #444;
                font-weight: bold;
            }
        """)
        content_layout.addWidget(self.result_label)
        
        # Image display - centered
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background-color: #1e1e1e; border: 2px solid #444; }")
        self.image_label.setMinimumSize(600, 400)
        content_layout.addWidget(self.image_label)
        
        # Confidence - Just the progress bar
        self.confidence_progress = QProgressBar()
        self.confidence_progress.setMinimum(0)
        self.confidence_progress.setMaximum(100)
        self.confidence_progress.setFormat("%p%")
        self.confidence_progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #444;
                border-radius: 8px;
                text-align: center;
                height: 30px;
                font-weight: bold;
                font-size: 14px;
            }
            QProgressBar::chunk {
                border-radius: 6px;
            }
        """)
        content_layout.addWidget(self.confidence_progress)
        
        content_group.setLayout(content_layout)
        main_layout.addWidget(content_group)
        
        # Button panel at bottom - responsive layout
        self.button_container = QWidget()
        self.button_layout = QHBoxLayout()
        self.button_layout.setSpacing(15)
        self.button_container.setLayout(self.button_layout)
        
        # Create buttons
        self._create_buttons()
        
        # Add button container to main layout
        main_layout.addWidget(self.button_container)
        
        # Keyboard shortcuts
        QShortcut(QKeySequence('Y'), self, lambda: self.set_result(True))
        QShortcut(QKeySequence('N'), self, lambda: self.set_result(False))
        QShortcut(QKeySequence('S'), self, lambda: self.set_result(None))
        QShortcut(QKeySequence('Q'), self, lambda: self.set_result('quit'))
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, lambda: self.set_result('quit'))
        
        # Apply dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QWidget {
                background-color: #2b2b2b;
                color: white;
            }
            QGroupBox {
                border: 2px solid #444;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QProgressBar {
                border: 2px solid #444;
                border-radius: 5px;
                text-align: center;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
    
    def _create_buttons(self):
        """Create all buttons with consistent styling."""
        button_style = """
            QPushButton {
                background-color: %s;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
                padding: 10px;
                text-align: center;
                icon-size: 20px;
                min-height: 50px;
            }
            QPushButton:hover {
                background-color: %s;
            }
        """
        
        # Create horizontal layout buttons
        self.approve_btn_h = QPushButton()
        self.approve_btn_h.setIcon(qta.icon('fa5s.check', color='white'))
        self.approve_btn_h.setText("Approve (Y)")
        self.approve_btn_h.setStyleSheet(button_style % ("#4CAF50", "#45a049"))
        self.approve_btn_h.clicked.connect(lambda: self.set_result(True))
        
        self.reject_btn_h = QPushButton()
        self.reject_btn_h.setIcon(qta.icon('fa5s.times', color='white'))
        self.reject_btn_h.setText("Reject (N)")
        self.reject_btn_h.setStyleSheet(button_style % ("#f44336", "#da190b"))
        self.reject_btn_h.clicked.connect(lambda: self.set_result(False))
        
        self.skip_btn_h = QPushButton()
        self.skip_btn_h.setIcon(qta.icon('fa5s.forward', color='white'))
        self.skip_btn_h.setText("Skip (S)")
        self.skip_btn_h.setStyleSheet(button_style % ("#FF9800", "#e68900"))
        self.skip_btn_h.clicked.connect(lambda: self.set_result(None))
        
        self.quit_btn_h = QPushButton()
        self.quit_btn_h.setIcon(qta.icon('fa5s.stop', color='white'))
        self.quit_btn_h.setText("Quit (Q)")
        self.quit_btn_h.setStyleSheet(button_style % ("#607D8B", "#546E7A"))
        self.quit_btn_h.clicked.connect(lambda: self.set_result('quit'))
        
        # Create grid layout buttons
        self.approve_btn_g = QPushButton()
        self.approve_btn_g.setIcon(qta.icon('fa5s.check', color='white'))
        self.approve_btn_g.setText("Approve (Y)")
        self.approve_btn_g.setStyleSheet(button_style % ("#4CAF50", "#45a049"))
        self.approve_btn_g.clicked.connect(lambda: self.set_result(True))
        
        self.reject_btn_g = QPushButton()
        self.reject_btn_g.setIcon(qta.icon('fa5s.times', color='white'))
        self.reject_btn_g.setText("Reject (N)")
        self.reject_btn_g.setStyleSheet(button_style % ("#f44336", "#da190b"))
        self.reject_btn_g.clicked.connect(lambda: self.set_result(False))
        
        self.skip_btn_g = QPushButton()
        self.skip_btn_g.setIcon(qta.icon('fa5s.forward', color='white'))
        self.skip_btn_g.setText("Skip (S)")
        self.skip_btn_g.setStyleSheet(button_style % ("#FF9800", "#e68900"))
        self.skip_btn_g.clicked.connect(lambda: self.set_result(None))
        
        self.quit_btn_g = QPushButton()
        self.quit_btn_g.setIcon(qta.icon('fa5s.stop', color='white'))
        self.quit_btn_g.setText("Quit (Q)")
        self.quit_btn_g.setStyleSheet(button_style % ("#607D8B", "#546E7A"))
        self.quit_btn_g.clicked.connect(lambda: self.set_result('quit'))
        
        # Show appropriate layout based on initial window size
        self._update_button_layout()
    
    def _update_button_layout(self):
        """Update button layout based on window width."""
        if self.width() < 1000:  # Threshold for switching to 2x2
            self._show_grid_layout()
        else:
            self._show_horizontal_layout()
    
    def _show_horizontal_layout(self):
        """Show horizontal button layout for wider windows."""
        # Hide grid widget
        if hasattr(self, 'grid_widget'):
            self.grid_widget.hide()
        
        # Show horizontal widget
        if hasattr(self, 'horizontal_widget'):
            self.horizontal_widget.show()
        else:
            self._create_horizontal_widget()
    
    def _show_grid_layout(self):
        """Show 2x2 grid button layout for narrower windows."""
        # Hide horizontal widget
        if hasattr(self, 'horizontal_widget'):
            self.horizontal_widget.hide()
        
        # Show grid widget
        if hasattr(self, 'grid_widget'):
            self.grid_widget.show()
        else:
            self._create_grid_widget()
    
    def _create_horizontal_widget(self):
        """Create the horizontal widget with buttons."""
        self.horizontal_widget = QWidget()
        horizontal_layout = QHBoxLayout(self.horizontal_widget)
        horizontal_layout.setSpacing(15)
        
        # Add horizontal buttons
        horizontal_layout.addWidget(self.approve_btn_h)
        horizontal_layout.addWidget(self.reject_btn_h)
        horizontal_layout.addWidget(self.skip_btn_h)
        horizontal_layout.addWidget(self.quit_btn_h)
        
        # Add horizontal widget to main button layout
        self.button_layout.addWidget(self.horizontal_widget)
    
    def _create_grid_widget(self):
        """Create the grid widget with buttons."""
        self.grid_widget = QWidget()
        grid_layout = QGridLayout(self.grid_widget)
        grid_layout.setSpacing(15)
        
        # Add buttons in 2x2 grid
        grid_layout.addWidget(self.approve_btn_g, 0, 0)
        grid_layout.addWidget(self.reject_btn_g, 0, 1)
        grid_layout.addWidget(self.skip_btn_g, 1, 0)
        grid_layout.addWidget(self.quit_btn_g, 1, 1)
        
        # Add grid widget to main button layout
        self.button_layout.addWidget(self.grid_widget)
    
    def resizeEvent(self, event):
        """Handle window resize events to update button layout."""
        super().resizeEvent(event)
        # Update button layout when window is resized
        self._update_button_layout()
    
    def set_result(self, result):
        """Set the validation result (don't close window)."""
        self.result = result
        # Don't close the window - it will be reused for next test
    
    def update_display(self, region_image: np.ndarray, parser_name: str, region_name: str,
                      parsed_value, confidence: float, preprocessing_method: str,
                      test_num: int, total_tests: int):
        """Update the display with new test data."""
        # Update progress
        self.progress_bar.setMaximum(total_tests)
        self.progress_bar.setValue(test_num)
        
        # Set dynamic parser icon based on parser type
        parser_icons = {
            'card': 'fa5s.heart',           # Card Detection - heart for cards
            'money': 'fa5s.dollar-sign',   # Money Amount - dollar sign
            'status': 'fa5s.user-tag',      # Player Status - user tag for status
            'name': 'fa5s.user',            # Player Name - user icon
            'table_info': 'fa5s.coins',     # Stake Amount - coins for stakes
            'hand_id': 'fa5s.hashtag',      # Hand ID - hashtag for ID
            'timer': 'fa5s.play-circle',    # Is Player Turn - play circle for turn
            'transparency': 'fa5s.eye-slash', # Player In Play - eye-slash for folded/not in play
            'dealer': 'fa5s.crown'          # Dealer Button - crown for dealer
        }
        
        # Map parser names to better labels
        parser_labels = {
            'transparency': 'Player In Play',
            'timer': 'Is Player Turn', 
            'table_info': 'Stake Amount',
            'card': 'Card Detection',
            'money': 'Money Amount',
            'status': 'Player Status',
            'name': 'Player Name',
            'hand_id': 'Hand ID',
            'dealer': 'Dealer Button'
        }
        
        # Update parser type with dynamic icon
        parser_label_text = parser_labels.get(parser_name.lower(), parser_name.upper())
        self.parser_label.setText(parser_label_text)
        
        parser_icon = parser_icons.get(parser_name.lower(), 'fa5s.cogs')
        self.parser_icon_label.setPixmap(qta.icon(parser_icon, color='#4CAF50').pixmap(32, 32))
        
        # Update button text based on parser type
        self.approve_btn_h.setText("Approve (Y)")
        self.reject_btn_h.setText("Reject (N)")
        self.approve_btn_g.setText("Approve (Y)")
        self.reject_btn_g.setText("Reject (N)")
        
        # Update result with enhanced styling
        if parsed_value is None or parsed_value == "" or str(parsed_value) == "None":
            self.result_label.setText("No Result Detected")
            self.result_label.setStyleSheet("""
                QLabel { 
                    color: #FF9800; 
                    padding: 25px; 
                    background-color: #2d2d2d; 
                    border-radius: 12px; 
                    border: 3px solid #FF9800;
                    font-weight: bold;
                }
            """)
        else:
            # Format result in a user-friendly way
            display_text = self._format_result_for_display(parsed_value, parser_name)
            self.result_label.setText(display_text)
            self.result_label.setStyleSheet("""
                QLabel { 
                    color: white; 
                    padding: 25px; 
                    background-color: #2d2d2d; 
                    border-radius: 12px; 
                    border: 3px solid #4CAF50;
                    font-weight: bold;
                }
            """)
        
        # Update confidence progress bar
        conf_percent = int(confidence * 100)
        self.confidence_progress.setValue(conf_percent)
        
        if confidence > 0.8:
            conf_color = "#4CAF50"  # Green
        elif confidence > 0.5:
            conf_color = "#FF9800"  # Orange
        else:
            conf_color = "#F44336"  # Red
        
        # Update progress bar color
        self.confidence_progress.setStyleSheet(f"""
            QProgressBar {{
                border: 2px solid #444;
                border-radius: 8px;
                text-align: center;
                height: 30px;
                font-weight: bold;
                font-size: 14px;
            }}
            QProgressBar::chunk {{
                background-color: {conf_color};
                border-radius: 6px;
            }}
        """)
        
        # Update main image
        self.display_image(region_image)
    
    def display_image(self, image: np.ndarray):
        """Display numpy image in QLabel."""
        # Convert BGR to RGB
        if len(image.shape) == 3:
            image_rgb = image[:, :, ::-1].copy()
        else:
            image_rgb = image
        
        # Scale image to fit display (maintain aspect ratio)
        height, width = image_rgb.shape[:2]
        max_width = 580
        max_height = 380
        
        scale = min(max_width / width, max_height / height, 4.0)  # Max 4x upscale
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Create QImage
        if len(image_rgb.shape) == 3:
            bytes_per_line = 3 * width
            q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        else:
            bytes_per_line = width
            q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        
        # Scale and display
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(new_width, new_height, Qt.AspectRatioMode.KeepAspectRatio, 
                                       Qt.TransformationMode.FastTransformation)
        self.image_label.setPixmap(scaled_pixmap)
    
    def _format_result_for_display(self, parsed_value, parser_name: str) -> str:
        """
        Format parser results in a user-friendly, readable way.
        
        Args:
            parsed_value: The parsed value from any parser
            parser_name: Name of the parser
            
        Returns:
            Formatted string for display
        """
        parser_name_lower = parser_name.lower()
        
        # Handle different parser types with appropriate formatting
        if parser_name_lower in ['card', 'cards']:
            return self._format_card_result(parsed_value)
        elif parser_name_lower == 'money':
            return self._format_money_result(parsed_value)
        elif parser_name_lower == 'status':
            return self._format_status_result(parsed_value)
        elif parser_name_lower == 'transparency':
            return self._format_boolean_result(parsed_value, "Player Folded", "Player In Play")
        elif parser_name_lower == 'timer':
            return self._format_boolean_result(parsed_value, "Player's Turn", "Not Turn")
        elif parser_name_lower == 'table_info':
            return self._format_table_info_result(parsed_value)
        elif parser_name_lower == 'name':
            return self._format_name_result(parsed_value)
        elif parser_name_lower == 'hand_id':
            return self._format_hand_id_result(parsed_value)
        elif parser_name_lower == 'dealer':
            return self._format_boolean_result(parsed_value, "Dealer Button", "Not Dealer")
        else:
            # Default formatting for unknown parsers
            return self._format_generic_result(parsed_value)
    
    def _format_card_result(self, parsed_value) -> str:
        """
        Format card results to use suit icons instead of suit names.
        
        Args:
            parsed_value: The parsed value from the card parser
            
        Returns:
            Formatted string with suit icons
        """
        # Handle None/empty cases first
        if parsed_value is None:
            return "No Card Detected"
        
        # Handle empty string
        if isinstance(parsed_value, str) and not parsed_value.strip():
            return "No Card Detected"
        
        # Suit name to Unicode symbol mapping
        suit_symbols = {
            'hearts': '♥',
            'diamonds': '♦', 
            'clubs': '♣',
            'spades': '♠'
        }
        
        # Handle different card result formats
        if hasattr(parsed_value, 'rank') and hasattr(parsed_value, 'suit'):
            # CardResult object
            rank = parsed_value.rank
            suit = parsed_value.suit
            
            # Handle missing rank or suit
            if not rank and not suit:
                return "No Card Detected"
            elif not rank:
                return f"?{suit_symbols.get(suit, suit)}"
            elif not suit:
                return f"{rank}?"
            else:
                suit_symbol = suit_symbols.get(suit, suit)
                return f"{rank}{suit_symbol}"
        
        elif isinstance(parsed_value, str):
            # String format - try to convert suit names to symbols
            result = parsed_value.strip()
            if not result:
                return "No Card Detected"
            
            # Handle common OCR errors
            if result.lower() in ['none', 'null', 'empty', '']:
                return "No Card Detected"
            
            # Convert suit names to symbols
            for suit_name, symbol in suit_symbols.items():
                result = result.replace(suit_name, symbol)
            return result
        
        else:
            # Fallback to string representation
            return str(parsed_value) if parsed_value else "No Card Detected"
    
    def _format_money_result(self, parsed_value) -> str:
        """Format money amounts with currency symbols and proper formatting."""
        if parsed_value is None:
            return "No Amount"
        
        # Handle empty string
        if isinstance(parsed_value, str) and not parsed_value.strip():
            return "No Amount"
        
        try:
            # Try to convert to float for formatting
            amount = float(parsed_value)
            if amount >= 1000:
                return f"${amount:,.0f}"
            elif amount >= 1:
                return f"${amount:.2f}"
            else:
                return f"${amount:.3f}"
        except (ValueError, TypeError):
            # Handle string representations
            if isinstance(parsed_value, str):
                cleaned = parsed_value.strip()
                if cleaned.lower() in ['none', 'null', 'empty', '']:
                    return "No Amount"
            return str(parsed_value)
    
    def _format_status_result(self, parsed_value) -> str:
        """Format player status in a readable way."""
        if parsed_value is None:
            return "No Status"
        
        if isinstance(parsed_value, str):
            cleaned = parsed_value.strip()
            if not cleaned or cleaned.lower() in ['none', 'null', 'empty', '']:
                return "No Status"
            
            status = cleaned.lower()
        else:
            status = str(parsed_value).lower()
        
        # Map statuses to more readable formats
        status_mapping = {
            '': 'No Action',
            'call': 'Call',
            'raise': 'Raise',
            'check': 'Check',
            'bb': 'Big Blind',
            'sb': 'Small Blind',
            'fold': 'Fold',
            'bet': 'Bet',
            'all-in': 'All-In',
            'away': 'Away',
            'straddle': 'Straddle',
            'show cards': 'Show Cards',
            'decline straddle': 'Decline Straddle'
        }
        
        return status_mapping.get(status, parsed_value.title() if parsed_value else "No Status")
    
    def _format_boolean_result(self, parsed_value, true_text: str, false_text: str) -> str:
        """Format boolean results with descriptive text."""
        if parsed_value is None:
            return "Unknown"
        
        # Handle various boolean representations
        if isinstance(parsed_value, bool):
            return true_text if parsed_value else false_text
        elif isinstance(parsed_value, str):
            lower_val = parsed_value.lower()
            if lower_val in ['true', '1', 'yes', 'on']:
                return true_text
            elif lower_val in ['false', '0', 'no', 'off']:
                return false_text
            else:
                return str(parsed_value)
        elif isinstance(parsed_value, (int, float)):
            return true_text if parsed_value else false_text
        else:
            return str(parsed_value)
    
    def _format_table_info_result(self, parsed_value) -> str:
        """Format table info results."""
        if parsed_value is None:
            return "No Table Info"
        
        if isinstance(parsed_value, dict):
            # Format dictionary results
            parts = []
            for key, value in parsed_value.items():
                if key.lower() in ['sb', 'small_blind']:
                    parts.append(f"SB: ${value}")
                elif key.lower() in ['bb', 'big_blind']:
                    parts.append(f"BB: ${value}")
                else:
                    parts.append(f"{key.title()}: {value}")
            return " | ".join(parts)
        else:
            return str(parsed_value)
    
    def _format_name_result(self, parsed_value) -> str:
        """Format player name results."""
        if parsed_value is None:
            return "No Name"
        
        if isinstance(parsed_value, str):
            cleaned = parsed_value.strip()
            if not cleaned or cleaned.lower() in ['none', 'null', 'empty', '']:
                return "No Name"
            return cleaned
        
        return str(parsed_value) if parsed_value else "No Name"
    
    def _format_hand_id_result(self, parsed_value) -> str:
        """Format hand ID results."""
        if parsed_value is None:
            return "No Hand ID"
        
        if isinstance(parsed_value, str):
            cleaned = parsed_value.strip()
            if not cleaned or cleaned.lower() in ['none', 'null', 'empty', '']:
                return "No Hand ID"
            return f"Hand #{cleaned}"
        
        if isinstance(parsed_value, (int, float)):
            return f"Hand #{int(parsed_value)}"
        
        return str(parsed_value) if parsed_value else "No Hand ID"
    
    def _format_generic_result(self, parsed_value) -> str:
        """Format generic results for unknown parsers."""
        if parsed_value is None:
            return "No Result"
        
        if isinstance(parsed_value, str):
            cleaned = parsed_value.strip()
            if not cleaned or cleaned.lower() in ['none', 'null', 'empty', '']:
                return "No Result"
            return cleaned
        
        return str(parsed_value) if parsed_value else "No Result"
    
    def get_result(self):
        """Get the validation result."""
        return self.result


# Global window instance to reuse
_global_window = None
_global_app = None


def show_visual_test(region_image: np.ndarray, parser_name: str, region_name: str,
                     parsed_value, confidence: float, preprocessing_method: str,
                     test_num: int = 1, total_tests: int = 1):
    """
    Show visual test window and get user validation.
    Reuses the same window for all tests.
    
    Returns:
        True if approved, False if rejected, None if skipped, 'quit' if user wants to quit
    """
    global _global_window, _global_app
    
    # Create QApplication if it doesn't exist
    if _global_app is None:
        _global_app = QApplication.instance()
        if _global_app is None:
            _global_app = QApplication(sys.argv)
    
    # Create window on first call
    if _global_window is None:
        _global_window = VisualTestWindow()
        _global_window.show()
    
    # Update display with new test data
    _global_window.result = None  # Reset result
    _global_window.update_display(region_image, parser_name, region_name, parsed_value, 
                                  confidence, preprocessing_method, test_num, total_tests)
    
    # Ensure window is visible and has focus
    _global_window.show()
    _global_window.raise_()
    _global_window.activateWindow()
    
    # Process events until user makes a choice
    import time
    while _global_window.result is None:
        _global_app.processEvents()
        time.sleep(0.01)  # Small delay to prevent CPU spinning
        if not _global_window.isVisible():
            # Window was closed
            return 'quit'
    
    result = _global_window.result
    
    # If user quit, close the window
    if result == 'quit':
        _global_window.close()
        _global_window = None
    
    return result


def close_visual_test_window():
    """Close the global visual test window if it exists."""
    global _global_window
    if _global_window is not None:
        _global_window.close()
        _global_window = None


if __name__ == "__main__":
    # Test the interface
    import cv2
    test_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
    
    # Test with mock CardResult object
    class MockCardResult:
        def __init__(self, rank, suit, confidence):
            self.rank = rank
            self.suit = suit
            self.confidence = confidence
    
    card_result = MockCardResult(rank="A", suit="spades", confidence=0.95)
    result = show_visual_test(test_image, "card", "player_1_hole_1", card_result, 0.95, "threshold", 1, 10)
    print(f"Result: {result}")

