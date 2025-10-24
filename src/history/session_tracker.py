#!/usr/bin/env python3
"""
SessionTracker for manual poker session management.

Handles session lifecycle (start/stop), hand recording, and statistics
calculation. Integrates with HandStorage for persistence.
"""

import logging
import uuid
from typing import Optional, Dict, Any
from datetime import datetime

from src.models.hand_record import HandRecord
from src.history.hand_storage import HandStorage
from src.config.settings import Settings

logger = logging.getLogger(__name__)


class SessionTracker:
    """Manual session management with statistics tracking."""
    
    def __init__(self, hand_storage: HandStorage):
        """
        Initialize session tracker.
        
        Args:
            hand_storage: HandStorage instance for persistence
        """
        self.hand_storage = hand_storage
        self.settings = Settings()
        
        # Current session state
        self.current_session_id: Optional[str] = None
        self.session_start_time: Optional[datetime] = None
        self.hands_in_session: int = 0
        self.session_profit: float = 0.0
        
        logger.info("SessionTracker initialized")
    
    def start_session(self) -> str:
        """
        Start a new poker session.
        
        Returns:
            session_id: UUID string for the new session
            
        Raises:
            RuntimeError: If a session is already active
        """
        if self.is_session_active():
            raise RuntimeError(
                f"Session {self.current_session_id} is already active. "
                "End current session before starting a new one."
            )
        
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        # Create session in database
        success = self.hand_storage.create_session(session_id, start_time)
        if not success:
            raise RuntimeError(f"Failed to create session {session_id} in database")
        
        # Initialize session state
        self.current_session_id = session_id
        self.session_start_time = start_time
        self.hands_in_session = 0
        self.session_profit = 0.0
        
        logger.info(f"Session {session_id} started at {start_time.isoformat()}")
        return session_id
    
    def end_session(self) -> Dict[str, Any]:
        """
        End the current session and return statistics.
        
        Returns:
            Dict with keys: session_id, start_time, end_time, duration_minutes,
                           total_hands, total_profit, avg_profit_per_hand
            
        Raises:
            RuntimeError: If no session is active
        """
        if not self.is_session_active():
            raise RuntimeError("No active session to end")
        
        end_time = datetime.now()
        duration = end_time - self.session_start_time
        duration_minutes = duration.total_seconds() / 60
        
        # Update session in database
        success = self.hand_storage.update_session(
            self.current_session_id,
            end_time,
            self.hands_in_session,
            self.session_profit
        )
        
        if not success:
            logger.warning(f"Failed to update session {self.current_session_id} in database")
        
        # Build statistics dict
        stats = {
            'session_id': self.current_session_id,
            'start_time': self.session_start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_minutes': round(duration_minutes, 2),
            'total_hands': self.hands_in_session,
            'total_profit': round(self.session_profit, 2),
            'avg_profit_per_hand': (
                round(self.session_profit / self.hands_in_session, 2)
                if self.hands_in_session > 0 else 0.0
            )
        }
        
        logger.info(
            f"Session {self.current_session_id} ended: "
            f"{self.hands_in_session} hands, "
            f"{duration_minutes:.1f} minutes, "
            f"P/L: {self.session_profit:.2f}"
        )
        
        # Reset session state
        self.current_session_id = None
        self.session_start_time = None
        self.hands_in_session = 0
        self.session_profit = 0.0
        
        return stats
    
    def is_session_active(self) -> bool:
        """
        Check if a session is currently running.
        
        Returns:
            True if session is active, False otherwise
        """
        return self.current_session_id is not None
    
    def record_hand_in_session(self, hand_record: HandRecord) -> bool:
        """
        Save a completed hand to the current session.
        
        Args:
            hand_record: Completed HandRecord from HandTracker
            
        Returns:
            True if saved successfully, False otherwise
        """
        if not self.is_session_active():
            logger.warning(
                f"No active session - hand {hand_record.hand_id} not persisted to database"
            )
            return False
        
        # Save hand to database
        success = self.hand_storage.save_hand(hand_record, self.current_session_id)
        
        if success:
            # Update session statistics
            self.hands_in_session += 1
            self.session_profit += hand_record.net_profit
            
            logger.info(
                f"Hand {hand_record.hand_id} recorded in session {self.current_session_id} "
                f"(Session totals: {self.hands_in_session} hands, P/L: {self.session_profit:.2f})"
            )
        else:
            logger.error(f"Failed to save hand {hand_record.hand_id} to database")
        
        return success
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Calculate statistics for a specific session.
        
        Args:
            session_id: Session to analyze
            
        Returns:
            Dict with: session_id, start_time, end_time, duration_minutes,
                      total_hands, total_profit, avg_profit_per_hand,
                      win_rate, bb_per_100_hands (if BB data available)
        """
        # Fetch session metadata
        session_data = self.hand_storage.get_session(session_id)
        if not session_data:
            logger.warning(f"Session {session_id} not found")
            return {}
        
        # Fetch all hands for session
        hands = self.hand_storage.get_hands_for_session(session_id)
        
        # Calculate statistics
        total_hands = len(hands)
        total_profit = sum(hand.net_profit for hand in hands)
        
        # Calculate win rate (hands won / total hands)
        won_hands = sum(1 for hand in hands if hand.result == 'won')
        win_rate = (won_hands / total_hands * 100) if total_hands > 0 else 0.0
        
        # Calculate average profit per hand
        avg_profit_per_hand = (total_profit / total_hands) if total_hands > 0 else 0.0
        
        # Calculate duration
        start_time = datetime.fromisoformat(session_data['start_time'])
        end_time = (
            datetime.fromisoformat(session_data['end_time'])
            if session_data['end_time'] else datetime.now()
        )
        duration = end_time - start_time
        duration_minutes = duration.total_seconds() / 60
        
        # Calculate BB/100 hands if we have table info
        bb_per_100 = None
        if hands and hands[0].table_info:
            bb = hands[0].table_info.bb
            bb_per_100 = (total_profit / bb / total_hands * 100) if total_hands > 0 else 0.0
        
        stats = {
            'session_id': session_id,
            'start_time': session_data['start_time'],
            'end_time': session_data['end_time'],
            'duration_minutes': round(duration_minutes, 2),
            'total_hands': total_hands,
            'total_profit': round(total_profit, 2),
            'avg_profit_per_hand': round(avg_profit_per_hand, 2),
            'win_rate': round(win_rate, 2),
            'won_hands': won_hands,
            'lost_hands': sum(1 for hand in hands if hand.result == 'lost'),
            'folded_hands': sum(1 for hand in hands if hand.result == 'folded')
        }
        
        if bb_per_100 is not None:
            stats['bb_per_100_hands'] = round(bb_per_100, 2)
        
        return stats
    
    def get_current_session_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get real-time stats for the active session.
        
        Returns:
            Dict with current session stats if active, None otherwise
        """
        if not self.is_session_active():
            return None
        
        duration = datetime.now() - self.session_start_time
        duration_minutes = duration.total_seconds() / 60
        
        return {
            'session_id': self.current_session_id,
            'start_time': self.session_start_time.isoformat(),
            'duration_minutes': round(duration_minutes, 2),
            'hands_played': self.hands_in_session,
            'current_profit': round(self.session_profit, 2),
            'avg_profit_per_hand': (
                round(self.session_profit / self.hands_in_session, 2)
                if self.hands_in_session > 0 else 0.0
            )
        }

