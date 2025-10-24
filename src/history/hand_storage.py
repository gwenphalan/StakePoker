#!/usr/bin/env python3
"""
HandStorage for SQLite-based hand history persistence.

Manages database operations for storing and retrieving poker hand records,
actions, and session metadata. Uses SQLite with proper schema design and
transaction handling.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from src.models.hand_record import HandRecord, Action
from src.models.card import Card
from src.models.table_info import TableInfo
from src.config.settings import Settings

logger = logging.getLogger(__name__)


class HandStorage:
    """SQLite database manager for hand history storage."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database connection and create tables.
        
        Args:
            db_path: Path to SQLite database. If None, uses Settings.
        """
        self.settings = Settings()
        
        if db_path is None:
            db_path = self.settings.get("history.database_path", "data/poker_history.db")
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        self._create_tables()
        
        logger.info(f"HandStorage initialized with database: {self.db_path}")
    
    def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        try:
            cursor = self.conn.cursor()
            
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    is_active INTEGER DEFAULT 1,
                    total_hands INTEGER DEFAULT 0,
                    total_profit REAL DEFAULT 0.0
                )
            """)
            
            # Hands table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hands (
                    hand_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    hero_seat INTEGER NOT NULL,
                    hero_position TEXT NOT NULL,
                    hero_cards TEXT NOT NULL,
                    table_info TEXT NOT NULL,
                    result TEXT,
                    net_profit REAL NOT NULL,
                    final_pot REAL NOT NULL,
                    showdown INTEGER NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)
            
            # Actions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hand_id TEXT NOT NULL,
                    seat_number INTEGER NOT NULL,
                    action_type TEXT NOT NULL,
                    amount REAL,
                    phase TEXT NOT NULL,
                    action_order INTEGER NOT NULL,
                    FOREIGN KEY (hand_id) REFERENCES hands(hand_id)
                )
            """)
            
            # Create indexes for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_hands_session 
                ON hands(session_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_actions_hand 
                ON actions(hand_id)
            """)
            
            self.conn.commit()
            logger.info("Database tables created successfully")
            
        except sqlite3.Error as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    def save_hand(self, hand: HandRecord, session_id: str) -> bool:
        """
        Save a HandRecord and its actions to the database.
        
        Args:
            hand: HandRecord to save
            session_id: Session ID to associate with this hand
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            
            # Begin transaction
            cursor.execute("BEGIN TRANSACTION")
            
            # Serialize complex fields
            hero_cards_json = self._serialize_cards(hand.hero_cards)
            table_info_json = self._serialize_table_info(hand.table_info)
            
            # Insert hand record
            cursor.execute("""
                INSERT INTO hands (
                    hand_id, session_id, timestamp, hero_seat, hero_position,
                    hero_cards, table_info, result, net_profit, final_pot, showdown
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                hand.hand_id,
                session_id,
                hand.timestamp.isoformat(),
                hand.hero_seat,
                hand.hero_position,
                hero_cards_json,
                table_info_json,
                hand.result,
                hand.net_profit,
                hand.final_pot,
                1 if hand.showdown else 0
            ))
            
            # Insert actions
            for idx, action in enumerate(hand.actions):
                cursor.execute("""
                    INSERT INTO actions (
                        hand_id, seat_number, action_type, amount, phase, action_order
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    hand.hand_id,
                    action.seat_number,
                    action.action_type,
                    action.amount,
                    action.phase,
                    idx
                ))
            
            # Commit transaction
            self.conn.commit()
            
            logger.info(f"Hand {hand.hand_id} saved to session {session_id} "
                       f"(P/L: {hand.net_profit:.2f}, {len(hand.actions)} actions)")
            return True
            
        except sqlite3.Error as e:
            self.conn.rollback()
            logger.error(f"Database error saving hand {hand.hand_id}: {e}")
            return False
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Unexpected error saving hand {hand.hand_id}: {e}")
            return False
    
    def get_hand(self, hand_id: str) -> Optional[HandRecord]:
        """
        Retrieve a single hand by ID.
        
        Args:
            hand_id: Unique hand identifier
            
        Returns:
            HandRecord if found, None otherwise
        """
        try:
            cursor = self.conn.cursor()
            
            # Fetch hand record
            cursor.execute("""
                SELECT * FROM hands WHERE hand_id = ?
            """, (hand_id,))
            
            hand_row = cursor.fetchone()
            if not hand_row:
                logger.warning(f"Hand {hand_id} not found in database")
                return None
            
            # Fetch actions
            cursor.execute("""
                SELECT * FROM actions 
                WHERE hand_id = ? 
                ORDER BY action_order
            """, (hand_id,))
            
            action_rows = cursor.fetchall()
            
            # Reconstruct HandRecord
            return self._reconstruct_hand_record(hand_row, action_rows)
            
        except sqlite3.Error as e:
            logger.error(f"Database error retrieving hand {hand_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error retrieving hand {hand_id}: {e}")
            return None
    
    def get_hands_for_session(self, session_id: str) -> List[HandRecord]:
        """
        Get all hands for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of HandRecords (empty list if none found)
        """
        try:
            cursor = self.conn.cursor()
            
            # Fetch all hands for session
            cursor.execute("""
                SELECT * FROM hands 
                WHERE session_id = ? 
                ORDER BY timestamp
            """, (session_id,))
            
            hand_rows = cursor.fetchall()
            
            if not hand_rows:
                logger.debug(f"No hands found for session {session_id}")
                return []
            
            # Reconstruct each hand with its actions
            hands = []
            for hand_row in hand_rows:
                hand_id = hand_row['hand_id']
                
                # Fetch actions for this hand
                cursor.execute("""
                    SELECT * FROM actions 
                    WHERE hand_id = ? 
                    ORDER BY action_order
                """, (hand_id,))
                
                action_rows = cursor.fetchall()
                
                # Reconstruct and add to list
                hand = self._reconstruct_hand_record(hand_row, action_rows)
                if hand:
                    hands.append(hand)
            
            logger.debug(f"Retrieved {len(hands)} hands for session {session_id}")
            return hands
            
        except sqlite3.Error as e:
            logger.error(f"Database error retrieving hands for session {session_id}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error retrieving hands for session {session_id}: {e}")
            return []
    
    def create_session(self, session_id: str, start_time: datetime) -> bool:
        """
        Create a new session record.
        
        Args:
            session_id: Unique session identifier
            start_time: Session start timestamp
            
        Returns:
            True if created successfully, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                INSERT INTO sessions (session_id, start_time, is_active)
                VALUES (?, ?, 1)
            """, (session_id, start_time.isoformat()))
            
            self.conn.commit()
            
            logger.info(f"Session {session_id} created at {start_time.isoformat()}")
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Database error creating session {session_id}: {e}")
            return False
    
    def update_session(self, session_id: str, end_time: datetime, 
                      total_hands: int, total_profit: float) -> bool:
        """
        Update session with final statistics.
        
        Args:
            session_id: Session identifier
            end_time: Session end timestamp
            total_hands: Total hands played
            total_profit: Total profit/loss
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                UPDATE sessions 
                SET end_time = ?, is_active = 0, total_hands = ?, total_profit = ?
                WHERE session_id = ?
            """, (end_time.isoformat(), total_hands, total_profit, session_id))
            
            self.conn.commit()
            
            logger.info(f"Session {session_id} updated: {total_hands} hands, "
                       f"P/L: {total_profit:.2f}")
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Database error updating session {session_id}: {e}")
            return False
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session metadata.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict with session data if found, None otherwise
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                SELECT * FROM sessions WHERE session_id = ?
            """, (session_id,))
            
            row = cursor.fetchone()
            if not row:
                logger.warning(f"Session {session_id} not found")
                return None
            
            return dict(row)
            
        except sqlite3.Error as e:
            logger.error(f"Database error retrieving session {session_id}: {e}")
            return None
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """
        Get all sessions ordered by start time (most recent first).
        
        Returns:
            List of session dicts (empty list if none found)
        """
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                SELECT * FROM sessions 
                ORDER BY start_time DESC
            """)
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
        except sqlite3.Error as e:
            logger.error(f"Database error retrieving all sessions: {e}")
            return []
    
    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    # Serialization helpers
    
    def _serialize_cards(self, cards: List[Card]) -> str:
        """Convert Card list to JSON string."""
        return json.dumps([{"rank": c.rank, "suit": c.suit} for c in cards])
    
    def _deserialize_cards(self, json_str: str) -> List[Card]:
        """Convert JSON string to Card list."""
        data = json.loads(json_str)
        return [Card(rank=c["rank"], suit=c["suit"]) for c in data]
    
    def _serialize_table_info(self, table_info: TableInfo) -> str:
        """Convert TableInfo to JSON string."""
        return json.dumps({"sb": table_info.sb, "bb": table_info.bb})
    
    def _deserialize_table_info(self, json_str: str) -> TableInfo:
        """Convert JSON string to TableInfo."""
        data = json.loads(json_str)
        return TableInfo(sb=data["sb"], bb=data["bb"])
    
    def _reconstruct_hand_record(self, hand_row: sqlite3.Row, 
                                 action_rows: List[sqlite3.Row]) -> Optional[HandRecord]:
        """
        Reconstruct a HandRecord from database rows.
        
        Args:
            hand_row: Row from hands table
            action_rows: Rows from actions table
            
        Returns:
            HandRecord if successful, None otherwise
        """
        try:
            # Deserialize complex fields
            hero_cards = self._deserialize_cards(hand_row['hero_cards'])
            table_info = self._deserialize_table_info(hand_row['table_info'])
            
            # Reconstruct actions
            actions = []
            for action_row in action_rows:
                actions.append(Action(
                    seat_number=action_row['seat_number'],
                    action_type=action_row['action_type'],
                    amount=action_row['amount'],
                    phase=action_row['phase']
                ))
            
            # Create HandRecord
            return HandRecord(
                hand_id=hand_row['hand_id'],
                timestamp=datetime.fromisoformat(hand_row['timestamp']),
                table_info=table_info,
                hero_position=hand_row['hero_position'],
                hero_seat=hand_row['hero_seat'],
                hero_cards=hero_cards,
                actions=actions,
                result=hand_row['result'],
                net_profit=hand_row['net_profit'],
                final_pot=hand_row['final_pot'],
                showdown=bool(hand_row['showdown'])
            )
            
        except Exception as e:
            logger.error(f"Error reconstructing hand record: {e}")
            return None

