#!/usr/bin/env python3
"""
HandExporter for CSV export of hand history data.

Provides functionality to export session data and hand records to CSV format
for external analysis in spreadsheet applications.
"""

import csv
import logging
from pathlib import Path
from typing import List, Dict, Any

from src.models.hand_record import HandRecord, Action
from src.models.card import Card
from src.history.hand_storage import HandStorage
from src.config.settings import Settings

logger = logging.getLogger(__name__)


class HandExporter:
    """CSV export functionality for hand history."""
    
    def __init__(self, hand_storage: HandStorage):
        """
        Initialize hand exporter.
        
        Args:
            hand_storage: HandStorage instance for data retrieval
        """
        self.hand_storage = hand_storage
        self.settings = Settings()
        
        logger.info("HandExporter initialized")
    
    def export_session_to_csv(self, session_id: str, output_path: str) -> bool:
        """
        Export a single session to CSV format.
        
        Args:
            session_id: Session to export
            output_path: Path to output CSV file
            
        Returns:
            True if export successful, False otherwise
            
        CSV Columns:
            hand_id, timestamp, session_id, hero_seat, hero_position,
            hero_card_1, hero_card_2, sb, bb, result, net_profit, final_pot,
            showdown, num_actions, action_sequence
        """
        # Fetch hands for session
        hands = self.hand_storage.get_hands_for_session(session_id)
        
        if not hands:
            logger.warning(f"No hands found for session {session_id}")
            return False
        
        try:
            # Ensure output directory exists
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Open CSV writer
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'hand_id', 'timestamp', 'session_id', 'hero_seat', 'hero_position',
                    'hero_card_1', 'hero_card_2', 'sb', 'bb', 'result', 'net_profit',
                    'final_pot', 'showdown', 'num_actions', 'action_sequence'
                ])
                
                writer.writeheader()
                
                # Write each hand as a row
                for hand in hands:
                    row = self._flatten_hand_to_row(hand, session_id)
                    writer.writerow(row)
            
            logger.info(f"Exported {len(hands)} hands from session {session_id} to {output_path}")
            return True
            
        except IOError as e:
            logger.error(f"IO error exporting session {session_id} to {output_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error exporting session {session_id}: {e}")
            return False
    
    def export_all_hands_to_csv(self, output_path: str) -> bool:
        """
        Export all hands from all sessions to CSV.
        
        Args:
            output_path: Path to output CSV file
            
        Returns:
            True if export successful, False otherwise
        """
        # Fetch all sessions
        sessions = self.hand_storage.get_all_sessions()
        
        if not sessions:
            logger.warning("No sessions found in database")
            return False
        
        try:
            # Ensure output directory exists
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Open CSV writer
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'hand_id', 'timestamp', 'session_id', 'hero_seat', 'hero_position',
                    'hero_card_1', 'hero_card_2', 'sb', 'bb', 'result', 'net_profit',
                    'final_pot', 'showdown', 'num_actions', 'action_sequence'
                ])
                
                writer.writeheader()
                
                total_hands = 0
                
                # Write hands from each session
                for session in sessions:
                    session_id = session['session_id']
                    hands = self.hand_storage.get_hands_for_session(session_id)
                    
                    for hand in hands:
                        row = self._flatten_hand_to_row(hand, session_id)
                        writer.writerow(row)
                        total_hands += 1
            
            logger.info(f"Exported {total_hands} hands from {len(sessions)} sessions to {output_path}")
            return True
            
        except IOError as e:
            logger.error(f"IO error exporting all hands to {output_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error exporting all hands: {e}")
            return False
    
    def export_session_summary_to_csv(self, output_path: str) -> bool:
        """
        Export session-level summary statistics to CSV.
        
        Args:
            output_path: Path to output CSV file
            
        Returns:
            True if export successful, False otherwise
            
        CSV Columns:
            session_id, start_time, end_time, duration_minutes, total_hands,
            total_profit, avg_profit_per_hand, win_rate, won_hands, lost_hands,
            folded_hands
        """
        # Fetch all sessions
        sessions = self.hand_storage.get_all_sessions()
        
        if not sessions:
            logger.warning("No sessions found in database")
            return False
        
        try:
            # Ensure output directory exists
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Open CSV writer
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'session_id', 'start_time', 'end_time', 'duration_minutes',
                    'total_hands', 'total_profit', 'avg_profit_per_hand',
                    'win_rate', 'won_hands', 'lost_hands', 'folded_hands'
                ])
                
                writer.writeheader()
                
                # Calculate and write stats for each session
                for session in sessions:
                    session_id = session['session_id']
                    
                    # Get hands for this session to calculate stats
                    hands = self.hand_storage.get_hands_for_session(session_id)
                    
                    if not hands:
                        continue
                    
                    # Calculate statistics
                    total_hands = len(hands)
                    total_profit = sum(hand.net_profit for hand in hands)
                    avg_profit = total_profit / total_hands if total_hands > 0 else 0.0
                    
                    won_hands = sum(1 for hand in hands if hand.result == 'won')
                    lost_hands = sum(1 for hand in hands if hand.result == 'lost')
                    folded_hands = sum(1 for hand in hands if hand.result == 'folded')
                    win_rate = (won_hands / total_hands * 100) if total_hands > 0 else 0.0
                    
                    # Calculate duration
                    from datetime import datetime
                    start_time = datetime.fromisoformat(session['start_time'])
                    end_time = (
                        datetime.fromisoformat(session['end_time'])
                        if session['end_time'] else datetime.now()
                    )
                    duration = (end_time - start_time).total_seconds() / 60
                    
                    row = {
                        'session_id': session_id,
                        'start_time': session['start_time'],
                        'end_time': session['end_time'] or 'Active',
                        'duration_minutes': round(duration, 2),
                        'total_hands': total_hands,
                        'total_profit': round(total_profit, 2),
                        'avg_profit_per_hand': round(avg_profit, 2),
                        'win_rate': round(win_rate, 2),
                        'won_hands': won_hands,
                        'lost_hands': lost_hands,
                        'folded_hands': folded_hands
                    }
                    
                    writer.writerow(row)
            
            logger.info(f"Exported summary for {len(sessions)} sessions to {output_path}")
            return True
            
        except IOError as e:
            logger.error(f"IO error exporting session summary to {output_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error exporting session summary: {e}")
            return False
    
    def _flatten_hand_to_row(self, hand: HandRecord, session_id: str) -> Dict[str, Any]:
        """
        Convert HandRecord to flat dictionary for CSV row.
        
        Args:
            hand: HandRecord to flatten
            session_id: Session ID to include in row
            
        Returns:
            Dict with flattened hand data
        """
        # Format hero cards
        hero_card_1 = self._format_card(hand.hero_cards[0]) if len(hand.hero_cards) > 0 else ''
        hero_card_2 = self._format_card(hand.hero_cards[1]) if len(hand.hero_cards) > 1 else ''
        
        # Format action sequence
        action_sequence = self._format_action_sequence(hand.actions)
        
        return {
            'hand_id': hand.hand_id,
            'timestamp': hand.timestamp.isoformat(),
            'session_id': session_id,
            'hero_seat': hand.hero_seat,
            'hero_position': hand.hero_position,
            'hero_card_1': hero_card_1,
            'hero_card_2': hero_card_2,
            'sb': hand.table_info.sb,
            'bb': hand.table_info.bb,
            'result': hand.result or '',
            'net_profit': round(hand.net_profit, 2),
            'final_pot': round(hand.final_pot, 2),
            'showdown': 'Yes' if hand.showdown else 'No',
            'num_actions': len(hand.actions),
            'action_sequence': action_sequence
        }
    
    def _format_card(self, card: Card) -> str:
        """
        Convert Card to string like 'Ah' or 'Kd'.
        
        Args:
            card: Card to format
            
        Returns:
            String representation (e.g., 'Ah', 'Kd')
        """
        # Map suit names to single character
        suit_map = {
            'hearts': 'h',
            'diamonds': 'd',
            'clubs': 'c',
            'spades': 's'
        }
        
        suit_char = suit_map.get(card.suit, card.suit[0])
        return f"{card.rank}{suit_char}"
    
    def _format_action_sequence(self, actions: List[Action]) -> str:
        """
        Convert actions to compact string: '1:raise(5.0),2:call,3:fold'.
        
        Args:
            actions: List of actions
            
        Returns:
            Compact string representation
        """
        if not actions:
            return ''
        
        action_strs = []
        for action in actions:
            if action.amount is not None:
                action_str = f"{action.seat_number}:{action.action_type}({action.amount:.2f})"
            else:
                action_str = f"{action.seat_number}:{action.action_type}"
            action_strs.append(action_str)
        
        return ','.join(action_strs)

