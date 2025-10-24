#!/usr/bin/env python3
"""
Parser test utilities and base classes for visual and unit testing.

Provides common functionality for testing parser components including:
- Visual testing with random video frames
- Interactive validation with keybinds
- Test result storage and aggregation
- Mock data generation
"""

import sys
import cv2
import numpy as np
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
src_path = str(project_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from capture.region_extractor import RegionExtractor
from capture.region_loader import RegionLoader

logger = logging.getLogger(__name__)


@dataclass
class ParserTestResult:
    """Result of a single parser test."""
    parser_name: str
    region_name: str
    video_file: str
    frame_number: int
    parsed_value: Any
    confidence: float
    preprocessing_method: str
    timestamp: str
    user_approved: Optional[bool] = None
    user_notes: str = ""
    debug_info: Optional[Dict[str, Any]] = None


@dataclass
class VisualTestSession:
    """Session data for visual testing."""
    session_id: str
    start_time: str
    end_time: Optional[str] = None
    total_tests: int = 0
    approved_tests: int = 0
    rejected_tests: int = 0
    skipped_tests: int = 0
    user_quit: bool = False


class VideoFrameExtractor:
    """Extract random frames from poker gameplay videos for testing."""
    
    def __init__(self, video_dir: Path = None):
        """Initialize with video directory."""
        self.video_dir = video_dir or Path("data/video")
        self.video_files = self._discover_video_files()
        
        logger.info(f"VideoFrameExtractor initialized with {len(self.video_files)} videos")
    
    def _discover_video_files(self) -> List[Path]:
        """Discover all video files in the video directory."""
        if not self.video_dir.exists():
            logger.warning(f"Video directory not found: {self.video_dir}")
            return []
        
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        video_files = []
        
        for file_path in self.video_dir.iterdir():
            if file_path.suffix.lower() in video_extensions:
                video_files.append(file_path)
        
        return sorted(video_files)
    
    def extract_random_frame(self, video_file: Path = None) -> Tuple[np.ndarray, int, Path]:
        """
        Extract a random frame from a video file.
        
        Args:
            video_file: Specific video file to use, or None for random selection
            
        Returns:
            Tuple of (frame_image, frame_number, video_file_path)
        """
        if not self.video_files:
            raise ValueError("No video files available for frame extraction")
        
        # Select video file
        if video_file is None:
            video_file = random.choice(self.video_files)
        elif video_file not in self.video_files:
            raise ValueError(f"Video file not found: {video_file}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_file}")
        
        try:
            # Get total frame count
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                raise ValueError(f"Invalid frame count for video: {video_file}")
            
            # Select random frame
            frame_number = random.randint(0, total_frames - 1)
            
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Could not read frame {frame_number} from {video_file}")
            
            logger.debug(f"Extracted frame {frame_number} from {video_file.name}")
            return frame, frame_number, video_file
            
        finally:
            cap.release()


# RegionTestExtractor removed - use RegionLoader from src.capture.region_loader instead


class VisualTestValidator:
    """Interactive visual testing with PyQt6 GUI."""
    
    def __init__(self, test_data_dir: Path = None, use_qt: bool = True):
        """Initialize with test data directory."""
        self.test_data_dir = test_data_dir or Path("data/tests")
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
        self.use_qt = use_qt
        
        # Key bindings (for OpenCV fallback)
        self.key_bindings = {
            ord('y'): True,   # Approve
            ord('n'): False,  # Reject
            ord('s'): None,   # Skip
            ord('q'): 'quit', # Quit
            27: 'quit'        # ESC to quit
        }
        
        # Track test progress for PyQt6
        self.current_test_num = 0
        self.total_tests = 0
        
        logger.info(f"VisualTestValidator initialized (PyQt6: {use_qt})")
    
    def validate_parser_result(self, region_image: np.ndarray, parser_name: str, 
                             region_name: str, parsed_value: Any, confidence: float,
                             preprocessing_method: str = "unknown") -> Optional[bool]:
        """
        Display parser result for manual validation.
        
        Args:
            region_image: Extracted region image
            parser_name: Name of parser that processed the image
            region_name: Name of region that was processed
            parsed_value: Value parsed from the image
            confidence: Confidence score from parser
            preprocessing_method: OCR preprocessing method used
            
        Returns:
            True if approved, False if rejected, None if skipped, 'quit' if user wants to quit
        """
        # Store parser name for help context
        self.current_parser_name = parser_name
        
        if self.use_qt:
            # Use PyQt6 interface
            try:
                from test.parser.visual_test_qt import show_visual_test
                self.current_test_num += 1
                return show_visual_test(
                    region_image, parser_name, region_name, parsed_value,
                    confidence, preprocessing_method,
                    self.current_test_num, self.total_tests
                )
            except Exception as e:
                logger.error(f"PyQt6 interface failed, falling back to OpenCV: {e}")
                self.use_qt = False
        
        # Fallback to OpenCV interface
        window_name = f"Parser Test - {parser_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Prepare display image
        display_image = self._prepare_display_image(
            region_image, parser_name, region_name, parsed_value, confidence, preprocessing_method
        )
        
        # Show image
        cv2.imshow(window_name, display_image)
        
        # Wait for key press
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key in self.key_bindings:
                result = self.key_bindings[key]
                cv2.destroyWindow(window_name)
                return result
            
            # Show help
            if key == ord('h'):
                self._show_help(parser_name)
        
        cv2.destroyWindow(window_name)
        return None
    
    def _prepare_display_image(self, region_image: np.ndarray, parser_name: str,
                             region_name: str, parsed_value: Any, confidence: float,
                             preprocessing_method: str) -> np.ndarray:
        """Prepare image for display with overlay text - improved layout."""
        # Get original dimensions
        orig_height, orig_width = region_image.shape[:2]
        
        # Scale up the region image for better visibility (minimum 400px wide)
        scale_factor = max(1.0, 400 / orig_width)
        if scale_factor > 1.0:
            new_width = int(orig_width * scale_factor)
            new_height = int(orig_height * scale_factor)
            region_image = cv2.resize(region_image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        
        height, width = region_image.shape[:2]
        
        # Create larger canvas with more space for info
        info_panel_height = 250
        canvas_width = max(width, 800)  # Minimum 800px wide
        canvas_height = height + info_panel_height
        
        # Create canvas with dark background
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 30
        
        # Center the region image horizontally
        x_offset = (canvas_width - width) // 2
        canvas[:height, x_offset:x_offset+width] = region_image
        
        # Draw separator line
        cv2.line(canvas, (0, height), (canvas_width, height), (100, 100, 100), 2)
        
        # Info panel starts below image
        y_start = height + 20
        
        # Use larger, clearer fonts
        font = cv2.FONT_HERSHEY_DUPLEX
        title_font_scale = 0.9
        info_font_scale = 0.7
        key_font_scale = 0.6
        
        # Colors
        title_color = (100, 255, 100)  # Green for titles
        value_color = (255, 255, 255)  # White for values
        key_color = (150, 200, 255)    # Light blue for keys
        
        # Title
        cv2.putText(canvas, f"PARSER: {parser_name.upper()}", 
                   (20, y_start), font, title_font_scale, title_color, 2)
        
        # Region name
        cv2.putText(canvas, f"Region: {region_name}", 
                   (20, y_start + 35), font, info_font_scale, value_color, 1)
        
        # Parsed value - highlight this prominently
        if parsed_value is None or parsed_value == "" or parsed_value == "None":
            value_str = "[EMPTY / NO RESULT]"
            result_color = (0, 150, 255)  # Orange for empty results
        else:
            value_str = str(parsed_value)
            result_color = value_color  # White for normal results
            
        cv2.putText(canvas, "RESULT:", 
                   (20, y_start + 75), font, title_font_scale, title_color, 2)
        cv2.putText(canvas, f"  {value_str}", 
                   (20, y_start + 110), font, title_font_scale, result_color, 2)
        
        # Confidence with color coding
        conf_color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255) if confidence > 0.5 else (0, 100, 255)
        cv2.putText(canvas, f"Confidence: {confidence:.1%}", 
                   (20, y_start + 145), font, info_font_scale, conf_color, 1)
        
        # Method
        cv2.putText(canvas, f"Method: {preprocessing_method}", 
                   (20, y_start + 175), font, key_font_scale, (200, 200, 200), 1)
        
        # Key bindings - make them stand out
        cv2.rectangle(canvas, (0, canvas_height - 40), (canvas_width, canvas_height), (50, 50, 50), -1)
        
        # Customize instructions based on parser type
        keys_text = "[Y] Approve  |  [N] Reject  |  [S] Skip  |  [Q] Quit  |  [H] Help"
            
        text_size = cv2.getTextSize(keys_text, font, key_font_scale, 1)[0]
        text_x = (canvas_width - text_size[0]) // 2
        cv2.putText(canvas, keys_text, 
                   (text_x, canvas_height - 15), font, key_font_scale, key_color, 1)
        
        return canvas
    
    def _show_help(self, parser_name: str = "general"):
        """Show help information."""
        # Create help window with better design
        help_canvas = np.ones((450, 700, 3), dtype=np.uint8) * 30
        font = cv2.FONT_HERSHEY_DUPLEX
        
        # Title
        title = "VISUAL TEST HELP"
        cv2.putText(help_canvas, title, (50, 50), font, 1.0, (100, 255, 100), 2)
        cv2.line(help_canvas, (50, 65), (650, 65), (100, 100, 100), 2)
        
        # Key bindings with clear descriptions - standard for all parsers
        keys = [
            ("Y", "Approve - Parser result is CORRECT", (0, 255, 0)),
            ("N", "Reject - Parser result is WRONG", (0, 100, 255)),
            ("S", "Skip - Can't determine / unclear", (0, 255, 255)),
            ("Q / ESC", "Quit - End testing session", (255, 100, 100)),
            ("H", "Help - Show this help screen", (150, 200, 255)),
        ]
        
        y_pos = 120
        for key, desc, color in keys:
            # Draw key box
            cv2.rectangle(help_canvas, (50, y_pos - 25), (120, y_pos + 5), color, 2)
            cv2.putText(help_canvas, key, (60, y_pos - 5), font, 0.7, color, 2)
            
            # Draw description
            cv2.putText(help_canvas, desc, (140, y_pos - 5), font, 0.6, (255, 255, 255), 1)
            
            y_pos += 55
        
        # Footer
        cv2.line(help_canvas, (50, 390), (650, 390), (100, 100, 100), 2)
        cv2.putText(help_canvas, "Press any key to continue...", 
                   (200, 425), font, 0.6, (150, 200, 255), 1)
        
        cv2.imshow("Help", help_canvas)
        cv2.waitKey(0)
        cv2.destroyWindow("Help")


class ParserTestResultManager:
    """Manage test results storage and aggregation."""
    
    def __init__(self, test_data_dir: Path = None):
        """Initialize with test data directory."""
        self.test_data_dir = test_data_dir or Path("data/tests")
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("ParserTestResultManager initialized")
    
    def save_test_result(self, result: ParserTestResult) -> None:
        """Save a single test result to parser-specific file in passed/failed directory structure."""
        # Determine target directory based on user_approved
        target_dir = self.test_data_dir / ("passed" if result.user_approved else "failed")
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Use parser-specific filename that accumulates all results
        parser_file = target_dir / f"{result.parser_name}_results.json"
        
        # Load existing results if file exists
        existing_results = []
        if parser_file.exists():
            try:
                with open(parser_file, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                existing_results = []
        
        # Add new result
        result_dict = asdict(result)
        existing_results.append(result_dict)
        
        # Save updated results
        with open(parser_file, 'w', encoding='utf-8') as f:
            json.dump(existing_results, f, indent=2)
        
        logger.debug(f"Saved test result to {parser_file} (total results: {len(existing_results)})")
    
    def save_session_results(self, session: VisualTestSession, results: List[ParserTestResult]) -> None:
        """
        Save a complete testing session.
        
        Saves to parser-specific files that accumulate all results for that parser,
        making it easy to track accuracy over time.
        """
        if not results:
            logger.info("No results to save")
            return
        
        # Group results by parser and approval status
        results_by_parser_and_status = {}
        for result in results:
            parser_name = result.parser_name
            status = "passed" if result.user_approved else "failed"
            key = (parser_name, status)
            
            if key not in results_by_parser_and_status:
                results_by_parser_and_status[key] = []
            results_by_parser_and_status[key].append(result)
        
        # Save/append to parser-specific files in appropriate directories
        for (parser_name, status), parser_results in results_by_parser_and_status.items():
            target_dir = self.test_data_dir / status
            target_dir.mkdir(parents=True, exist_ok=True)
            parser_file = target_dir / f"{parser_name}_results.json"
            
            # Load existing results if file exists
            existing_results = []
            if parser_file.exists():
                try:
                    with open(parser_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                        if isinstance(existing_data, list):
                            existing_results = existing_data
                except Exception as e:
                    logger.warning(f"Could not load existing results for {parser_name} ({status}): {e}")
            
            # Append new results
            all_results = existing_results + [asdict(r) for r in parser_results]
            
            # Save back to file
            with open(parser_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2)
            
            logger.info(f"Saved {len(parser_results)} {status} results for {parser_name} (total: {len(all_results)})")
        
        # Also save session metadata to both directories for reference
        session_dict = asdict(session)
        session_dict['num_results'] = len(results)
        session_dict['parsers_tested'] = list(set(r.parser_name for r in results))
        
        for status in ["passed", "failed"]:
            target_dir = self.test_data_dir / status
            target_dir.mkdir(parents=True, exist_ok=True)
            session_file = target_dir / f"session_{session.session_id}.json"
            
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_dict, f, indent=2)
        
        logger.info(f"Saved session {session.session_id} summary")
    
    def load_test_results(self, parser_name: str = None, status: str = None) -> List[ParserTestResult]:
        """
        Load test results, optionally filtered by parser name and/or status.
        
        Args:
            parser_name: Filter by specific parser (e.g., 'card', 'money')
            status: Filter by status ('passed', 'failed', or None for both)
        """
        results = []
        
        # Determine which directories to search
        if status:
            search_dirs = [self.test_data_dir / status]
        else:
            search_dirs = [
                self.test_data_dir / "passed",
                self.test_data_dir / "failed"
            ]
        
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
                
            if parser_name:
                # Load from parser-specific file in specific directory
                parser_file = search_dir / f"{parser_name}_results.json"
                if parser_file.exists():
                    try:
                        with open(parser_file, 'r', encoding='utf-8') as f:
                            results_data = json.load(f)
                            if isinstance(results_data, list):
                                for result_dict in results_data:
                                    try:
                                        result = ParserTestResult(**result_dict)
                                        results.append(result)
                                    except Exception as e:
                                        logger.warning(f"Could not parse result: {e}")
                    except Exception as e:
                        logger.warning(f"Could not load results for {parser_name} from {search_dir}: {e}")
            else:
                # Load all parser result files from directory
                for file_path in search_dir.glob("*_results.json"):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            results_data = json.load(f)
                            if isinstance(results_data, list):
                                for result_dict in results_data:
                                    try:
                                        result = ParserTestResult(**result_dict)
                                        results.append(result)
                                    except Exception as e:
                                        logger.warning(f"Could not parse result: {e}")
                    except Exception as e:
                        logger.warning(f"Could not load results from {file_path}: {e}")
        
        logger.info(f"Loaded {len(results)} test results")
        return results
    
    def get_parser_statistics(self, parser_name: str) -> Dict[str, Any]:
        """Get statistics for a specific parser."""
        results = self.load_test_results(parser_name)
        
        if not results:
            return {
                'total_tests': 0,
                'approved': 0,
                'rejected': 0,
                'skipped': 0,
                'approval_rate': 0.0,
                'avg_confidence': 0.0
            }
        
        total_tests = len(results)
        approved = sum(1 for r in results if r.user_approved is True)
        rejected = sum(1 for r in results if r.user_approved is False)
        skipped = sum(1 for r in results if r.user_approved is None)
        
        # Calculate approval rate (excluding skipped)
        validated_tests = approved + rejected
        approval_rate = approved / validated_tests if validated_tests > 0 else 0.0
        
        # Calculate average confidence
        avg_confidence = sum(r.confidence for r in results) / total_tests
        
        return {
            'total_tests': total_tests,
            'approved': approved,
            'rejected': rejected,
            'skipped': skipped,
            'approval_rate': approval_rate,
            'avg_confidence': avg_confidence
        }


class ParserTestBase:
    """Base class for parser tests with common functionality."""
    
    def __init__(self):
        """Initialize test base."""
        self.frame_extractor = VideoFrameExtractor()
        self.region_loader = RegionLoader()
        self.regions = self.region_loader.load_regions()
        self.validator = VisualTestValidator()
        self.result_manager = ParserTestResultManager()
        
        logger.info("ParserTestBase initialized")
    
    def create_test_session(self) -> VisualTestSession:
        """Create a new visual test session."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        return VisualTestSession(
            session_id=session_id,
            start_time=datetime.now().isoformat()
        )
    
    def run_visual_test(self, parser_instance, parser_name: str, region_name: str,
                       num_tests: int = 10) -> VisualTestSession:
        """
        Run visual tests for a parser.
        
        Args:
            parser_instance: Parser instance to test
            parser_name: Name of the parser
            region_name: Region to test
            num_tests: Number of tests to run
            
        Returns:
            VisualTestSession with results
        """
        session = self.create_test_session()
        results = []
        
        # Set total tests for progress tracking
        self.validator.total_tests = num_tests
        self.validator.current_test_num = 0
        
        logger.info(f"Starting visual test session for {parser_name} on region {region_name}")
        
        try:
            for test_num in range(num_tests):
                logger.info(f"Running test {test_num + 1}/{num_tests}")
                
                # Extract random frame and region
                frame, frame_number, video_file = self.frame_extractor.extract_random_frame()
                
                # Extract region using RegionLoader
                if region_name not in self.regions:
                    logger.warning(f"Region '{region_name}' not found in loaded regions")
                    continue
                
                region_model = self.regions[region_name]
                x, y, w, h = region_model.x, region_model.y, region_model.width, region_model.height
                
                # Check bounds
                if (x + w > frame.shape[1] or y + h > frame.shape[0] or x < 0 or y < 0):
                    logger.warning(f"Region '{region_name}' bounds exceed frame dimensions")
                    continue
                
                region_image = frame[y:y+h, x:x+w]
                logger.debug(f"Extracted region '{region_name}': {w}x{h} at ({x}, {y})")
                
                if region_image is None or region_image.size == 0:
                    logger.warning(f"Could not extract region {region_name}, skipping test")
                    continue
                
                # Parse the region
                parsed_result = self._parse_region(parser_instance, region_image)
                
                # Extract result data (even if None - user needs to validate empty results too)
                if parsed_result is None:
                    parsed_value = None
                    confidence = 0.0
                    preprocessing_method = "none"
                    logger.debug(f"Parser returned no result - showing for validation")
                else:
                    parsed_value, confidence, preprocessing_method = self._extract_result_data(parsed_result)
                
                # Get debug information if available
                debug_info = None
                if hasattr(self, '_get_debug_info'):
                    debug_info = self._get_debug_info(parser_instance, region_image, parsed_result)
                
                # Create test result
                test_result = ParserTestResult(
                    parser_name=parser_name,
                    region_name=region_name,
                    video_file=video_file.name,
                    frame_number=frame_number,
                    parsed_value=parsed_value,
                    confidence=confidence,
                    preprocessing_method=preprocessing_method,
                    timestamp=datetime.now().isoformat(),
                    debug_info=debug_info
                )
                
                # Validate visually
                validation_result = self.validator.validate_parser_result(
                    region_image, parser_name, region_name, parsed_value, 
                    confidence, preprocessing_method
                )
                
                if validation_result == 'quit':
                    logger.info("User requested to quit testing session")
                    session.user_quit = True
                    break
                
                # Update test result
                test_result.user_approved = validation_result
                results.append(test_result)
                
                # Update session stats
                session.total_tests += 1
                if validation_result is True:
                    session.approved_tests += 1
                elif validation_result is False:
                    session.rejected_tests += 1
                else:
                    session.skipped_tests += 1
                
                logger.info(f"Test {test_num + 1} completed: {validation_result}")
        
        except KeyboardInterrupt:
            logger.info("Testing interrupted by user")
        
        finally:
            # Finalize session
            session.end_time = datetime.now().isoformat()
            
            # Close PyQt6 window if it's open
            if self.validator.use_qt:
                try:
                    from test.parser.visual_test_qt import close_visual_test_window
                    close_visual_test_window()
                except Exception:
                    pass
            
            # Save results
            self.result_manager.save_session_results(session, results)
            
            # Print summary
            self._print_session_summary(session)
        
        return session
    
    def _parse_region(self, parser_instance, region_image: np.ndarray) -> Any:
        """Parse a region image using the parser instance."""
        # This method should be overridden by specific parser test classes
        # to call the appropriate parsing method
        raise NotImplementedError("Subclasses must implement _parse_region")
    
    def _extract_result_data(self, parsed_result: Any) -> Tuple[Any, float, str]:
        """Extract value, confidence, and preprocessing method from parser result."""
        # This method should be overridden by specific parser test classes
        # to extract the appropriate data from the parser result
        raise NotImplementedError("Subclasses must implement _extract_result_data")
    
    def _print_session_summary(self, session: VisualTestSession):
        """Print a summary of the testing session."""
        print("\n" + "="*60)
        print(f"Visual Test Session Summary: {session.session_id}")
        print("="*60)
        print(f"Total Tests: {session.total_tests}")
        print(f"Approved: {session.approved_tests}")
        print(f"Rejected: {session.rejected_tests}")
        print(f"Skipped: {session.skipped_tests}")
        
        if session.total_tests > 0:
            approval_rate = session.approved_tests / (session.approved_tests + session.rejected_tests)
            print(f"Approval Rate: {approval_rate:.1%}")
        
        print(f"Start Time: {session.start_time}")
        print(f"End Time: {session.end_time}")
        print("="*60)
