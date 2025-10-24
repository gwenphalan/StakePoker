#!/usr/bin/env python3
"""
Test utilities for capture module testing.

Provides common fixtures, helpers, and utilities for testing capture functionality.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any
import json


class CaptureTestUtils:
    """Utility class for capture module tests."""
    
    @staticmethod
    def create_test_frame(width: int = 1920, height: int = 1080, 
                         channels: int = 3) -> np.ndarray:
        """Create a test frame with specified dimensions."""
        return np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)
    
    @staticmethod
    def create_patterned_frame(width: int = 1920, height: int = 1080) -> np.ndarray:
        """Create a frame with known patterns for testing."""
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some test patterns
        # Red rectangle in top-left
        frame[50:100, 50:150] = [255, 0, 0]
        
        # Green rectangle in center
        frame[400:450, 800:900] = [0, 255, 0]
        
        # Blue rectangle in bottom-right
        frame[900:950, 1600:1700] = [0, 0, 255]
        
        return frame
    
    @staticmethod
    def create_sample_regions() -> Dict[str, Dict[str, int]]:
        """Create sample region definitions for testing."""
        return {
            "player_1_name": {"x": 100, "y": 200, "width": 150, "height": 30},
            "player_1_stack": {"x": 100, "y": 250, "width": 100, "height": 25},
            "player_2_name": {"x": 300, "y": 200, "width": 150, "height": 30},
            "player_2_stack": {"x": 300, "y": 250, "width": 100, "height": 25},
            "pot": {"x": 400, "y": 300, "width": 120, "height": 40},
            "table_info": {"x": 50, "y": 50, "width": 200, "height": 30},
            "dealer_button": {"x": 200, "y": 150, "width": 20, "height": 20}
        }
    
    @staticmethod
    def save_test_regions(regions: Dict[str, Dict[str, int]], 
                         file_path: Path) -> None:
        """Save test regions to a JSON file."""
        with open(file_path, 'w') as f:
            json.dump(regions, f, indent=2)
    
    @staticmethod
    def compare_images(img1: np.ndarray, img2: np.ndarray, 
                      threshold: float = 0.95) -> bool:
        """Compare two images using structural similarity."""
        try:
            from skimage.metrics import structural_similarity as ssim
            
            # Convert to grayscale if needed
            if len(img1.shape) == 3:
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            if len(img2.shape) == 3:
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            # Ensure same size
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
            score = ssim(img1, img2)
            return score >= threshold
            
        except ImportError:
            # Fallback to simple pixel comparison
            if img1.shape != img2.shape:
                return False
            return np.allclose(img1, img2, atol=10)
    
    @staticmethod
    def create_monitor_bounds(left: int = 0, top: int = 0, 
                            width: int = 1920, height: int = 1080) -> Dict[str, int]:
        """Create monitor bounds dictionary."""
        return {
            'left': left,
            'top': top,
            'width': width,
            'height': height
        }


@pytest.fixture
def capture_test_utils():
    """Provide CaptureTestUtils instance."""
    return CaptureTestUtils()


@pytest.fixture
def sample_frame():
    """Provide a sample test frame."""
    return CaptureTestUtils.create_test_frame()


@pytest.fixture
def patterned_frame():
    """Provide a frame with known patterns."""
    return CaptureTestUtils.create_patterned_frame()


@pytest.fixture
def sample_regions():
    """Provide sample region definitions."""
    return CaptureTestUtils.create_sample_regions()


@pytest.fixture
def test_regions_file(tmp_path, sample_regions):
    """Provide a temporary regions file."""
    regions_file = tmp_path / "test_regions.json"
    CaptureTestUtils.save_test_regions(sample_regions, regions_file)
    return regions_file


@pytest.fixture
def mock_monitor_bounds():
    """Provide mock monitor bounds."""
    return CaptureTestUtils.create_monitor_bounds()


@pytest.fixture
def mock_monitor_bounds_secondary():
    """Provide mock secondary monitor bounds."""
    return CaptureTestUtils.create_monitor_bounds(
        left=1920, top=0, width=2560, height=1440
    )


class VisualTestValidator:
    """Helper for visual testing validation."""
    
    def __init__(self, test_data_dir: Path):
        self.test_data_dir = test_data_dir
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
    
    def save_reference_image(self, image: np.ndarray, name: str) -> Path:
        """Save a reference image for comparison."""
        ref_path = self.test_data_dir / f"{name}_reference.png"
        cv2.imwrite(str(ref_path), image)
        return ref_path
    
    def compare_with_reference(self, image: np.ndarray, name: str, 
                              threshold: float = 0.95) -> bool:
        """Compare image with saved reference."""
        ref_path = self.test_data_dir / f"{name}_reference.png"
        
        if not ref_path.exists():
            # Save as reference if doesn't exist
            self.save_reference_image(image, name)
            return True
        
        reference = cv2.imread(str(ref_path))
        return CaptureTestUtils.compare_images(image, reference, threshold)
    
    def create_comparison_image(self, original: np.ndarray, 
                              processed: np.ndarray) -> np.ndarray:
        """Create side-by-side comparison image."""
        # Ensure same height
        if original.shape[0] != processed.shape[0]:
            processed = cv2.resize(processed, 
                                 (processed.shape[1], original.shape[0]))
        
        # Concatenate horizontally
        comparison = np.hstack([original, processed])
        
        # Add separator line
        separator_x = original.shape[1]
        cv2.line(comparison, (separator_x, 0), 
                (separator_x, comparison.shape[0]), (255, 255, 255), 2)
        
        return comparison


@pytest.fixture
def visual_validator(tmp_path):
    """Provide VisualTestValidator instance."""
    return VisualTestValidator(tmp_path / "visual_tests")


# Performance testing utilities
class PerformanceTracker:
    """Track performance metrics during tests."""
    
    def __init__(self):
        self.metrics = {}
    
    def start_timer(self, name: str):
        """Start timing an operation."""
        import time
        self.metrics[name] = {'start': time.time()}
    
    def end_timer(self, name: str):
        """End timing an operation."""
        import time
        if name in self.metrics:
            self.metrics[name]['end'] = time.time()
            self.metrics[name]['duration'] = (
                self.metrics[name]['end'] - self.metrics[name]['start']
            )
    
    def get_duration(self, name: str) -> float:
        """Get duration of a timed operation."""
        return self.metrics.get(name, {}).get('duration', 0.0)
    
    def assert_faster_than(self, name: str, max_duration: float):
        """Assert that an operation completed within time limit."""
        duration = self.get_duration(name)
        assert duration < max_duration, (
            f"Operation '{name}' took {duration:.3f}s, "
            f"expected < {max_duration:.3f}s"
        )


@pytest.fixture
def performance_tracker():
    """Provide PerformanceTracker instance."""
    return PerformanceTracker()


if __name__ == '__main__':
    # Test the utilities
    utils = CaptureTestUtils()
    
    # Test frame creation
    frame = utils.create_test_frame()
    print(f"Test frame shape: {frame.shape}")
    
    # Test patterned frame
    pattern_frame = utils.create_patterned_frame()
    print(f"Patterned frame shape: {pattern_frame.shape}")
    
    # Test regions
    regions = utils.create_sample_regions()
    print(f"Sample regions: {list(regions.keys())}")
    
    print("All utilities working correctly!")
