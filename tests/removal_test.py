#!/usr/bin/env python3
"""
Background removal test script for poker game images.

Tests all background subtraction methods on debug frames using the reference
poker_game_background.png. Saves results to tests/data/images/ for analysis.

Usage:
    python tests/removal_test.py
"""

import cv2
import numpy as np
import os
import logging
from pathlib import Path
from typing import List, Tuple, Dict
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.parser.background_subtractor import BackgroundSubtractor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BackgroundRemovalTester:
    """Test all background removal methods on debug frames."""
    
    def __init__(self):
        """Initialize tester with paths and subtractor."""
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data" / "photo"
        self.output_dir = self.project_root / "tests" / "data" / "images"
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize background subtractor
        self.subtractor = BackgroundSubtractor()
        
        # Define test images
        self.background_image_path = self.data_dir / "poker_game_background.png"
        self.debug_frames = [
            "debug_frame_1.png",
            "debug_frame_2.png", 
            "debug_frame_3.png",
            "debug_frame_4.png"
        ]
        
        # Define methods to test
        self.methods = [
            "transparency_simulation",
            "alpha_extraction",
            "simple_difference",
            "statistical_subtraction", 
            "correlation_detection",
            "hsv_analysis",
            "multi_method_subtraction"
        ]
        
        logger.info(f"Initialized tester with {len(self.debug_frames)} debug frames and {len(self.methods)} methods")
    
    def load_images(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Load background image and all debug frames.
        
        Returns:
            Tuple of (background_image, list_of_debug_frames)
        """
        logger.info("Loading images...")
        
        # Load background image
        if not self.background_image_path.exists():
            raise FileNotFoundError(f"Background image not found: {self.background_image_path}")
        
        background = cv2.imread(str(self.background_image_path))
        if background is None:
            raise ValueError(f"Failed to load background image: {self.background_image_path}")
        
        logger.info(f"Loaded background image: {background.shape}")
        
        # Load debug frames
        debug_frames = []
        for frame_name in self.debug_frames:
            frame_path = self.data_dir / frame_name
            if not frame_path.exists():
                logger.warning(f"Debug frame not found: {frame_path}")
                continue
            
            frame = cv2.imread(str(frame_path))
            if frame is None:
                logger.warning(f"Failed to load debug frame: {frame_path}")
                continue
            
            debug_frames.append(frame)
            logger.info(f"Loaded debug frame: {frame_name} - {frame.shape}")
        
        if not debug_frames:
            raise ValueError("No debug frames could be loaded")
        
        return background, debug_frames
    
    def test_all_methods(self) -> None:
        """Test all background removal methods on all debug frames."""
        logger.info("Starting background removal tests...")
        
        try:
            # Load images
            background, debug_frames = self.load_images()
            
            # Test each debug frame
            for i, debug_frame in enumerate(debug_frames, 1):
                logger.info(f"\n=== Testing Debug Frame {i} ===")
                
                # Ensure frames are same size
                if debug_frame.shape != background.shape:
                    logger.warning(f"Resizing debug frame {i} to match background")
                    debug_frame = cv2.resize(debug_frame, (background.shape[1], background.shape[0]))
                
                # Test each method
                for method_name in self.methods:
                    logger.info(f"Testing method: {method_name}")
                    
                    try:
                        # Get the method from subtractor
                        method = getattr(self.subtractor, method_name)
                        
                        # Apply background removal
                        result = method(debug_frame, background)
                        
                        # Save results
                        self._save_results(i, method_name, result, debug_frame, background)
                        
                        logger.info(f"✓ {method_name} completed (confidence: {result.confidence:.3f})")
                        
                    except Exception as e:
                        logger.error(f"✗ {method_name} failed: {e}")
                        continue
            
            logger.info("\n=== All tests completed ===")
            
        except Exception as e:
            logger.error(f"Test failed: {e}")
            raise
    
    def _save_results(self, frame_num: int, method_name: str, result, original_frame: np.ndarray, background_image: np.ndarray) -> None:
        """
        Save results from background removal method.
        
        Args:
            frame_num: Debug frame number (1-4)
            method_name: Name of the method used
            result: SubtractionResult object
            original_frame: Original debug frame
            background_image: Background image for reconstruction tests
        """
        # Create filename prefix
        prefix = f"debug_frame_{frame_num}_{method_name}"
        
        # Save foreground image
        if result.foreground_image.size > 0:
            foreground_path = self.output_dir / f"{prefix}_foreground.png"
            cv2.imwrite(str(foreground_path), result.foreground_image)
            logger.debug(f"Saved foreground: {foreground_path}")
        
        # Save foreground mask
        if result.foreground_mask.size > 0:
            mask_path = self.output_dir / f"{prefix}_mask.png"
            cv2.imwrite(str(mask_path), result.foreground_mask)
            logger.debug(f"Saved mask: {mask_path}")
        
        # Create enhanced version for OCR
        enhanced_path = self.output_dir / f"{prefix}_enhanced.png"
        enhanced_image = self._enhance_for_ocr(result.foreground_image, result.foreground_mask)
        cv2.imwrite(str(enhanced_path), enhanced_image)
        logger.debug(f"Saved enhanced: {enhanced_path}")
        
        # Create comparison image (side by side)
        comparison_path = self.output_dir / f"{prefix}_comparison.png"
        comparison_image = self._create_comparison(original_frame, result.foreground_image, enhanced_image)
        cv2.imwrite(str(comparison_path), comparison_image)
        logger.debug(f"Saved comparison: {comparison_path}")
        
        # For transparency simulation methods, create reconstruction test
        if method_name in ["transparency_simulation", "alpha_extraction"]:
            reconstruction_path = self.output_dir / f"{prefix}_reconstruction.png"
            reconstruction_image = self._test_reconstruction(result.foreground_image, background_image)
            cv2.imwrite(str(reconstruction_path), reconstruction_image)
            logger.debug(f"Saved reconstruction test: {reconstruction_path}")
    
    def _enhance_for_ocr(self, foreground_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Enhance foreground image for better OCR performance.
        
        Args:
            foreground_image: Extracted foreground image
            mask: Foreground mask
            
        Returns:
            Enhanced image optimized for OCR
        """
        if foreground_image.size == 0:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Convert mask to 3-channel if needed
        if len(mask.shape) == 2:
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            mask_3ch = mask
        
        # Apply mask to foreground
        masked_foreground = cv2.bitwise_and(foreground_image, mask_3ch)
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(masked_foreground, cv2.COLOR_BGR2GRAY)
        
        # Apply morphological operations to clean up text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(cleaned)
        
        # Apply sharpening
        kernel_sharpen = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
        
        # Convert back to BGR for consistency
        enhanced_bgr = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        
        return enhanced_bgr
    
    def _create_comparison(self, original: np.ndarray, foreground: np.ndarray, enhanced: np.ndarray) -> np.ndarray:
        """
        Create side-by-side comparison image.
        
        Args:
            original: Original debug frame
            foreground: Extracted foreground
            enhanced: Enhanced version for OCR
            
        Returns:
            Combined comparison image
        """
        # Ensure all images are same height
        target_height = original.shape[0]
        
        # Resize images to same height
        foreground_resized = cv2.resize(foreground, (original.shape[1], target_height)) if foreground.size > 0 else np.zeros_like(original)
        enhanced_resized = cv2.resize(enhanced, (original.shape[1], target_height)) if enhanced.size > 0 else np.zeros_like(original)
        
        # Create side-by-side comparison
        comparison = np.hstack([original, foreground_resized, enhanced_resized])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 255, 255)
        thickness = 2
        
        # Add text labels
        cv2.putText(comparison, "Original", (10, 30), font, font_scale, color, thickness)
        cv2.putText(comparison, "Foreground", (original.shape[1] + 10, 30), font, font_scale, color, thickness)
        cv2.putText(comparison, "Enhanced", (original.shape[1] * 2 + 10, 30), font, font_scale, color, thickness)
        
        return comparison
    
    def _test_reconstruction(self, simulated_image: np.ndarray, background_image: np.ndarray) -> np.ndarray:
        """
        Test reconstruction by compositing simulated image over background.
        
        This simulates what would happen if you dropped the simulated image
        (transparent elements over black) back onto the original background.
        It should recreate something close to the original composite image.
        
        Args:
            simulated_image: Image with transparent elements over black background
            background_image: Original background image
            
        Returns:
            Reconstructed composite image
        """
        # Convert to float for alpha blending
        sim_float = simulated_image.astype(np.float32) / 255.0
        bg_float = background_image.astype(np.float32) / 255.0
        
        # Simple alpha blending: result = foreground + background * (1 - alpha)
        # Where alpha is estimated from the brightness of the simulated image
        alpha = np.mean(sim_float, axis=2)  # Use average brightness as alpha estimate
        
        # Create reconstruction
        reconstructed = np.zeros_like(sim_float)
        for c in range(3):  # For each color channel
            reconstructed[:, :, c] = sim_float[:, :, c] + bg_float[:, :, c] * (1 - alpha)
        
        # Convert back to uint8
        reconstructed = np.clip(reconstructed * 255, 0, 255).astype(np.uint8)
        
        return reconstructed
    
    def generate_summary_report(self) -> None:
        """Generate a summary report of all test results."""
        logger.info("Generating summary report...")
        
        report_path = self.output_dir / "test_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("Background Removal Test Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Test Date: {Path(__file__).stat().st_mtime}\n")
            f.write(f"Background Image: {self.background_image_path}\n")
            f.write(f"Debug Frames: {len(self.debug_frames)}\n")
            f.write(f"Methods Tested: {len(self.methods)}\n\n")
            
            f.write("Methods:\n")
            for i, method in enumerate(self.methods, 1):
                f.write(f"  {i}. {method}\n")
            
            f.write(f"\nDebug Frames:\n")
            for i, frame in enumerate(self.debug_frames, 1):
                f.write(f"  {i}. {frame}\n")
            
            f.write(f"\nOutput Directory: {self.output_dir}\n")
            f.write(f"Total Files Generated: {len(list(self.output_dir.glob('*.png')))} images\n")
        
        logger.info(f"Summary report saved: {report_path}")


def main():
    """Main test execution."""
    logger.info("Starting Background Removal Tests")
    logger.info("=" * 50)
    
    try:
        # Create tester
        tester = BackgroundRemovalTester()
        
        # Run all tests
        tester.test_all_methods()
        
        # Generate summary
        tester.generate_summary_report()
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
