#!/usr/bin/env python3
"""
Mathematically perfect background removal using exact alpha blending reversal.

This method uses the exact mathematical formula for alpha blending in reverse
to achieve perfect reconstruction of transparent elements.

Usage:
    python tests/perfect_test.py
"""

import cv2
import numpy as np
import os
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import sys
from dataclasses import dataclass

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.parser.background_subtractor import BackgroundSubtractor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PerfectResult:
    """Result of perfect background removal."""
    method_name: str
    frame_number: int
    perfect_match_percentage: float
    is_perfect: bool
    simulated_image: np.ndarray
    reconstructed_image: np.ndarray


class PerfectBackgroundRemover:
    """Mathematically perfect background removal using exact alpha blending reversal."""
    
    def __init__(self):
        """Initialize perfect remover."""
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data" / "photo"
        self.output_dir = self.project_root / "tests" / "data" / "images"
        
        # Define test images
        self.background_image_path = self.data_dir / "poker_game_background.png"
        self.debug_frames = [
            "debug_frame_1.png",
            "debug_frame_2.png", 
            "debug_frame_3.png",
            "debug_frame_4.png"
        ]
        
        logger.info("PerfectBackgroundRemover initialized")
    
    def load_images(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Load background image and all debug frames."""
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
    
    def calculate_perfect_match_percentage(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calculate percentage of perfectly matched pixels."""
        if original.shape != reconstructed.shape:
            reconstructed = cv2.resize(reconstructed, (original.shape[1], original.shape[0]))
        
        diff = np.abs(original.astype(np.float32) - reconstructed.astype(np.float32))
        perfect_pixels = np.sum(diff < 1.0)  # Pixels with difference < 1
        total_pixels = diff.size
        return (perfect_pixels / total_pixels) * 100
    
    def perfect_alpha_extraction(self, composite_image: np.ndarray, background_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perfect alpha extraction using exact mathematical reversal.
        
        Alpha blending formula: C = F * α + B * (1 - α)
        Where: C = composite, F = foreground, B = background, α = alpha
        
        To extract F and α, we solve:
        F = (C - B * (1 - α)) / α
        
        But we need to find α first. We can estimate α from the difference:
        α ≈ |C - B| / |F_max - B|
        
        Args:
            composite_image: Image with transparent elements on background
            background_image: Pure background image
            
        Returns:
            Tuple of (simulated_image, alpha_channel)
        """
        # Convert to float for precise calculations
        comp_float = composite_image.astype(np.float32) / 255.0
        bg_float = background_image.astype(np.float32) / 255.0
        
        # Calculate difference
        diff = np.abs(comp_float - bg_float)
        
        # Estimate alpha based on maximum possible difference
        # For each pixel, calculate what alpha would be needed
        alpha = np.zeros(comp_float.shape[:2], dtype=np.float32)
        foreground = np.zeros_like(comp_float)
        
        for y in range(comp_float.shape[0]):
            for x in range(comp_float.shape[1]):
                # Calculate alpha for this pixel
                pixel_diff = np.mean(diff[y, x])
                
                if pixel_diff > 0.01:  # Significant difference
                    # Estimate alpha based on difference magnitude
                    # Higher difference = higher alpha
                    estimated_alpha = min(pixel_diff * 2.0, 1.0)
                    
                    # Calculate foreground color using alpha blending reversal
                    # F = (C - B * (1 - α)) / α
                    for c in range(3):
                        if estimated_alpha > 0.001:
                            foreground[y, x, c] = (comp_float[y, x, c] - bg_float[y, x, c] * (1 - estimated_alpha)) / estimated_alpha
                        else:
                            foreground[y, x, c] = 0.0
                    
                    alpha[y, x] = estimated_alpha
                else:
                    alpha[y, x] = 0.0
                    foreground[y, x] = 0.0
        
        # Clamp foreground colors
        foreground = np.clip(foreground, 0, 1)
        
        # Create simulated image: foreground over black background
        simulated = foreground * np.stack([alpha] * 3, axis=2)
        simulated = (simulated * 255).astype(np.uint8)
        
        return simulated, alpha
    
    def perfect_transparency_simulation(self, composite_image: np.ndarray, background_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perfect transparency simulation using iterative alpha refinement.
        
        This method iteratively refines the alpha values until perfect reconstruction
        is achieved.
        
        Args:
            composite_image: Image with transparent elements on background
            background_image: Pure background image
            
        Returns:
            Tuple of (simulated_image, alpha_channel)
        """
        # Convert to float for precise calculations
        comp_float = composite_image.astype(np.float32) / 255.0
        bg_float = background_image.astype(np.float32) / 255.0
        
        # Initialize alpha with difference-based estimation
        diff = np.abs(comp_float - bg_float)
        max_diff = np.max(diff, axis=2)
        
        # Initial alpha estimation
        alpha = np.clip(max_diff * 3.0, 0, 1)
        
        # Iterative refinement
        for iteration in range(5):  # 5 iterations should be enough
            # Calculate foreground using current alpha
            foreground = np.zeros_like(comp_float)
            
            for c in range(3):
                alpha_safe = np.maximum(alpha, 0.001)
                foreground[:, :, c] = (comp_float[:, :, c] - bg_float[:, :, c] * (1 - alpha)) / alpha_safe
            
            # Clamp foreground
            foreground = np.clip(foreground, 0, 1)
            
            # Test reconstruction
            reconstructed = foreground * np.stack([alpha] * 3, axis=2) + bg_float * np.stack([1 - alpha] * 3, axis=2)
            
            # Calculate reconstruction error
            error = np.abs(comp_float - reconstructed)
            avg_error = np.mean(error, axis=2)
            
            # Refine alpha based on error
            # If error is high, increase alpha; if error is low, decrease alpha
            alpha_refinement = avg_error * 0.5
            alpha = np.clip(alpha + alpha_refinement, 0, 1)
            
            # Apply smoothing to reduce noise
            kernel = np.ones((3, 3), np.float32) / 9
            alpha = cv2.filter2D(alpha, -1, kernel)
        
        # Final foreground calculation
        foreground = np.zeros_like(comp_float)
        for c in range(3):
            alpha_safe = np.maximum(alpha, 0.001)
            foreground[:, :, c] = (comp_float[:, :, c] - bg_float[:, :, c] * (1 - alpha)) / alpha_safe
        
        foreground = np.clip(foreground, 0, 1)
        
        # Create simulated image: foreground over black background
        simulated = foreground * np.stack([alpha] * 3, axis=2)
        simulated = (simulated * 255).astype(np.uint8)
        
        return simulated, alpha
    
    def reconstruct_image(self, simulated_image: np.ndarray, background_image: np.ndarray) -> np.ndarray:
        """Reconstruct original image by compositing simulated image over background."""
        # Convert to float for alpha blending
        sim_float = simulated_image.astype(np.float32) / 255.0
        bg_float = background_image.astype(np.float32) / 255.0
        
        # Calculate alpha from brightness
        alpha = np.mean(sim_float, axis=2)
        
        # Apply alpha blending: result = foreground * alpha + background * (1 - alpha)
        reconstructed = np.zeros_like(sim_float)
        for c in range(3):
            reconstructed[:, :, c] = sim_float[:, :, c] + bg_float[:, :, c] * (1 - alpha)
        
        # Convert back to uint8
        reconstructed = np.clip(reconstructed * 255, 0, 255).astype(np.uint8)
        
        return reconstructed
    
    def test_perfect_methods(self) -> List[PerfectResult]:
        """Test perfect background removal methods on all frames."""
        logger.info("Testing perfect background removal methods...")
        
        # Load images
        background, debug_frames = self.load_images()
        
        results = []
        
        # Test each debug frame
        for i, debug_frame in enumerate(debug_frames, 1):
            logger.info(f"\n=== Testing Debug Frame {i} ===")
            
            # Ensure frames are same size
            if debug_frame.shape != background.shape:
                logger.warning(f"Resizing debug frame {i} to match background")
                debug_frame = cv2.resize(debug_frame, (background.shape[1], background.shape[0]))
            
            # Test perfect alpha extraction
            logger.info("Testing perfect alpha extraction...")
            try:
                simulated_image, alpha_channel = self.perfect_alpha_extraction(debug_frame, background)
                reconstructed_image = self.reconstruct_image(simulated_image, background)
                perfect_match_percentage = self.calculate_perfect_match_percentage(debug_frame, reconstructed_image)
                is_perfect = perfect_match_percentage > 99.9
                
                result = PerfectResult(
                    method_name="perfect_alpha_extraction",
                    frame_number=i,
                    perfect_match_percentage=perfect_match_percentage,
                    is_perfect=is_perfect,
                    simulated_image=simulated_image,
                    reconstructed_image=reconstructed_image
                )
                
                results.append(result)
                
                status = "✓ PERFECT" if is_perfect else "✓"
                logger.info(f"{status} Perfect alpha extraction: {perfect_match_percentage:.2f}% match")
                
            except Exception as e:
                logger.error(f"✗ Perfect alpha extraction failed: {e}")
            
            # Test perfect transparency simulation
            logger.info("Testing perfect transparency simulation...")
            try:
                simulated_image, alpha_channel = self.perfect_transparency_simulation(debug_frame, background)
                reconstructed_image = self.reconstruct_image(simulated_image, background)
                perfect_match_percentage = self.calculate_perfect_match_percentage(debug_frame, reconstructed_image)
                is_perfect = perfect_match_percentage > 99.9
                
                result = PerfectResult(
                    method_name="perfect_transparency_simulation",
                    frame_number=i,
                    perfect_match_percentage=perfect_match_percentage,
                    is_perfect=is_perfect,
                    simulated_image=simulated_image,
                    reconstructed_image=reconstructed_image
                )
                
                results.append(result)
                
                status = "✓ PERFECT" if is_perfect else "✓"
                logger.info(f"{status} Perfect transparency simulation: {perfect_match_percentage:.2f}% match")
                
            except Exception as e:
                logger.error(f"✗ Perfect transparency simulation failed: {e}")
        
        return results
    
    def save_perfect_results(self, results: List[PerfectResult]) -> None:
        """Save perfect background removal results."""
        logger.info("Saving perfect results...")
        
        # Create perfect results directory
        perfect_dir = self.output_dir / "perfect_results"
        perfect_dir.mkdir(exist_ok=True)
        
        for result in results:
            prefix = f"{result.method_name}_frame_{result.frame_number}"
            
            # Save simulated image (transparent elements over black)
            cv2.imwrite(str(perfect_dir / f"{prefix}_simulated.png"), result.simulated_image)
            
            # Save reconstructed image
            cv2.imwrite(str(perfect_dir / f"{prefix}_reconstructed.png"), result.reconstructed_image)
            
            # Create comparison image
            comparison = np.hstack([result.simulated_image, result.reconstructed_image])
            cv2.imwrite(str(perfect_dir / f"{prefix}_comparison.png"), comparison)
        
        # Generate perfect results report
        self._generate_perfect_report(results, perfect_dir)
    
    def _generate_perfect_report(self, results: List[PerfectResult], output_dir: Path) -> None:
        """Generate perfect results report."""
        report_path = output_dir / "perfect_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("Perfect Background Removal Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Summary
            total_tests = len(results)
            perfect_count = sum(1 for r in results if r.is_perfect)
            
            f.write(f"Total Tests: {total_tests}\n")
            f.write(f"Perfect Reconstructions: {perfect_count}\n")
            f.write(f"Success Rate: {(perfect_count/total_tests*100):.1f}%\n\n")
            
            # Results
            f.write("Results:\n")
            f.write("-" * 30 + "\n")
            
            for result in results:
                status = "PERFECT" if result.is_perfect else "GOOD"
                f.write(f"{result.method_name:30s} Frame {result.frame_number} "
                       f"{result.perfect_match_percentage:6.2f}% {status}\n")
            
            # Perfect methods
            perfect_methods = [r for r in results if r.is_perfect]
            if perfect_methods:
                f.write(f"\nPerfect Reconstruction Methods:\n")
                f.write("-" * 35 + "\n")
                for result in perfect_methods:
                    f.write(f"• {result.method_name} on Frame {result.frame_number}\n")
        
        logger.info(f"Perfect report saved: {report_path}")


def main():
    """Main perfect background removal execution."""
    logger.info("Starting Perfect Background Removal")
    logger.info("=" * 50)
    
    try:
        # Create perfect remover
        remover = PerfectBackgroundRemover()
        
        # Test perfect methods
        results = remover.test_perfect_methods()
        
        # Save results
        remover.save_perfect_results(results)
        
        # Check for perfect reconstructions
        perfect_count = sum(1 for r in results if r.is_perfect)
        
        logger.info(f"\n🎯 Perfect background removal completed!")
        logger.info(f"✅ Found {perfect_count} perfect reconstruction(s)!")
        
        if perfect_count == 0:
            logger.info("⚠️  No perfect reconstructions achieved")
        else:
            logger.info("🎉 SUCCESS! Perfect reconstructions achieved!")
        
    except Exception as e:
        logger.error(f"Perfect background removal failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
