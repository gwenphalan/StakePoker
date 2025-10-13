#!/usr/bin/env python3
"""
Iterative refinement system for perfect background removal.

Uses difference analysis to iteratively improve transparency simulation
methods until perfect reconstruction is achieved.

Usage:
    python tests/refinement_test.py
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
class RefinementResult:
    """Result of iterative refinement."""
    iteration: int
    method_name: str
    frame_number: int
    perfect_match_percentage: float
    improvement: float
    is_perfect: bool
    refined_image: np.ndarray


class IterativeRefiner:
    """Iteratively refines background removal methods for perfect reconstruction."""
    
    def __init__(self):
        """Initialize refiner with paths and subtractor."""
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data" / "photo"
        self.output_dir = self.project_root / "tests" / "data" / "images"
        
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
        
        logger.info("IterativeRefiner initialized")
    
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
    
    def reconstruct_image(self, simulated_image: np.ndarray, background_image: np.ndarray) -> np.ndarray:
        """Reconstruct original image by compositing simulated image over background."""
        # Convert to float for alpha blending
        sim_float = simulated_image.astype(np.float32) / 255.0
        bg_float = background_image.astype(np.float32) / 255.0
        
        # Calculate alpha from brightness (brighter = more opaque)
        alpha = np.mean(sim_float, axis=2)
        
        # Apply alpha blending: result = foreground * alpha + background * (1 - alpha)
        reconstructed = np.zeros_like(sim_float)
        for c in range(3):  # For each color channel
            reconstructed[:, :, c] = sim_float[:, :, c] + bg_float[:, :, c] * (1 - alpha)
        
        # Convert back to uint8
        reconstructed = np.clip(reconstructed * 255, 0, 255).astype(np.uint8)
        
        return reconstructed
    
    def refined_alpha_extraction(self, composite_image: np.ndarray, background_image: np.ndarray, 
                                iteration: int = 0) -> np.ndarray:
        """
        Refined alpha extraction with iterative improvement.
        
        Uses difference analysis to refine alpha calculation for better reconstruction.
        """
        # Convert to float
        comp_float = composite_image.astype(np.float32) / 255.0
        bg_float = background_image.astype(np.float32) / 255.0
        
        # Calculate initial alpha
        diff = np.abs(comp_float - bg_float)
        pixel_diff = np.mean(diff, axis=2)
        
        # Refine alpha calculation based on iteration
        if iteration == 0:
            # Initial calculation
            alpha = np.clip(pixel_diff * 4.0, 0, 1)
        else:
            # Refined calculation with better scaling
            # Use more sophisticated alpha estimation
            alpha = np.clip(pixel_diff * (4.0 + iteration * 0.5), 0, 1)
            
            # Apply smoothing to reduce noise
            kernel = np.ones((3, 3), np.float32) / 9
            alpha = cv2.filter2D(alpha, -1, kernel)
        
        # Calculate foreground colors with improved formula
        foreground = np.zeros_like(comp_float)
        
        # Avoid division by zero
        alpha_safe = np.maximum(alpha, 0.001)
        
        # Improved foreground calculation
        for c in range(3):  # For each color channel
            # Use more sophisticated blending formula
            foreground[:, :, c] = (comp_float[:, :, c] - bg_float[:, :, c] * (1 - alpha)) / alpha_safe
        
        # Clamp foreground colors
        foreground = np.clip(foreground, 0, 1)
        
        # Create simulated image: foreground over black background
        simulated = foreground * np.stack([alpha] * 3, axis=2)
        simulated = (simulated * 255).astype(np.uint8)
        
        return simulated
    
    def refined_transparency_simulation(self, composite_image: np.ndarray, background_image: np.ndarray,
                                      iteration: int = 0) -> np.ndarray:
        """
        Refined transparency simulation with iterative improvement.
        
        Uses difference analysis to refine transparency calculation.
        """
        # Convert to float for precise calculations
        comp_float = composite_image.astype(np.float32) / 255.0
        bg_float = background_image.astype(np.float32) / 255.0
        
        # Calculate alpha (transparency) for each pixel
        diff = np.abs(comp_float - bg_float)
        max_diff = np.max(diff, axis=2)
        
        # Refine threshold based on iteration
        base_threshold = self.subtractor.settings.get("parser.background_subtraction.pixel_threshold") / 255.0
        threshold = base_threshold * (1.0 - iteration * 0.1)  # Reduce threshold each iteration
        
        # Calculate alpha for non-transparent pixels
        alpha = np.zeros(comp_float.shape[:2], dtype=np.float32)
        
        for y in range(comp_float.shape[0]):
            for x in range(comp_float.shape[1]):
                pixel_diff = max_diff[y, x]
                
                if pixel_diff > threshold:
                    # Refined alpha calculation
                    alpha[y, x] = min(pixel_diff * (3.0 + iteration * 0.3), 1.0)
                else:
                    alpha[y, x] = 0.0
        
        # Apply smoothing for better results
        if iteration > 0:
            kernel = np.ones((3, 3), np.float32) / 9
            alpha = cv2.filter2D(alpha, -1, kernel)
        
        # Create the simulated image: foreground elements over black background
        simulated_image = np.zeros_like(comp_float)
        
        # For pixels with alpha > 0, calculate the foreground color
        mask = alpha > 0
        
        if np.any(mask):
            # Improved foreground color calculation
            foreground_colors = comp_float[mask] / np.stack([alpha[mask]] * 3, axis=1)
            
            # Clamp to valid range
            foreground_colors = np.clip(foreground_colors, 0, 1)
            
            # Apply alpha blending with black background
            simulated_image[mask] = foreground_colors * np.stack([alpha[mask]] * 3, axis=1)
        
        # Convert back to uint8
        simulated_image = (simulated_image * 255).astype(np.uint8)
        
        return simulated_image
    
    def iterative_refinement(self, method_name: str, frame_number: int, 
                           max_iterations: int = 10) -> List[RefinementResult]:
        """
        Perform iterative refinement on a specific method and frame.
        
        Args:
            method_name: Name of the method to refine
            frame_number: Debug frame number (1-4)
            max_iterations: Maximum number of refinement iterations
            
        Returns:
            List of RefinementResult objects showing improvement over iterations
        """
        logger.info(f"Starting iterative refinement: {method_name} on Frame {frame_number}")
        
        # Load images
        background, debug_frames = self.load_images()
        debug_frame = debug_frames[frame_number - 1]
        
        # Ensure frames are same size
        if debug_frame.shape != background.shape:
            debug_frame = cv2.resize(debug_frame, (background.shape[1], background.shape[0]))
        
        refinement_results = []
        best_percentage = 0.0
        
        for iteration in range(max_iterations):
            logger.info(f"  Iteration {iteration + 1}/{max_iterations}")
            
            try:
                # Apply refined method
                if method_name == "alpha_extraction":
                    simulated_image = self.refined_alpha_extraction(debug_frame, background, iteration)
                elif method_name == "transparency_simulation":
                    simulated_image = self.refined_transparency_simulation(debug_frame, background, iteration)
                else:
                    # Use original method for other methods
                    method = getattr(self.subtractor, method_name)
                    result = method(debug_frame, background)
                    simulated_image = result.foreground_image
                
                # Reconstruct and validate
                reconstructed = self.reconstruct_image(simulated_image, background)
                perfect_match_percentage = self.calculate_perfect_match_percentage(debug_frame, reconstructed)
                
                # Calculate improvement
                improvement = perfect_match_percentage - best_percentage
                if improvement > 0:
                    best_percentage = perfect_match_percentage
                
                # Check if perfect
                is_perfect = perfect_match_percentage > 99.9
                
                # Create result
                result = RefinementResult(
                    iteration=iteration,
                    method_name=method_name,
                    frame_number=frame_number,
                    perfect_match_percentage=perfect_match_percentage,
                    improvement=improvement,
                    is_perfect=is_perfect,
                    refined_image=simulated_image
                )
                
                refinement_results.append(result)
                
                # Log progress
                status = "✓ PERFECT" if is_perfect else "✓"
                logger.info(f"    {status} {perfect_match_percentage:.2f}% match (improvement: {improvement:+.2f}%)")
                
                # Stop if perfect
                if is_perfect:
                    logger.info(f"🎯 PERFECT RECONSTRUCTION ACHIEVED!")
                    break
                
                # Stop if no improvement for 3 iterations
                if iteration >= 3 and improvement <= 0:
                    logger.info(f"    No improvement for 3 iterations, stopping")
                    break
                
            except Exception as e:
                logger.error(f"    ✗ Iteration {iteration + 1} failed: {e}")
                break
        
        return refinement_results
    
    def refine_all_methods(self) -> Dict[str, List[RefinementResult]]:
        """
        Refine all methods on all frames.
        
        Returns:
            Dictionary mapping method names to lists of refinement results
        """
        logger.info("Starting comprehensive iterative refinement...")
        
        # Focus on the best performing methods from validation
        methods_to_refine = ["alpha_extraction", "transparency_simulation"]
        
        all_results = {}
        
        for method_name in methods_to_refine:
            logger.info(f"\n=== Refining {method_name} ===")
            method_results = []
            
            for frame_number in range(1, 5):  # Frames 1-4
                logger.info(f"\n--- Frame {frame_number} ---")
                
                try:
                    frame_results = self.iterative_refinement(method_name, frame_number)
                    method_results.extend(frame_results)
                    
                    # Check if we achieved perfect reconstruction
                    perfect_results = [r for r in frame_results if r.is_perfect]
                    if perfect_results:
                        logger.info(f"🎯 PERFECT RECONSTRUCTION on Frame {frame_number}!")
                
                except Exception as e:
                    logger.error(f"Failed to refine {method_name} on Frame {frame_number}: {e}")
                    continue
            
            all_results[method_name] = method_results
        
        return all_results
    
    def save_refinement_results(self, all_results: Dict[str, List[RefinementResult]]) -> None:
        """Save refinement results and analysis."""
        logger.info("Saving refinement results...")
        
        # Create refinement directory
        refinement_dir = self.output_dir / "refinement_results"
        refinement_dir.mkdir(exist_ok=True)
        
        for method_name, results in all_results.items():
            logger.info(f"Saving results for {method_name}...")
            
            # Group results by frame
            frame_results = {}
            for result in results:
                if result.frame_number not in frame_results:
                    frame_results[result.frame_number] = []
                frame_results[result.frame_number].append(result)
            
            # Save results for each frame
            for frame_number, frame_result_list in frame_results.items():
                # Sort by iteration
                frame_result_list.sort(key=lambda x: x.iteration)
                
                # Save the best result
                best_result = max(frame_result_list, key=lambda x: x.perfect_match_percentage)
                
                prefix = f"{method_name}_frame_{frame_number}"
                
                # Save best refined image
                cv2.imwrite(str(refinement_dir / f"{prefix}_best_refined.png"), best_result.refined_image)
                
                # Save iteration progression
                if len(frame_result_list) > 1:
                    # Create progression image
                    progression_images = []
                    for result in frame_result_list:
                        progression_images.append(result.refined_image)
                    
                    # Concatenate horizontally
                    progression = np.hstack(progression_images)
                    cv2.imwrite(str(refinement_dir / f"{prefix}_progression.png"), progression)
        
        # Generate refinement report
        self._generate_refinement_report(all_results, refinement_dir)
    
    def _generate_refinement_report(self, all_results: Dict[str, List[RefinementResult]], 
                                  output_dir: Path) -> None:
        """Generate comprehensive refinement report."""
        report_path = output_dir / "refinement_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("Iterative Refinement Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Summary
            total_tests = sum(len(results) for results in all_results.values())
            perfect_count = sum(1 for results in all_results.values() 
                              for result in results if result.is_perfect)
            
            f.write(f"Total Refinement Tests: {total_tests}\n")
            f.write(f"Perfect Reconstructions: {perfect_count}\n")
            f.write(f"Success Rate: {(perfect_count/total_tests*100):.1f}%\n\n")
            
            # Results by method
            for method_name, results in all_results.items():
                f.write(f"{method_name.upper()} Results:\n")
                f.write("-" * 30 + "\n")
                
                # Group by frame
                frame_results = {}
                for result in results:
                    if result.frame_number not in frame_results:
                        frame_results[result.frame_number] = []
                    frame_results[result.frame_number].append(result)
                
                for frame_number, frame_result_list in frame_results.items():
                    frame_result_list.sort(key=lambda x: x.iteration)
                    best_result = max(frame_result_list, key=lambda x: x.perfect_match_percentage)
                    
                    f.write(f"Frame {frame_number}: {best_result.perfect_match_percentage:.2f}% "
                           f"({len(frame_result_list)} iterations)\n")
                    
                    if best_result.is_perfect:
                        f.write(f"  ✓ PERFECT RECONSTRUCTION!\n")
                
                f.write("\n")
        
        logger.info(f"Refinement report saved: {report_path}")


def main():
    """Main refinement execution."""
    logger.info("Starting Iterative Refinement")
    logger.info("=" * 50)
    
    try:
        # Create refiner
        refiner = IterativeRefiner()
        
        # Run comprehensive refinement
        all_results = refiner.refine_all_methods()
        
        # Save results
        refiner.save_refinement_results(all_results)
        
        # Check for perfect reconstructions
        perfect_count = sum(1 for results in all_results.values() 
                          for result in results if result.is_perfect)
        
        logger.info(f"\n🎯 Refinement completed!")
        logger.info(f"✅ Found {perfect_count} perfect reconstruction(s)!")
        
        if perfect_count == 0:
            logger.info("⚠️  No perfect reconstructions achieved - check refinement analysis")
        
    except Exception as e:
        logger.error(f"Refinement failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
