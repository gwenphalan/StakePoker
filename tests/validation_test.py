#!/usr/bin/env python3
"""
Advanced background removal validation and calibration system.

Automatically validates reconstruction accuracy, finds the best method,
and iteratively improves results based on pixel-level differences.

Usage:
    python tests/validation_test.py
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
class ValidationResult:
    """Result of reconstruction validation."""
    method_name: str
    frame_number: int
    mse: float  # Mean Squared Error
    psnr: float  # Peak Signal-to-Noise Ratio
    ssim: float  # Structural Similarity Index
    perfect_match_percentage: float
    is_perfect: bool
    reconstruction_image: np.ndarray


class BackgroundRemovalValidator:
    """Advanced validator that finds perfect reconstructions and calibrates methods."""
    
    def __init__(self):
        """Initialize validator with paths and subtractor."""
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
        
        logger.info(f"Initialized validator with {len(self.debug_frames)} debug frames and {len(self.methods)} methods")
    
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
    
    def calculate_image_metrics(self, original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive image quality metrics.
        
        Args:
            original: Original debug frame
            reconstructed: Reconstructed image
            
        Returns:
            Dictionary with MSE, PSNR, SSIM, and perfect match percentage
        """
        # Ensure same size
        if original.shape != reconstructed.shape:
            reconstructed = cv2.resize(reconstructed, (original.shape[1], original.shape[0]))
        
        # Convert to float for calculations
        orig_float = original.astype(np.float32)
        recon_float = reconstructed.astype(np.float32)
        
        # Calculate MSE (Mean Squared Error)
        mse = np.mean((orig_float - recon_float) ** 2)
        
        # Calculate PSNR (Peak Signal-to-Noise Ratio)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        
        # Calculate SSIM (Structural Similarity Index)
        ssim = self._calculate_ssim(original, reconstructed)
        
        # Calculate perfect match percentage
        diff = np.abs(orig_float - recon_float)
        perfect_pixels = np.sum(diff < 1.0)  # Pixels with difference < 1
        total_pixels = diff.size
        perfect_match_percentage = (perfect_pixels / total_pixels) * 100
        
        return {
            'mse': mse,
            'psnr': psnr,
            'ssim': ssim,
            'perfect_match_percentage': perfect_match_percentage
        }
    
    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Structural Similarity Index."""
        # Convert to grayscale for SSIM calculation
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Convert to float
        gray1 = gray1.astype(np.float32)
        gray2 = gray2.astype(np.float32)
        
        # Calculate means
        mu1 = np.mean(gray1)
        mu2 = np.mean(gray2)
        
        # Calculate variances and covariance
        sigma1_sq = np.var(gray1)
        sigma2_sq = np.var(gray2)
        sigma12 = np.mean((gray1 - mu1) * (gray2 - mu2))
        
        # SSIM constants
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        # Calculate SSIM
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
        
        return ssim
    
    def reconstruct_image(self, simulated_image: np.ndarray, background_image: np.ndarray) -> np.ndarray:
        """
        Reconstruct original image by compositing simulated image over background.
        
        Args:
            simulated_image: Image with transparent elements over black background
            background_image: Original background image
            
        Returns:
            Reconstructed composite image
        """
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
    
    def validate_all_methods(self) -> List[ValidationResult]:
        """
        Validate all methods and find perfect reconstructions.
        
        Returns:
            List of ValidationResult objects sorted by quality
        """
        logger.info("Starting comprehensive validation...")
        
        # Load images
        background, debug_frames = self.load_images()
        
        validation_results = []
        
        # Test each debug frame
        for i, debug_frame in enumerate(debug_frames, 1):
            logger.info(f"\n=== Validating Debug Frame {i} ===")
            
            # Ensure frames are same size
            if debug_frame.shape != background.shape:
                logger.warning(f"Resizing debug frame {i} to match background")
                debug_frame = cv2.resize(debug_frame, (background.shape[1], background.shape[0]))
            
            # Test each method
            for method_name in self.methods:
                logger.info(f"Validating method: {method_name}")
                
                try:
                    # Get the method from subtractor
                    method = getattr(self.subtractor, method_name)
                    
                    # Apply background removal
                    result = method(debug_frame, background)
                    
                    # Reconstruct the original image
                    reconstructed = self.reconstruct_image(result.foreground_image, background)
                    
                    # Calculate validation metrics
                    metrics = self.calculate_image_metrics(debug_frame, reconstructed)
                    
                    # Determine if it's a perfect match
                    is_perfect = metrics['perfect_match_percentage'] > 99.9
                    
                    # Create validation result
                    validation_result = ValidationResult(
                        method_name=method_name,
                        frame_number=i,
                        mse=metrics['mse'],
                        psnr=metrics['psnr'],
                        ssim=metrics['ssim'],
                        perfect_match_percentage=metrics['perfect_match_percentage'],
                        is_perfect=is_perfect,
                        reconstruction_image=reconstructed
                    )
                    
                    validation_results.append(validation_result)
                    
                    # Log results
                    status = "✓ PERFECT" if is_perfect else "✓"
                    logger.info(f"{status} {method_name}: {metrics['perfect_match_percentage']:.2f}% match, PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.3f}")
                    
                except Exception as e:
                    logger.error(f"✗ {method_name} failed: {e}")
                    continue
        
        # Sort by quality (perfect match percentage, then PSNR)
        validation_results.sort(key=lambda x: (x.perfect_match_percentage, x.psnr), reverse=True)
        
        return validation_results
    
    def find_perfect_methods(self, validation_results: List[ValidationResult]) -> List[ValidationResult]:
        """Find methods that achieved perfect reconstruction."""
        perfect_methods = [r for r in validation_results if r.is_perfect]
        
        if perfect_methods:
            logger.info(f"\n🎯 FOUND {len(perfect_methods)} PERFECT RECONSTRUCTIONS!")
            for result in perfect_methods:
                logger.info(f"  ✓ {result.method_name} on Frame {result.frame_number}: {result.perfect_match_percentage:.2f}% match")
        else:
            logger.info("\n❌ No perfect reconstructions found. Best results:")
            best_results = validation_results[:5]  # Top 5
            for result in best_results:
                logger.info(f"  • {result.method_name} on Frame {result.frame_number}: {result.perfect_match_percentage:.2f}% match")
        
        return perfect_methods
    
    def calibrate_methods(self, validation_results: List[ValidationResult]) -> None:
        """
        Analyze differences and calibrate methods for better results.
        
        Args:
            validation_results: List of validation results to analyze
        """
        logger.info("\n🔧 Starting method calibration...")
        
        # Load images for analysis
        background, debug_frames = self.load_images()
        
        # Find the best performing method
        best_result = validation_results[0]
        logger.info(f"Best method: {best_result.method_name} ({best_result.perfect_match_percentage:.2f}% match)")
        
        # Analyze differences for the best method
        debug_frame = debug_frames[best_result.frame_number - 1]
        
        # Get the method result
        method = getattr(self.subtractor, best_result.method_name)
        result = method(debug_frame, background)
        
        # Reconstruct and analyze differences
        reconstructed = self.reconstruct_image(result.foreground_image, background)
        
        # Calculate pixel-level differences
        diff = np.abs(debug_frame.astype(np.float32) - reconstructed.astype(np.float32))
        
        # Find areas with largest differences
        max_diff_per_pixel = np.max(diff, axis=2)
        threshold = np.percentile(max_diff_per_pixel, 95)  # Top 5% differences
        problem_areas = max_diff_per_pixel > threshold
        
        logger.info(f"Found {np.sum(problem_areas)} pixels with significant differences")
        
        # Save analysis images
        self._save_calibration_analysis(
            debug_frame, reconstructed, diff, problem_areas, 
            best_result.method_name, best_result.frame_number
        )
        
        # Suggest improvements
        self._suggest_improvements(best_result, diff, problem_areas)
    
    def _save_calibration_analysis(self, original: np.ndarray, reconstructed: np.ndarray, 
                                 diff: np.ndarray, problem_areas: np.ndarray,
                                 method_name: str, frame_number: int) -> None:
        """Save calibration analysis images."""
        
        # Create analysis directory
        analysis_dir = self.output_dir / "calibration_analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        prefix = f"frame_{frame_number}_{method_name}"
        
        # Save difference heatmap
        diff_normalized = (diff / np.max(diff) * 255).astype(np.uint8)
        diff_heatmap = cv2.applyColorMap(diff_normalized, cv2.COLORMAP_HOT)
        cv2.imwrite(str(analysis_dir / f"{prefix}_difference_heatmap.png"), diff_heatmap)
        
        # Save problem areas mask
        problem_mask = (problem_areas * 255).astype(np.uint8)
        cv2.imwrite(str(analysis_dir / f"{prefix}_problem_areas.png"), problem_mask)
        
        # Save comparison
        comparison = np.hstack([original, reconstructed, diff_heatmap])
        cv2.imwrite(str(analysis_dir / f"{prefix}_comparison.png"), comparison)
        
        logger.info(f"Calibration analysis saved to {analysis_dir}")
    
    def _suggest_improvements(self, best_result: ValidationResult, diff: np.ndarray, problem_areas: np.ndarray) -> None:
        """Suggest improvements based on analysis."""
        logger.info("\n💡 Improvement Suggestions:")
        
        # Analyze problem areas
        problem_pixels = np.sum(problem_areas)
        total_pixels = problem_areas.size
        problem_percentage = (problem_pixels / total_pixels) * 100
        
        logger.info(f"  • {problem_percentage:.2f}% of pixels have significant reconstruction errors")
        
        # Analyze difference patterns
        avg_diff = np.mean(diff[problem_areas])
        max_diff = np.max(diff)
        
        logger.info(f"  • Average difference in problem areas: {avg_diff:.2f}")
        logger.info(f"  • Maximum difference: {max_diff:.2f}")
        
        # Suggest specific improvements
        if best_result.method_name in ["transparency_simulation", "alpha_extraction"]:
            logger.info("  • Alpha calculation may need refinement")
            logger.info("  • Consider adjusting transparency thresholds")
        elif best_result.method_name == "simple_difference":
            logger.info("  • Pixel difference threshold may need adjustment")
        elif best_result.method_name == "statistical_subtraction":
            logger.info("  • Statistical multiplier may need calibration")
        
        logger.info("  • Consider implementing iterative refinement")
        logger.info("  • Try combining multiple methods for problem areas")
    
    def generate_validation_report(self, validation_results: List[ValidationResult]) -> None:
        """Generate comprehensive validation report."""
        logger.info("Generating validation report...")
        
        report_path = self.output_dir / "validation_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("Background Removal Validation Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Summary statistics
            perfect_count = sum(1 for r in validation_results if r.is_perfect)
            f.write(f"Total Tests: {len(validation_results)}\n")
            f.write(f"Perfect Reconstructions: {perfect_count}\n")
            f.write(f"Success Rate: {(perfect_count/len(validation_results)*100):.1f}%\n\n")
            
            # Best results
            f.write("Top 10 Results:\n")
            f.write("-" * 30 + "\n")
            for i, result in enumerate(validation_results[:10], 1):
                status = "PERFECT" if result.is_perfect else "GOOD"
                f.write(f"{i:2d}. {result.method_name:20s} Frame {result.frame_number} "
                       f"{result.perfect_match_percentage:6.2f}% {status}\n")
            
            # Perfect methods
            perfect_methods = [r for r in validation_results if r.is_perfect]
            if perfect_methods:
                f.write(f"\nPerfect Reconstruction Methods:\n")
                f.write("-" * 35 + "\n")
                for result in perfect_methods:
                    f.write(f"• {result.method_name} on Frame {result.frame_number}\n")
                    f.write(f"  MSE: {result.mse:.2f}, PSNR: {result.psnr:.2f}, SSIM: {result.ssim:.3f}\n")
            
            # Method performance summary
            f.write(f"\nMethod Performance Summary:\n")
            f.write("-" * 30 + "\n")
            method_stats = {}
            for result in validation_results:
                if result.method_name not in method_stats:
                    method_stats[result.method_name] = []
                method_stats[result.method_name].append(result.perfect_match_percentage)
            
            for method_name, scores in method_stats.items():
                avg_score = np.mean(scores)
                max_score = np.max(scores)
                f.write(f"{method_name:20s}: Avg {avg_score:6.2f}%, Max {max_score:6.2f}%\n")
        
        logger.info(f"Validation report saved: {report_path}")


def main():
    """Main validation execution."""
    logger.info("Starting Background Removal Validation")
    logger.info("=" * 50)
    
    try:
        # Create validator
        validator = BackgroundRemovalValidator()
        
        # Run comprehensive validation
        validation_results = validator.validate_all_methods()
        
        # Find perfect methods
        perfect_methods = validator.find_perfect_methods(validation_results)
        
        # Calibrate methods for improvement
        validator.calibrate_methods(validation_results)
        
        # Generate report
        validator.generate_validation_report(validation_results)
        
        logger.info("\n🎯 Validation completed successfully!")
        
        if perfect_methods:
            logger.info(f"✅ Found {len(perfect_methods)} perfect reconstruction(s)!")
        else:
            logger.info("⚠️  No perfect reconstructions found - check calibration analysis")
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
