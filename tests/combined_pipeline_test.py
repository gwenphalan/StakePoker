#!/usr/bin/env python3
"""
Combined background removal + text enhancement pipeline.

Applies text enhancement operations ON TOP OF background removal results
to maximize text legibility for OCR on poker game interfaces.

Usage:
    python tests/combined_pipeline_test.py
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
class PipelineResult:
    """Result of combined background removal + text enhancement pipeline."""
    method_name: str
    frame_number: int
    background_removed_image: np.ndarray
    enhanced_image: np.ndarray
    confidence_score: float
    processing_time: float


class CombinedPipelineProcessor:
    """Combined background removal + text enhancement processor."""
    
    def __init__(self):
        """Initialize combined pipeline processor."""
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
        
        logger.info("CombinedPipelineProcessor initialized")
    
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
    
    def unsharp_masking(self, image: np.ndarray, strength: float = 1.5) -> np.ndarray:
        """Apply unsharp masking to enhance text edges."""
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (0, 0), 2.0)
        
        # Calculate unsharp mask
        unsharp_mask = cv2.addWeighted(gray, 1.0 + strength, blurred, -strength, 0)
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(unsharp_mask, cv2.COLOR_GRAY2BGR)
        
        return enhanced
    
    def adaptive_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Apply Adaptive Histogram Equalization (CLAHE) for better contrast."""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def gamma_correction(self, image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
        """Apply gamma correction to improve text visibility."""
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        
        # Apply gamma correction
        enhanced = cv2.LUT(image, table)
        
        return enhanced
    
    def text_specific_sharpening(self, image: np.ndarray) -> np.ndarray:
        """Apply text-specific sharpening using high-pass filter."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Laplacian filter for edge detection
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Convert back to uint8
        laplacian = np.uint8(np.absolute(laplacian))
        
        # Add sharpening effect
        sharpened = cv2.addWeighted(gray, 1.0, laplacian, 0.3, 0)
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        
        return enhanced
    
    def morphological_text_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Use morphological operations to enhance text structure."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply morphological closing to connect text regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Apply morphological opening to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)
        
        return enhanced
    
    def contrast_stretching(self, image: np.ndarray) -> np.ndarray:
        """Apply contrast stretching to maximize dynamic range."""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Apply contrast stretching to L channel
        l_channel = lab[:, :, 0]
        
        # Find min and max values
        min_val = np.min(l_channel)
        max_val = np.max(l_channel)
        
        # Stretch contrast
        if max_val > min_val:
            stretched = ((l_channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            lab[:, :, 0] = stretched
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def apply_text_enhancement(self, background_removed_image: np.ndarray, enhancement_method: str) -> np.ndarray:
        """
        Apply specific text enhancement method to background-removed image.
        
        Args:
            background_removed_image: Image with transparent elements over black background
            enhancement_method: Name of enhancement method to apply
            
        Returns:
            Enhanced image with improved text legibility
        """
        if enhancement_method == "unsharp_masking":
            return self.unsharp_masking(background_removed_image, strength=1.5)
        elif enhancement_method == "adaptive_histogram_equalization":
            return self.adaptive_histogram_equalization(background_removed_image)
        elif enhancement_method == "gamma_correction":
            return self.gamma_correction(background_removed_image, gamma=1.2)
        elif enhancement_method == "text_specific_sharpening":
            return self.text_specific_sharpening(background_removed_image)
        elif enhancement_method == "morphological_text_enhancement":
            return self.morphological_text_enhancement(background_removed_image)
        elif enhancement_method == "contrast_stretching":
            return self.contrast_stretching(background_removed_image)
        elif enhancement_method == "combined_enhancement":
            # Apply multiple enhancements in sequence
            enhanced = self.unsharp_masking(background_removed_image, strength=1.5)
            enhanced = self.adaptive_histogram_equalization(enhanced)
            enhanced = self.gamma_correction(enhanced, gamma=1.1)
            enhanced = self.text_specific_sharpening(enhanced)
            return enhanced
        else:
            return background_removed_image
    
    def calculate_enhancement_confidence(self, original: np.ndarray, enhanced: np.ndarray) -> float:
        """Calculate confidence score for enhancement quality."""
        # Convert to grayscale
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        # Calculate contrast improvement
        orig_contrast = np.std(orig_gray)
        enh_contrast = np.std(enh_gray)
        contrast_improvement = enh_contrast / (orig_contrast + 1e-6)
        
        # Calculate edge enhancement
        orig_edges = cv2.Canny(orig_gray, 50, 150)
        enh_edges = cv2.Canny(enh_gray, 50, 150)
        edge_improvement = np.sum(enh_edges) / (np.sum(orig_edges) + 1e-6)
        
        # Calculate overall confidence
        confidence = min(contrast_improvement * 0.6 + edge_improvement * 0.4, 2.0)
        
        return confidence
    
    def test_combined_pipeline(self) -> List[PipelineResult]:
        """Test combined background removal + text enhancement pipeline."""
        logger.info("Testing combined background removal + text enhancement pipeline...")
        
        # Load images
        background, debug_frames = self.load_images()
        
        # Define enhancement methods to test
        enhancement_methods = [
            "unsharp_masking",
            "adaptive_histogram_equalization", 
            "gamma_correction",
            "text_specific_sharpening",
            "morphological_text_enhancement",
            "contrast_stretching",
            "combined_enhancement"
        ]
        
        results = []
        
        # Test each debug frame
        for i, debug_frame in enumerate(debug_frames, 1):
            logger.info(f"\n=== Processing Debug Frame {i} ===")
            
            # Ensure frames are same size
            if debug_frame.shape != background.shape:
                logger.warning(f"Resizing debug frame {i} to match background")
                debug_frame = cv2.resize(debug_frame, (background.shape[1], background.shape[0]))
            
            # Step 1: Background removal using alpha extraction (best method)
            logger.info("Step 1: Background removal...")
            bg_removal_result = self.subtractor.alpha_extraction(debug_frame, background)
            background_removed_image = bg_removal_result.foreground_image
            
            # Test each enhancement method on the background-removed image
            for enhancement_method in enhancement_methods:
                logger.info(f"Step 2: Applying {enhancement_method}...")
                
                try:
                    import time
                    start_time = time.time()
                    
                    # Apply text enhancement to background-removed image
                    enhanced_image = self.apply_text_enhancement(background_removed_image, enhancement_method)
                    
                    processing_time = time.time() - start_time
                    
                    # Calculate confidence score
                    confidence_score = self.calculate_enhancement_confidence(background_removed_image, enhanced_image)
                    
                    # Create result
                    result = PipelineResult(
                        method_name=enhancement_method,
                        frame_number=i,
                        background_removed_image=background_removed_image,
                        enhanced_image=enhanced_image,
                        confidence_score=confidence_score,
                        processing_time=processing_time
                    )
                    
                    results.append(result)
                    
                    logger.info(f"✓ {enhancement_method}: confidence={confidence_score:.3f}, time={processing_time:.3f}s")
                    
                except Exception as e:
                    logger.error(f"✗ {enhancement_method} failed: {e}")
                    continue
        
        return results
    
    def save_pipeline_results(self, results: List[PipelineResult]) -> None:
        """Save combined pipeline results."""
        logger.info("Saving combined pipeline results...")
        
        # Create pipeline directory
        pipeline_dir = self.output_dir / "combined_pipeline"
        pipeline_dir.mkdir(exist_ok=True)
        
        # Group results by method
        method_results = {}
        for result in results:
            if result.method_name not in method_results:
                method_results[result.method_name] = []
            method_results[result.method_name].append(result)
        
        # Save results for each method
        for method_name, method_result_list in method_results.items():
            logger.info(f"Saving results for {method_name}...")
            
            # Create method directory
            method_dir = pipeline_dir / method_name
            method_dir.mkdir(exist_ok=True)
            
            for result in method_result_list:
                frame_num = result.frame_number
                
                # Save background-removed image
                cv2.imwrite(str(method_dir / f"frame_{frame_num}_background_removed.png"), 
                           result.background_removed_image)
                
                # Save enhanced image
                cv2.imwrite(str(method_dir / f"frame_{frame_num}_enhanced.png"), 
                           result.enhanced_image)
                
                # Create triple comparison (original | background_removed | enhanced)
                # Load original frame
                frame_path = self.data_dir / f"debug_frame_{frame_num}.png"
                if frame_path.exists():
                    original = cv2.imread(str(frame_path))
                    if original is not None:
                        # Ensure same size
                        if original.shape != result.background_removed_image.shape:
                            original = cv2.resize(original, (result.background_removed_image.shape[1], 
                                                           result.background_removed_image.shape[0]))
                        
                        # Create triple comparison
                        comparison = np.hstack([original, result.background_removed_image, result.enhanced_image])
                        cv2.imwrite(str(method_dir / f"frame_{frame_num}_triple_comparison.png"), comparison)
        
        # Generate pipeline report
        self._generate_pipeline_report(results, pipeline_dir)
    
    def _generate_pipeline_report(self, results: List[PipelineResult], output_dir: Path) -> None:
        """Generate comprehensive pipeline report."""
        report_path = output_dir / "pipeline_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("Combined Background Removal + Text Enhancement Pipeline Report\n")
            f.write("=" * 70 + "\n\n")
            
            # Summary statistics
            f.write("Pipeline Performance Summary:\n")
            f.write("-" * 50 + "\n")
            
            # Group by method
            method_stats = {}
            for result in results:
                if result.method_name not in method_stats:
                    method_stats[result.method_name] = []
                method_stats[result.method_name].append(result)
            
            for method_name, method_results in method_stats.items():
                avg_confidence = np.mean([r.confidence_score for r in method_results])
                avg_time = np.mean([r.processing_time for r in method_results])
                f.write(f"{method_name:30s}: Confidence={avg_confidence:.3f}, Time={avg_time:.3f}s\n")
            
            # Best methods
            f.write(f"\nBest Enhancement Methods (on background-removed images):\n")
            f.write("-" * 60 + "\n")
            
            method_scores = []
            for method_name, method_results in method_stats.items():
                avg_confidence = np.mean([r.confidence_score for r in method_results])
                method_scores.append((method_name, avg_confidence))
            
            method_scores.sort(key=lambda x: x[1], reverse=True)
            
            for i, (method_name, score) in enumerate(method_scores, 1):
                f.write(f"{i}. {method_name}: {score:.3f}\n")
            
            # Pipeline description
            f.write(f"\nPipeline Description:\n")
            f.write("-" * 25 + "\n")
            f.write("1. Background Removal: Extract transparent elements over black background\n")
            f.write("2. Text Enhancement: Apply enhancement operations to improve text legibility\n")
            f.write("3. Result: Clean, legible text ready for OCR\n")
        
        logger.info(f"Pipeline report saved: {report_path}")


def main():
    """Main combined pipeline execution."""
    logger.info("Starting Combined Background Removal + Text Enhancement Pipeline")
    logger.info("=" * 70)
    
    try:
        # Create processor
        processor = CombinedPipelineProcessor()
        
        # Test combined pipeline
        results = processor.test_combined_pipeline()
        
        # Save results
        processor.save_pipeline_results(results)
        
        logger.info(f"\n🎯 Combined pipeline completed!")
        logger.info(f"✅ Generated enhanced images for all methods!")
        
        # Show best method
        if results:
            best_result = max(results, key=lambda x: x.confidence_score)
            logger.info(f"🏆 Best method: {best_result.method_name} (confidence={best_result.confidence_score:.3f})")
        
    except Exception as e:
        logger.error(f"Combined pipeline failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
