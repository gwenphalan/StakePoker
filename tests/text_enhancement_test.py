#!/usr/bin/env python3
"""
Advanced text enhancement system for improving OCR legibility.

Applies multiple image processing techniques to enhance semi-transparent text
and improve OCR accuracy on poker game interfaces.

Usage:
    python tests/text_enhancement_test.py
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
class EnhancementResult:
    """Result of text enhancement operation."""
    method_name: str
    enhanced_image: np.ndarray
    confidence_score: float
    processing_time: float


class TextEnhancementProcessor:
    """Advanced text enhancement processor for OCR optimization."""
    
    def __init__(self):
        """Initialize text enhancement processor."""
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data" / "photo"
        self.output_dir = self.project_root / "tests" / "data" / "images"
        
        # Initialize background subtractor for transparency simulation
        self.subtractor = BackgroundSubtractor()
        
        # Define test images
        self.background_image_path = self.data_dir / "poker_game_background.png"
        self.debug_frames = [
            "debug_frame_1.png",
            "debug_frame_2.png", 
            "debug_frame_3.png",
            "debug_frame_4.png"
        ]
        
        logger.info("TextEnhancementProcessor initialized")
    
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
    
    def adaptive_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Adaptive Histogram Equalization (CLAHE) for better contrast.
        
        Especially effective for semi-transparent text by enhancing local contrast.
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def unsharp_masking(self, image: np.ndarray, strength: float = 1.5) -> np.ndarray:
        """
        Apply unsharp masking to enhance text edges.
        
        Creates a high-pass filter effect that makes text appear sharper.
        """
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (0, 0), 2.0)
        
        # Calculate unsharp mask
        unsharp_mask = cv2.addWeighted(gray, 1.0 + strength, blurred, -strength, 0)
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(unsharp_mask, cv2.COLOR_GRAY2BGR)
        
        return enhanced
    
    def morphological_text_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Use morphological operations to enhance text structure.
        
        Connects broken text regions and removes noise.
        """
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
    
    def edge_preserving_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Apply edge-preserving filter to smooth background while preserving text edges.
        
        Reduces noise in background areas while keeping text sharp.
        """
        # Apply edge-preserving filter
        enhanced = cv2.edgePreservingFilter(image, flags=1, sigma_s=50, sigma_r=0.4)
        
        return enhanced
    
    def gamma_correction(self, image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
        """
        Apply gamma correction to improve text visibility.
        
        Gamma > 1 brightens the image, making dark text more visible.
        """
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        
        # Apply gamma correction
        enhanced = cv2.LUT(image, table)
        
        return enhanced
    
    def contrast_stretching(self, image: np.ndarray) -> np.ndarray:
        """
        Apply contrast stretching to maximize dynamic range.
        
        Maps the darkest and brightest pixels to 0 and 255 respectively.
        """
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
    
    def bilateral_filtering(self, image: np.ndarray) -> np.ndarray:
        """
        Apply bilateral filtering to reduce noise while preserving edges.
        
        Good for cleaning up semi-transparent text artifacts.
        """
        enhanced = cv2.bilateralFilter(image, 9, 75, 75)
        
        return enhanced
    
    def text_specific_sharpening(self, image: np.ndarray) -> np.ndarray:
        """
        Apply text-specific sharpening using high-pass filter.
        
        Enhances text edges while minimizing background artifacts.
        """
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
    
    def multi_scale_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Apply multi-scale enhancement for different text sizes.
        
        Combines multiple enhancement techniques at different scales.
        """
        # Apply different enhancements at different scales
        enhanced1 = self.adaptive_histogram_equalization(image)
        enhanced2 = self.unsharp_masking(image, strength=1.0)
        enhanced3 = self.gamma_correction(image, gamma=1.1)
        
        # Combine enhancements
        combined = cv2.addWeighted(enhanced1, 0.4, enhanced2, 0.3, 0)
        combined = cv2.addWeighted(combined, 0.7, enhanced3, 0.3, 0)
        
        return combined
    
    def transparency_aware_enhancement(self, image: np.ndarray, background: np.ndarray) -> np.ndarray:
        """
        Apply transparency-aware enhancement using background subtraction.
        
        First extracts transparent elements, then enhances them specifically.
        """
        # Extract transparent elements using alpha extraction
        result = self.subtractor.alpha_extraction(image, background)
        transparent_elements = result.foreground_image
        
        # Apply enhancement to transparent elements only
        enhanced_elements = self.multi_scale_enhancement(transparent_elements)
        
        # Create mask for transparent areas
        mask = cv2.cvtColor(result.foreground_mask, cv2.COLOR_GRAY2BGR) / 255.0
        
        # Blend enhanced elements with original background
        enhanced = image.astype(np.float32) * (1 - mask) + enhanced_elements.astype(np.float32) * mask
        
        return enhanced.astype(np.uint8)
    
    def advanced_text_enhancement(self, image: np.ndarray, background: np.ndarray) -> np.ndarray:
        """
        Apply advanced text enhancement pipeline.
        
        Combines multiple techniques in an optimal sequence for text legibility.
        """
        # Step 1: Transparency-aware enhancement
        enhanced = self.transparency_aware_enhancement(image, background)
        
        # Step 2: Adaptive histogram equalization
        enhanced = self.adaptive_histogram_equalization(enhanced)
        
        # Step 3: Edge-preserving filtering
        enhanced = self.edge_preserving_filter(enhanced)
        
        # Step 4: Text-specific sharpening
        enhanced = self.text_specific_sharpening(enhanced)
        
        # Step 5: Gamma correction
        enhanced = self.gamma_correction(enhanced, gamma=1.15)
        
        # Step 6: Morphological enhancement
        enhanced = self.morphological_text_enhancement(enhanced)
        
        return enhanced
    
    def test_all_enhancement_methods(self) -> Dict[str, List[EnhancementResult]]:
        """Test all enhancement methods on all debug frames."""
        logger.info("Testing all text enhancement methods...")
        
        # Load images
        background, debug_frames = self.load_images()
        
        # Define enhancement methods
        enhancement_methods = {
            "adaptive_histogram_equalization": self.adaptive_histogram_equalization,
            "unsharp_masking": lambda img: self.unsharp_masking(img, 1.5),
            "morphological_text_enhancement": self.morphological_text_enhancement,
            "edge_preserving_filter": self.edge_preserving_filter,
            "gamma_correction": lambda img: self.gamma_correction(img, 1.2),
            "contrast_stretching": self.contrast_stretching,
            "bilateral_filtering": self.bilateral_filtering,
            "text_specific_sharpening": self.text_specific_sharpening,
            "multi_scale_enhancement": self.multi_scale_enhancement,
            "transparency_aware_enhancement": lambda img: self.transparency_aware_enhancement(img, background),
            "advanced_text_enhancement": lambda img: self.advanced_text_enhancement(img, background)
        }
        
        all_results = {}
        
        # Test each method
        for method_name, method_func in enhancement_methods.items():
            logger.info(f"\n=== Testing {method_name} ===")
            method_results = []
            
            for i, debug_frame in enumerate(debug_frames, 1):
                logger.info(f"Processing Frame {i}...")
                
                try:
                    import time
                    start_time = time.time()
                    
                    # Apply enhancement
                    enhanced_image = method_func(debug_frame)
                    
                    processing_time = time.time() - start_time
                    
                    # Calculate confidence score (simplified)
                    confidence_score = self._calculate_enhancement_confidence(debug_frame, enhanced_image)
                    
                    # Create result
                    result = EnhancementResult(
                        method_name=method_name,
                        enhanced_image=enhanced_image,
                        confidence_score=confidence_score,
                        processing_time=processing_time
                    )
                    
                    method_results.append(result)
                    
                    logger.info(f"✓ Frame {i}: confidence={confidence_score:.3f}, time={processing_time:.3f}s")
                    
                except Exception as e:
                    logger.error(f"✗ Frame {i} failed: {e}")
                    continue
            
            all_results[method_name] = method_results
        
        return all_results
    
    def _calculate_enhancement_confidence(self, original: np.ndarray, enhanced: np.ndarray) -> float:
        """
        Calculate confidence score for enhancement quality.
        
        Higher scores indicate better text enhancement.
        """
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
    
    def save_enhancement_results(self, all_results: Dict[str, List[EnhancementResult]]) -> None:
        """Save enhancement results and analysis."""
        logger.info("Saving enhancement results...")
        
        # Create enhancement directory
        enhancement_dir = self.output_dir / "text_enhancement"
        enhancement_dir.mkdir(exist_ok=True)
        
        for method_name, results in all_results.items():
            logger.info(f"Saving results for {method_name}...")
            
            # Create method directory
            method_dir = enhancement_dir / method_name
            method_dir.mkdir(exist_ok=True)
            
            for i, result in enumerate(results, 1):
                # Save enhanced image
                cv2.imwrite(str(method_dir / f"frame_{i}_enhanced.png"), result.enhanced_image)
                
                # Create comparison image
                # Load original frame for comparison
                frame_path = self.data_dir / f"debug_frame_{i}.png"
                if frame_path.exists():
                    original = cv2.imread(str(frame_path))
                    if original is not None:
                        comparison = np.hstack([original, result.enhanced_image])
                        cv2.imwrite(str(method_dir / f"frame_{i}_comparison.png"), comparison)
        
        # Generate enhancement report
        self._generate_enhancement_report(all_results, enhancement_dir)
    
    def _generate_enhancement_report(self, all_results: Dict[str, List[EnhancementResult]], 
                                   output_dir: Path) -> None:
        """Generate comprehensive enhancement report."""
        report_path = output_dir / "enhancement_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("Text Enhancement Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Summary statistics
            f.write("Method Performance Summary:\n")
            f.write("-" * 40 + "\n")
            
            for method_name, results in all_results.items():
                if results:
                    avg_confidence = np.mean([r.confidence_score for r in results])
                    avg_time = np.mean([r.processing_time for r in results])
                    f.write(f"{method_name:30s}: Confidence={avg_confidence:.3f}, Time={avg_time:.3f}s\n")
            
            # Best methods
            f.write(f"\nBest Enhancement Methods:\n")
            f.write("-" * 30 + "\n")
            
            method_scores = []
            for method_name, results in all_results.items():
                if results:
                    avg_confidence = np.mean([r.confidence_score for r in results])
                    method_scores.append((method_name, avg_confidence))
            
            method_scores.sort(key=lambda x: x[1], reverse=True)
            
            for i, (method_name, score) in enumerate(method_scores[:5], 1):
                f.write(f"{i}. {method_name}: {score:.3f}\n")
        
        logger.info(f"Enhancement report saved: {report_path}")


def main():
    """Main text enhancement execution."""
    logger.info("Starting Text Enhancement Processing")
    logger.info("=" * 50)
    
    try:
        # Create processor
        processor = TextEnhancementProcessor()
        
        # Test all enhancement methods
        all_results = processor.test_all_enhancement_methods()
        
        # Save results
        processor.save_enhancement_results(all_results)
        
        logger.info(f"\n🎯 Text enhancement completed!")
        logger.info(f"✅ Generated enhanced images for all methods!")
        
    except Exception as e:
        logger.error(f"Text enhancement failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
