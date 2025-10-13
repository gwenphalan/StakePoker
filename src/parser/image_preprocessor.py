
#!/usr/bin/env python3
"""
Image preprocessing module for OCR optimization.

Provides various preprocessing methods to improve OCR accuracy by enhancing
text regions and reducing noise. All methods are configurable via settings.

Usage:
    from src.parser.image_preprocessor import ImagePreprocessor
    
    preprocessor = ImagePreprocessor()
    
    # Get all preprocessing variants
    variants = preprocessor.preprocess_all_methods(image)
    for method_name, processed_image in variants:
        # Try OCR on each variant
        result = ocr_engine.extract(processed_image)
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple

from src.config.settings import Settings

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Image preprocessing for OCR optimization.
    
    Applies various preprocessing techniques to improve text recognition.
    All methods and parameters are configurable via settings.
    """
    
    def __init__(self):
        """Initialize preprocessor and create default settings."""
        self.settings = Settings()
        
        # Create preprocessing settings with defaults
        self.settings.create("parser.preprocessing.enabled", default=True)
        self.settings.create("parser.preprocessing.methods", default=[
            "original",
            "grayscale",
            "threshold",
            "adaptive_threshold",
            "otsu_threshold",
            "denoise",
            "contrast",
            "sharpen"
        ])
        
        # Threshold values
        self.settings.create("parser.preprocessing.threshold_value", default=127)
        self.settings.create("parser.preprocessing.threshold_max", default=255)
        
        # Adaptive threshold
        self.settings.create("parser.preprocessing.adaptive_block_size", default=11)
        self.settings.create("parser.preprocessing.adaptive_c", default=2)
        
        # Contrast enhancement (CLAHE)
        self.settings.create("parser.preprocessing.clahe_clip_limit", default=2.0)
        self.settings.create("parser.preprocessing.clahe_tile_size", default=8)
        
        # Morphology kernel size
        self.settings.create("parser.preprocessing.morph_kernel_size", default=2)
        
        # Denoise strength
        self.settings.create("parser.preprocessing.denoise_strength", default=10)
        
        # Load settings
        self.enabled = self.settings.get("parser.preprocessing.enabled")
        self.methods = self.settings.get("parser.preprocessing.methods")
        
        logger.info(f"ImagePreprocessor initialized (enabled={self.enabled}, methods={len(self.methods)})")
    
    def preprocess_all_methods(self, image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """
        Apply all enabled preprocessing methods to an image.
        
        Args:
            image: Input image (BGR or grayscale)
        
        Returns:
            List of (method_name, processed_image) tuples
        
        Example:
            variants = preprocessor.preprocess_all_methods(image)
            for method_name, processed_img in variants:
                result = ocr_engine.extract(processed_img)
        """
        if not self.enabled:
            logger.debug("Preprocessing disabled, returning original only")
            return [("original", image.copy())]
        
        results = []
        
        # Always include original as first option
        if "original" in self.methods:
            results.append(("original", image.copy()))
            logger.debug("Added preprocessing variant: original")
        
        # Apply each enabled method
        for method_name in self.methods:
            if method_name == "original":
                continue  # Already added
            
            try:
                # Get the preprocessing method
                method = getattr(self, method_name, None)
                if method is None:
                    logger.warning(f"Unknown preprocessing method: {method_name}")
                    continue
                
                # Apply preprocessing
                processed = method(image)
                results.append((method_name, processed))
                logger.debug(f"Added preprocessing variant: {method_name}")
                
            except Exception as e:
                logger.error(f"Failed to apply preprocessing method '{method_name}': {e}")
                continue
        
        logger.debug(f"Generated {len(results)} preprocessing variants")
        return results
    
    def grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale.
        
        Args:
            image: Input image (BGR or grayscale)
        
        Returns:
            Grayscale image
        """
        if len(image.shape) == 2:
            # Already grayscale
            return image.copy()
        
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def threshold(self, image: np.ndarray) -> np.ndarray:
        """
        Apply binary threshold.
        
        Args:
            image: Input image
        
        Returns:
            Binary thresholded image
        """
        gray = self.grayscale(image)
        
        threshold_value = self.settings.get("parser.preprocessing.threshold_value")
        threshold_max = self.settings.get("parser.preprocessing.threshold_max")
        
        _, binary = cv2.threshold(gray, threshold_value, threshold_max, cv2.THRESH_BINARY)
        return binary
    
    def adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive binary threshold.
        
        Better for images with varying lighting conditions.
        
        Args:
            image: Input image
        
        Returns:
            Adaptive thresholded image
        """
        gray = self.grayscale(image)
        
        block_size = self.settings.get("parser.preprocessing.adaptive_block_size")
        c_value = self.settings.get("parser.preprocessing.adaptive_c")
        
        # Ensure block_size is odd
        if block_size % 2 == 0:
            block_size += 1
        
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c_value
        )
        return binary
    
    def otsu_threshold(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Otsu's binarization.
        
        Automatically determines optimal threshold value.
        
        Args:
            image: Input image
        
        Returns:
            Otsu thresholded image
        """
        gray = self.grayscale(image)
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Remove noise from image.
        
        Uses Non-local Means Denoising algorithm.
        
        Args:
            image: Input image
        
        Returns:
            Denoised image
        """
        strength = self.settings.get("parser.preprocessing.denoise_strength")
        
        if len(image.shape) == 2:
            # Grayscale
            return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)
        else:
            # Color
            return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Args:
            image: Input image
        
        Returns:
            Contrast enhanced image
        """
        gray = self.grayscale(image)
        
        clip_limit = self.settings.get("parser.preprocessing.clahe_clip_limit")
        tile_size = self.settings.get("parser.preprocessing.clahe_tile_size")
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        enhanced = clahe.apply(gray)
        return enhanced
    
    def sharpen(self, image: np.ndarray) -> np.ndarray:
        """
        Sharpen image edges.
        
        Args:
            image: Input image
        
        Returns:
            Sharpened image
        """
        # Sharpening kernel
        kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])
        
        if len(image.shape) == 2:
            # Grayscale
            return cv2.filter2D(image, -1, kernel)
        else:
            # Apply to each channel
            sharpened = cv2.filter2D(image, -1, kernel)
            return sharpened
    
    def dilate(self, image: np.ndarray) -> np.ndarray:
        """
        Apply morphological dilation.
        
        Useful for connecting broken text regions.
        
        Args:
            image: Input image
        
        Returns:
            Dilated image
        """
        kernel_size = self.settings.get("parser.preprocessing.morph_kernel_size")
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        return cv2.dilate(image, kernel, iterations=1)
    
    def erode(self, image: np.ndarray) -> np.ndarray:
        """
        Apply morphological erosion.
        
        Useful for removing small noise.
        
        Args:
            image: Input image
        
        Returns:
            Eroded image
        """
        kernel_size = self.settings.get("parser.preprocessing.morph_kernel_size")
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        return cv2.erode(image, kernel, iterations=1)
    
    def invert(self, image: np.ndarray) -> np.ndarray:
        """
        Invert image colors.
        
        Useful when text is lighter than background.
        
        Args:
            image: Input image
        
        Returns:
            Inverted image
        """
        return cv2.bitwise_not(image)
    
    def contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Alias for enhance_contrast for backward compatibility.
        
        Args:
            image: Input image
        
        Returns:
            Contrast enhanced image
        """
        return self.enhance_contrast(image)
