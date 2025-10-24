
import numpy as np
import logging
from typing import Dict

from capture.region_loader import RegionLoader
from capture.screen_capture import ScreenCapture

logger = logging.getLogger(__name__)

class RegionExtractor:
    """
    Orchestrates screen capture and extraction of all defined poker regions.

    This class acts as a high-level manager, handling the instantiation of
    ScreenCapture and RegionLoader to provide a simple interface for getting
    all region images from a single screen capture.

    It is designed to be used as a context manager to ensure proper
    resource cleanup.
    """
    def __init__(self):
        """Initializes the RegionExtractor."""
        self.screen_capture = ScreenCapture()
        self.region_loader = RegionLoader()
        self.regions = self.region_loader.load_regions()
        logger.info("RegionExtractor initialized with %d regions.", len(self.regions))

    def extract_all_regions(self) -> Dict[str, np.ndarray]:
        """
        Captures a single frame and extracts all defined regions from it.

        Returns:
            A dictionary mapping each region's name (str) to its
            corresponding image data (np.ndarray).
        """
        frame = self.screen_capture.capture_frame()
        
        extracted_regions = {}
        for name, region_model in self.regions.items():
            x, y, w, h = region_model.x, region_model.y, region_model.width, region_model.height
            region_img = frame[y:y+h, x:x+w]
            extracted_regions[name] = region_img
            
        logger.debug("Extracted %d regions from new frame.", len(extracted_regions))
        return extracted_regions

    def __enter__(self):
        """Enter the runtime context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context and close resources."""
        self.screen_capture.close()
