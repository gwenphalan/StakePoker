import mss
import numpy as np
import cv2
import logging
from src.capture.monitor_config import MonitorConfig

logger = logging.getLogger(__name__)

class ScreenCapture:
    """
    Handles on-demand screen capture of a specific monitor.
    Uses MonitorConfig to determine which monitor to capture.
    """
    def __init__(self):
        """Initializes the ScreenCapture instance."""
        self.monitor_config = MonitorConfig()
        self.monitor_bounds = self.monitor_config.get_monitor_bounds()  # cache this
        self.sct = mss.mss()
        logger.info("ScreenCapture initialized.")
    
    def close(self):
        """Closes the screen capture instance."""
        self.sct.close()
        logger.info("ScreenCapture closed.")
    
    def capture_frame(self) -> np.ndarray:
        """
        Captures a single frame from the configured poker monitor.
        
        Returns:
            np.ndarray: Captured frame in BGR format
            
        Raises:
            RuntimeError: If capture fails
        """
        try:
            logger.debug(f"Capturing frame from monitor bounds: {self.monitor_bounds}")
            sct_img = self.sct.grab(self.monitor_bounds)
            frame = np.array(sct_img)
            
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            logger.debug(f"Frame captured successfully. Shape: {frame.shape}")
            return frame
            
        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            raise RuntimeError(f"Screen capture failed: {e}")
    
    def __enter__(self):
        """Enter the runtime context."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context and close resources."""
        self.close()
        logger.info("ScreenCapture resources released.")
