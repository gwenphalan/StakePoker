#!/usr/bin/env python3
"""
Visual testing script for parser components.

This script provides an interactive testing interface for validating parser accuracy
using random frames from poker gameplay videos. It allows manual validation of parser
results with keybinds and stores results for analysis.

Usage:
    python test/parser/visual_test.py --parser card --region card_1 --tests 20
    python test/parser/visual_test.py --parser money --region pot --tests 10
    python test/parser/visual_test.py --list-parsers
    python test/parser/visual_test.py --stats card
"""

import argparse
import logging
import sys
import numpy as np
import cv2
from pathlib import Path

# Add src and test/parser to path for imports
project_root = Path(__file__).parent.parent.parent
src_path = str(project_root / "src")
test_parser_path = str(Path(__file__).parent)

# Add src path FIRST so src.parser takes precedence over test.parser
if src_path not in sys.path:
    sys.path.insert(0, src_path)
if test_parser_path not in sys.path:
    sys.path.insert(0, test_parser_path)

from test_utils import (
    ParserTestBase, VideoFrameExtractor, 
    ParserTestResultManager, VisualTestValidator, VisualTestSession, ParserTestResult
)
from capture.region_loader import RegionLoader

# Import all parsers - use importlib to avoid conflicts with test/parser package
import importlib.util

def load_parser_module(module_name):
    """Load a parser module from src/parser/"""
    module_path = project_root / "src" / "parser" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"parser.{module_name}", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

card_parser_module = load_parser_module("card_parser")
dealer_detector_module = load_parser_module("dealer_detector")
hand_id_parser_module = load_parser_module("hand_id_parser")
money_parser_module = load_parser_module("money_parser")
name_parser_module = load_parser_module("name_parser")
status_parser_module = load_parser_module("status_parser")
table_info_parser_module = load_parser_module("table_info_parser")
timer_detector_module = load_parser_module("timer_detector")
transparency_detector_module = load_parser_module("transparency_detector")

CardParser = card_parser_module.CardParser
CardResult = card_parser_module.CardResult
DealerDetector = dealer_detector_module.DealerDetector
DealerResult = dealer_detector_module.DealerResult
HandIdParser = hand_id_parser_module.HandIdParser
HandIdResult = hand_id_parser_module.HandIdResult
MoneyParser = money_parser_module.MoneyParser
AmountResult = money_parser_module.AmountResult
NameParser = name_parser_module.NameParser
NameResult = name_parser_module.NameResult
StatusParser = status_parser_module.StatusParser
StatusResult = status_parser_module.StatusResult
TableInfoParser = table_info_parser_module.TableInfoParser
TableInfoResult = table_info_parser_module.TableInfoResult
TimerDetector = timer_detector_module.TimerDetector
TimerResult = timer_detector_module.TimerResult
TransparencyDetector = transparency_detector_module.TransparencyDetector
TransparencyResult = transparency_detector_module.TransparencyResult

logger = logging.getLogger(__name__)


class CardParserTest(ParserTestBase):
    """Test class for CardParser."""
    
    def _parse_region(self, parser_instance, region_image):
        """Parse card region."""
        return parser_instance.parse_card(region_image)
    
    def _extract_result_data(self, parsed_result):
        """Extract data from CardResult."""
        if isinstance(parsed_result, CardResult):
            return f"{parsed_result.rank}{parsed_result.suit}", parsed_result.confidence, "card_parsing"
        return None, 0.0, "none"
    
    def _get_debug_info(self, parser_instance, region_image, parsed_result):
        """Collect comprehensive debug info for card parsing."""
        debug_info = {}
        
        try:
            # Basic image properties
            debug_info["image_properties"] = {
                "shape": list(region_image.shape),
                "dtype": str(region_image.dtype),
                "min_value": float(region_image.min()),
                "max_value": float(region_image.max()),
                "mean_value": float(region_image.mean()),
                "std_value": float(region_image.std()),
                "median_value": float(np.median(region_image)),
                "total_pixels": int(region_image.size),
                "non_zero_pixels": int(np.count_nonzero(region_image)),
                "zero_pixels": int(np.count_nonzero(region_image == 0)),
                "white_pixels": int(np.count_nonzero(region_image == 255)),
                "black_pixels": int(np.count_nonzero(region_image == 0))
            }
            
            # Color analysis
            if len(region_image.shape) == 3:
                debug_info["color_analysis"] = {
                    "b_channel": {
                        "mean": float(region_image[:,:,0].mean()),
                        "std": float(region_image[:,:,0].std()),
                        "min": float(region_image[:,:,0].min()),
                        "max": float(region_image[:,:,0].max())
                    },
                    "g_channel": {
                        "mean": float(region_image[:,:,1].mean()),
                        "std": float(region_image[:,:,1].std()),
                        "min": float(region_image[:,:,1].min()),
                        "max": float(region_image[:,:,1].max())
                    },
                    "r_channel": {
                        "mean": float(region_image[:,:,2].mean()),
                        "std": float(region_image[:,:,2].std()),
                        "min": float(region_image[:,:,2].min()),
                        "max": float(region_image[:,:,2].max())
                    }
                }
                
                # HSV analysis
                hsv = cv2.cvtColor(region_image, cv2.COLOR_BGR2HSV)
                debug_info["hsv_analysis"] = {
                    "h_channel": {
                        "mean": float(hsv[:,:,0].mean()),
                        "std": float(hsv[:,:,0].std()),
                        "min": float(hsv[:,:,0].min()),
                        "max": float(hsv[:,:,0].max())
                    },
                    "s_channel": {
                        "mean": float(hsv[:,:,1].mean()),
                        "std": float(hsv[:,:,1].std()),
                        "min": float(hsv[:,:,1].min()),
                        "max": float(hsv[:,:,1].max())
                    },
                    "v_channel": {
                        "mean": float(hsv[:,:,2].mean()),
                        "std": float(hsv[:,:,2].std()),
                        "min": float(hsv[:,:,2].min()),
                        "max": float(hsv[:,:,2].max())
                    }
                }
            
            # Edge analysis
            gray = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY) if len(region_image.shape) == 3 else region_image
            edges = cv2.Canny(gray, 50, 150)
            edge_pixels = np.count_nonzero(edges)
            
            debug_info["edge_analysis"] = {
                "edge_pixel_count": int(edge_pixels),
                "edge_density": float(edge_pixels / gray.size),
                "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                "sobel_mean_magnitude": float(np.mean(cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)))
            }
            
            # Histogram analysis
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            debug_info["histogram_analysis"] = {
                "histogram_mean": float(np.mean(hist)),
                "histogram_std": float(np.std(hist)),
                "histogram_max": float(np.max(hist)),
                "histogram_min": float(np.min(hist)),
                "peak_count": int(len(cv2.findNonZero(hist > np.mean(hist) + np.std(hist)) or [])),
                "peak_indices": [int(i) for i in np.where(hist > np.mean(hist) + np.std(hist))[0]]
            }
            
            # Texture analysis
            kernel = np.ones((3,3), np.float32) / 9
            local_variance = cv2.filter2D(gray.astype(np.float32)**2, -1, kernel) - cv2.filter2D(gray.astype(np.float32), -1, kernel)**2
            debug_info["texture_analysis"] = {
                "local_variance_mean": float(np.mean(local_variance)),
                "local_variance_std": float(np.std(local_variance)),
                "local_variance_max": float(np.max(local_variance)),
                "local_variance_min": float(np.min(local_variance))
            }
            
            # OCR debug info if available
            if hasattr(parser_instance, 'ocr_engine'):
                try:
                    text, ocr_confidence, method = parser_instance.ocr_engine.extract_text(region_image)
                    debug_info["ocr_debug"] = {
                        "extracted_text": text,
                        "ocr_confidence": ocr_confidence,
                        "preprocessing_method": method
                    }
                except Exception as e:
                    debug_info["ocr_debug"] = {"error": str(e)}
            
            # Parser settings
            if hasattr(parser_instance, 'settings'):
                debug_info["parser_settings"] = {
                    "min_confidence": getattr(parser_instance, 'min_confidence', None),
                    "valid_ranks": getattr(parser_instance, 'valid_ranks', None),
                    "valid_suits": getattr(parser_instance, 'valid_suits', None)
                }
            
            # Parsed result details
            if parsed_result:
                debug_info["parsed_result"] = {
                    "rank": parsed_result.rank if hasattr(parsed_result, 'rank') else None,
                    "suit": parsed_result.suit if hasattr(parsed_result, 'suit') else None,
                    "confidence": parsed_result.confidence if hasattr(parsed_result, 'confidence') else None
                }
                
        except Exception as e:
            debug_info['debug_error'] = str(e)
            import traceback
            debug_info['debug_traceback'] = traceback.format_exc()
        
        return debug_info


class DealerDetectorTest(ParserTestBase):
    """Test class for DealerDetector."""
    
    def _parse_region(self, parser_instance, region_image):
        """Parse dealer region."""
        return parser_instance.detect_dealer_button(region_image)
    
    def _extract_result_data(self, parsed_result):
        """Extract data from DealerResult."""
        if isinstance(parsed_result, DealerResult):
            return parsed_result.has_dealer, parsed_result.confidence, "dealer_detection"
        return None, 0.0, "none"
    
    def _get_debug_info(self, parser_instance, region_image, parsed_result):
        """Collect comprehensive debug info for dealer detection."""
        debug_info = {}
        
        try:
            # Basic image properties
            debug_info["image_properties"] = {
                "shape": list(region_image.shape),
                "dtype": str(region_image.dtype),
                "min_value": float(region_image.min()),
                "max_value": float(region_image.max()),
                "mean_value": float(region_image.mean()),
                "std_value": float(region_image.std()),
                "median_value": float(np.median(region_image)),
                "total_pixels": int(region_image.size),
                "non_zero_pixels": int(np.count_nonzero(region_image)),
                "zero_pixels": int(np.count_nonzero(region_image == 0)),
                "white_pixels": int(np.count_nonzero(region_image == 255)),
                "black_pixels": int(np.count_nonzero(region_image == 0))
            }
            
            # Color analysis
            if len(region_image.shape) == 3:
                debug_info["color_analysis"] = {
                    "b_channel": {
                        "mean": float(region_image[:,:,0].mean()),
                        "std": float(region_image[:,:,0].std()),
                        "min": float(region_image[:,:,0].min()),
                        "max": float(region_image[:,:,0].max())
                    },
                    "g_channel": {
                        "mean": float(region_image[:,:,1].mean()),
                        "std": float(region_image[:,:,1].std()),
                        "min": float(region_image[:,:,1].min()),
                        "max": float(region_image[:,:,1].max())
                    },
                    "r_channel": {
                        "mean": float(region_image[:,:,2].mean()),
                        "std": float(region_image[:,:,2].std()),
                        "min": float(region_image[:,:,2].min()),
                        "max": float(region_image[:,:,2].max())
                    }
                }
                
                # HSV analysis
                hsv = cv2.cvtColor(region_image, cv2.COLOR_BGR2HSV)
                debug_info["hsv_analysis"] = {
                    "h_channel": {
                        "mean": float(hsv[:,:,0].mean()),
                        "std": float(hsv[:,:,0].std()),
                        "min": float(hsv[:,:,0].min()),
                        "max": float(hsv[:,:,0].max())
                    },
                    "s_channel": {
                        "mean": float(hsv[:,:,1].mean()),
                        "std": float(hsv[:,:,1].std()),
                        "min": float(hsv[:,:,1].min()),
                        "max": float(hsv[:,:,1].max())
                    },
                    "v_channel": {
                        "mean": float(hsv[:,:,2].mean()),
                        "std": float(hsv[:,:,2].std()),
                        "min": float(hsv[:,:,2].min()),
                        "max": float(hsv[:,:,2].max())
                    }
                }
            
            # Edge analysis
            gray = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY) if len(region_image.shape) == 3 else region_image
            edges = cv2.Canny(gray, 50, 150)
            edge_pixels = np.count_nonzero(edges)
            
            debug_info["edge_analysis"] = {
                "edge_pixel_count": int(edge_pixels),
                "edge_density": float(edge_pixels / gray.size),
                "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                "sobel_mean_magnitude": float(np.mean(cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)))
            }
            
            # Histogram analysis
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            debug_info["histogram_analysis"] = {
                "histogram_mean": float(np.mean(hist)),
                "histogram_std": float(np.std(hist)),
                "histogram_max": float(np.max(hist)),
                "histogram_min": float(np.min(hist)),
                "peak_count": int(len(cv2.findNonZero(hist > np.mean(hist) + np.std(hist)) or [])),
                "peak_indices": [int(i) for i in np.where(hist > np.mean(hist) + np.std(hist))[0]]
            }
            
            # Texture analysis
            kernel = np.ones((3,3), np.float32) / 9
            local_variance = cv2.filter2D(gray.astype(np.float32)**2, -1, kernel) - cv2.filter2D(gray.astype(np.float32), -1, kernel)**2
            debug_info["texture_analysis"] = {
                "local_variance_mean": float(np.mean(local_variance)),
                "local_variance_std": float(np.std(local_variance)),
                "local_variance_max": float(np.max(local_variance)),
                "local_variance_min": float(np.min(local_variance))
            }
            
            # Parser settings
            if hasattr(parser_instance, 'settings'):
                debug_info["parser_settings"] = {
                    "min_confidence": getattr(parser_instance, 'min_confidence', None),
                    "dealer_color_ranges": getattr(parser_instance, 'dealer_color_ranges', None)
                }
            
            # Parsed result details
            if parsed_result:
                debug_info["parsed_result"] = {
                    "has_dealer": parsed_result.has_dealer if hasattr(parsed_result, 'has_dealer') else None,
                    "confidence": parsed_result.confidence if hasattr(parsed_result, 'confidence') else None
                }
                
        except Exception as e:
            debug_info['debug_error'] = str(e)
            import traceback
            debug_info['debug_traceback'] = traceback.format_exc()
        
        return debug_info


class HandIdParserTest(ParserTestBase):
    """Test class for HandIdParser."""
    
    def _parse_region(self, parser_instance, region_image):
        """Parse hand ID region."""
        return parser_instance.parse_hand_id(region_image)
    
    def _extract_result_data(self, parsed_result):
        """Extract data from HandIdResult."""
        if isinstance(parsed_result, HandIdResult):
            return parsed_result.hand_id, parsed_result.confidence, "hand_id_parsing"
        return None, 0.0, "none"
    
    def _get_debug_info(self, parser_instance, region_image, parsed_result):
        """Collect comprehensive debug info for hand ID parsing."""
        debug_info = {}
        
        try:
            # Basic image properties
            debug_info["image_properties"] = {
                "shape": list(region_image.shape),
                "dtype": str(region_image.dtype),
                "min_value": float(region_image.min()),
                "max_value": float(region_image.max()),
                "mean_value": float(region_image.mean()),
                "std_value": float(region_image.std()),
                "median_value": float(np.median(region_image)),
                "total_pixels": int(region_image.size),
                "non_zero_pixels": int(np.count_nonzero(region_image)),
                "zero_pixels": int(np.count_nonzero(region_image == 0)),
                "white_pixels": int(np.count_nonzero(region_image == 255)),
                "black_pixels": int(np.count_nonzero(region_image == 0))
            }
            
            # Color analysis
            if len(region_image.shape) == 3:
                debug_info["color_analysis"] = {
                    "b_channel": {
                        "mean": float(region_image[:,:,0].mean()),
                        "std": float(region_image[:,:,0].std()),
                        "min": float(region_image[:,:,0].min()),
                        "max": float(region_image[:,:,0].max())
                    },
                    "g_channel": {
                        "mean": float(region_image[:,:,1].mean()),
                        "std": float(region_image[:,:,1].std()),
                        "min": float(region_image[:,:,1].min()),
                        "max": float(region_image[:,:,1].max())
                    },
                    "r_channel": {
                        "mean": float(region_image[:,:,2].mean()),
                        "std": float(region_image[:,:,2].std()),
                        "min": float(region_image[:,:,2].min()),
                        "max": float(region_image[:,:,2].max())
                    }
                }
                
                # HSV analysis
                hsv = cv2.cvtColor(region_image, cv2.COLOR_BGR2HSV)
                debug_info["hsv_analysis"] = {
                    "h_channel": {
                        "mean": float(hsv[:,:,0].mean()),
                        "std": float(hsv[:,:,0].std()),
                        "min": float(hsv[:,:,0].min()),
                        "max": float(hsv[:,:,0].max())
                    },
                    "s_channel": {
                        "mean": float(hsv[:,:,1].mean()),
                        "std": float(hsv[:,:,1].std()),
                        "min": float(hsv[:,:,1].min()),
                        "max": float(hsv[:,:,1].max())
                    },
                    "v_channel": {
                        "mean": float(hsv[:,:,2].mean()),
                        "std": float(hsv[:,:,2].std()),
                        "min": float(hsv[:,:,2].min()),
                        "max": float(hsv[:,:,2].max())
                    }
                }
            
            # Edge analysis
            gray = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY) if len(region_image.shape) == 3 else region_image
            edges = cv2.Canny(gray, 50, 150)
            edge_pixels = np.count_nonzero(edges)
            
            debug_info["edge_analysis"] = {
                "edge_pixel_count": int(edge_pixels),
                "edge_density": float(edge_pixels / gray.size),
                "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                "sobel_mean_magnitude": float(np.mean(cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)))
            }
            
            # Histogram analysis
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            debug_info["histogram_analysis"] = {
                "histogram_mean": float(np.mean(hist)),
                "histogram_std": float(np.std(hist)),
                "histogram_max": float(np.max(hist)),
                "histogram_min": float(np.min(hist)),
                "peak_count": int(len(cv2.findNonZero(hist > np.mean(hist) + np.std(hist)) or [])),
                "peak_indices": [int(i) for i in np.where(hist > np.mean(hist) + np.std(hist))[0]]
            }
            
            # Texture analysis
            kernel = np.ones((3,3), np.float32) / 9
            local_variance = cv2.filter2D(gray.astype(np.float32)**2, -1, kernel) - cv2.filter2D(gray.astype(np.float32), -1, kernel)**2
            debug_info["texture_analysis"] = {
                "local_variance_mean": float(np.mean(local_variance)),
                "local_variance_std": float(np.std(local_variance)),
                "local_variance_max": float(np.max(local_variance)),
                "local_variance_min": float(np.min(local_variance))
            }
            
            # OCR debug info if available
            if hasattr(parser_instance, 'ocr_engine'):
                try:
                    text, ocr_confidence, method = parser_instance.ocr_engine.extract_text(region_image)
                    debug_info["ocr_debug"] = {
                        "extracted_text": text,
                        "ocr_confidence": ocr_confidence,
                        "preprocessing_method": method
                    }
                except Exception as e:
                    debug_info["ocr_debug"] = {"error": str(e)}
            
            # Parser settings
            if hasattr(parser_instance, 'settings'):
                debug_info["parser_settings"] = {
                    "min_confidence": getattr(parser_instance, 'min_confidence', None),
                    "valid_patterns": getattr(parser_instance, 'valid_patterns', None)
                }
            
            # Parsed result details
            if parsed_result:
                debug_info["parsed_result"] = {
                    "hand_id": parsed_result.hand_id if hasattr(parsed_result, 'hand_id') else None,
                    "confidence": parsed_result.confidence if hasattr(parsed_result, 'confidence') else None
                }
                
        except Exception as e:
            debug_info['debug_error'] = str(e)
            import traceback
            debug_info['debug_traceback'] = traceback.format_exc()
        
        return debug_info


class MoneyParserTest(ParserTestBase):
    """Test class for MoneyParser."""
    
    def _parse_region(self, parser_instance, region_image):
        """Parse money region."""
        results = parser_instance.parse_amounts(region_image)
        return results[0] if results else None
    
    def _extract_result_data(self, parsed_result):
        """Extract data from AmountResult."""
        if isinstance(parsed_result, AmountResult):
            return parsed_result.value, parsed_result.confidence, "money_parsing"
        return None, 0.0, "none"


class NameParserTest(ParserTestBase):
    """Test class for NameParser."""
    
    def _parse_region(self, parser_instance, region_image):
        """Parse name region."""
        return parser_instance.parse_player_name(region_image, 1)  # Player 1
    
    def _extract_result_data(self, parsed_result):
        """Extract data from NameResult."""
        if isinstance(parsed_result, NameResult):
            return parsed_result.name, parsed_result.confidence, "name_parsing"
        return None, 0.0, "none"


class StatusParserTest(ParserTestBase):
    """Test class for StatusParser."""
    
    def _parse_region(self, parser_instance, region_image):
        """Parse status region."""
        return parser_instance.parse_status(region_image)
    
    def _extract_result_data(self, parsed_result):
        """Extract data from StatusResult."""
        if isinstance(parsed_result, StatusResult):
            return parsed_result.status, parsed_result.confidence, "status_parsing"
        return None, 0.0, "none"


class TableInfoParserTest(ParserTestBase):
    """Test class for TableInfoParser."""
    
    def _parse_region(self, parser_instance, region_image):
        """Parse table info region."""
        return parser_instance.parse_table_info(region_image)
    
    def _extract_result_data(self, parsed_result):
        """Extract data from TableInfoResult."""
        if isinstance(parsed_result, TableInfoResult):
            return f"{parsed_result.sb}/{parsed_result.bb}", parsed_result.confidence, "table_info_parsing"
        return None, 0.0, "none"
    
    def _get_debug_info(self, parser_instance, region_image, parsed_result):
        """Get debug information for table info parsing."""
        debug_info = {
            "image_properties": {
                "shape": list(region_image.shape),
                "dtype": str(region_image.dtype),
                "min_value": float(region_image.min()),
                "max_value": float(region_image.max()),
                "mean_value": float(region_image.mean())
            },
            "parser_settings": {
                "min_stake": parser_instance.min_stake,
                "max_stake": parser_instance.max_stake
            }
        }
        
        # Add OCR debug info if available
        try:
            # Extract text to show what OCR is seeing
            text, ocr_confidence, method = parser_instance._extract_text_from_image(region_image)
            
            # Also try different preprocessing methods to see what works best
            ocr_methods = {}
            try:
                # Try different preprocessing methods using the correct ImagePreprocessor API
                preprocessor = parser_instance.ocr_engine.preprocessor
                methods = ["original", "grayscale", "threshold", "adaptive_threshold", "otsu_threshold"]
                
                for prep_method in methods:
                    try:
                        # Use the correct method names from ImagePreprocessor
                        if prep_method == "original":
                            processed_image = region_image.copy()
                        else:
                            # Get the preprocessing method from the preprocessor
                            method_func = getattr(preprocessor, prep_method, None)
                            if method_func:
                                processed_image = method_func(region_image)
                            else:
                                ocr_methods[prep_method] = {"error": f"Method {prep_method} not found"}
                                continue
                        
                        # Run OCR on the processed image
                        temp_text, temp_conf = parser_instance.ocr_engine._run_ocr(processed_image)
                        ocr_methods[prep_method] = {
                            "text": temp_text,
                            "confidence": temp_conf
                        }
                    except Exception as e:
                        ocr_methods[prep_method] = {"error": str(e)}
            except Exception as e:
                ocr_methods = {"error": f"Could not test methods: {e}"}
            
            debug_info["ocr_debug"] = {
                "extracted_text": text,
                "ocr_confidence": ocr_confidence,
                "preprocessing_method": method,
                "all_methods": ocr_methods
            }
        except Exception as e:
            debug_info["ocr_debug"] = {
                "error": str(e)
            }
        
        # Add parsed result details
        if parsed_result:
            debug_info["parsed_result"] = {
                "sb": parsed_result.sb,
                "bb": parsed_result.bb,
                "confidence": parsed_result.confidence
            }
        
        return debug_info


class TimerDetectorTest(ParserTestBase):
    """Test class for TimerDetector."""
    
    def _parse_region(self, parser_instance, region_image):
        """Parse timer region."""
        return parser_instance.detect_timer(region_image)
    
    def _extract_result_data(self, parsed_result):
        """Extract data from TimerResult."""
        if isinstance(parsed_result, TimerResult):
            return parsed_result.turn_state, parsed_result.confidence, "timer_detection"
        return None, 0.0, "none"


class TransparencyDetectorTest(ParserTestBase):
    """Test class for TransparencyDetector."""
    
    def _parse_region(self, parser_instance, region_image):
        """Parse transparency region."""
        return parser_instance.detect_transparency(region_image)
    
    def _extract_result_data(self, parsed_result):
        """Extract data from TransparencyResult."""
        if isinstance(parsed_result, TransparencyResult):
            return parsed_result.is_transparent, parsed_result.confidence, "transparency_detection"
        return None, 0.0, "none"
    
    def _get_debug_info(self, parser_instance, region_image, parsed_result):
        """Get comprehensive debug information for transparency detection."""
        debug_info = {}
        
        try:
            # Get the raw metrics used in detection
            if hasattr(parser_instance, '_analyze_transparency_metrics'):
                metrics = parser_instance._analyze_transparency_metrics(region_image)
                debug_info['metrics'] = metrics
            
            # Get settings used
            if hasattr(parser_instance, 'settings'):
                debug_info['settings'] = {
                    'contrast_threshold': parser_instance.settings.get("parser.transparency.contrast_threshold"),
                    'saturation_threshold': parser_instance.settings.get("parser.transparency.saturation_threshold"),
                    'brightness_min': parser_instance.settings.get("parser.transparency.brightness_min"),
                    'brightness_max': parser_instance.settings.get("parser.transparency.brightness_max"),
                    'require_all_criteria': parser_instance.settings.get("parser.transparency.require_all_criteria")
                }
            
            # Get comprehensive image properties
            if region_image is not None and region_image.size > 0:
                debug_info['image_properties'] = {
                    'shape': region_image.shape,
                    'dtype': str(region_image.dtype),
                    'min_value': float(np.min(region_image)),
                    'max_value': float(np.max(region_image)),
                    'mean_value': float(np.mean(region_image)),
                    'std_value': float(np.std(region_image)),
                    'median_value': float(np.median(region_image)),
                    'total_pixels': int(region_image.size),
                    'non_zero_pixels': int(np.count_nonzero(region_image)),
                    'zero_pixels': int(np.count_nonzero(region_image == 0)),
                    'white_pixels': int(np.count_nonzero(region_image >= 240)),
                    'black_pixels': int(np.count_nonzero(region_image <= 15))
                }
            
            # Color analysis (BGR channels)
            if region_image is not None and len(region_image.shape) == 3:
                b_channel = region_image[:, :, 0]
                g_channel = region_image[:, :, 1]
                r_channel = region_image[:, :, 2]
                
                debug_info['color_analysis'] = {
                    'b_channel': {
                        'mean': float(np.mean(b_channel)),
                        'std': float(np.std(b_channel)),
                        'min': float(np.min(b_channel)),
                        'max': float(np.max(b_channel))
                    },
                    'g_channel': {
                        'mean': float(np.mean(g_channel)),
                        'std': float(np.std(g_channel)),
                        'min': float(np.min(g_channel)),
                        'max': float(np.max(g_channel))
                    },
                    'r_channel': {
                        'mean': float(np.mean(r_channel)),
                        'std': float(np.std(r_channel)),
                        'min': float(np.min(r_channel)),
                        'max': float(np.max(r_channel))
                    }
                }
            
            # HSV analysis
            if region_image is not None and len(region_image.shape) == 3:
                hsv = cv2.cvtColor(region_image, cv2.COLOR_BGR2HSV)
                h_channel = hsv[:, :, 0]
                s_channel = hsv[:, :, 1]
                v_channel = hsv[:, :, 2]
                
                debug_info['hsv_analysis'] = {
                    'h_channel': {
                        'mean': float(np.mean(h_channel)),
                        'std': float(np.std(h_channel)),
                        'min': float(np.min(h_channel)),
                        'max': float(np.max(h_channel))
                    },
                    's_channel': {
                        'mean': float(np.mean(s_channel)),
                        'std': float(np.std(s_channel)),
                        'min': float(np.min(s_channel)),
                        'max': float(np.max(s_channel))
                    },
                    'v_channel': {
                        'mean': float(np.mean(v_channel)),
                        'std': float(np.std(v_channel)),
                        'min': float(np.min(v_channel)),
                        'max': float(np.max(v_channel))
                    }
                }
            
            # Edge detection analysis
            if region_image is not None:
                gray = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY) if len(region_image.shape) == 3 else region_image
                
                # Canny edge detection
                edges = cv2.Canny(gray, 50, 150)
                edge_pixels = np.count_nonzero(edges)
                
                # Laplacian variance (texture measure)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                # Sobel edge strength
                sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
                sobel_mean = float(np.mean(sobel_magnitude))
                
                debug_info['edge_analysis'] = {
                    'edge_pixel_count': int(edge_pixels),
                    'edge_density': float(edge_pixels / gray.size),
                    'laplacian_variance': float(laplacian_var),
                    'sobel_mean_magnitude': sobel_mean
                }
            
            # Histogram analysis
            if region_image is not None:
                gray = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY) if len(region_image.shape) == 3 else region_image
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                
                # Find histogram peaks and valleys
                hist_flat = hist.flatten()
                peak_indices = []
                for i in range(1, len(hist_flat) - 1):
                    if hist_flat[i] > hist_flat[i-1] and hist_flat[i] > hist_flat[i+1] and hist_flat[i] > np.mean(hist_flat) * 2:
                        peak_indices.append(i)
                
                debug_info['histogram_analysis'] = {
                    'histogram_mean': float(np.mean(hist)),
                    'histogram_std': float(np.std(hist)),
                    'histogram_max': float(np.max(hist)),
                    'histogram_min': float(np.min(hist)),
                    'peak_count': len(peak_indices),
                    'peak_indices': peak_indices[:10]  # Limit to first 10 peaks
                }
            
            # Texture analysis using Local Binary Patterns
            if region_image is not None:
                gray = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY) if len(region_image.shape) == 3 else region_image
                
                # Calculate local variance
                kernel = np.ones((3, 3), np.float32) / 9
                local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
                local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
                
                debug_info['texture_analysis'] = {
                    'local_variance_mean': float(np.mean(local_variance)),
                    'local_variance_std': float(np.std(local_variance)),
                    'local_variance_max': float(np.max(local_variance)),
                    'local_variance_min': float(np.min(local_variance))
                }
            
            # Brightness distribution analysis
            if region_image is not None:
                gray = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY) if len(region_image.shape) == 3 else region_image
                
                # Calculate brightness percentiles
                brightness_percentiles = {
                    'p10': float(np.percentile(gray, 10)),
                    'p25': float(np.percentile(gray, 25)),
                    'p50': float(np.percentile(gray, 50)),
                    'p75': float(np.percentile(gray, 75)),
                    'p90': float(np.percentile(gray, 90))
                }
                
                # Brightness range analysis
                brightness_ranges = {
                    'very_dark': int(np.count_nonzero(gray <= 50)),
                    'dark': int(np.count_nonzero((gray > 50) & (gray <= 100))),
                    'medium': int(np.count_nonzero((gray > 100) & (gray <= 150))),
                    'bright': int(np.count_nonzero((gray > 150) & (gray <= 200))),
                    'very_bright': int(np.count_nonzero(gray > 200))
                }
                
                debug_info['brightness_distribution'] = {
                    'percentiles': brightness_percentiles,
                    'ranges': brightness_ranges,
                    'range_counts': brightness_ranges
                }
            
            # Get detection stats if available
            if hasattr(parser_instance, 'get_detection_stats'):
                debug_info['detection_stats'] = parser_instance.get_detection_stats()
                
        except Exception as e:
            debug_info['debug_error'] = str(e)
            import traceback
            debug_info['debug_traceback'] = traceback.format_exc()
        
        return debug_info


# Parser registry - maps parser names to (ParserClass, TestClass, [regions])
PARSER_REGISTRY = {
    'card': (CardParser, CardParserTest, [
        # Community cards
        'community_card_1', 'community_card_2', 'community_card_3', 'community_card_4', 'community_card_5',
        # Player hole cards (all 8 players)
        'player_1_hole_1', 'player_1_hole_2', 'player_2_hole_1', 'player_2_hole_2',
        'player_3_hole_1', 'player_3_hole_2', 'player_4_hole_1', 'player_4_hole_2',
        'player_5_hole_1', 'player_5_hole_2', 'player_6_hole_1', 'player_6_hole_2',
        'player_7_hole_1', 'player_7_hole_2', 'player_8_hole_1', 'player_8_hole_2'
    ]),
    'dealer': (DealerDetector, DealerDetectorTest, [
        # Dealer button positions for all 8 players
        'player_1_dealer', 'player_2_dealer', 'player_3_dealer', 'player_4_dealer',
        'player_5_dealer', 'player_6_dealer', 'player_7_dealer', 'player_8_dealer'
    ]),
    'hand_id': (HandIdParser, HandIdParserTest, [
        # Hand number region
        'hand_num'
    ]),
    'money': (MoneyParser, MoneyParserTest, [
        # Pot and split bets
        'pot_total', 'split_bets',
        # Player banks (stacks)
        'player_1_bank', 'player_2_bank', 'player_3_bank', 'player_4_bank',
        'player_5_bank', 'player_6_bank', 'player_7_bank', 'player_8_bank',
        # Player bets
        'player_1_bet', 'player_2_bet', 'player_3_bet', 'player_4_bet',
        'player_5_bet', 'player_6_bet', 'player_7_bet', 'player_8_bet'
    ]),
    'name': (NameParser, NameParserTest, [
        # Player names for all 8 players
        'player_1_name', 'player_2_name', 'player_3_name', 'player_4_name',
        'player_5_name', 'player_6_name', 'player_7_name', 'player_8_name'
    ]),
    'status': (StatusParser, StatusParserTest, [
        # Player status for all 8 players
        'player_1_status', 'player_2_status', 'player_3_status', 'player_4_status',
        'player_5_status', 'player_6_status', 'player_7_status', 'player_8_status'
    ]),
    'timer': (TimerDetector, TimerDetectorTest, [
        # Timer detection uses nameplate regions for all 8 players
        'player_1_nameplate', 'player_2_nameplate', 'player_3_nameplate', 'player_4_nameplate',
        'player_5_nameplate', 'player_6_nameplate', 'player_7_nameplate', 'player_8_nameplate'
    ]),
    'transparency': (TransparencyDetector, TransparencyDetectorTest, [
        # Transparency detection uses nameplate regions for all 8 players
        'player_1_nameplate', 'player_2_nameplate', 'player_3_nameplate', 'player_4_nameplate',
        'player_5_nameplate', 'player_6_nameplate', 'player_7_nameplate', 'player_8_nameplate'
    ]),
    'table_info': (TableInfoParser, TableInfoParserTest, [
        # Table info region for blinds/stakes
        'table_info'
    ]),
}


def list_available_parsers():
    """List all available parsers and their regions."""
    print("Available Parsers:")
    print("=" * 50)
    
    for parser_name, (parser_class, test_class, regions) in PARSER_REGISTRY.items():
        print(f"\n{parser_name}:")
        print(f"  Class: {parser_class.__name__}")
        print(f"  Test Class: {test_class.__name__}")
        print(f"  Regions: {', '.join(regions)}")


def list_available_regions():
    """List all available regions."""
    try:
        region_loader = RegionLoader()
        regions = region_loader.load_regions()
        
        print("Available Regions:")
        print("=" * 30)
        for region in sorted(regions.keys()):
            print(f"  {region}")
    except Exception as e:
        print(f"Error loading regions: {e}")


def show_parser_statistics(parser_name: str):
    """Show statistics for a specific parser."""
    if parser_name not in PARSER_REGISTRY:
        print(f"Unknown parser: {parser_name}")
        return
    
    try:
        result_manager = ParserTestResultManager()
        stats = result_manager.get_parser_statistics(parser_name)
        
        print(f"Statistics for {parser_name} parser:")
        print("=" * 40)
        print(f"Total Tests: {stats['total_tests']}")
        print(f"Approved: {stats['approved']}")
        print(f"Rejected: {stats['rejected']}")
        print(f"Skipped: {stats['skipped']}")
        print(f"Approval Rate: {stats['approval_rate']:.1%}")
        print(f"Average Confidence: {stats['avg_confidence']:.3f}")
        
    except Exception as e:
        print(f"Error loading statistics: {e}")


def run_visual_test(parser_name: str, region_name: str, num_tests: int):
    """Run visual tests for a specific parser and region."""
    if parser_name not in PARSER_REGISTRY:
        print(f"Unknown parser: {parser_name}")
        return
    
    parser_class, test_class, valid_regions = PARSER_REGISTRY[parser_name]
    
    if region_name not in valid_regions:
        print(f"Invalid region '{region_name}' for parser '{parser_name}'")
        print(f"Valid regions: {', '.join(valid_regions)}")
        return
    
    try:
        # Initialize parser and test class
        parser_instance = parser_class()
        test_instance = test_class()
        
        print(f"Starting visual test for {parser_name} parser on region {region_name}")
        print(f"Running {num_tests} tests...")
        print("\nKey Bindings:")
        print("  Y - Approve result")
        print("  N - Reject result")
        print("  S - Skip test")
        print("  Q - Quit testing")
        print("  H - Show help")
        print("  ESC - Quit testing")
        print("\nPress any key to start...")
        input()
        
        # Run tests
        session = test_instance.run_visual_test(
            parser_instance, parser_name, region_name, num_tests
        )
        
        print(f"\nTesting completed! Session ID: {session.session_id}")
        
    except Exception as e:
        print(f"Error running visual test: {e}")
        logger.error(f"Visual test error: {e}", exc_info=True)


def _run_single_random_test(parser_instance, parser_name: str, region_name: str, test_class,
                           frame_extractor, regions, validator, result_manager):
    """
    Run a single random test using shared components to maintain window continuity.
    
    Args:
        parser_instance: Parser instance to test
        parser_name: Name of the parser
        region_name: Region to test
        test_class: Test class for this parser type
        frame_extractor: Shared VideoFrameExtractor instance
        regions: Loaded regions dictionary
        validator: Shared VisualTestValidator instance
        result_manager: Shared ParserTestResultManager instance
        
    Returns:
        VisualTestSession with results
    """
    from datetime import datetime
    
    # Create a minimal session for this single test
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session = VisualTestSession(
        session_id=session_id,
        start_time=datetime.now().isoformat()
    )
    
    try:
        # Extract random frame and region
        frame, frame_number, video_file = frame_extractor.extract_random_frame()
        
        # Extract region using RegionLoader
        if region_name not in regions:
            logger.warning(f"Region '{region_name}' not found in loaded regions")
            return session
        
        region_model = regions[region_name]
        x, y, w, h = region_model.x, region_model.y, region_model.width, region_model.height
        
        # Check bounds
        if (x + w > frame.shape[1] or y + h > frame.shape[0] or x < 0 or y < 0):
            logger.warning(f"Region '{region_name}' bounds exceed frame dimensions")
            return session
        
        region_image = frame[y:y+h, x:x+w]
        logger.debug(f"Extracted region '{region_name}': {w}x{h} at ({x}, {y})")
        
        if region_image is None or region_image.size == 0:
            logger.warning(f"Could not extract region {region_name}, skipping test")
            return session
        
        # Create a temporary test instance to use its parsing methods
        temp_test_instance = test_class()
        
        # Parse the region
        parsed_result = temp_test_instance._parse_region(parser_instance, region_image)
        
        # Extract result data (even if None - user needs to validate empty results too)
        if parsed_result is None:
            parsed_value = None
            confidence = 0.0
            preprocessing_method = "none"
            logger.debug(f"Parser returned no result - showing for validation")
        else:
            parsed_value, confidence, preprocessing_method = temp_test_instance._extract_result_data(parsed_result)
        
        # Create test result
        test_result = ParserTestResult(
            parser_name=parser_name,
            region_name=region_name,
            video_file=video_file.name,
            frame_number=frame_number,
            parsed_value=parsed_value,
            confidence=confidence,
            preprocessing_method=preprocessing_method,
            timestamp=datetime.now().isoformat()
        )
        
        # Collect debug info if available
        if hasattr(temp_test_instance, '_get_debug_info'):
            try:
                debug_info = temp_test_instance._get_debug_info(parser_instance, region_image, parsed_result)
                test_result.debug_info = debug_info
            except Exception as e:
                logger.warning(f"Failed to collect debug info: {e}")
                test_result.debug_info = None
        
        # Validate visually using shared validator
        validation_result = validator.validate_parser_result(
            region_image, parser_name, region_name, parsed_value, 
            confidence, preprocessing_method
        )
        
        if validation_result == 'quit':
            logger.info("User requested to quit testing session")
            session.user_quit = True
            return session
        
        # Standard parser validation for all parsers
        test_result.user_approved = validation_result
        
        # Update session stats
        session.total_tests += 1
        if validation_result is True:
            session.approved_tests += 1
        elif validation_result is False:
            session.rejected_tests += 1
        else:
            session.skipped_tests += 1
        
        # Save the single test result
        result_manager.save_test_result(test_result)
        
        logger.info(f"Test completed: {validation_result}")
        
    except Exception as e:
        logger.error(f"Error during single test: {e}")
    
    finally:
        # Finalize session
        session.end_time = datetime.now().isoformat()
    
    return session


def run_random_visual_tests(num_tests: int):
    """Run visual tests with random parser and region selection."""
    import random
    
    print(f"Starting randomized visual testing with {num_tests} total tests")
    print("=" * 60)
    print("\nKey Bindings:")
    print("  Y - Approve result")
    print("  N - Reject result")
    print("  S - Skip test")
    print("  Q - Quit testing")
    print("  H - Show help")
    print("  ESC - Quit testing")
    print("\nPress any key to start...")
    try:
        input()
    except EOFError:
        # Handle non-interactive environments (like automated testing)
        print("(Non-interactive mode - starting immediately)")
    
    tests_completed = 0
    
    # Create shared components once to maintain window continuity
    frame_extractor = VideoFrameExtractor()
    region_loader = RegionLoader()
    regions = region_loader.load_regions()
    validator = VisualTestValidator()
    result_manager = ParserTestResultManager()
    
    # Set total tests for progress tracking
    validator.total_tests = num_tests
    validator.current_test_num = 0
    
    try:
        while tests_completed < num_tests:
            # Randomly select a parser
            parser_name = random.choice(list(PARSER_REGISTRY.keys()))
            parser_class, test_class, valid_regions = PARSER_REGISTRY[parser_name]
            
            # Randomly select a region for that parser
            region_name = random.choice(valid_regions)
            
            print(f"\n{'='*60}")
            print(f"Test {tests_completed + 1}/{num_tests}")
            print(f"Parser: {parser_name} | Region: {region_name}")
            print(f"{'='*60}\n")
            
            # Initialize parser instance (but reuse validator)
            parser_instance = parser_class()
            
            # Run a single test using shared components
            try:
                session = _run_single_random_test(
                    parser_instance, parser_name, region_name, test_class,
                    frame_extractor, regions, validator, result_manager
                )
                
                # Check if user quit
                if hasattr(session, 'user_quit') and session.user_quit:
                    print("\n\nTesting stopped by user (Quit pressed)")
                    break
                
                tests_completed += 1
                    
            except KeyboardInterrupt:
                print("\n\nTesting interrupted by user (Ctrl+C)")
                break
            except Exception as e:
                logger.error(f"Error during test: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"Randomized testing completed! Total tests: {tests_completed}")
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print(f"\n\nTesting interrupted. Completed {tests_completed}/{num_tests} tests")
    except Exception as e:
        print(f"Error during random testing: {e}")
        logger.error(f"Random test error: {e}", exc_info=True)
    
    finally:
        # Close PyQt6 window if it's open
        if validator.use_qt:
            try:
                from test.parser.visual_test_qt import close_visual_test_window
                close_visual_test_window()
            except Exception:
                pass


def run_transparency_random_regions(num_tests: int):
    """Run transparency tests with random nameplate region selection."""
    import random
    
    print(f"Starting transparency testing with random nameplate regions ({num_tests} tests)")
    print("=" * 60)
    print("\nKey Bindings:")
    print("  Y - Approve result")
    print("  N - Reject result")
    print("  S - Skip test")
    print("  Q - Quit testing")
    print("  H - Show help")
    print("  ESC - Quit testing")
    print("\nPress any key to start...")
    try:
        input()
    except EOFError:
        print("(Non-interactive mode - starting immediately)")
    
    tests_completed = 0
    
    # Create shared components once to maintain window continuity
    frame_extractor = VideoFrameExtractor()
    region_loader = RegionLoader()
    regions = region_loader.load_regions()
    validator = VisualTestValidator()
    result_manager = ParserTestResultManager()
    
    # Set total tests for progress tracking
    validator.total_tests = num_tests
    validator.current_test_num = 0
    
    # Get transparency parser info
    parser_name = "transparency"
    parser_class, test_class, valid_regions = PARSER_REGISTRY[parser_name]
    
    # Filter to only nameplate regions
    nameplate_regions = [r for r in valid_regions if "nameplate" in r]
    
    try:
        while tests_completed < num_tests:
            # Randomly select a nameplate region
            region_name = random.choice(nameplate_regions)
            
            print(f"\n{'='*60}")
            print(f"Test {tests_completed + 1}/{num_tests}")
            print(f"Parser: {parser_name} | Region: {region_name}")
            print(f"{'='*60}\n")
            
            # Initialize parser instance (but reuse validator)
            parser_instance = parser_class()
            
            # Run a single test using shared components
            try:
                session = _run_single_random_test(
                    parser_instance, parser_name, region_name, test_class,
                    frame_extractor, regions, validator, result_manager
                )
                
                # Check if user quit
                if hasattr(session, 'user_quit') and session.user_quit:
                    print("\n\nTesting stopped by user (Quit pressed)")
                    break
                
                tests_completed += 1
                    
            except KeyboardInterrupt:
                print("\n\nTesting interrupted by user (Ctrl+C)")
                break
            except Exception as e:
                logger.error(f"Error during test: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"Transparency testing completed! Total tests: {tests_completed}")
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print(f"\n\nTesting interrupted. Completed {tests_completed}/{num_tests} tests")
    except Exception as e:
        print(f"Error during transparency testing: {e}")
        logger.error(f"Transparency test error: {e}", exc_info=True)
    
    finally:
        # Close PyQt6 window if it's open
        if validator.use_qt:
            try:
                from test.parser.visual_test_qt import close_visual_test_window
                close_visual_test_window()
            except Exception:
                pass


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visual testing tool for parser components",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test/parser/visual_test.py --parser card --region card_1 --tests 20
  python test/parser/visual_test.py --parser money --region pot --tests 10
  python test/parser/visual_test.py --random --tests 50
  python test/parser/visual_test.py --list-parsers
  python test/parser/visual_test.py --list-regions
  python test/parser/visual_test.py --stats card
        """
    )
    
    parser.add_argument(
        '--parser', '-p',
        choices=list(PARSER_REGISTRY.keys()),
        help='Parser to test'
    )
    
    parser.add_argument(
        '--region', '-r',
        help='Region to test'
    )
    
    parser.add_argument(
        '--tests', '-t',
        type=int,
        default=10,
        help='Number of tests to run (default: 10)'
    )
    
    parser.add_argument(
        '--list-parsers',
        action='store_true',
        help='List all available parsers'
    )
    
    parser.add_argument(
        '--list-regions',
        action='store_true',
        help='List all available regions'
    )
    
    parser.add_argument(
        '--stats',
        help='Show statistics for a specific parser'
    )
    
    parser.add_argument(
        '--random',
        action='store_true',
        help='Run tests with random parser and region selection'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)8s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handle different commands
    if args.list_parsers:
        list_available_parsers()
    elif args.list_regions:
        list_available_regions()
    elif args.stats:
        show_parser_statistics(args.stats)
    elif args.random:
        run_random_visual_tests(args.tests)
    elif args.parser and args.region:
        run_visual_test(args.parser, args.region, args.tests)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
