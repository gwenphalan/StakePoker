#!/usr/bin/env python3
"""
Interactive region mapper for poker table
Click and drag to define regions on your poker video frame

This tool allows you to:
- Load random frames from videos in /data/video
- Click and drag to define regions
- Save regions to config/poker_regions.json
- Navigate through different frames and videos
"""

import cv2
import json
import random
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

class VideoFrameLoader:
    """Helper class to load random frames from video files"""
    
    def __init__(self, video_dir: Path):
        self.video_dir = video_dir
        self.video_files = list(video_dir.glob("*.mp4"))
        if not self.video_files:
            raise FileNotFoundError(f"No video files found in {video_dir}")
        print(f"Found {len(self.video_files)} video files")
    
    def get_random_frame(self) -> Tuple[np.ndarray, str]:
        """Get a random frame from a random video file"""
        video_file = random.choice(self.video_files)
        cap = cv2.VideoCapture(str(video_file))
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_file}")
        
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            raise RuntimeError(f"Video file has no frames: {video_file}")
        
        # Get random frame
        random_frame_number = random.randint(0, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise RuntimeError(f"Could not read frame from video: {video_file}")
        
        return frame, f"{video_file.name} (frame {random_frame_number})"
    
    def get_frame_at_time(self, video_file: Path, time_seconds: float) -> np.ndarray:
        """Get a specific frame at a given time in seconds"""
        cap = cv2.VideoCapture(str(video_file))
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_file}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(time_seconds * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise RuntimeError(f"Could not read frame at time {time_seconds}s from video: {video_file}")
        
        return frame

class RegionMapper:
    """Interactive region mapper for poker table regions"""
    
    def __init__(self, video_dir: Path, regions_file: Path):
        self.video_dir = video_dir
        self.regions_file = regions_file
        self.video_loader = VideoFrameLoader(video_dir)
        
        # Load a random frame to start with
        self.frame, self.current_video_info = self.video_loader.get_random_frame()
        self.original_frame = self.frame.copy()
        self.regions = {}
        self.drawing = False
        self.start_point = None
        self.current_video_file = None
        self.current_time = 0.0
        
        # Load existing regions if they exist
        self.load_regions()
        
        print(f"Loaded frame from: {self.current_video_info}")
        print(f"Frame size: {self.frame.shape[1]}x{self.frame.shape[0]}")
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for region drawing"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # Show preview rectangle
                temp_frame = self.original_frame.copy()
                cv2.rectangle(temp_frame, self.start_point, (x, y), (0, 255, 0), 2)
                cv2.imshow('Region Mapper', temp_frame)
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                end_point = (x, y)
                
                # Get region name from user
                region_name = input(f"\nEnter name for region ({self.start_point} to {end_point}): ")
                if region_name:
                    self.regions[region_name] = {
                        'x': min(self.start_point[0], end_point[0]),
                        'y': min(self.start_point[1], end_point[1]),
                        'width': abs(end_point[0] - self.start_point[0]),
                        'height': abs(end_point[1] - self.start_point[1])
                    }
                    print(f"Added region: {region_name}")
                
                # Redraw with all regions
                self.redraw_frame()
    
    def redraw_frame(self):
        """Redraw the frame with all defined regions"""
        self.frame = self.original_frame.copy()
        
        # Draw all regions
        for name, region in self.regions.items():
            cv2.rectangle(self.frame, 
                         (region['x'], region['y']),
                         (region['x'] + region['width'], region['y'] + region['height']),
                         (0, 255, 0), 2)
            cv2.putText(self.frame, name,
                       (region['x'], region['y'] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add video info
        info_text = f"Video: {self.current_video_info}"
        cv2.putText(self.frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add region count
        region_count_text = f"Regions: {len(self.regions)}"
        cv2.putText(self.frame, region_count_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Region Mapper', self.frame)
    
    def load_random_frame(self):
        """Load a new random frame from any video"""
        try:
            self.frame, self.current_video_info = self.video_loader.get_random_frame()
            self.original_frame = self.frame.copy()
            self.redraw_frame()
            print(f"Loaded new random frame from: {self.current_video_info}")
        except Exception as e:
            print(f"Error loading random frame: {e}")
    
    def load_regions(self):
        """Load regions from the regions file"""
        try:
            if self.regions_file.exists():
                with open(self.regions_file, 'r') as f:
                    self.regions = json.load(f)
                print(f"Loaded {len(self.regions)} regions from {self.regions_file}")
            else:
                print("No existing regions file found, starting with empty regions")
                self.regions = {}
        except json.JSONDecodeError as e:
            print(f"Error reading regions file: {e}")
            self.regions = {}
        except Exception as e:
            print(f"Error loading regions: {e}")
            self.regions = {}
    
    def save_regions(self):
        """Save regions to the regions file"""
        try:
            # Ensure the directory exists
            self.regions_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.regions_file, 'w') as f:
                json.dump(self.regions, f, indent=2)
            print(f"Regions saved to {self.regions_file}")
            print(f"Total regions saved: {len(self.regions)}")
        except Exception as e:
            print(f"Error saving regions: {e}")
    
    def list_regions(self):
        """List all currently defined regions"""
        if not self.regions:
            print("No regions defined yet")
            return
        
        print("\nCurrent regions:")
        for name, region in self.regions.items():
            print(f"  {name}: x={region['x']}, y={region['y']}, "
                  f"w={region['width']}, h={region['height']}")
    
    def delete_region(self, region_name: str):
        """Delete a specific region"""
        if region_name in self.regions:
            del self.regions[region_name]
            print(f"Deleted region: {region_name}")
            self.redraw_frame()
        else:
            print(f"Region '{region_name}' not found")
    
    def clear_all_regions(self):
        """Clear all regions"""
        self.regions = {}
        print("Cleared all regions")
        self.redraw_frame()

def print_instructions():
    """Print usage instructions"""
    print("\n" + "="*60)
    print("POKER REGION MAPPER")
    print("="*60)
    print("Instructions:")
    print("- Click and drag to define regions")
    print("- Press 'r' to load a new random frame")
    print("- Press 's' to save regions")
    print("- Press 'l' to list current regions")
    print("- Press 'd' to delete a region (will prompt for name)")
    print("- Press 'c' to clear all regions")
    print("- Press 'q' to quit")
    print("="*60)

def main():
    """Main function"""
    # Set up paths
    project_root = Path(__file__).parent.parent
    video_dir = project_root / "data" / "video"
    regions_file = project_root / "config" / "poker_regions.json"
    
    # Check if video directory exists
    if not video_dir.exists():
        print(f"Error: Video directory not found: {video_dir}")
        print("Please ensure you have video files in the /data/video directory")
        return 1
    
    try:
        # Initialize the region mapper
        mapper = RegionMapper(video_dir, regions_file)
        
        # Set up the display
        cv2.namedWindow('Region Mapper', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Region Mapper', mapper.mouse_callback)
        
        # Show initial frame
        mapper.redraw_frame()
        
        # Print instructions
        print_instructions()
        
        # Main interaction loop
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                mapper.save_regions()
            elif key == ord('r'):
                mapper.load_random_frame()
            elif key == ord('l'):
                mapper.list_regions()
            elif key == ord('c'):
                confirm = input("Are you sure you want to clear all regions? (y/N): ")
                if confirm.lower() == 'y':
                    mapper.clear_all_regions()
            elif key == ord('d'):
                region_name = input("Enter region name to delete: ")
                if region_name:
                    mapper.delete_region(region_name)
        
        cv2.destroyAllWindows()
        
        # Ask if user wants to save before exiting
        if mapper.regions:
            save_confirm = input("\nSave regions before exiting? (Y/n): ")
            if save_confirm.lower() != 'n':
                mapper.save_regions()
        
        print("Region mapper closed.")
        return 0
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
