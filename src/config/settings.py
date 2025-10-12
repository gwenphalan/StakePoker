#!/usr/bin/env python3
"""
Dynamic settings management system for StakePoker.

Supports hierarchical settings with dot notation (e.g., "capture.monitor.index"),
automatic JSON persistence, default values, and group-based queries.

Usage:
    from src.config.settings import Settings
    
    # Initialize settings (do this once at app startup)
    settings = Settings()
    
    # Create settings with defaults
    settings.create("capture.monitor.index", default=1)
    settings.create("parser.ocr.confidence_threshold", default=0.7)
    settings.create("advisor.gto.enabled", default=True)
    
    # Update settings
    settings.update("capture.monitor.index", 2)
    
    # Get individual setting
    monitor_index = settings.get("capture.monitor.index")
    
    # Get all settings in a group
    capture_settings = settings.get_group("capture")
    ocr_settings = settings.get_group("parser.ocr")
    
    # Get all settings
    all_settings = settings.get_all()
    
    # Reset to defaults
    settings.reset("capture.monitor.index")
    settings.reset_group("parser.ocr")
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, List, Union
from threading import Lock
import logging
from box import Box

logger = logging.getLogger(__name__)


class Settings:
    """
    Dynamic hierarchical settings manager with JSON persistence.
    
    Settings are stored in dot notation (e.g., "group.subgroup.setting")
    and automatically persisted to settings.json. Uses python-box for
    elegant nested dictionary access.
    
    Thread-safe singleton implementation.
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls, settings_file: Optional[Path] = None):
        """Singleton pattern to ensure only one Settings instance exists."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Settings, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, settings_file: Optional[Path] = None):
        """
        Initialize the settings manager.
        
        Args:
            settings_file: Path to settings JSON file. Defaults to data/settings.json
        """
        if self._initialized:
            return
        
        self._initialized = True
        
        # Set default settings file path
        if settings_file is None:
            self.settings_file = Path(__file__).parent.parent.parent / "data" / "settings.json"
        else:
            self.settings_file = Path(settings_file)
        
        # Ensure data directory exists
        self.settings_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Storage for settings values and defaults
        self._settings = Box(default_box=True, box_dots=True)
        self._defaults = Box(default_box=True, box_dots=True)
        
        # Load existing settings from file
        self._load_from_file()
        
        logger.info(f"Settings initialized from {self.settings_file}")
    
    def create(self, setting_name: str, default: Any) -> None:
        """
        Create a new setting with a default value.
        
        If the setting already exists in the JSON file, the file value is preserved.
        Otherwise, the default value is used and saved to the file.
        
        Args:
            setting_name: Dot-notation path (e.g., "capture.monitor.index")
            default: Default value for the setting
        
        Example:
            settings.create("capture.monitor.index", default=1)
            settings.create("parser.ocr.enabled", default=True)
            settings.create("parser.ocr.confidence_threshold", default=0.7)
        """
        # Store the default value
        self._set_nested(self._defaults, setting_name, default)
        
        # Check if setting already exists in loaded settings
        existing_value = self._get_nested(self._settings, setting_name)
        
        # Check if it's actually a value or just an empty Box/dict
        is_empty = (
            existing_value is None or 
            (isinstance(existing_value, (dict, Box)) and len(existing_value) == 0)
        )
        
        if is_empty:
            # Setting doesn't exist in file or is empty, use default value
            self._set_nested(self._settings, setting_name, default)
            self._save_to_file()
            logger.debug(f"Created setting '{setting_name}' with default value: {default}")
        else:
            # Setting exists in file with a real value, preserve it
            logger.debug(f"Setting '{setting_name}' already exists with value: {existing_value} (default: {default})")
    
    def update(self, setting_name: str, value: Any) -> None:
        """
        Update an existing setting's value.
        
        Args:
            setting_name: Dot-notation path (e.g., "capture.monitor.index")
            value: New value for the setting
        
        Raises:
            KeyError: If the setting doesn't exist
        
        Example:
            settings.update("capture.monitor.index", 2)
        """
        # Check if setting exists
        if self._get_nested(self._settings, setting_name) is None:
            raise KeyError(f"Setting '{setting_name}' does not exist. Use create() first.")
        
        old_value = self._get_nested(self._settings, setting_name)
        self._set_nested(self._settings, setting_name, value)
        self._save_to_file()
        
        logger.debug(f"Updated setting '{setting_name}': {old_value} -> {value}")
    
    def get(self, setting_name: str, fallback: Any = None) -> Any:
        """
        Get a setting's value.
        
        Args:
            setting_name: Dot-notation path (e.g., "capture.monitor.index")
            fallback: Value to return if setting doesn't exist
        
        Returns:
            The setting's value, or fallback if not found
        
        Example:
            monitor_index = settings.get("capture.monitor.index")
            threshold = settings.get("parser.ocr.threshold", 0.7)
        """
        value = self._get_nested(self._settings, setting_name)
        
        if value is None:
            if fallback is not None:
                return fallback
            # Try to get default value
            default = self._get_nested(self._defaults, setting_name)
            return default if default is not None else fallback
        
        return value
    
    def get_group(self, group_path: str) -> Dict[str, Any]:
        """
        Get all settings within a group or subgroup.
        
        Args:
            group_path: Dot-notation path to group (e.g., "capture" or "parser.ocr")
        
        Returns:
            Dictionary containing all settings in the group
        
        Example:
            capture_settings = settings.get_group("capture")
            # Returns: {"monitor": {"index": 2}, "fps": 2, ...}
            
            ocr_settings = settings.get_group("parser.ocr")
            # Returns: {"enabled": True, "confidence_threshold": 0.8, ...}
        """
        group_data = self._get_nested(self._settings, group_path)
        
        if group_data is None:
            logger.warning(f"Group '{group_path}' not found")
            return {}
        
        # Convert Box to regular dict for return
        if isinstance(group_data, Box):
            return group_data.to_dict()
        
        return group_data if isinstance(group_data, dict) else {}
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all settings as a dictionary.
        
        Returns:
            Dictionary containing all settings
        
        Example:
            all_settings = settings.get_all()
        """
        return self._settings.to_dict()
    
    def reset(self, setting_name: str) -> None:
        """
        Reset a setting to its default value.
        
        Args:
            setting_name: Dot-notation path (e.g., "capture.monitor.index")
        
        Raises:
            KeyError: If the setting doesn't exist
        
        Example:
            settings.reset("capture.monitor.index")
        """
        default_value = self._get_nested(self._defaults, setting_name)
        
        if default_value is None:
            raise KeyError(f"Setting '{setting_name}' has no default value")
        
        self._set_nested(self._settings, setting_name, default_value)
        self._save_to_file()
        
        logger.info(f"Reset setting '{setting_name}' to default: {default_value}")
    
    def reset_group(self, group_path: str) -> None:
        """
        Reset all settings in a group to their default values.
        
        Args:
            group_path: Dot-notation path to group (e.g., "capture" or "parser.ocr")
        
        Example:
            settings.reset_group("parser.ocr")
        """
        defaults_group = self._get_nested(self._defaults, group_path)
        
        if defaults_group is None:
            logger.warning(f"No defaults found for group '{group_path}'")
            return
        
        # Reset each setting in the group
        self._reset_nested_group(group_path, defaults_group)
        self._save_to_file()
        
        logger.info(f"Reset group '{group_path}' to defaults")
    
    def exists(self, setting_name: str) -> bool:
        """
        Check if a setting exists.
        
        Args:
            setting_name: Dot-notation path (e.g., "capture.monitor.index")
        
        Returns:
            True if setting exists, False otherwise
        
        Example:
            if settings.exists("capture.monitor.index"):
                index = settings.get("capture.monitor.index")
        """
        return self._get_nested(self._settings, setting_name) is not None
    
    def delete(self, setting_name: str) -> None:
        """
        Delete a setting.
        
        Args:
            setting_name: Dot-notation path (e.g., "capture.monitor.index")
        
        Example:
            settings.delete("capture.monitor.index")
        """
        self._delete_nested(self._settings, setting_name)
        self._delete_nested(self._defaults, setting_name)
        self._save_to_file()
        
        logger.info(f"Deleted setting '{setting_name}'")
    
    def list_groups(self, parent_path: str = "") -> List[str]:
        """
        List all groups and subgroups under a parent path.
        
        Args:
            parent_path: Parent path to search under (empty string for root)
        
        Returns:
            List of group names
        
        Example:
            root_groups = settings.list_groups()  # ["capture", "parser", "advisor"]
            parser_groups = settings.list_groups("parser")  # ["ocr", "cards", "money"]
        """
        if parent_path:
            parent_data = self._get_nested(self._settings, parent_path)
        else:
            parent_data = self._settings
        
        if parent_data is None or not isinstance(parent_data, (dict, Box)):
            return []
        
        groups = []
        for key, value in parent_data.items():
            if isinstance(value, (dict, Box)):
                groups.append(key)
        
        return sorted(groups)
    
    def list_settings(self, group_path: str = "") -> List[str]:
        """
        List all setting names (leaf nodes) under a group path.
        
        Args:
            group_path: Group path to search under (empty string for root)
        
        Returns:
            List of setting names (not including nested groups)
        
        Example:
            capture_settings = settings.list_settings("capture")
            # ["fps", "quality"] (excludes "monitor" which is a group)
        """
        if group_path:
            group_data = self._get_nested(self._settings, group_path)
        else:
            group_data = self._settings
        
        if group_data is None or not isinstance(group_data, (dict, Box)):
            return []
        
        settings_list = []
        for key, value in group_data.items():
            if not isinstance(value, (dict, Box)):
                settings_list.append(key)
        
        return sorted(settings_list)
    
    def _get_nested(self, data: Box, path: str) -> Any:
        """Get a value from nested dictionary using dot notation."""
        try:
            keys = path.split('.')
            current = data
            for key in keys:
                if isinstance(current, Box):
                    current = current.get(key)
                elif isinstance(current, dict):
                    current = current.get(key)
                else:
                    return None
            
            if current is None:
                return None
        
            # If we got an empty Box/dict, treat it as None
            if isinstance(current, (dict, Box)) and len(current) == 0:
                return None
            
            return current
        except (KeyError, AttributeError, TypeError):
            return None
    
    def _set_nested(self, data: Box, path: str, value: Any) -> None:
        """Set a value in nested dictionary using dot notation."""
        keys = path.split('.')
        current = data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = Box(default_box=True, box_dots=True)
            current = current[key]
        
        current[keys[-1]] = value
    
    def _delete_nested(self, data: Box, path: str) -> None:
        """Delete a value from nested dictionary using dot notation."""
        keys = path.split('.')
        current = data
        
        try:
            for key in keys[:-1]:
                current = current[key]
            
            if keys[-1] in current:
                del current[keys[-1]]
        except (KeyError, AttributeError, TypeError):
            pass
    
    def _reset_nested_group(self, base_path: str, defaults_data: Union[Box, Dict]) -> None:
        """Recursively reset all settings in a group to defaults."""
        if not isinstance(defaults_data, (dict, Box)):
            return
        
        for key, value in defaults_data.items():
            full_path = f"{base_path}.{key}" if base_path else key
            
            if isinstance(value, (dict, Box)):
                # Recursively reset nested groups
                self._reset_nested_group(full_path, value)
            else:
                # Reset individual setting
                self._set_nested(self._settings, full_path, value)
    
    def _load_from_file(self) -> None:
        """Load settings from JSON file."""
        if not self.settings_file.exists():
            logger.info(f"Settings file not found, creating new: {self.settings_file}")
            self._save_to_file()
            return
        
        try:
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Extract settings from the data structure
                if isinstance(data, dict) and "settings" in data:
                    settings_data = data["settings"]
                else:
                    settings_data = data
                
                self._settings = Box(settings_data, default_box=True, box_dots=True)
            
            logger.info(f"Loaded settings from {self.settings_file}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse settings file: {e}")
            logger.warning("Using empty settings")
            self._settings = Box(default_box=True, box_dots=True)
        except Exception as e:
            logger.error(f"Failed to load settings file: {e}")
            self._settings = Box(default_box=True, box_dots=True)
    
    def _save_to_file(self) -> None:
        """
        Save current settings to file.
        
        Creates the settings file and any necessary parent directories.
        """
        try:
            # Ensure parent directory exists
            self.settings_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert Box to dict for proper JSON serialization
            settings_dict = self._settings.to_dict()
            
            # Prepare data for saving
            data = {
                "settings": settings_dict,
                "metadata": {
                    "version": "1.0",
                    "last_modified": self._get_timestamp()
                }
            }
            
            # Write to file with pretty formatting
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Settings saved to {self.settings_file}")
            
        except PermissionError as e:
            logger.error(f"Permission denied writing settings file: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            raise

    def _get_timestamp(self) -> str:
        """Get current timestamp as ISO format string."""
        from datetime import datetime
        return datetime.now().isoformat()