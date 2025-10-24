#!/usr/bin/env python3
"""
Unit tests for the Settings class in src/config/settings.py.

Tests all functionality including singleton pattern, CRUD operations,
group operations, file persistence, and error handling.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
import threading
import time

# Add src to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from config.settings import Settings


class TestSettingsSingleton(unittest.TestCase):
    """Test singleton pattern implementation."""
    
    def setUp(self):
        """Reset singleton instance before each test."""
        Settings._instance = None
        Settings._initialized = False
    
    def test_singleton_instance(self):
        """Test that only one Settings instance is created."""
        settings1 = Settings()
        settings2 = Settings()
        
        self.assertIs(settings1, settings2)
        self.assertIsInstance(settings1, Settings)
    
    def test_singleton_thread_safety(self):
        """Test singleton pattern is thread-safe."""
        instances = []
        
        def create_instance():
            instances.append(Settings())
        
        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=create_instance)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All instances should be the same
        self.assertEqual(len(instances), 10)
        for instance in instances:
            self.assertIs(instance, instances[0])
    
    def test_singleton_initialization_once(self):
        """Test that initialization only happens once."""
        settings1 = Settings()
        settings2 = Settings()
        
        # Both should have the same initialization state
        self.assertTrue(settings1._initialized)
        self.assertTrue(settings2._initialized)


class TestSettingsInitialization(unittest.TestCase):
    """Test Settings initialization and file handling."""
    
    def setUp(self):
        """Reset singleton instance before each test."""
        Settings._instance = None
        Settings._initialized = False
    
    def tearDown(self):
        """Clean up after each test."""
        Settings._instance = None
        Settings._initialized = False
    
    def test_default_settings_file_path(self):
        """Test default settings file path construction."""
        settings = Settings()
        expected_path = Path(__file__).parent.parent.parent / "src" / "config" / ".." / ".." / "config" / "settings.json"
        self.assertEqual(settings.settings_file, expected_path.resolve())
    
    def test_custom_settings_file_path(self):
        """Test custom settings file path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_file = Path(temp_dir) / "custom_settings.json"
            settings = Settings(custom_file)
            self.assertEqual(settings.settings_file, custom_file)
    
    def test_load_nonexistent_file(self):
        """Test loading from non-existent file creates new file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = Path(temp_dir) / "nonexistent.json"
            settings = Settings(settings_file)
            
            # File should be created
            self.assertTrue(settings_file.exists())
            
            # Should contain empty settings structure
            with open(settings_file, 'r') as f:
                data = json.load(f)
                self.assertIn("settings", data)
                self.assertIn("metadata", data)
    
    def test_load_existing_file(self):
        """Test loading from existing file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = Path(temp_dir) / "existing.json"
            
            # Create test data
            test_data = {
                "settings": {
                    "test": {
                        "value": 42,
                        "nested": {
                            "setting": "hello"
                        }
                    }
                },
                "metadata": {
                    "version": "1.0",
                    "last_modified": "2023-01-01T00:00:00"
                }
            }
            
            with open(settings_file, 'w') as f:
                json.dump(test_data, f)
            
            settings = Settings(settings_file)
            
            # Should load existing data
            self.assertEqual(settings.get("test.value"), 42)
            self.assertEqual(settings.get("test.nested.setting"), "hello")
    
    def test_load_invalid_json(self):
        """Test handling of invalid JSON file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = Path(temp_dir) / "invalid.json"
            
            # Create invalid JSON
            with open(settings_file, 'w') as f:
                f.write("invalid json content")
            
            # Should not raise exception, should use empty settings
            settings = Settings(settings_file)
            self.assertIsNotNone(settings)
            self.assertEqual(settings.get_all(), {})


class TestSettingsCRUD(unittest.TestCase):
    """Test Settings CRUD operations."""
    
    def setUp(self):
        """Reset singleton instance before each test."""
        Settings._instance = None
        Settings._initialized = False
    
    def tearDown(self):
        """Clean up after each test."""
        Settings._instance = None
        Settings._initialized = False
    
    def test_create_setting_with_default(self):
        """Test creating a setting with default value."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = Path(temp_dir) / "test.json"
            settings = Settings(settings_file)
            
            # Create setting
            settings.create("test.setting", default=42)
            
            # Should be retrievable
            self.assertEqual(settings.get("test.setting"), 42)
            
            # Should be saved to file
            self.assertTrue(settings_file.exists())
    
    def test_create_existing_setting_preserves_value(self):
        """Test creating setting that already exists preserves file value."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = Path(temp_dir) / "test.json"
            
            # Pre-create file with existing value
            existing_data = {
                "settings": {
                    "test": {
                        "setting": 100
                    }
                },
                "metadata": {"version": "1.0", "last_modified": "2023-01-01T00:00:00"}
            }
            
            with open(settings_file, 'w') as f:
                json.dump(existing_data, f)
            
            settings = Settings(settings_file)
            
            # Create setting with different default
            settings.create("test.setting", default=42)
            
            # Should preserve existing value, not use default
            self.assertEqual(settings.get("test.setting"), 100)
    
    def test_update_existing_setting(self):
        """Test updating an existing setting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = Path(temp_dir) / "test.json"
            settings = Settings(settings_file)
            
            # Create setting
            settings.create("test.setting", default=42)
            
            # Update setting
            settings.update("test.setting", 100)
            
            # Should have new value
            self.assertEqual(settings.get("test.setting"), 100)
    
    def test_update_nonexistent_setting_raises_error(self):
        """Test updating non-existent setting raises KeyError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = Path(temp_dir) / "test.json"
            settings = Settings(settings_file)
            
            with self.assertRaises(KeyError):
                settings.update("nonexistent.setting", 100)
    
    def test_get_setting_with_fallback(self):
        """Test getting setting with fallback value."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = Path(temp_dir) / "test.json"
            settings = Settings(settings_file)
            
            # Get non-existent setting with fallback
            value = settings.get("nonexistent.setting", fallback=99)
            self.assertEqual(value, 99)
    
    def test_get_setting_with_default(self):
        """Test getting setting returns default when not set."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = Path(temp_dir) / "test.json"
            settings = Settings(settings_file)
            
            # Create setting with default
            settings.create("test.setting", default=42)
            
            # Get setting should return default
            self.assertEqual(settings.get("test.setting"), 42)
    
    def test_delete_setting(self):
        """Test deleting a setting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = Path(temp_dir) / "test.json"
            settings = Settings(settings_file)
            
            # Create setting
            settings.create("test.setting", default=42)
            self.assertTrue(settings.exists("test.setting"))
            
            # Delete setting
            settings.delete("test.setting")
            self.assertFalse(settings.exists("test.setting"))
    
    def test_exists_setting(self):
        """Test checking if setting exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = Path(temp_dir) / "test.json"
            settings = Settings(settings_file)
            
            # Non-existent setting
            self.assertFalse(settings.exists("nonexistent.setting"))
            
            # Create setting
            settings.create("test.setting", default=42)
            self.assertTrue(settings.exists("test.setting"))


class TestSettingsGroups(unittest.TestCase):
    """Test Settings group operations."""
    
    def setUp(self):
        """Reset singleton instance before each test."""
        Settings._instance = None
        Settings._initialized = False
    
    def tearDown(self):
        """Clean up after each test."""
        Settings._instance = None
        Settings._initialized = False
    
    def test_get_group(self):
        """Test getting all settings in a group."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = Path(temp_dir) / "test.json"
            settings = Settings(settings_file)
            
            # Create multiple settings in a group
            settings.create("capture.monitor.index", default=1)
            settings.create("capture.fps", default=30)
            settings.create("capture.quality", default="high")
            
            # Get group
            capture_settings = settings.get_group("capture")
            
            # Should contain all settings in group
            expected = {
                "monitor": {"index": 1},
                "fps": 30,
                "quality": "high"
            }
            self.assertEqual(capture_settings, expected)
    
    def test_get_nonexistent_group(self):
        """Test getting non-existent group returns empty dict."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = Path(temp_dir) / "test.json"
            settings = Settings(settings_file)
            
            # Get non-existent group
            result = settings.get_group("nonexistent")
            self.assertEqual(result, {})
    
    def test_get_nested_group(self):
        """Test getting nested group."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = Path(temp_dir) / "test.json"
            settings = Settings(settings_file)
            
            # Create nested settings
            settings.create("parser.ocr.confidence", default=0.8)
            settings.create("parser.ocr.enabled", default=True)
            settings.create("parser.cards.detection", default="auto")
            
            # Get nested group
            ocr_settings = settings.get_group("parser.ocr")
            
            expected = {
                "confidence": 0.8,
                "enabled": True
            }
            self.assertEqual(ocr_settings, expected)
    
    def test_reset_group(self):
        """Test resetting all settings in a group to defaults."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = Path(temp_dir) / "test.json"
            settings = Settings(settings_file)
            
            # Create settings with defaults
            settings.create("capture.monitor.index", default=1)
            settings.create("capture.fps", default=30)
            
            # Update settings
            settings.update("capture.monitor.index", 2)
            settings.update("capture.fps", 60)
            
            # Reset group
            settings.reset_group("capture")
            
            # Should be back to defaults
            self.assertEqual(settings.get("capture.monitor.index"), 1)
            self.assertEqual(settings.get("capture.fps"), 30)
    
    def test_reset_nonexistent_group(self):
        """Test resetting non-existent group does nothing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = Path(temp_dir) / "test.json"
            settings = Settings(settings_file)
            
            # Should not raise exception
            settings.reset_group("nonexistent")
    
    def test_list_groups(self):
        """Test listing groups."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = Path(temp_dir) / "test.json"
            settings = Settings(settings_file)
            
            # Create settings in different groups
            settings.create("capture.monitor.index", default=1)
            settings.create("parser.ocr.confidence", default=0.8)
            settings.create("advisor.gto.enabled", default=True)
            
            # List root groups
            root_groups = settings.list_groups()
            self.assertIn("capture", root_groups)
            self.assertIn("parser", root_groups)
            self.assertIn("advisor", root_groups)
            
            # List nested groups
            parser_groups = settings.list_groups("parser")
            self.assertIn("ocr", parser_groups)
    
    def test_list_settings(self):
        """Test listing settings (leaf nodes)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = Path(temp_dir) / "test.json"
            settings = Settings(settings_file)
            
            # Create settings
            settings.create("capture.fps", default=30)
            settings.create("capture.quality", default="high")
            settings.create("capture.monitor.index", default=1)
            
            # List settings in capture group
            capture_settings = settings.list_settings("capture")
            self.assertIn("fps", capture_settings)
            self.assertIn("quality", capture_settings)
            # monitor is a group, not a setting
            self.assertNotIn("monitor", capture_settings)
            
            # List settings in monitor subgroup
            monitor_settings = settings.list_settings("capture.monitor")
            self.assertIn("index", monitor_settings)


class TestSettingsFilePersistence(unittest.TestCase):
    """Test Settings file persistence and error handling."""
    
    def setUp(self):
        """Reset singleton instance before each test."""
        Settings._instance = None
        Settings._initialized = False
    
    def tearDown(self):
        """Clean up after each test."""
        Settings._instance = None
        Settings._initialized = False
    
    def test_save_to_file(self):
        """Test saving settings to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = Path(temp_dir) / "test.json"
            settings = Settings(settings_file)
            
            # Create settings
            settings.create("test.setting1", default=42)
            settings.create("test.nested.setting2", default="hello")
            
            # File should exist and contain data
            self.assertTrue(settings_file.exists())
            
            with open(settings_file, 'r') as f:
                data = json.load(f)
                
                self.assertIn("settings", data)
                self.assertIn("metadata", data)
                self.assertEqual(data["settings"]["test"]["setting1"], 42)
                self.assertEqual(data["settings"]["test"]["nested"]["setting2"], "hello")
    
    def test_save_with_permission_error(self):
        """Test handling of permission error when saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = Path(temp_dir) / "readonly.json"
            
            # Create readonly file
            settings_file.touch()
            settings_file.chmod(0o444)  # Read-only
            
            settings = Settings(settings_file)
            
            # Should raise PermissionError
            with self.assertRaises(PermissionError):
                settings.create("test.setting", default=42)
    
    def test_load_corrupted_file(self):
        """Test handling of corrupted file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = Path(temp_dir) / "corrupted.json"
            
            # Create corrupted file
            with open(settings_file, 'w') as f:
                f.write("{ invalid json")
            
            # Should not raise exception, should use empty settings
            settings = Settings(settings_file)
            self.assertIsNotNone(settings)
            self.assertEqual(settings.get_all(), {})


class TestSettingsEdgeCases(unittest.TestCase):
    """Test Settings edge cases and validation."""
    
    def setUp(self):
        """Reset singleton instance before each test."""
        Settings._instance = None
        Settings._initialized = False
    
    def tearDown(self):
        """Clean up after each test."""
        Settings._instance = None
        Settings._initialized = False
    
    def test_empty_setting_name(self):
        """Test handling of empty setting name."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = Path(temp_dir) / "test.json"
            settings = Settings(settings_file)
            
            # Empty setting name should work
            settings.create("", default=42)
            self.assertEqual(settings.get(""), 42)
    
    def test_deeply_nested_setting(self):
        """Test deeply nested settings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = Path(temp_dir) / "test.json"
            settings = Settings(settings_file)
            
            # Create deeply nested setting
            deep_path = "a.b.c.d.e.f.g.h.i.j"
            settings.create(deep_path, default="deep")
            
            self.assertEqual(settings.get(deep_path), "deep")
    
    def test_special_characters_in_setting_name(self):
        """Test settings with special characters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = Path(temp_dir) / "test.json"
            settings = Settings(settings_file)
            
            # Create setting with special characters
            settings.create("test-special.setting_name", default=42)
            self.assertEqual(settings.get("test-special.setting_name"), 42)
    
    def test_none_values(self):
        """Test handling of None values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = Path(temp_dir) / "test.json"
            settings = Settings(settings_file)
            
            # Create setting with None value
            settings.create("test.none_value", default=None)
            self.assertIsNone(settings.get("test.none_value"))
    
    def test_complex_data_types(self):
        """Test handling of complex data types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = Path(temp_dir) / "test.json"
            settings = Settings(settings_file)
            
            # Test list
            settings.create("test.list", default=[1, 2, 3])
            self.assertEqual(settings.get("test.list"), [1, 2, 3])
            
            # Test dict
            settings.create("test.dict", default={"key": "value"})
            self.assertEqual(settings.get("test.dict"), {"key": "value"})
            
            # Test boolean
            settings.create("test.bool", default=True)
            self.assertTrue(settings.get("test.bool"))
    
    def test_reset_nonexistent_setting(self):
        """Test resetting non-existent setting raises KeyError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = Path(temp_dir) / "test.json"
            settings = Settings(settings_file)
            
            with self.assertRaises(KeyError):
                settings.reset("nonexistent.setting")
    
    def test_get_all_settings(self):
        """Test getting all settings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = Path(temp_dir) / "test.json"
            settings = Settings(settings_file)
            
            # Create multiple settings
            settings.create("test.setting1", default=42)
            settings.create("test.nested.setting2", default="hello")
            settings.create("other.setting3", default=True)
            
            all_settings = settings.get_all()
            
            self.assertIn("test", all_settings)
            self.assertIn("other", all_settings)
            self.assertEqual(all_settings["test"]["setting1"], 42)
            self.assertEqual(all_settings["test"]["nested"]["setting2"], "hello")
            self.assertTrue(all_settings["other"]["setting3"])


if __name__ == "__main__":
    # Configure logging to reduce noise during tests
    import logging
    logging.getLogger("config.settings").setLevel(logging.WARNING)
    
    # Run tests
    unittest.main(verbosity=2)
