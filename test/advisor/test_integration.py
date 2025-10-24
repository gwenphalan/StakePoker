#!/usr/bin/env python3
"""
Simple integration test to verify the advisor test suite works.

This test verifies that all test files can be imported and basic functionality works.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

def test_imports():
    """Test that all test modules can be imported."""
    test_dir = Path(__file__).parent
    
    test_modules = [
        "test_config",
        "test_gto_loader", 
        "test_equity_calculator",
        "test_range_estimator", 
        "test_postflop_solver",
        "test_decision_engine"
    ]
    
    for module_name in test_modules:
        try:
            __import__(module_name)
            print(f"[OK] {module_name} imports successfully")
        except ImportError as e:
            print(f"[FAIL] {module_name} import failed: {e}")
            return False
    
    return True

def test_fixtures():
    """Test that test fixtures work correctly."""
    try:
        from test_config import AdvisorTestFixtures, MockAdvisorComponents
        
        # Test card creation
        cards = AdvisorTestFixtures.create_test_cards()
        assert len(cards) > 0
        assert cards[0].rank == 'A'
        assert cards[0].suit == 'hearts'
        print("[OK] Test cards creation works")
        
        # Test hero creation
        hero = AdvisorTestFixtures.create_test_hero()
        assert hero.is_hero is True
        assert hero.username == "Hero"
        assert len(hero.hole_cards) == 2
        print("[OK] Test hero creation works")
        
        # Test game state creation
        game_state = AdvisorTestFixtures.create_test_game_state()
        assert len(game_state.players) == 2
        assert game_state.pot == 100
        print("[OK] Test game state creation works")
        
        # Test mock components
        mock_gto = MockAdvisorComponents.mock_gto_loader()
        assert mock_gto is not None
        print("[OK] Mock components creation works")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Test fixtures failed: {e}")
        return False

def test_temp_charts():
    """Test temporary GTO chart creation."""
    try:
        from test_config import AdvisorTestFixtures
        
        temp_dir = AdvisorTestFixtures.create_temp_gto_charts()
        assert temp_dir.exists()
        
        # Check that files were created
        metadata_file = temp_dir / "metadata.json"
        assert metadata_file.exists()
        
        utg_file = temp_dir / "UTG.json"
        assert utg_file.exists()
        
        utg_6max_file = temp_dir / "UTG_6max.json"
        assert utg_6max_file.exists()
        
        print("[OK] Temporary GTO charts creation works")
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Temporary charts test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("Running Advisor Test Suite Integration Tests")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Fixture Tests", test_fixtures), 
        ("Temp Charts Tests", test_temp_charts)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"[FAIL] {test_name} failed")
    
    print(f"\n{'='*50}")
    print(f"Integration Tests: {passed}/{total} passed")
    
    if passed == total:
        print("[OK] All integration tests passed!")
        return 0
    else:
        print("[FAIL] Some integration tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
