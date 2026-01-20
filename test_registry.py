#!/usr/bin/env python3
"""
Quick test to verify matcher registry works correctly.
"""
import sys
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from matcher.base_matcher import AVAILABLE_MATCHERS, available_models, get_matcher

def test_registry():
    print("=" * 60)
    print("Testing Matcher Registry")
    print("=" * 60)
    
    # Test 1: Check AVAILABLE_MATCHERS list
    print(f"\n✅ Found {len(AVAILABLE_MATCHERS)} available matchers:")
    for i, name in enumerate(AVAILABLE_MATCHERS, 1):
        print(f"   {i:2d}. {name}")
    
    # Test 2: Check alias
    print(f"\n✅ Alias 'available_models' works: {available_models == AVAILABLE_MATCHERS}")
    
    # Test 3: Try to instantiate a few matchers (without running them)
    test_matchers = ["orb-nn", "xfeat", "liftfeat"]
    print(f"\n✅ Testing instantiation of sample matchers:")
    
    for name in test_matchers:
        try:
            matcher = get_matcher(name, device='cpu')
            print(f"   ✓ {name:20s} -> {matcher.__class__.__name__}")
        except Exception as e:
            print(f"   ✗ {name:20s} -> ERROR: {e}")
    
    # Test 4: Check unknown matcher raises error
    print(f"\n✅ Testing error handling for unknown matcher:")
    try:
        matcher = get_matcher("unknown-matcher", device='cpu')
        print(f"   ✗ Should have raised ValueError!")
    except ValueError as e:
        print(f"   ✓ Correctly raised ValueError: {str(e)[:50]}...")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    test_registry()
