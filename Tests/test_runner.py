#!/usr/bin/env python3
"""
Simple test runner for blackjack game
Run this to execute all tests and see results
"""

import subprocess
import sys
import os

def run_tests():
    """Run all tests and display results"""
    print("=" * 60)
    print("RUNNING BLACKJACK TESTS")
    print("=" * 60)
    
    # Check if pytest is installed
    try:
        import pytest
    except ImportError:
        print("❌ pytest not found! Please install it with:")
        print("   pip install pytest")
        return False
    
    # Run tests with verbose output
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "Tests/test_play.py",  # Replace with your test file name
            "-v",  # Verbose output
            "--tb=short",  # Shorter traceback format
            "--color=yes"  # Colored output
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print("\n✅ All tests passed!")
            return True
        else:
            print(f"\n❌ Tests failed with return code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def run_specific_test(test_name):
    """Run a specific test or test class"""
    print(f"Running specific test: {test_name}")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            f"test_blackjack.py::{test_name}",
            "-v"
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
    except Exception as e:
        print(f"❌ Error running test: {e}")

def main():
    """Main function to handle command line arguments"""
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1]
        run_specific_test(test_name)
    else:
        # Run all tests
        run_tests()

if __name__ == "__main__":
    main()