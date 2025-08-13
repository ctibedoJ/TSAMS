#!/usr/bin/env python3
"""
Prepare the TSAMS Prime Generator package for GitHub.

This script:
1. Verifies that all required files are present
2. Runs basic tests to ensure functionality
3. Creates a simple example output to demonstrate the package
4. Provides instructions for pushing to GitHub
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path


def check_file_exists(filepath):
    """Check if a file exists and print status."""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"{status} {filepath}")
    return exists


def run_command(command):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr


def main():
    """Main function to prepare the package for GitHub."""
    print("Preparing TSAMS Prime Generator for GitHub...\n")
    
    # Check if we're in the right directory
    if not os.path.exists("setup.py"):
        print("Error: This script must be run from the prime_generator directory.")
        print("Current directory:", os.getcwd())
        sys.exit(1)
    
    # Check required files
    print("Checking required files:")
    required_files = [
        "setup.py",
        "README.md",
        "LICENSE",
        ".gitignore",
        "prime_generator/__init__.py",
        "prime_generator/algorithms/__init__.py",
        "prime_generator/algorithms/generation.py",
        "prime_generator/algorithms/testing.py",
        "prime_generator/algorithms/operations.py",
        "prime_generator/algorithms/tsams_primes.py",
        "prime_generator/visualization/__init__.py",
        "prime_generator/visualization/spirals.py",
        "prime_generator/visualization/distribution.py",
        "prime_generator/visualization/interactive.py",
        "prime_generator/visualization/tsams_visualizations.py",
        "prime_generator/utils/__init__.py",
        "prime_generator/utils/statistics.py",
        "prime_generator/utils/special_primes.py",
        "prime_generator/utils/number_theory.py",
        "prime_generator/utils/tsams_utils.py",
        "prime_generator/tests/__init__.py",
        "prime_generator/tests/test_basic.py",
        "prime_generator/tests/test_tsams.py",
        "examples/basic_usage.py",
        "examples/visualizations.py"
    ]
    
    all_files_exist = True
    for filepath in required_files:
        if not check_file_exists(filepath):
            all_files_exist = False
    
    if not all_files_exist:
        print("\nError: Some required files are missing.")
        sys.exit(1)
    
    print("\nAll required files are present.")
    
    # Run basic tests
    print("\nRunning basic tests...")
    success, output = run_command("python -m unittest discover -s prime_generator/tests")
    
    if not success:
        print("Error: Tests failed.")
        print(output)
        sys.exit(1)
    
    print("All tests passed successfully.")
    
    # Create a simple example output
    print("\nGenerating example output...")
    
    try:
        # Try to import the package
        spec = importlib.util.spec_from_file_location("prime_generator", "prime_generator/__init__.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Generate some primes using different methods
        from prime_generator import sieve_of_eratosthenes, cyclotomic_sieve
        
        limit = 50
        classical_primes = sieve_of_eratosthenes(limit)
        cyclotomic_primes = cyclotomic_sieve(limit, conductor=8)
        
        print(f"Classical sieve found {len(classical_primes)} primes up to {limit}:")
        print(classical_primes)
        
        print(f"\nCyclotomic sieve found {len(cyclotomic_primes)} primes up to {limit}:")
        print(cyclotomic_primes)
        
        print("\nPackage imported and executed successfully.")
        
    except Exception as e:
        print(f"Error: Failed to import and run the package: {e}")
        sys.exit(1)
    
    # Provide instructions for GitHub
    print("\nInstructions for pushing to GitHub:")
    print("1. Create a new repository on GitHub named 'tsams-prime-generator'")
    print("2. Initialize git and push the package:")
    print("   git init")
    print("   git add .")
    print("   git commit -m 'Initial commit of TSAMS Prime Generator'")
    print("   git branch -M main")
    print("   git remote add origin https://github.com/yourusername/tsams-prime-generator.git")
    print("   git push -u origin main")
    print("\nThe TSAMS Prime Generator package is ready for GitHub!")


if __name__ == "__main__":
    main()