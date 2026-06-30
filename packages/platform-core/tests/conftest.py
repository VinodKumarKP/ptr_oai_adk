"""Pytest configuration for platform-core.

Ensures that the local src/ directory is prioritized for imports over
installed packages in site-packages.
"""
import sys
import os

# Add src directory to the beginning of sys.path so local code is imported first
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
