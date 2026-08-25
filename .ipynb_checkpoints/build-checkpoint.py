#!/usr/bin/env python3
"""
Deepity build entrypoint.

All real logic lives in the deepity_build package (config resolution, the
CMake command builder, and the rich/plain reporters). This file just wires
argv to that package so `python build.py ...` keeps working exactly as
before.
"""

from deepity_build.cli import main

if __name__ == "__main__":
    main()