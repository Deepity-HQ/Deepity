#!/bin/bash

set -e

BUILD_DIR="build/Release"
CACHE_FILE="$BUILD_DIR/CMakeCache.txt"

# Conditionally configure the build directory
if [ ! -f "$CACHE_FILE" ]; then
    echo "Configuration not found or incomplete. Generating build system with Ninja..."
    # -G Ninja specifies the generator
    # -DCMAKE_BUILD_TYPE=Release ensures compiler optimizations are enabled
    cmake -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release -DHAVE_STD_REGEX=ON
else
    echo "Build system already configured. Skipping to build..."
fi

# Build project
echo "Building the project..."
# -j "$(nproc)" automatically counts your CPU cores and uses all of them
cmake --build "$BUILD_DIR" -j "$(nproc)"

echo "Done!"