#!/bin/bash
set -e

BUILD_DIR="build/Release"

# Robustness check: If the dependency folder is missing, we MUST re-configure
if [ ! -d "$BUILD_DIR/_deps" ]; then
    echo "Dependencies missing. Forcing re-configuration..."
    rm -rf "$BUILD_DIR"
fi

if [ ! -f "$BUILD_DIR/build.ninja" ]; then
    echo "Configuring build system with Ninja..."
    cmake -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release
fi

# Build project
echo "Building the project..."
cmake --build "$BUILD_DIR" -j "$(nproc)"

echo "Done!"