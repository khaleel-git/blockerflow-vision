#!/bin/bash
# Builds build/blockerflow-vision, a universal binary (Apple silicon + Intel), macOS 12.3 or newer.
#   ./build.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
export CGO_ENABLED=1 MACOSX_DEPLOYMENT_TARGET=12.3
GOOS=darwin GOARCH=arm64 CC="clang -arch arm64" go build -trimpath -ldflags "-s -w" -o build/blockerflow-vision-arm64 .
GOOS=darwin GOARCH=amd64 CC="clang -arch x86_64" go build -trimpath -ldflags "-s -w" -o build/blockerflow-vision-amd64 .
lipo -create -output build/blockerflow-vision build/blockerflow-vision-arm64 build/blockerflow-vision-amd64
rm build/blockerflow-vision-arm64 build/blockerflow-vision-amd64
echo "Built build/blockerflow-vision ($(lipo -archs build/blockerflow-vision))"
