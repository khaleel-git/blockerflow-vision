#!/bin/bash
# Builds build/windows-amd64/blockerflow-vision.exe and build/windows-arm64/blockerflow-vision.exe.
# Pure Go (no C compiler): ONNX Runtime is called through its C API from model_windows.go.
#   ./build-windows.sh
set -euo pipefail
cd "$(dirname "$0")"
for arch in amd64 arm64; do
  mkdir -p "build/windows-$arch"
  CGO_ENABLED=0 GOOS=windows GOARCH=$arch go build -trimpath -ldflags "-s -w" -o "build/windows-$arch/blockerflow-vision.exe" .
done
ls -l build/windows-*/blockerflow-vision.exe
