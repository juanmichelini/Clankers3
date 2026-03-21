#!/usr/bin/env bash
# Build WASM as ES module for main-thread rendering
set -e
export PATH="$PATH:$HOME/.cargo/bin"
wasm-pack build --target web --out-dir ../web/wasm
echo "Build done"
