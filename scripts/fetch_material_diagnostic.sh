#!/usr/bin/env bash
# Run on the local computer to collect actual images and comparison JSON.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs/material-diagnostic-3h4h-20261014
scp -P 39291 -o ControlPath=/private/tmp/ftmaterial-autodl-%C -r \
  root@region-9.autodl.pro:/root/autodl-tmp/saves/qwen3-8b/lora/material-diagnostic-3h4h-20261014/diagnostics \
  outputs/material-diagnostic-3h4h-20261014/
