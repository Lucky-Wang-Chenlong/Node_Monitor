#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <user> [--output-dir ./collector_data] [--push-url http://center:18080/api/collector] [--push-token xxx] [--interval 10] [--gpu-timeout 30] [--gpu-retries 2]"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
user="$1"
shift || true

exec "$PYTHON_BIN" "$SCRIPT_DIR/user_gpu_collector.py" --user "$user" "$@"
