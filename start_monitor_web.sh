#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 user1 [user2 ...] [--partitions p1 p2 ...] [--port 18080] [--refresh 15]"
  echo "or:    $0 --users user1 user2 [--collector-token xxx] [--allow-remote-ui]"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ "${1:-}" = "--users" ]; then
  exec "$PYTHON_BIN" "$SCRIPT_DIR/slurm_monitor_center.py" --serve "$@"
fi

users=()
rest=()
seen_opt=0
for arg in "$@"; do
  if [ "$seen_opt" -eq 0 ] && [[ "$arg" == --* ]]; then
    seen_opt=1
  fi
  if [ "$seen_opt" -eq 0 ]; then
    users+=("$arg")
  else
    rest+=("$arg")
  fi
done

if [ "${#users[@]}" -eq 0 ]; then
  echo "Error: at least one user is required."
  exit 1
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/slurm_monitor_center.py" --serve --users "${users[@]}" "${rest[@]}"
