#!/usr/bin/env bash
# Runtime-validate a deployed Space or local server (OpenEnv CLI).
set -euo pipefail

ENV_URL="${1:-http://127.0.0.1:8000}"
ROOT="${2:-.}"

cd "${ROOT}"

if ! command -v openenv >/dev/null 2>&1; then
  echo "openenv CLI not found. Install with: pip install openenv-core"
  exit 1
fi

echo "Using ENV_URL=${ENV_URL} (cwd=$(pwd))"
openenv validate --url "${ENV_URL}" --timeout 30 -v
