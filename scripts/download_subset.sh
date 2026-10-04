#!/usr/bin/env bash
# Smart-subset download via bearing-datasets (resumable, cached in data/_raw).
# Covers: 2 healthy + 2 artificial + 4 real (IR + OR mix) = all 3 classes x both damage origins.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export BEARING_DATASETS_ROOT="${BEARING_DATASETS_ROOT:-$ROOT/data}"

uv run bearing-datasets root "$ROOT/data" || true
uv run bearing-datasets build paderborn \
  --files "K001,K002,KA01,KA15,KI01,KI14,KB23,KB27" \
  || {
  echo "--- partial/glob build failed, trying full paderborn build (needs ~10GB free) ---"
  uv run bearing-datasets build paderborn
}
uv run bearing-datasets list
