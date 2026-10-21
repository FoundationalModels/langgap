#!/usr/bin/env bash
# Collect all seven originals and nine extensions from identical scene snapshots.
set -euo pipefail
scripts_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$scripts_dir/../../../.." && pwd)"
exec bash "$project_root/langgap/collect/missing_scripts/collect_matched_libero_goal.sh" \
    --include-originals "$@"
