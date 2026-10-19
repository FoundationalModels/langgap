#!/usr/bin/env bash
set -euo pipefail
scripts_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$scripts_dir/../../.." && pwd)"
source "$project_root/activate_env.sh" langgap
exec "$environment_dir/bin/python" -u "$scripts_dir/collect_matched_libero_goal.py" "$@"
