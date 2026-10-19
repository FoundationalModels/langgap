#!/usr/bin/env bash
# Collect and render every registered libero_goal extended task.
set -euo pipefail
scripts_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$scripts_dir/../../.." && pwd)"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Usage: ./langgap/collect/missing_scripts/collect_libero_goal.sh [options]

Collect and render all nine libero_goal extended task IDs.
  --episodes N      Successful demos per task (default: 1)
  --attempts N      Maximum attempts per task (default: 4 * episodes)
  --output-dir DIR Base output directory (default: langgap/data/scripted_runs)

Uses the root .venv-langgap. Tasks sample initial scenes independently.
See collect_and_render.sh --help for output details.
EOF
    exit 0
fi
for ((i=1; i<=$#; i+=2)); do
    arg="${!i}"
    case "$arg" in
        --episodes|--attempts|--output-dir) ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
    if ((i == $#)); then echo "Missing value for $arg" >&2; exit 2; fi
done
source "$project_root/activate_env.sh" langgap
mapfile -t task_ids < <("$environment_dir/bin/python" - "$project_root/langgap" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from task_registry import EXTENDED_TASK_MAP
for task_id, task in sorted(EXTENDED_TASK_MAP.items()):
    if task['suite'] == 'libero_goal':
        print(task_id)
PY
)
if [[ ${#task_ids[@]} -eq 0 ]]; then echo "No libero_goal tasks found." >&2; exit 1; fi
failed=()
for task_id in "${task_ids[@]}"; do
    echo "Collecting libero_goal task $task_id"
    if ! "$scripts_dir/collect_and_render.sh" --task-id "$task_id" "$@"; then
        failed+=("$task_id")
    fi
done
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "Failed task IDs: ${failed[*]}" >&2
    exit 1
fi
echo "Collected and rendered all ${#task_ids[@]} libero_goal tasks."
