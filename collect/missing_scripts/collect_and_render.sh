#!/usr/bin/env bash
# Collect scripted LangGap demos and replay every saved demo into an MP4.
set -euo pipefail

scripts_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
collect_dir="$(cd -- "$scripts_dir/.." && pwd)"
langgap_dir="$(cd -- "$collect_dir/.." && pwd)"
project_root="$(cd -- "$langgap_dir/.." && pwd)"

usage() {
    cat <<'EOF'
Usage: ./langgap/collect/missing_scripts/collect_and_render.sh [options]

  --task-id ID       Extended task to collect (default: 55)
  --episodes N       Successful demos requested (default: 1)
  --attempts N       Maximum collection attempts (default: 4 * episodes)
  --output-dir DIR  Base directory for timestamped runs (default: langgap/data/scripted_runs)
  -h, --help        Show this help

Each run saves demos.hdf5, videos/episode_*.mp4, collect.log, and replay.log
in a new subdirectory. Uses the root .venv-langgap environment automatically.
EOF
}

task_id=55
episodes=1
attempts=""
output_base="$langgap_dir/data/scripted_runs"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --task-id|--episodes|--attempts|--output-dir)
            if [[ $# -lt 2 || -z "$2" ]]; then
                echo "Missing value for $1" >&2
                exit 2
            fi
            case "$1" in
                --task-id) task_id="$2" ;;
                --episodes) episodes="$2" ;;
                --attempts) attempts="$2" ;;
                --output-dir) output_base="$2" ;;
            esac
            shift 2
            ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! "$task_id" =~ ^[1-9][0-9]*$ || ! "$episodes" =~ ^[1-9][0-9]*$ ]]; then
    echo "--task-id and --episodes must be positive integers." >&2
    exit 2
fi
attempts="${attempts:-$((episodes * 4))}"
if [[ ! "$attempts" =~ ^[1-9][0-9]*$ || "$attempts" -lt "$episodes" ]]; then
    echo "--attempts must be a positive integer at least as large as --episodes." >&2
    exit 2
fi

source "$project_root/activate_env.sh" langgap
python_bin="$environment_dir/bin/python"

collector="$scripts_dir/scripted_collect_task${task_id}.py"
if [[ ! -f "$collector" ]]; then
    collector="$collect_dir/scripted_collect_task${task_id}.py"
fi
if [[ ! -f "$collector" ]]; then
    echo "No scripted collector is available for task $task_id." >&2
    exit 2
fi

bddl_path="$("$python_bin" - "$project_root" "$task_id" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
sys.path.insert(0, str(root / "langgap"))
from task_registry import EXTENDED_TASK_MAP

task_id = int(sys.argv[2])
if task_id not in EXTENDED_TASK_MAP:
    sys.exit(f"Unknown extended task ID: {task_id}")
task = EXTENDED_TASK_MAP[task_id]
path = root / "langgap/data/bddl_files" / task["suite"] / "extended" / task["dim"] / (task["bddl"] + ".bddl")
if not path.is_file():
    sys.exit(f"BDDL file not found: {path}")
print(path)
PY
)"

mkdir -p "$output_base/task_$task_id"
run_dir="$(mktemp -d "$output_base/task_$task_id/$(date +%Y%m%d_%H%M%S)_XXXXXX")"
run_dir="$(cd -- "$run_dir" && pwd)"
hdf5_path="$run_dir/demos.hdf5"
echo "Run directory: $run_dir"
echo "Collecting up to $episodes successful demos for task $task_id ($attempts attempts maximum)..."
"$python_bin" -u "$collector" \
    --bddl "$bddl_path" --target "$episodes" --num "$attempts" \
    --output "$hdf5_path" 2>&1 | tee "$run_dir/collect.log"

n_demos="$("$python_bin" - "$hdf5_path" <<'PY'
import sys
import h5py

with h5py.File(sys.argv[1], "r") as f:
    count = sum(name.startswith("demo_") for name in f["data"])
if count == 0:
    sys.exit("Collector saved no demos.")
print(count)
PY
)"
if [[ "$n_demos" -lt "$episodes" ]]; then
    echo "Collected $n_demos/$episodes requested demos; rendering all saved demos."
fi

echo "Rendering $n_demos demos to videos at 20 FPS..."
"$python_bin" -u "$langgap_dir/process/replay_dataset.py" \
    --dataset "$hdf5_path" --bddl "$bddl_path" \
    --episodes "$n_demos" --frame_skip 1 --fps 20 \
    --output_dir "$run_dir/videos" 2>&1 | tee "$run_dir/replay.log"

shopt -s nullglob
videos=("$run_dir"/videos/episode_*.mp4)
if [[ "${#videos[@]}" -ne "$n_demos" ]]; then
    echo "Expected $n_demos videos, found ${#videos[@]}. See $run_dir/replay.log" >&2
    exit 1
fi

echo "Saved $n_demos demos and ${#videos[@]} videos in: $run_dir"
