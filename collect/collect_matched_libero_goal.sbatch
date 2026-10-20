#!/usr/bin/env bash
#SBATCH --job-name=matched-libero-goal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=matched-libero-goal-%j.out
#SBATCH --error=matched-libero-goal-%j.err

# Submit from the checkout root (or a directory inside it):
#   sbatch langgap/collect/collect_matched_libero_goal.sbatch
# Add your cluster's account / CPU partition on the sbatch command line.
# Optional environment overrides:
#   MATCHED_PROJECT_ROOT=/shared/path/VLA-Embd-Analysis
#   MATCHED_OUTPUT_DIR=/scratch/$USER/matched_goal_150  # must not exist
#   MATCHED_MAX_SCENES=600  MATCHED_SEED=0
#   MATCHED_REPO_ID=YOUR_ORG/libero_goal_matched_150  # metadata only; no upload
# Collects 150 matched scenes, then renders/packages both training cameras.
# Extra collector arguments follow the script path, e.g. --random-init.
# --videos additionally creates standalone replay videos (separate from training).
# For GPU rendering: MUJOCO_GL=egl sbatch --partition=gpu --gres=gpu:1 ...
set -euo pipefail

# Slurm executes a spool copy, so BASH_SOURCE alone cannot locate the checkout.
if [[ -n "${MATCHED_PROJECT_ROOT:-}" ]]; then
    matched_root="$(cd -- "$MATCHED_PROJECT_ROOT" && pwd)"
else
    matched_root="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
    while [[ ! -f "$matched_root/activate_env.sh" ||
             ! -f "$matched_root/langgap/collect/missing_scripts/collect_matched_libero_goal.sh" ]]; do
        if [[ "$matched_root" == / ]]; then
            echo 'Cannot locate checkout. Submit from inside it or set MATCHED_PROJECT_ROOT.' >&2
            exit 1
        fi
        matched_root="$(dirname -- "$matched_root")"
    done
fi

matched_launcher="$matched_root/langgap/collect/missing_scripts/collect_matched_libero_goal.sh"
if [[ ! -f "$matched_launcher" || ! -x "$matched_root/.venv-langgap/bin/python" ]]; then
    echo "Missing collector or root .venv-langgap. Run ./setup_env.sh langgap in $matched_root before submission." >&2
    exit 1
fi
cd -- "$matched_root"
for matched_argument in "$@"; do
    case "$matched_argument" in
        --episodes|--episodes=*|--output-dir|--output-dir=*)
            echo 'This batch job requires 150 scenes and owns its output path. Use MATCHED_OUTPUT_DIR to choose the location.' >&2
            exit 1 ;;
    esac
done

# CPU rendering needs OSMesa on the compute node; EGL is available as an override.
# Keep BLAS single-threaded: this job runs one serial matched collector, not nine
# independent collectors, so all tasks receive exactly the same accepted scenes.
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# The root environment and vendored LeRobot are used for both job stages.
source "$matched_root/activate_env.sh" langgap
export PYTHONPATH="$matched_root/langgap/lerobot/src:$matched_root/langgap${PYTHONPATH:+:$PYTHONPATH}"
matched_python="$matched_root/.venv-langgap/bin/python"
matched_converter="$matched_root/langgap/convert/convert_matched_libero_goal.py"
if [[ ! -f "$matched_converter" ]]; then
    echo "Missing matched conversion helper: $matched_converter" >&2
    exit 1
fi
# Check converter imports before spending time on collection.
"$matched_python" "$matched_root/langgap/convert/source_to_lerobot.py" --help >/dev/null
for matched_argument in "$@"; do
    if [[ "$matched_argument" == --help || "$matched_argument" == -h ]]; then
        exec bash "$matched_launcher" --help
    fi
done

matched_run="$(date -u +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local_$$}"
matched_output="${MATCHED_OUTPUT_DIR:-$matched_root/langgap/data/scripted_runs/matched_libero_goal/$matched_run}"
if [[ -e "$matched_output" ]]; then
    echo "Output directory must be new: $matched_output" >&2
    exit 1
fi
# The Python collector creates the run directory exclusively.
mkdir -p -- "$(dirname -- "$matched_output")"

printf 'Job: %s\nCheckout: %s\nOutput: %s\nRenderer: %s\n' \
    "${SLURM_JOB_ID:-local}" "$matched_root" "$matched_output" "$MUJOCO_GL"
printf 'Requesting 150 shared scenes across tasks 54,55,56,57,77,78,79,80,81 (150 demos per task).\n'

# The existing launcher activates the root .venv-langgap. Collection exits nonzero
# if the candidate budget cannot produce 150 complete nine-task bundles.
bash "$matched_launcher" \
    --episodes 150 \
    --max-scenes "${MATCHED_MAX_SCENES:-600}" \
    --seed "${MATCHED_SEED:-0}" \
    --output-dir "$matched_output" \
    "$@"


printf '\nCollection succeeded. Rendering every timestep from both cameras in HF LIBERO orientation.\n'
"$matched_python" -u "$matched_converter" \
    --run-dir "$matched_output" \
    --repo-id "${MATCHED_REPO_ID:-local/libero_goal_matched_150}" \
    --expected-scenes 150
printf '\nTraining dataset ready: %s/lerobot\n' "$matched_output"
