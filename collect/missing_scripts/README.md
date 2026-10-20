# Scripted demo collection

Run these commands from the repository root.

## Matched collection: identical starts across all nine goal tasks

```bash
./langgap/collect/missing_scripts/collect_matched_libero_goal.sh --episodes 5
```

This requests **5 complete matched scenes × 9 tasks = 45 demos**. The runner uses
the root `.venv-langgap`. Add `--videos` to render the exported demos.

The runner checks that tasks 54, 55, 56, 57, 77, 78, 79, 80, and 81 have matching
physical BDDL definitions. Task instructions, goals, and objects of interest may
differ. For each candidate it uses a row from task 54's existing `.pruned_init`
bank, samples fixture poses once, settles once, and captures the full scene.
Use `--random-init` to sample the canonical scene from BDDL instead, or
`--init-file PATH` to choose a compatible starting-state bank.

Before each policy runs, it restores model body poses, simulation time,
positions/velocities, control inputs, warm-start state, and gripper state, and
resets controller goals to the restored pose. It asserts exact initial-state and
initial-camera-image equality. The collectors skip their normal reset/settling
block in this mode, so the first recorded state is identical across tasks.
`--no-verify-images` skips pixel verification and rendering while retaining exact
model/state checks.

A candidate is accepted only if **all nine policies succeed**. If one fails, the
entire bundle is rejected. The runner tries up to `4 * episodes` candidates by
default; use `--max-scenes N` to change that budget. If it falls short, it exits
with an error and exports only the complete matched bundles collected so far.

Default runs live under `langgap/data/scripted_runs/matched_libero_goal/`:

- `scene_bank.hdf5`: reusable candidate snapshots, including reference images.
- `matched_demos.hdf5`: complete accepted scenes, each containing all nine tasks.
- `task54.hdf5`, …, `task81.hdf5`: standard per-task collector exports.
- `manifest.json`: accepted/rejected scenes, hashes, failed task IDs, and run status.
- `videos/task<ID>/`: optional action replay videos.

Per-task episodes carry `scene_id`, `scene_hash`, and `initial_image_hash`.
`scene_id` is local to the bank; `scene_hash` distinguishes scenes across banks.
Match examples using these attributes, rather than assuming independent demo
numbers correspond.

Reuse exactly the same candidate scenes in another run:

```bash
./langgap/collect/missing_scripts/collect_matched_libero_goal.sh \
  --episodes 5 \
  --scene-bank /path/to/previous/run/scene_bank.hdf5 \
  --output-dir langgap/data/scripted_runs/matched_goal_repeat
```

The source bank is opened read-only. `--output-dir` must be a new directory.
Task init files 56, 80, and 81 are not required: all tasks restore the shared bank.

Validation: one complete nine-task scene passed exact state and pixel checks.
Reusing its bank reproduced bit-identical state/action recordings for all nine
tasks. An injected policy failure produced no partial task exports and left the
source bank unchanged.

## Collect 150 matched scenes on Slurm

Set up the root `.venv-langgap` on the cluster with `./setup_env.sh langgap`
before submitting. From the repository root:

```bash
sbatch langgap/collect/collect_matched_libero_goal.sbatch
```

This submits one serial matched collector: **150 shared configurations × nine
tasks = 150 demos per task**. It allows 600 candidate scenes by default, requests
four CPUs, 16 GB RAM and 48 hours, and uses CPU OSMesa rendering. Initial physical
state and reference image checks remain enabled. The compute node needs OSMesa;
load your cluster's Mesa module if required. Add your cluster's account,
partition or time limit as `sbatch` options. For GPU EGL rendering, use:

```bash
MUJOCO_GL=egl sbatch --partition=gpu --gres=gpu:1 \
  langgap/collect/collect_matched_libero_goal.sbatch
```

Choose a shared/scratch output directory and a larger candidate budget with:

```bash
MATCHED_OUTPUT_DIR=/scratch/$USER/matched_goal_150 MATCHED_MAX_SCENES=900 \
  sbatch langgap/collect/collect_matched_libero_goal.sbatch
```

The output directory must not exist. By default it is
`langgap/data/scripted_runs/matched_libero_goal/<UTC timestamp>_<job ID>/`.
Slurm writes `matched-libero-goal-<job ID>.out` and `.err` in the submission
working directory. The job uses the existing root environment and does not install
packages. Submit from inside the checkout, or set `MATCHED_PROJECT_ROOT` to its
absolute cluster path.

After successful collection, the same job automatically converts all nine task
files into a **training-ready LeRobot dataset** at `<run directory>/lerobot/`.
It renders agent-view and wrist-camera images at every recorded timestep
(`frame_skip=1`), rotated 180 degrees from the raw simulator arrays to match
original HF LIBERO training images. The videos, robot observations, actions and
instructions are packaged together. Collection and rendering run sequentially;
images are generated during this automatic conversion stage.

The dataset includes `meta/matched_demos.hdf5`, `meta/matched_episodes.json` and
`meta/image_convention.json` for restoring and identifying the same 150 scenes
at evaluation time. The job verifies 1,350 nonempty episodes and that every task
contains each scene exactly once, including the scene hashes. It also checks
that both training camera videos exist. It exits nonzero on collection or
conversion failure.

Set `MATCHED_REPO_ID=YOUR_ORG/YOUR_DATASET` when submitting to choose the identity
recorded in dataset metadata; the default is `local/libero_goal_matched_150`.
The batch job does not upload to Hugging Face. Add `--videos` after the script
path only if you also want standalone replay videos; the two training camera
videos are always generated automatically. `--episodes` and `--output-dir`
cannot be overridden as collector arguments in this fixed 150-scene batch job.

If collection has already succeeded, conversion can also be run independently:

```bash
source activate_env.sh langgap
python langgap/convert/convert_matched_libero_goal.py \
  --run-dir /path/to/completed/collection \
  --repo-id YOUR_ORG/YOUR_DATASET
```

The helper refuses to overwrite an existing `lerobot/` dataset.

## Evaluate all 150 training configurations

Collect 150 accepted scenes with `collect_matched_libero_goal.sh --episodes 150`.
This produces 150 demos **per task**, sharing the same 150 initial configurations
across all nine tasks. Keep `matched_demos.hdf5` with the trained checkpoint;
copy this archive to the evaluation machine as well.

```bash
source activate_env.sh langgap
python langgap/eval/unified_eval.py \
  --model_path /path/to/trained/pretrained_model \
  --matched-scenes /path/to/training/run/matched_demos.hdf5 \
  --max_steps 500 \
  --output_dir langgap/eval/results/matched_goal_150
```

The matched flag selects the nine extended goal tasks automatically, requires
exactly 150 complete accepted scenes, and runs each once per task: **1,350 policy
episodes**. Omit `--episodes`; there is no modulo cycling through the 50-row init
bank. Use `--task_id 56` to evaluate a task subset over all 150 scenes. For a
smaller validation bank, specify its actual count with `--expected-scenes N`.
`--dry_run` validates the archive and lists selected tasks without loading a model.

Before the first policy action, the evaluator restores and verifies the saved
robot joint/gripper configuration, object poses, fixture model poses, simulation
state and integration inputs. It resets policy/processor state between episodes.
It performs no settling actions after snapshot restoration. Each episode in
`results.json` records `scene_id` and `scene_hash`; `evaluation_scenes.json` records
the complete scene schedule. Restoration errors make the evaluation exit with an
error rather than fall back to randomly initialized scenes.

The archive is self-contained: collection-machine paths in its metadata do not
need to exist on the evaluation machine. Use the same simulator/package versions,
robot configuration and assets. Exact physical model/state checks always run;
add `--verify-scene-images` to require pixel equality too. Different rendering
hardware can change pixels, and matching the initial state does not guarantee
identical subsequent physics across different machines.

## Camera orientation for fine-tuning

The converter stores both cameras rotated 180 degrees from raw simulator arrays,
matching original HF LIBERO images and the evaluation environment processor.
Fine-tuning then resizes/normalizes stored images without adding another flip.
Earlier vertical-only converted extended videos need regeneration from HDF5.
See [the image-convention audit](../../convert/IMAGE_CONVENTIONS.md) for the
OpenPI/LeRobot source trace and regression checks.

## Restore from the Hugging Face training dataset only

When converting all nine matched task HDF5 files with
`convert/source_to_lerobot.py`, add:

```bash
--matched-scenes /path/to/run/matched_demos.hdf5
```

The converter retains every matched scene for all nine tasks, preserves fixture
poses while rendering, and writes these additional dataset files:

- `meta/matched_demos.hdf5`: self-contained accepted snapshots and source demos.
- `meta/matched_episodes.json`: converted episode index → source demo, BDDL,
  scene ID and hash.

These files are included by the converter's existing `--push` option. Matched
inputs require `--matched-scenes`, and trimming or partial scene conversion is rejected.
No upload occurs unless `--push` is explicitly supplied.

On the evaluation machine:

```bash
source activate_env.sh langgap
python langgap/eval/unified_eval.py \
  --model_path /path/to/trained/pretrained_model \
  --matched-dataset YOUR_ORG/YOUR_TRAINING_DATASET \
  --dataset-revision TRAINING_DATASET_COMMIT \
  --max_steps 500
```

The evaluator downloads `meta/matched_demos.hdf5` from that dataset revision,
checks all 150 scenes, and evaluates all nine tasks from those saved starts.
Pin the dataset commit used for training. Private datasets use the machine's
normal Hugging Face authentication. A local collection directory, original init
banks and collector scripts are not needed on the evaluation machine.

Existing datasets containing only images, policy observation vectors and actions
cannot reconstruct exact simulator scenes: those observations omit object and
fixture configurations. Upload the original matched archive into the same dataset
at `meta/matched_demos.hdf5`, or reconvert with the option above. The simulator,
compatible assets and restoration helper still need to be installed locally.

## All libero_goal extended tasks

Every registered extended `libero_goal` task has a collector. Existing collectors
stay in `langgap/collect/`; new collectors are in `missing_scripts/`.

| Task | Instruction | Collector directory |
|---|---|---|
| 54 | Cream cheese onto stove | `collect/` |
| 55 | Wine bottle onto plate | `collect/missing_scripts/` |
| 56 | Open top drawer and put cream cheese inside | `collect/missing_scripts/` |
| 57 | Open bottom drawer | `collect/missing_scripts/` |
| 77 | Wine bottle onto stove | `collect/missing_scripts/` |
| 78 | Cream cheese onto cabinet | `collect/` |
| 79 | Cream cheese onto plate | `collect/` |
| 80 | Black bowl in front of stove | `collect/missing_scripts/` |
| 81 | Wine bottle in front of stove | `collect/missing_scripts/` |

Collector filenames are `scripted_collect_task<ID>.py`.

Collect and render all nine tasks:

```bash
./langgap/collect/missing_scripts/collect_libero_goal.sh --episodes 5
```

Or choose one task:

```bash
./langgap/collect/missing_scripts/collect_and_render.sh --task-id 77 --episodes 5
```

The batch launcher derives task IDs and BDDL paths from `task_registry.py`,
continues after individual task failures, and exits with an error if any task
fails. Task starts are sampled independently; this does not enforce matching
initial scenes across tasks.

These goal collectors save pre-action simulator states, actions, sampled model
body poses, and ordered body names in HDF5. The replay script restores those model
poses before the initial simulator state, preserving stove and cabinet placement.
Older HDF5 files without model-pose metadata remain readable. Placement policies
check actual object location and contact after release and retreat, in addition
to LIBERO's goal predicate.

Validation: three randomized attempts per task saved 26/27 successful demos.
Every task passed action replay of a saved demo with independent physical-goal
checks. Validation artifacts are under `langgap/data/scripted_runs/libero_goal_validation/`.

## Collect and save replay videos

```bash
./langgap/collect/missing_scripts/collect_and_render.sh
```

The wrapper activates the root `.venv-langgap` environment, collects one task-55
demo (up to four attempts), and replays every saved demo into an MP4 at 20 FPS.
Each run creates a timestamped directory under
`langgap/data/scripted_runs/task_55/`, containing:

- `demos.hdf5`: simulator states and actions.
- `videos/episode_000.mp4`, etc.: action replay videos.
- `collect.log` and `replay.log`: collection and replay results.

```bash
./langgap/collect/missing_scripts/collect_and_render.sh --episodes 5
./langgap/collect/missing_scripts/collect_and_render.sh --task-id 43 --episodes 3 --attempts 20
./langgap/collect/missing_scripts/collect_and_render.sh --output-dir /tmp/langgap-runs
```

Other task IDs require an existing `scripted_collect_task<ID>.py` collector.
All saved demos are rendered, including partial collections that fall short of
the requested count. Errors stop the wrapper and retain the run's files.

## Task 55: wine bottle onto plate

The task-specific collector follows the existing collectors' CLI and HDF5 format:

```bash
source activate_env.sh langgap
python langgap/collect/missing_scripts/scripted_collect_task55.py \
  --bddl langgap/data/bddl_files/libero_goal/extended/dim1_change_object/ext_06_wine_bottle_to_plate.bddl \
  --target 5 --num 20
```

Use `--debug` for one attempt with waypoint diagnostics, and `--output` to choose
a new HDF5 path. Existing output files are not overwritten. The wrapper is the
convenient option when you also want videos.

## Task 56: cream cheese into top drawer

The robot grasps and pulls the top-drawer handle, releases it, then picks up the
cream cheese and places it in the exposed drawer. Only completed, successful
rollouts are saved; the drawer is opened through recorded robot actions.

```bash
./langgap/collect/missing_scripts/collect_and_render.sh --task-id 56 --episodes 5
```

For collection without video rendering:

```bash
source activate_env.sh langgap
python langgap/collect/missing_scripts/scripted_collect_task56.py \
  --bddl langgap/data/bddl_files/libero_goal/extended/dim1_change_object/ext_09_cream_cheese_to_top_drawer.bddl \
  --target 5 --num 20
```

`--debug` runs one attempt with waypoint diagnostics. `--grasp_offset` and
`--place_offset` adjust the cheese grasp height and release clearance in meters.
The gripper orientation is computed from the handle bar's actual collision-box
orientation and the drawer slide axis, so it follows sloped or rotated handles.
`--handle_tilt` controls a clearance angle in that handle frame: 0° points down
and 90° points straight into the drawer. The default 60° is 30° more upright
than the original horizontal grasp for this task's horizontal handle. It is
not a fixed world-space orientation. The grasp includes a
small forward clearance so the rear finger avoids the drawer front. The drawer
phase waits for orientation error below 3° before advancing waypoints.

Handle orientation checks cover slopes from −45° to +45°. Failed attempts are
omitted.

## Additional collectors

Task 57 uses a closed-finger hook behind the bottom handle, then pulls and
retreats before checking that the drawer is open. Task 77 reuses task 55's
bottle-neck grasp, targeting the stove's physical burner surface. Task 80 uses
the existing bowl collectors' rim-grasp strategy and compensates the measured
gripper-to-bowl offset. Task 81 uses the bottle-neck grasp and places the bottle
inside the table's stove-front region. The table-region collectors verify both
region membership and actual table contact.
