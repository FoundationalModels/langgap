# LIBERO image convention for VLA fine-tuning

For both agent-view and wrist cameras, store RGB images in the same orientation
as the original Hugging Face LIBERO demonstrations:

```python
stored_training_image = raw_simulator_image[::-1, ::-1].copy()
```

Apply this rotation exactly once when converting raw simulator observations.
Already-converted Hugging Face training images must not be rotated again.
Raw simulator evaluation observations receive the same rotation through the
LIBERO environment processor. Policy-specific resizing and numeric normalization
then apply to both training and evaluation images without an orientation change.
A vertical-only flip is an upright display convention, but it does not match
this training convention. Preview/reference images are not training inputs.

## Verified reference implementations

### Physical Intelligence OpenPI pi0.5

- The `pi05_libero` configuration uses `physical-intelligence/libero`, with
  `LeRobotLiberoDataConfig`. This is a different repository/schema from
  LeRobot's `lerobot/libero`.
- The example RLDS-to-LeRobot converter copies image and wrist-image arrays
  without adding a flip.
- `LiberoInputs` changes image dtype/layout and camera key names, preserving
  spatial orientation. Model transforms resize with padding to 224 x 224;
  image values are mapped to [-1, 1].
- The JAX training observation preprocessor optionally adds a 95% crop and
  +/-5-degree rotation for non-wrist views, plus color jitter. It does not add
  a horizontal/vertical flip or 180-degree rotation to stored dataset images.
- The evaluation client explicitly rotates both raw simulator camera arrays
  180 degrees before resizing, to match the training orientation.

Sources: [data configuration](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py),
[conversion](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/convert_libero_data_to_lerobot.py),
[dataset adapter](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/policies/libero_policy.py),
[model preprocessing](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/model.py),
[evaluation](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py).
Local OpenPI revision reviewed: `215abfb217dbac7d5f1273282331b9b1866c0479`.
Upstream main was also checked on 2026-10-19.

### Hugging Face LeRobot pi0.5 (our fine-tuning path)

- The official LIBERO quickstart uses `lerobot/libero`. Its agent-view and wrist
  images are 256 x 256 and are resized to 224 x 224 for pi0.5.
- Dataset loading decodes frames into channel-first float tensors in [0, 1].
  No LIBERO environment rotation is applied to these stored training frames.
- The pi0.5 processor retains the visual features' identity normalization.
  The model image preprocessor resizes/pads and maps values to [-1, 1], without
  mirroring or rotating images.
- Our `dataset/train_lora_goal_all.sbatch` does not enable dataset image
  augmentation; LeRobot's `image_transforms.enable` defaults to false.
- For evaluation, `LiberoProcessorStep` rotates raw images by flipping both
  height and width. This processor belongs on raw environment observations,
  not on already-oriented dataset frames.

Sources: [official pi0.5 training recipe](https://huggingface.co/docs/lerobot/pi05#quickstart-on-libero),
[pi0.5 model](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi05/modeling_pi05.py),
[pi0.5 processors](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi05/processor_pi05.py),
[LIBERO environment processor](https://huggingface.co/docs/lerobot/env_processor#why-these-transformations).
Local LeRobot revision reviewed: `2a687cc17e2ab1ac5ed6f8411e87d3234cacea0e`.

## Local correction and regression checks

`source_to_lerobot.py` now rotates both cameras on every stored frame, including
matched snapshots. Earlier versions used only a vertical flip, leaving the
extended demonstrations horizontally mirrored relative to the original dataset.
Regenerate affected extended-task videos from their source HDF5 recordings;
leave original Hugging Face LIBERO frames unchanged. Simulator states/actions
and the saved matched-scene snapshots do not need to be modified.

Tests:

```bash
source activate_env.sh langgap
python -m unittest discover -s langgap/convert/tests -v
```

They compare every converted frame/camera against `LiberoProcessorStep`, cover
matched-scene initialization, and exercise the actual pi0.5 image-preprocessing
method without loading weights to verify normalization does not flip images.

An actual Hugging Face original training episode and the local before/after
comparison are recorded under `simulation/comparisons/libero_goal_orientation/`.
