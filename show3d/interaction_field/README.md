# Interaction Field Estimation

The SHOW3D Interaction Field Estimation Challenge is hosted at the HANDS workshop
at ECCV 2026. The task is per-frame: given an egocentric frame, predict, for each
hand, a **hand-anchored 3D vector field** -- the offset from every one of the 21
hand joints to the nearest surface point of the manipulated object. The
prediction is a fixed `(21, 3)` per hand, independent of the object.

The path through this README: run the [demo](#quickstart) (no data needed),
[build a baseline](#build-your-baseline), [train](#training-set) on the shipped
train manifest, then [produce and submit](#submit-your-predictions) predictions
on the test manifest. `show3d.interaction_field` builds on `show3d.dataset` (see
the [top-level README](../../README.md) for install and dataset setup) and owns
the task-specific pieces: the frame manifests, label generation, the submission
schema, and the local evaluator.

## Rules

* **Declare your input view** (single-view = headset0 only, multi-view = both
  headsets; see [below](#single-view-or-multi-view)).
* **Declare external training data.** Training on other datasets (for example,
  ARCTIC) is allowed, but you must declare what you used. The live Codabench
  leaderboard is combined; organizers publish separate final tables for methods
  with and without external training data.

## Quickstart

The end-to-end demo runs the whole loop with **no data download** (it falls back
to a small synthetic scene). Load frames, run a model, evaluate:

```bash
python -m show3d.interaction_field.demo
```

```
Loaded 6 frames from the manifest (5 valid: headset tracked + object/hand pose present).
  views: headset0, headset1 | image 1408x1408

Interaction field for S000/mug_grab_demo:000000 (hand joint -> nearest object-surface point, mm):
  left_to_object  : 21x3 field | joint0 gt=(    -0,     7,    -6) pred=(    13,   -13,    64)

Evaluation (interaction field, naive random model):
  left_to_object  : ADE    163.5 mm | recall 1.00 (5/5) | acc@10mm=0.00 acc@50mm=0.02 acc@100mm=0.18
  right_to_object : no valid targets
  mean ADE: 163.5 mm
  mean recall: 1.00 (fraction of valid targets predicted)
```

Run it on your own mirror with `--root DIR --manifest MANIFEST.jsonl`.

## Build your baseline

The end-to-end demo is the template. To turn it into a real baseline, edit one
function, `naive_model` in `show3d/interaction_field/demo.py`:

```python
def naive_model(example, rng):
    # example.views[name].image      -> decoded frame (with decode_images=True)
    # example.views[name].calibration -> pinhole intrinsics + world-from-camera
    ...
    return {"left_to_object": left_vecs, "right_to_object": right_vecs}  # each (21, 3)
```

The demo writes `predictions.jsonl` and scores it with the official ADE /
accuracy evaluator. For fast training, pre-extract frames with
`show3d.extract_images --manifest show3d/interaction_field/train_manifest_202607.jsonl`
(the shipped list of training recordings; see the top-level README).

## Training set

Training uses **train subjects only**. The shipped manifest
`train_manifest_202607.jsonl` lists the exact **468 training recordings**; labels
(object pose + hand pose) are released for these on Hugging Face, and
test-subject labels are withheld.

* **468 recordings** = 10 subjects x 21 objects; 830,597 source frames at 60 fps
  (about 138,579 frames per view at the default 10 fps sampling).
* **Per subject** (recordings): ASC023 62, SPI102 62, LWA828 58, YZH016 58,
  XXI103 45, PCW023 43, MHA016 42, MMO925 40, XYZ109 32, LYA722 26.
* **Per object** (recordings): birdhousetoy 47, dinotoy 42, keyboard 42, mug 41,
  vase 37, dumbbell 36, milk 33, brushholder 30, balandabowl 26, orangejuice 26,
  aria 23, mustard 18, bbq 16, ranch 16, mouse 8, canparmesan 5, cansoup 5,
  cantomatosauce 5, vegetables 5, waffles 5, mug2 2.

### Building a fast training set (images + labels)

`Show3DInteractionFieldDataset` decodes frames from the MP4s on the fly, which is
fine for evaluation but slow for training. For training, pre-extract the frames
and materialize the targets in one command:

1. **Download** the training recordings' `scenes/` (videos), `object_pose/v1`,
   and `hand_pose/v2` from Hugging Face -- the manifest names the exact scenes.
2. **Extract + label** in one pass:

   ```bash
   python -m show3d.extract_images \
       --root /path/to/show3d --out frames/ --fps 10 \
       --manifest show3d/interaction_field/train_manifest_202607.jsonl \
       --save-labels
   ```

   This writes `frames/index.jsonl` (one row per frame *and view*: the image
   path) and `frames/labels.jsonl` (one row per frame: the `(21, 3)` target per
   hand, deduped across views), both keyed by the same `sample_id`.
3. **Train** from a plain map-style dataset: read `index.jsonl` for image paths,
   look up the target in `labels.jsonl` by `sample_id`, skip frames absent from
   `labels.jsonl` (no valid target). No re-decoding, no pose parsing per step.

`labels.jsonl` is the evaluator's reference-label format, so
`evaluate_label_jsonl("frames/labels.jsonl", "predictions.jsonl")` also scores a
submission against it directly. Only train-subject labels exist; the held-out
test frames have none.

**Storage** for the full 468-recording training set at the default JPEG quality
(`--quality 90`). Frame counts are exact; bytes per frame are the measured mean
grayscale JPEG size over a 10-recording sample (about 140 KB at q90):

| fps | frames / view | single-view | both views |
| --- | --- | --- | --- |
| 5  | 69,296  | 9 GiB   | 19 GiB  |
| 10 | 138,579 | 19 GiB  | 37 GiB  |
| 15 | 207,853 | 28 GiB  | 56 GiB  |
| 30 | 415,404 | 56 GiB  | 111 GiB |
| 60 | 830,597 | 111 GiB | 222 GiB |

`--quality 80` is about a third smaller (about 93 KB/frame); `--format png` is
lossless but roughly 3x larger (about 430 KB/frame).

## Submit your predictions

The test set is the shipped **test manifest** -- one row per frame you must
predict, keyed by `sample_id` (`"<SUBJECT>/<scene_id>:<frame_index:06d>"`), with
no labels:

```text
show3d/interaction_field/test_manifest_5fps_202607.jsonl
```

Download the videos + calibration for the manifest's scenes into your `root`
mirror (the held-out test subjects; no labels are released for them).

**1. Format.** Write one JSON object per line to `predictions.jsonl`, keyed by
`sample_id`. Each of `left_to_object` / `right_to_object` is a fixed `(21, 3)`
array of millimeter vectors -- one per hand joint, the predicted offset from that
joint to the nearest object-surface point, in world space. There is no
object-side field.

```json
{"sample_id": "S001/mug_grab_a1b2:000120", "left_to_object": [[x, y, z], ...], "right_to_object": [[x, y, z], ...]}
```

**Predict both hands.** You do not know at test time which hands have a valid
target, so a missing or `null` field lowers recall and affects the official
ranking (see [Evaluation](#evaluation)).

**2. Produce it.** Run your model over the test manifest and write the file:

```python
from show3d.interaction_field import (
    PredictionRecord,
    Show3DInteractionFieldDataset,
    write_submission_jsonl,
)

manifest = "show3d/interaction_field/test_manifest_5fps_202607.jsonl"
# Inputs only: there are no test labels, so load_labels=False.
dataset = Show3DInteractionFieldDataset(root, manifest, load_labels=False, decode_images=True)

records = []
for index in range(len(dataset)):
    example = dataset[index]
    left, right = your_model(example)  # each a (21, 3) array, world mm
    records.append(PredictionRecord(
        sample_id=example.sample.sample_id,
        fields={"left_to_object": left, "right_to_object": right},
    ))
write_submission_jsonl("predictions.jsonl", records)
```

**3. Validate, then upload.** Self-check before submitting:

```bash
python -m show3d.interaction_field.validate_submission \
    --manifest show3d/interaction_field/test_manifest_5fps_202607.jsonl \
    --submission predictions.jsonl
```

It flags any `sample_id` not in the manifest and any field that is not `(21, 3)`,
and reports how many frames you left unpredicted. Upload a zip with
`predictions.jsonl` at its root.

## Evaluation

Every field with a valid target (object pose confident and that hand valid) is
scored; invalid ground-truth fields are ignored. Two metric families are
reported, per field (`left_to_object`, `right_to_object`) and averaged across the
two hands:

* **Accuracy on the targets you predicted:**
  * **`ade_mm`**: mean per-joint endpoint error
    `|| prediction_vector - target_vector ||`, in millimeters
    (`mean_ade_mm` averages the two hands; lower is better).
  * accuracy at 10 / 50 / 100 mm: the fraction of joints within each threshold.
* **`recall`** (`mean_recall`): the fraction of valid targets you predicted at
  all. Skipping frames lifts your accuracy on what remains but lowers recall, so
  coverage stays visible and abstaining does not hide.

A prediction whose shape is not `(21, 3)` (wrong joint count) is rejected. The
final leaderboard ranking uses a withheld aggregate of accuracy and coverage,
so omitting hard frames does not pay off. The leaderboard shows its
`official_score`, but the formula is not disclosed.

`evaluate_submission_jsonl(dataset, "predictions.jsonl")` reproduces the public
ADE, accuracy, and recall metrics offline; the official ranking aggregate is
server-side only.

## Single-view or multi-view

Choose the view mode at construction with the `multiview` bool:

```python
Show3DInteractionFieldDataset(root, manifest)                   # both headsets (default)
Show3DInteractionFieldDataset(root, manifest, multiview=False)  # single-view (headset0)
```

Each frame's `frame_data.views` is a dict keyed by the selected view name. Each
`ViewFrame` has `video_path` (the MP4) and a parsed `calibration`
(`CameraCalibration`: pinhole `fx, fy, cx, cy`, image size, and the 4×4
`t_world_from_camera` for that frame). The interaction-field target is in 3D
world space, so it is view-independent; the choice only affects the inputs.
Declare which setting (single-view or multi-view) you used when you submit.

Frames decode on demand: pass `decode_images=True` and each `ViewFrame.image`
is the decoded RGB frame `(H, W, 3)` uint8; otherwise `image` is `None` and you
get only `video_path`.

## Frame validity

Some frames have unreliable ground truth. `frame_data.headset_pose_valid` is
the frame-level verdict from the calibration release. The dataset builds no
target (`labels=None`) when it is false. Use `example.is_valid` for the full
check: it also requires a valid object pose and at least one valid hand.

`frame_data.headset_tracking_valid` remains available for compatibility. It
uses the legacy `is_synthesized` flag and can disagree with the release verdict.
The projection and visualization helpers use the accepted pose when the file
declares pose contract version 1.

## Object geometry (self-contained)

The interaction field is the nearest-neighbor vector field between hand landmarks
and the object surface. The object surface for a frame is the object's canonical
mesh posed by the released per-frame `R,t`:

```python
world_mm = object_pose.pose_vertices(canonical_vertices_object_mm)
```

To stay self-contained, `show3d/assets/objects/<alias>.glb` bundles one GLB mesh
per object (object frame, mm) for the 21 challenge objects, so
`Show3DInteractionFieldDataset` builds labels out of the box, with no HOT3D
checkout and no multi-gigabyte per-frame vertices. Pass your own
`object_mesh_provider` to override.

The bundled meshes are the HOT3D object models (BOP HOT3D release,
`object_models_eval`); see `show3d/assets/objects/ATTRIBUTION.md`.
