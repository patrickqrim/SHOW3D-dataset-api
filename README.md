# SHOW3D dataset API

Standalone Python starter APIs for the SHOW3D dataset
(`facebook/show3d-dataset`) and the Interaction Field Estimation Challenge at the
HANDS workshop, ECCV 2026.

## Install

```bash
pip install -r requirements.txt
# or: pip install numpy opencv-python   (add matplotlib for the visualization demo)
```

## Quickstart

See the whole interaction-field loop -- load frames, run a model, evaluate -- in
one command, with **no data download** (it falls back to a tiny synthetic scene):

```bash
python -m show3d.interaction_field.demo
```

Then head to the
**[Interaction Field Estimation challenge](show3d/interaction_field/README.md)**
for the task, baseline, training, and submission flow.

## The dataset

Download SHOW3D from Hugging Face:
[`facebook/show3d-dataset`](https://huggingface.co/datasets/facebook/show3d-dataset).
`show3d.dataset` is the generic, task-agnostic loader: frame references and
manifest JSONL helpers, path resolution, and per-frame object-pose / hand-pose /
calibration loading. Point it at your local mirror:

```python
from show3d.dataset import Show3DDataset

dataset = Show3DDataset.from_manifest_jsonl("/path/to/show3d", "manifest.jsonl")
frame = dataset[0]   # views (video paths + calibration), object_pose, left/right hand
```

The loader expects the released on-disk layout under `root`:

```
<root>/
├── scenes/<subject>/<scene>/
│   ├── headset0.mp4, headset1.mp4              # egocentric videos
│   ├── camera_calibration/headset{0,1}.json   # intrinsics + per-frame pose
│   └── metadata/frame_info.json
├── object_pose/<version>/scenes/<subject>/<scene>/object_pose.json
└── hand_pose/<version>/scenes/<subject>/<scene>/hand_pose.json
```

### Headset pose validity

`is_synthesized` still means that interpolation produced the pose.

- **Unversioned:** The loader returns `t_world_from_camera` when it exists and
  `is_synthesized=false`.
- **Version 1:** The loader returns `t_world_from_camera` when it exists and
  `is_pose_valid=true`. `pose_source` records `mocap`, `vio`,
  `endpoint_interpolation`, `smooth_mocap_interpolation`, or
  `legacy_unspecified`.
- `Show3DFrameData.headset_pose_valid` reports frame-level headset-pose
  validity.

Older loaders read only `is_synthesized`. Pin the data and loader revisions to
keep the same frame set.

## Interaction Field Estimation challenge

The Interaction Field Estimation Challenge is hosted at the HANDS workshop at ECCV
2026: predict a per-frame, hand-anchored 3D interaction field. Its API, the
end-to-end demo, a baseline template, and evaluation live in
[`show3d/interaction_field/README.md`](show3d/interaction_field/README.md).

## Visualize a frame

`show3d.demo_viz` renders three views of a frame (needs `matplotlib`); the
projection and drawing primitives live in `show3d.camera` and `show3d.viz`:

```bash
python -m show3d.demo_viz --mode overlay  --out overlay.png    # skeleton + object on the frame
python -m show3d.demo_viz --mode geometry --out geometry.png   # 3D skeleton + object
python -m show3d.demo_viz --mode field    --out field.png      # 3D interaction field
```

Add `--root DIR --manifest MANIFEST.jsonl` to visualize your own mirror (overlay
decodes the frame).

**overlay**: the hand skeleton and object projected onto the egocentric frame.

![overlay](docs/overlay.png)

**geometry**: the hand skeleton and object surface in 3D. **field**: the
interaction field, arrows from each hand joint to the nearest object point.

![geometry](docs/geometry.png)

![field](docs/field.png)

## Extracting frames for training

Training straight from the MP4s is decode-bound (random per-frame seeking is
slow). `show3d.extract_images` pre-extracts frames at an fps you choose into a
random-access image store -- one JPEG per frame plus an `index.jsonl` a map-style
`Dataset` can shuffle over; `--save-labels` also materializes the interaction-
field targets:

```bash
python -m show3d.extract_images --root /path/to/show3d --out frames/ --fps 10 \
    --manifest show3d/interaction_field/train_manifest_202607.jsonl --save-labels
# --views headset0 | --format png | --quality 80 | --workers 16
```

`--fps` is required (no default) and must be a whole divisor of the 60 fps source
(1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60). The interaction-field training set,
the end-to-end train flow, and measured storage sizes live in the
[challenge README](show3d/interaction_field/README.md#training-set).

## Repository layout

```
show3d/
├── dataset.py                    # generic SHOW3D dataloader API
├── camera.py                     # pinhole projection helpers
├── viz.py                        # drawing / rendering library
├── demo_viz.py                   # visualization CLI (overlay / geometry / field)
├── extract_images.py             # pre-extract frames (+ labels) at a chosen fps
├── interaction_field/
│   ├── README.md                 # the challenge: task, baseline, train, submit, eval
│   ├── __init__.py               # challenge API (dataset, labels, eval, submission)
│   ├── demo.py                   # end-to-end demo: load -> model -> eval
│   ├── validate_submission.py    # check a predictions.jsonl before you upload
│   ├── train_manifest_202607.jsonl       # training recordings (train subjects)
│   └── test_manifest_5fps_202607.jsonl   # test frames to predict on (no labels)
├── assets/objects/               # bundled object meshes (.glb, HOT3D-derived)
└── tests/                        # unit tests
```

## Running the tests

```bash
python -m unittest discover
```

## License

The code and the bundled object meshes are released under Creative Commons
Attribution-NonCommercial 4.0 International (CC BY-NC 4.0); see `LICENSE`. The
bundled meshes are derived from the HOT3D object models (BOP HOT3D release,
`object_models_eval`); see `show3d/assets/objects/ATTRIBUTION.md`.

## Citation

If you use SHOW3D, please cite:

```bibtex
@InProceedings{Rim_2026_CVPR,
    author    = {Rim, Patrick and Harris, Kevin and Copple, Braden and Han, Shangchen and Xie, Xu and Shugurov, Ivan and An, Sizhe and Wen, He and Wong, Alex and Hodan, Tomas and He, Kun},
    title     = {SHOW3D: Capturing Scenes of 3D Hands and Objects in the Wild},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {7111-7120}
}
```
