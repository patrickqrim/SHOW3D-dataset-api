# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Training data for the InterField baseline.

Joins the two files that ``show3d.extract_images --save-labels`` writes:

  * ``index.jsonl``  -- one row per extracted (frame, view): the image path.
  * ``labels.jsonl`` -- one row per frame with a valid target: the ``(21, 3)``
    world-space field per hand, keyed by the same ``sample_id``.

and reads the per-frame ``R_world_from_camera`` from the mirror's
``camera_calibration/headset0.json`` so it can express the target in the camera
frame (see ``model.py`` for why). A ``subjects`` filter gives a clean
subject-level train/val split.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ...dataset import Show3DPaths, load_camera_calibration
from ...dataset import Show3DFrameRef
from .model import IMAGENET_MEAN, IMAGENET_STD, NUM_JOINTS


def _read_rotation_world_from_camera(
    calib_path: Path, frame_index: int, cache: dict
) -> np.ndarray | None:
    """The 3x3 world-from-camera rotation for a frame, or None if unavailable."""
    calibration = load_camera_calibration(calib_path, frame_index, cache=cache)
    if calibration is None or calibration.t_world_from_camera is None:
        return None
    return np.asarray(calibration.t_world_from_camera, dtype=np.float64)[:3, :3]


class InterFieldFrameDataset(Dataset):
    """Extracted frames + camera-frame interaction-field targets.

    Each item is ``(image, target, hand_mask, meta)`` where ``image`` is
    ``(3, H, W)`` float, ``target`` is ``(2, 21, 3)`` mm in the **camera frame**
    ordered ``[left, right]``, and ``hand_mask`` is ``(2,)`` (1 where a hand has
    a valid target).
    """

    def __init__(
        self,
        frames_dir: str | Path,
        root: str | Path,
        *,
        subjects: Sequence[str] | None = None,
        image_size: int = 224,
        train: bool = False,
        view: str = "headset0",
        require_labels: bool = True,
    ) -> None:
        self.frames_dir = Path(frames_dir)
        self.paths = Show3DPaths(root)
        self.image_size = image_size
        self.train = train
        self.view = view
        self._calib_cache: dict = {}

        labels = _read_jsonl(self.frames_dir / "labels.jsonl")
        self._labels = {row["sample_id"]: row for row in labels}
        subject_set = set(subjects) if subjects is not None else None

        self.items: list[dict] = []
        for row in _read_jsonl(self.frames_dir / "index.jsonl"):
            if row.get("view") != view:
                continue
            if subject_set is not None and row["subject_id"] not in subject_set:
                continue
            if require_labels and row["sample_id"] not in self._labels:
                continue  # no valid target on this frame
            self.items.append(row)

    def __len__(self) -> int:
        return len(self.items)

    def _augment(self, arr: np.ndarray) -> np.ndarray:
        """Photometric augmentation on a [0,1] grayscale image (train only).

        Geometric augmentation is avoided on purpose: the target is a 3D field, so
        a 2D crop/flip that moves or removes the hand would break the label. Only
        appearance is perturbed, to curb overfitting to per-subject lighting/skin.
        """
        rng = np.random.default_rng()
        arr = arr * rng.uniform(0.7, 1.3)  # brightness
        mean = float(arr.mean())
        arr = (arr - mean) * rng.uniform(0.7, 1.3) + mean  # contrast
        arr = np.clip(arr, 0.0, 1.0) ** rng.uniform(0.7, 1.4)  # gamma
        arr = arr + rng.normal(0.0, 0.02, size=arr.shape)  # sensor noise
        return np.clip(arr, 0.0, 1.0).astype(np.float32)

    def _load_image(self, rel_path: str) -> torch.Tensor:
        img = cv2.imread(str(self.frames_dir / rel_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(self.frames_dir / rel_path)
        img = cv2.resize(
            img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA
        )
        arr = img.astype(np.float32) / 255.0
        if self.train:
            arr = self._augment(arr)
        arr = np.repeat(arr[None, :, :], 3, axis=0)  # gray -> 3 channels
        mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
        std = np.asarray(IMAGENET_STD, dtype=np.float32)[:, None, None]
        arr = (arr - mean) / std
        return torch.from_numpy(arr)

    def __getitem__(self, index: int):
        row = self.items[index]
        subject_id, scene_id = row["subject_id"], row["scene_id"]
        frame_index = int(row["frame_index"])
        image = self._load_image(row["image"])

        frame_ref = Show3DFrameRef(
            subject_id=subject_id, scene_id=scene_id, frame_index=frame_index
        )
        calib_path = self.paths.camera_calibration_path(frame_ref, self.view)
        rotation = _read_rotation_world_from_camera(
            calib_path, frame_index, self._calib_cache
        )

        label = self._labels.get(row["sample_id"], {})  # {} in predict (no-label) mode
        target = np.zeros((2, NUM_JOINTS, 3), dtype=np.float32)
        mask = np.zeros((2,), dtype=np.float32)
        for hand_idx, field_name in enumerate(("left_to_object", "right_to_object")):
            world = label.get(field_name)
            if world is None or rotation is None:
                continue
            world_arr = np.asarray(world, dtype=np.float64)  # (21, 3) world mm
            if world_arr.shape != (NUM_JOINTS, 3):
                continue
            # V_cam = R_world_from_camera^T @ V_world  (row vectors: V_world @ R)
            cam = world_arr @ rotation
            target[hand_idx] = cam.astype(np.float32)
            mask[hand_idx] = 1.0

        meta = {
            "sample_id": row["sample_id"],
            "subject_id": subject_id,
            "scene_id": scene_id,
            "frame_index": frame_index,
        }
        return image, torch.from_numpy(target), torch.from_numpy(mask), meta


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def collate(batch):
    images = torch.stack([b[0] for b in batch])
    targets = torch.stack([b[1] for b in batch])
    masks = torch.stack([b[2] for b in batch])
    metas = [b[3] for b in batch]
    return images, targets, masks, metas
