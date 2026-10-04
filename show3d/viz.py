# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Visualization library for SHOW3D: hand skeleton, object, and interaction field.

Three renders, each writing a PNG:

* :func:`render_overlay`  -- hand skeleton + object projected onto the frame (2D).
* :func:`render_geometry` -- hand skeleton + object surface in 3D.
* :func:`render_field`    -- interaction field (skeleton + object + arrows) in 3D.

:func:`run_visualization` picks a suitable frame from a dataset and dispatches to
one of them. :func:`render_hand_meshes` draws the posed hand meshes of one hand
model (:mod:`show3d.hand_mesh`) on a frame or a frame range. Projection lives in
:mod:`show3d.camera`; the CLI is :mod:`show3d.demo_viz`. Needs matplotlib.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from . import camera
from .dataset import (
    CameraCalibration,
    default_object_mesh_provider,
    DEFAULT_VIDEO_FPS,
    EGOCENTRIC_VIEWS,
    FloatArray,
    load_camera_calibration,
    Show3DFrameRef,
    Show3DPaths,
)
from .hand_mesh import (
    DEFAULT_MESH_HAND_POSE_VERSION,
    HandMesh,
    HandMeshScene,
    LEFT_SLOT,
    RIGHT_SLOT,
)
from .interaction_field import (
    InteractionFieldExample,
    LEFT_TO_OBJECT,
    RIGHT_TO_OBJECT,
    Show3DInteractionFieldDataset,
)

MODES: tuple[str, ...] = ("overlay", "geometry", "field")

# 21-joint UmeTrack / HOT3D hand landmark order used by SHOW3D hand_pose:
# fingertips are 0-4, the wrist is 5, and the palm center is 20 (not a bone
# endpoint). Each finger chain is wrist -> proximal -> intermediate -> distal ->
# fingertip; the last four edges are the palm arch across the knuckles.
HAND_EDGES: tuple[tuple[int, int], ...] = (
    (5, 17),
    (17, 18),
    (18, 19),
    (19, 4),  # pinky
    (5, 14),
    (14, 15),
    (15, 16),
    (16, 3),  # ring
    (5, 11),
    (11, 12),
    (12, 13),
    (13, 2),  # middle
    (5, 8),
    (8, 9),
    (9, 10),
    (10, 1),  # index
    (5, 6),
    (6, 7),
    (7, 0),  # thumb
    (6, 8),
    (8, 11),
    (11, 14),
    (14, 17),  # palm arch
)
NUM_HAND_LANDMARKS: int = 21

# (label, color, frame_data attribute, interaction-field name) per hand.
_HANDS: tuple[tuple[str, str, str, str], ...] = (
    ("left hand", "tab:blue", "left_hand", LEFT_TO_OBJECT),
    ("right hand", "tab:green", "right_hand", RIGHT_TO_OBJECT),
)
# Field arrows use one distinct color so they read separately from the skeletons.
FIELD_COLOR: str = "crimson"
# Hand mesh RGB per slot, orange for the left hand and cyan for the right: the
# colors of the released viz_*.mp4 videos.
MESH_COLORS: tuple[tuple[int, int, int], tuple[int, int, int]] = (
    (235, 104, 52),
    (42, 184, 214),
)
# A vertex barely in front of the camera plane projects millions of pixels away.
# Pulling it to this distance from the principal point along its own direction
# keeps OpenCV's fixed-point coordinates in int32 and moves the triangle's edges
# by only a few pixels inside the image.
_MAX_PIXEL: float = 1e5
_SUBPIXEL_BITS: int = 4


# ----------------------------------------------------------------------------
# drawing primitives
# ----------------------------------------------------------------------------
def draw_object_3d(ax: Any, surface_mm: FloatArray, *, max_points: int = 4000) -> None:
    """Scatter the object surface points (thinned for a light plot)."""
    step = max(1, surface_mm.shape[0] // max_points)
    ax.scatter(
        surface_mm[::step, 0],
        surface_mm[::step, 1],
        surface_mm[::step, 2],
        s=2,
        c="0.6",
        label="object surface",
    )


def draw_hand_skeleton_3d(
    ax: Any, joints_mm: FloatArray, color: str, label: str
) -> None:
    """Draw the hand as connected bones plus joint markers, in 3D."""
    for a, b in HAND_EDGES:
        ax.plot(
            [joints_mm[a, 0], joints_mm[b, 0]],
            [joints_mm[a, 1], joints_mm[b, 1]],
            [joints_mm[a, 2], joints_mm[b, 2]],
            c=color,
            linewidth=2.0,
        )
    ax.scatter(
        joints_mm[:, 0],
        joints_mm[:, 1],
        joints_mm[:, 2],
        s=18,
        c=color,
        label=label,
        depthshade=False,
    )


def draw_field_3d(
    ax: Any,
    joints_mm: FloatArray,
    field_mm: FloatArray,
    color: str = FIELD_COLOR,
    label: str | None = None,
) -> None:
    """Draw the interaction field as arrows from each joint to the object."""
    ax.quiver(
        joints_mm[:, 0],
        joints_mm[:, 1],
        joints_mm[:, 2],
        field_mm[:, 0],
        field_mm[:, 1],
        field_mm[:, 2],
        color=color,
        arrow_length_ratio=0.15,
        linewidth=1.0,
        label=label,
    )


def set_equal_aspect_3d(ax: Any, points_mm: FloatArray) -> None:
    """Make the 3D box proportional to the data so geometry is not distorted."""
    span = points_mm.max(axis=0) - points_mm.min(axis=0)
    ax.set_box_aspect(tuple(float(s) if s > 0 else 1.0 for s in span))


def draw_object_2d(ax: Any, object_uv: FloatArray, valid: NDArray[np.bool_]) -> None:
    """Overlay projected object-surface points on an image axis."""
    pts = object_uv[valid]
    ax.scatter(pts[:, 0], pts[:, 1], s=2, c="gold", alpha=0.35, label="object")


def draw_hand_skeleton_2d(
    ax: Any, joints_uv: FloatArray, valid: NDArray[np.bool_], color: str, label: str
) -> None:
    """Overlay the hand skeleton (bones + joints) on an image axis."""
    for a, b in HAND_EDGES:
        if valid[a] and valid[b]:
            ax.plot(
                [joints_uv[a, 0], joints_uv[b, 0]],
                [joints_uv[a, 1], joints_uv[b, 1]],
                c=color,
                linewidth=2.0,
            )
    shown = joints_uv[valid]
    ax.scatter(shown[:, 0], shown[:, 1], s=18, c=color, label=label)


def draw_hand_meshes(
    image_rgb: NDArray[np.uint8],
    meshes: Sequence[tuple[HandMesh, tuple[int, int, int]]],
    calibration: CameraCalibration,
    *,
    alpha: float = 0.7,
) -> NDArray[np.uint8]:
    """Draw ``(mesh, rgb)`` pairs on a copy of the frame as shaded triangles.

    Triangles of all meshes are painted together from far to near, so a hand
    occludes the other. A triangle with a vertex behind the camera is skipped.
    """
    t_world_from_camera = calibration.t_world_from_camera
    if t_world_from_camera is None:
        raise ValueError("frame has no valid t_world_from_camera")
    depths: list[FloatArray] = []
    points: list[NDArray[np.int32]] = []
    colors: list[FloatArray] = []
    for mesh, rgb in meshes:
        triangles = camera.world_to_camera(mesh.vertices_world_mm, t_world_from_camera)[
            mesh.faces
        ]
        front = (triangles[:, :, 2] > 0.0).all(axis=1)
        triangles = triangles[front]
        uv, _valid = camera.project_to_image(mesh.vertices_world_mm, calibration)
        uv = uv[mesh.faces[front]]
        center = np.array([calibration.cx, calibration.cy])
        offset = uv - center
        distance = np.linalg.norm(offset, axis=-1, keepdims=True)
        uv = np.where(
            distance > _MAX_PIXEL,
            center + offset * (_MAX_PIXEL / np.maximum(distance, _MAX_PIXEL)),
            uv,
        )
        normals = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        centers = triangles.mean(axis=1)
        # Lambert shading with the light at the camera.
        cosine = np.abs((normals * centers).sum(axis=1)) / np.maximum(
            np.linalg.norm(normals, axis=1) * np.linalg.norm(centers, axis=1), 1e-12
        )
        depths.append(centers[:, 2])
        points.append(np.round(uv * (1 << _SUBPIXEL_BITS)).astype(np.int32))
        colors.append((0.3 + 0.7 * cosine)[:, None] * np.asarray(rgb, np.float64))
    out = image_rgb.copy()
    if not depths:
        return out
    depth = np.concatenate(depths)
    triangle_points = np.concatenate(points)
    triangle_colors = np.concatenate(colors)
    painted = image_rgb.copy()
    covered = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    for index in np.argsort(-depth):
        cv2.fillConvexPoly(
            painted,
            triangle_points[index],
            triangle_colors[index].tolist(),
            shift=_SUBPIXEL_BITS,
        )
        cv2.fillConvexPoly(covered, triangle_points[index], 255, shift=_SUBPIXEL_BITS)
    mask = covered > 0
    out[mask] = (alpha * painted[mask] + (1.0 - alpha) * image_rgb[mask]).astype(
        np.uint8
    )
    return out


# ----------------------------------------------------------------------------
# example -> geometry
# ----------------------------------------------------------------------------
def object_surface(example: InteractionFieldExample) -> FloatArray | None:
    """The object's canonical mesh posed into world space, or None."""
    object_pose = example.frame_data.object_pose
    alias = example.sample.object_alias
    mesh = default_object_mesh_provider()(alias) if alias is not None else None
    if mesh is None or object_pose is None:
        return None
    return object_pose.pose_vertices(mesh)


def present_hands(
    example: InteractionFieldExample,
) -> list[tuple[str, str, FloatArray, FloatArray | None]]:
    """``(label, color, joints_world_mm, field_or_none)`` for each present hand."""
    frame_data = example.frame_data
    labels = example.labels
    out: list[tuple[str, str, FloatArray, FloatArray | None]] = []
    for label, color, attr, field_name in _HANDS:
        hand = getattr(frame_data, attr)
        if hand is None or hand.landmarks_world_mm is None:
            continue
        field = labels.get(field_name) if labels is not None else None
        out.append((label, color, hand.landmarks_world_mm, field))
    return out


def _finish_3d(ax: Any, extent: list[FloatArray], title: str) -> None:
    if extent:
        set_equal_aspect_3d(ax, np.concatenate(extent, axis=0))
    ax.view_init(elev=18, azim=-70)
    ax.set_title(title)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.legend(loc="upper right", fontsize="small")


# ----------------------------------------------------------------------------
# renders
# ----------------------------------------------------------------------------
def render_geometry(example: InteractionFieldExample, out_path: str | Path) -> None:
    """3D hand skeleton + object surface."""
    fig = Figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")
    extent: list[FloatArray] = []
    surface = object_surface(example)
    if surface is not None:
        draw_object_3d(ax, surface)
        extent.append(surface)
    for label, color, joints, _field in present_hands(example):
        draw_hand_skeleton_3d(ax, joints, color, label)
        extent.append(joints)
    _finish_3d(ax, extent, f"SHOW3D hand + object: {example.sample.sample_id}")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")


def render_field(example: InteractionFieldExample, out_path: str | Path) -> None:
    """3D hand skeleton + object surface + interaction-field arrows."""
    fig = Figure(figsize=(9, 7))
    ax = fig.add_subplot(projection="3d")
    extent: list[FloatArray] = []
    surface = object_surface(example)
    if surface is not None:
        draw_object_3d(ax, surface)
        extent.append(surface)
    field_labeled = False
    for label, color, joints, field in present_hands(example):
        draw_hand_skeleton_3d(ax, joints, color, label)
        extent.append(joints)
        if field is not None:
            draw_field_3d(
                ax,
                joints,
                field,
                label=None if field_labeled else "interaction field",
            )
            field_labeled = True
            extent.append(joints + field)
    _finish_3d(ax, extent, f"SHOW3D interaction field: {example.sample.sample_id}")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")


def _decode_frame(video_path: Path, frame_index: int) -> FloatArray | None:
    if not video_path.exists():
        return None
    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def render_overlay(
    example: InteractionFieldExample, view_name: str, out_path: str | Path
) -> None:
    """Hand skeleton + object projected onto the egocentric frame (2D)."""
    view = example.frame_data.views.get(view_name)
    if view is None or view.calibration is None:
        raise ValueError(f"view {view_name!r} has no calibration to overlay")
    calibration = view.calibration
    if calibration.t_world_from_camera is None:
        raise ValueError(f"view {view_name!r} has no valid t_world_from_camera")

    fig = Figure(figsize=(8, 8 * calibration.image_height / calibration.image_width))
    ax: Any = fig.add_subplot()
    image = _decode_frame(view.video_path, example.sample.frame_index)
    if image is not None:
        ax.imshow(image)
    else:
        ax.set_facecolor("0.9")

    surface = object_surface(example)
    if surface is not None:
        object_uv, object_valid = camera.project_to_image(surface, calibration)
        draw_object_2d(ax, object_uv, object_valid)
    for label, color, joints, _field in present_hands(example):
        joints_uv, valid = camera.project_to_image(joints, calibration)
        if bool(valid.any()):
            draw_hand_skeleton_2d(ax, joints_uv, valid, color, label)

    ax.set_xlim(0, calibration.image_width)
    ax.set_ylim(calibration.image_height, 0)  # image row 0 at top
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"SHOW3D {view_name} overlay: {example.sample.sample_id}")
    ax.legend(loc="upper right", fontsize="small")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")


# ----------------------------------------------------------------------------
# frame selection + entry point
# ----------------------------------------------------------------------------
def _projects_into_view(example: InteractionFieldExample, view_name: str) -> bool:
    view = example.frame_data.views.get(view_name)
    if view is None or view.calibration is None:
        return False
    if view.calibration.t_world_from_camera is None:
        return False
    for _label, _color, joints, _field in present_hands(example):
        _uv, valid = camera.project_to_image(joints, view.calibration)
        if int(valid.sum()) >= 8:
            return True
    return False


def _pick_example(
    dataset: Show3DInteractionFieldDataset, mode: str, view_name: str
) -> InteractionFieldExample | None:
    fallback: InteractionFieldExample | None = None
    for index in range(len(dataset)):
        example = dataset[index]
        if not example.is_valid:
            continue
        if mode != "overlay":
            return example
        view = example.frame_data.views.get(view_name)
        if view is None or view.calibration is None:
            continue
        if view.calibration.t_world_from_camera is None:
            continue
        if fallback is None:
            fallback = example
        if _projects_into_view(example, view_name):
            return example
    return fallback


def run_visualization(
    root: str | Path,
    manifest_path: str | Path,
    out_path: str | Path,
    *,
    mode: str = "field",
    view: str = "headset0",
    verbose: bool = False,
) -> Path:
    """Pick a suitable frame from the dataset and render ``mode`` to ``out_path``."""
    dataset = Show3DInteractionFieldDataset(root, manifest_path)
    example = _pick_example(dataset, mode, view)
    if example is None:
        raise ValueError(f"no suitable frame to visualize for mode={mode!r}")

    out = Path(out_path)
    if mode == "overlay":
        render_overlay(example, view, out)
    elif mode == "geometry":
        render_geometry(example, out)
    else:
        render_field(example, out)
    if verbose:
        print(f"[{mode}] {example.sample.sample_id} -> {out}")
    return out


def _hand_meshes(
    scene: HandMeshScene, frame_index: int
) -> list[tuple[HandMesh, tuple[int, int, int]]]:
    meshes: list[tuple[HandMesh, tuple[int, int, int]]] = []
    for slot in (LEFT_SLOT, RIGHT_SLOT):
        mesh = scene.mesh(frame_index, slot)
        if mesh is not None:
            meshes.append((mesh, MESH_COLORS[slot]))
    return meshes


def _start_frame(scene: HandMeshScene, frame_index: int | None) -> int:
    """``frame_index``, or the scene's first frame with a hand mesh."""
    if frame_index is None:
        for key in sorted(scene.frames, key=int):
            if _hand_meshes(scene, int(key)):
                return int(key)
        raise ValueError("the scene has no hand mesh")
    if frame_index < 0:
        raise ValueError(f"frame_index must be non-negative, got {frame_index}")
    return frame_index


def _overlay_hand_meshes(
    scene: HandMeshScene,
    frame_index: int,
    frame_bgr: NDArray[np.uint8],
    calibration: CameraCalibration | None,
) -> NDArray[np.uint8]:
    hands = _hand_meshes(scene, frame_index)
    if not hands or calibration is None or calibration.t_world_from_camera is None:
        return frame_bgr
    image = draw_hand_meshes(
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), hands, calibration
    )
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def _open_video_writer(
    out: Path, capture: cv2.VideoCapture, frame_bgr: NDArray[np.uint8]
) -> cv2.VideoWriter:
    height, width = frame_bgr.shape[:2]
    fps = capture.get(cv2.CAP_PROP_FPS) or DEFAULT_VIDEO_FPS
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise OSError(f"could not open {out} for writing")
    return writer


def _write_frames(
    scene: HandMeshScene,
    video_path: Path,
    calibration_path: Path,
    out: Path,
    frame_index: int,
    num_frames: int | None,
) -> int:
    """Draw and write the frames; return how many were written."""
    calibration_cache: dict[Path, Mapping[str, object]] = {}
    capture = cv2.VideoCapture(str(video_path))
    writer: cv2.VideoWriter | None = None
    written = 0
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        for index in range(frame_index, frame_index + (num_frames or 1)):
            ok, frame_bgr = capture.read()
            if not ok:
                break
            calibration = load_camera_calibration(
                calibration_path, index, cache=calibration_cache
            )
            frame_bgr = _overlay_hand_meshes(scene, index, frame_bgr, calibration)
            if num_frames is None:
                if not cv2.imwrite(str(out), frame_bgr):
                    raise OSError(f"could not write {out}")
            else:
                if writer is None:
                    writer = _open_video_writer(out, capture, frame_bgr)
                writer.write(frame_bgr)
            written += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    return written


def render_hand_meshes(
    root: str | Path,
    subject: str,
    scene: str,
    model: str,
    out_path: str | Path,
    *,
    view: str = "headset0",
    frame_index: int | None = None,
    num_frames: int | None = None,
    asset_dir: str | Path | None = None,
    hand_pose_version: str = DEFAULT_MESH_HAND_POSE_VERSION,
    verbose: bool = False,
) -> Path:
    """Draw ``model``'s hand meshes on frames of ``view``.

    With ``num_frames``, write that many frames from ``frame_index`` to an MP4;
    without it, write frame ``frame_index`` to an image. With no
    ``frame_index``, start at the scene's first frame with a hand mesh.
    """
    if num_frames is not None and num_frames < 1:
        raise ValueError(f"num_frames must be at least 1, got {num_frames}")
    meshes = HandMeshScene(
        root, subject, scene, model, asset_dir=asset_dir, version=hand_pose_version
    )
    start = _start_frame(meshes, frame_index)
    paths = Show3DPaths(root)
    frame = Show3DFrameRef(subject_id=subject, scene_id=scene, frame_index=start)
    video_path = paths.headset_path(frame, EGOCENTRIC_VIEWS.index(view))
    out = Path(out_path)
    written = _write_frames(
        meshes,
        video_path,
        paths.camera_calibration_path(frame, view),
        out,
        start,
        num_frames,
    )
    if written == 0:
        raise ValueError(f"could not read frame {start} of {video_path}")
    if verbose:
        print(f"[{model}] {subject}/{scene} {view} frames {start}+{written} -> {out}")
    return out
