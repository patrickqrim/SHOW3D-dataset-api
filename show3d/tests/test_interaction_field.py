# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
import tempfile
import unittest
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from ..interaction_field import (
    DEFAULT_VIDEO_FPS,
    evaluate_label_jsonl,
    evaluate_prediction_records,
    evaluate_submission_jsonl,
    InteractionFieldLabels,
    InteractionFieldSample,
    LabelRecord,
    LEFT_TO_OBJECT,
    make_manifest_rows,
    NUM_HAND_LANDMARKS,
    PredictionRecord,
    read_manifest_jsonl,
    RIGHT_TO_OBJECT,
    sampled_frame_indices,
    Show3DInteractionFieldDataset,
    validate_submission,
    write_label_jsonl,
    write_manifest_jsonl,
    write_submission_jsonl,
)


class Show3DInteractionApiTest(unittest.TestCase):
    def test_sampling_uses_source_frame_stride_for_integer_rates(self) -> None:
        self.assertEqual(
            sampled_frame_indices(13, sampling_fps=10.0, video_fps=DEFAULT_VIDEO_FPS),
            [0, 6, 12],
        )
        self.assertEqual(
            sampled_frame_indices(13, sampling_fps=5.0, video_fps=DEFAULT_VIDEO_FPS),
            [0, 12],
        )

    def test_manifest_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "manifest.jsonl"
            rows = make_manifest_rows(
                "S001",
                "mug_grab_a1b2",
                num_frames=13,
                sampling_fps=10.0,
            )
            write_manifest_jsonl(path, rows)

            self.assertEqual(
                read_manifest_jsonl(path),
                [
                    InteractionFieldSample(
                        sample_id="S001/mug_grab_a1b2:000000",
                        subject_id="S001",
                        scene_id="mug_grab_a1b2",
                        frame_index=0,
                        object_alias="mug",
                    ),
                    InteractionFieldSample(
                        sample_id="S001/mug_grab_a1b2:000006",
                        subject_id="S001",
                        scene_id="mug_grab_a1b2",
                        frame_index=6,
                        object_alias="mug",
                    ),
                    InteractionFieldSample(
                        sample_id="S001/mug_grab_a1b2:000012",
                        subject_id="S001",
                        scene_id="mug_grab_a1b2",
                        frame_index=12,
                        object_alias="mug",
                    ),
                ],
            )

    def test_dataset_labels_and_submission_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            subject_id = "S001"
            scene_id = "mug_grab_a1b2"
            sample = InteractionFieldSample(
                sample_id=f"{subject_id}/{scene_id}:000000",
                subject_id=subject_id,
                scene_id=scene_id,
                frame_index=0,
                object_alias="mug",
            )
            manifest_path = root / "manifest.jsonl"
            write_manifest_jsonl(manifest_path, [sample])
            self._write_scene_files(root, subject_id, scene_id)

            dataset = Show3DInteractionFieldDataset(
                root,
                manifest_path,
                object_mesh_provider=lambda alias: np.asarray(
                    [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]
                ),
            )
            example = dataset[0]

            self.assertEqual(example.sample, sample)
            self.assertIsNotNone(example.labels)
            labels = example.labels
            assert labels is not None
            np.testing.assert_allclose(
                labels.left_to_object,
                np.asarray(
                    [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]] * 10 + [[-1.0, 0.0, 0.0]]
                ),
            )
            self.assertIsNone(labels.right_to_object)
            self.assertTrue(labels.is_valid)
            self.assertTrue(example.is_valid)
            self.assertTrue(example.headset_tracking_valid)
            self.assertTrue(example.headset_pose_valid)

            submission_path = root / "submission.jsonl"
            left_to_object = cast(NDArray[np.float64], labels.left_to_object)
            write_submission_jsonl(
                submission_path,
                [
                    PredictionRecord(
                        sample_id=sample.sample_id,
                        fields={LEFT_TO_OBJECT: left_to_object},
                    )
                ],
            )
            result = evaluate_submission_jsonl(dataset, submission_path)

            self.assertEqual(
                result.fields[LEFT_TO_OBJECT].num_points,
                NUM_HAND_LANDMARKS,
            )
            self.assertEqual(result.fields[LEFT_TO_OBJECT].missing_predictions, 0)
            self.assertEqual(result.fields[LEFT_TO_OBJECT].ade_mm, 0.0)
            self.assertEqual(result.fields[LEFT_TO_OBJECT].recall, 1.0)
            self.assertEqual(
                result.fields[LEFT_TO_OBJECT].accuracy_by_threshold_mm[10.0],
                1.0,
            )
            self.assertEqual(result.fields[RIGHT_TO_OBJECT].num_points, 0)
            # Right hand has no valid target here -> recall is undefined, not 0.
            self.assertIsNone(result.fields[RIGHT_TO_OBJECT].recall)

            label_path = root / "labels.jsonl"
            write_label_jsonl(
                label_path,
                [LabelRecord(sample_id=sample.sample_id, labels=labels)],
            )
            label_result = evaluate_label_jsonl(label_path, submission_path)
            self.assertEqual(label_result.mean_ade_mm, 0.0)

    def test_missing_prediction_lowers_recall_not_ade(self) -> None:
        # Two frames carry a valid left-hand target; predict only the first.
        target = np.zeros((NUM_HAND_LANDMARKS, 3), dtype=np.float64)
        references = [
            LabelRecord(
                sample_id="S001/mug_grab_a1b2:000000",
                labels=InteractionFieldLabels(left_to_object=target),
            ),
            LabelRecord(
                sample_id="S001/mug_grab_a1b2:000006",
                labels=InteractionFieldLabels(left_to_object=target),
            ),
        ]
        predictions = {
            "S001/mug_grab_a1b2:000000": PredictionRecord(
                sample_id="S001/mug_grab_a1b2:000000",
                fields={LEFT_TO_OBJECT: target},
            )
        }
        result = evaluate_prediction_records(references, predictions)
        left = result.fields[LEFT_TO_OBJECT]
        # Coverage drops (one of two valid targets predicted); recall records it.
        self.assertEqual(left.num_samples, 2)
        self.assertEqual(left.missing_predictions, 1)
        self.assertEqual(left.recall, 0.5)
        # ADE / accuracy are over the predicted target only -- the miss is NOT
        # folded into the error here (the penalty lives in the withheld aggregate).
        self.assertEqual(left.num_points, NUM_HAND_LANDMARKS)
        self.assertEqual(left.ade_mm, 0.0)
        self.assertEqual(result.mean_recall, 0.5)

    def test_evaluator_rejects_unknown_sample_id(self) -> None:
        target = np.zeros((NUM_HAND_LANDMARKS, 3), dtype=np.float64)
        references = [
            LabelRecord(
                sample_id="S001/mug_grab_a1b2:000000",
                labels=InteractionFieldLabels(left_to_object=target),
            )
        ]
        predictions = {
            "S001/mug_grab_a1b2:999999": PredictionRecord(
                sample_id="S001/mug_grab_a1b2:999999",
                fields={LEFT_TO_OBJECT: target},
            )
        }

        with self.assertRaisesRegex(ValueError, "unknown sample_id"):
            evaluate_prediction_records(references, predictions)

    def test_synthesized_frame_yields_no_labels(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            subject_id = "S001"
            scene_id = "mug_grab_a1b2"
            sample = InteractionFieldSample(
                sample_id=f"{subject_id}/{scene_id}:000000",
                subject_id=subject_id,
                scene_id=scene_id,
                frame_index=0,
                object_alias="mug",
            )
            manifest_path = root / "manifest.jsonl"
            write_manifest_jsonl(manifest_path, [sample])
            self._write_scene_files(root, subject_id, scene_id)
            self._write_calibration(
                root,
                subject_id,
                scene_id,
                is_synthesized=True,
            )

            dataset = Show3DInteractionFieldDataset(
                root,
                manifest_path,
                object_mesh_provider=lambda alias: np.asarray(
                    [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]
                ),
            )
            example = dataset[0]

            self.assertFalse(example.headset_tracking_valid)
            self.assertFalse(example.headset_pose_valid)
            self.assertFalse(example.is_valid)
            self.assertIsNone(example.labels)

            submission_path = root / "submission.jsonl"
            prediction = np.zeros((NUM_HAND_LANDMARKS, 3), dtype=np.float64)
            write_submission_jsonl(
                submission_path,
                [
                    PredictionRecord(
                        sample_id=sample.sample_id,
                        fields={
                            LEFT_TO_OBJECT: prediction,
                            RIGHT_TO_OBJECT: prediction,
                        },
                    )
                ],
            )
            result = evaluate_submission_jsonl(dataset, submission_path)

            for field_name in (LEFT_TO_OBJECT, RIGHT_TO_OBJECT):
                metrics = result.fields[field_name]
                self.assertEqual(metrics.num_samples, 0)
                self.assertEqual(metrics.num_points, 0)
                self.assertIsNone(metrics.recall)

            malformed_path = root / "malformed_submission.jsonl"
            write_submission_jsonl(
                malformed_path,
                [
                    PredictionRecord(
                        sample_id=sample.sample_id,
                        fields={LEFT_TO_OBJECT: np.zeros((5, 3))},
                    )
                ],
            )
            with self.assertRaisesRegex(ValueError, "must contain 21 vectors"):
                evaluate_submission_jsonl(dataset, malformed_path)

    def test_explicit_pose_validity_gates_labels(self) -> None:
        scenarios = (
            ("endpoint_interpolation", True, True, False, True),
            ("legacy_unspecified", False, False, True, False),
        )
        for (
            pose_source,
            is_synthesized,
            is_pose_valid,
            tracking_valid,
            expected_valid,
        ) in scenarios:
            with (
                self.subTest(pose_source=pose_source),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                subject_id = "S001"
                scene_id = "mug_grab_a1b2"
                sample = InteractionFieldSample(
                    sample_id=f"{subject_id}/{scene_id}:000000",
                    subject_id=subject_id,
                    scene_id=scene_id,
                    frame_index=0,
                    object_alias="mug",
                )
                manifest_path = root / "manifest.jsonl"
                write_manifest_jsonl(manifest_path, [sample])
                self._write_scene_files(root, subject_id, scene_id)
                self._write_calibration(
                    root,
                    subject_id,
                    scene_id,
                    is_synthesized=is_synthesized,
                    pose_source=pose_source,
                    is_pose_valid=is_pose_valid,
                )

                example = Show3DInteractionFieldDataset(
                    root,
                    manifest_path,
                    object_mesh_provider=lambda alias: np.asarray(
                        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]
                    ),
                )[0]

                self.assertEqual(example.headset_tracking_valid, tracking_valid)
                self.assertEqual(example.headset_pose_valid, is_pose_valid)
                self.assertEqual(example.is_valid, expected_valid)
                self.assertEqual(example.labels is not None, expected_valid)

    def _write_scene_files(self, root: Path, subject_id: str, scene_id: str) -> None:
        scene_dir = root / "scenes" / subject_id / scene_id
        scene_dir.mkdir(parents=True)
        (scene_dir / "headset0.mp4").write_bytes(b"")
        (scene_dir / "headset1.mp4").write_bytes(b"")
        self._write_calibration(root, subject_id, scene_id, is_synthesized=False)

        object_pose_dir = root / "object_pose" / "v1" / "scenes" / subject_id / scene_id
        object_pose_dir.mkdir(parents=True)
        with (object_pose_dir / "object_pose.json").open("w") as f:
            json.dump(
                {
                    "0": {
                        "confidence": 0.99,
                        "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        "t": [[0.0], [0.0], [0.0]],
                    }
                },
                f,
            )

        hand_pose_dir = root / "hand_pose" / "v2" / "scenes" / subject_id / scene_id
        hand_pose_dir.mkdir(parents=True)
        with (hand_pose_dir / "hand_pose.json").open("w") as f:
            json.dump(
                {
                    "0": {
                        "hand_poses": {
                            "0": {
                                "confidence": 0.99,
                                "landmarks_3d_mm": [
                                    [1.0, 0.0, 0.0]
                                    if index % 2 == 0
                                    else [9.0, 0.0, 0.0]
                                    for index in range(NUM_HAND_LANDMARKS)
                                ],
                            },
                            "1": {
                                "confidence": 0.0,
                                "landmarks_3d_mm": None,
                            },
                        }
                    }
                },
                f,
            )

    def _write_calibration(
        self,
        root: Path,
        subject_id: str,
        scene_id: str,
        *,
        is_synthesized: bool,
        pose_source: str | None = None,
        is_pose_valid: bool | None = None,
    ) -> None:
        calibration_dir = root / "scenes" / subject_id / scene_id / "camera_calibration"
        calibration_dir.mkdir(parents=True, exist_ok=True)
        identity = [[1.0 if r == c else 0.0 for c in range(4)] for r in range(4)]
        entry: dict[str, object] = {
            "index": 0,
            "T_WorldFromCamera": identity,
            "is_synthesized": is_synthesized,
        }
        if pose_source is not None:
            entry["pose_source"] = pose_source
        if is_pose_valid is not None:
            entry["is_pose_valid"] = is_pose_valid
        payload: dict[str, object] = {
            "ImageSizeX": 1024,
            "ImageSizeY": 1280,
            "fx": 450.0,
            "fy": 450.0,
            "cx": 512.0,
            "cy": 640.0,
            "DistortionModel": "PinholePlane",
            "T_WorldFromCamera_by_index": {"0": entry},
        }
        if pose_source is not None or is_pose_valid is not None:
            payload["pose_contract_version"] = 1
        for name in ("headset0", "headset1"):
            with (calibration_dir / f"{name}.json").open("w") as f:
                json.dump(payload, f)

    def test_validate_submission_reports_coverage_and_errors(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = root / "test_manifest.jsonl"
            write_manifest_jsonl(
                manifest,
                [
                    InteractionFieldSample(
                        sample_id="S001/mug_grab_a1b2:000000",
                        subject_id="S001",
                        scene_id="mug_grab_a1b2",
                        frame_index=0,
                    ),
                    InteractionFieldSample(
                        sample_id="S001/mug_grab_a1b2:000006",
                        subject_id="S001",
                        scene_id="mug_grab_a1b2",
                        frame_index=6,
                    ),
                ],
            )
            good = np.zeros((NUM_HAND_LANDMARKS, 3), dtype=np.float64)

            # Predict only the first sample (both hands): ok, one missing.
            sub = root / "predictions.jsonl"
            write_submission_jsonl(
                sub,
                [
                    PredictionRecord(
                        sample_id="S001/mug_grab_a1b2:000000",
                        fields={LEFT_TO_OBJECT: good, RIGHT_TO_OBJECT: good},
                    )
                ],
            )
            report = validate_submission(manifest, sub)
            self.assertTrue(report.ok)
            self.assertEqual(report.num_matched_samples, 1)
            self.assertEqual(report.missing_sample_ids, ["S001/mug_grab_a1b2:000006"])
            self.assertEqual(report.left_predicted, 1)
            self.assertEqual(report.right_predicted, 1)

            # A sample_id not in the manifest -> invalid.
            unknown = root / "unknown.jsonl"
            write_submission_jsonl(
                unknown,
                [
                    PredictionRecord(
                        sample_id="S001/mug_grab_a1b2:999999",
                        fields={LEFT_TO_OBJECT: good},
                    )
                ],
            )
            self.assertFalse(validate_submission(manifest, unknown).ok)

            # A field with the wrong joint count -> invalid, reported.
            malformed = root / "malformed.jsonl"
            write_submission_jsonl(
                malformed,
                [
                    PredictionRecord(
                        sample_id="S001/mug_grab_a1b2:000000",
                        fields={LEFT_TO_OBJECT: np.zeros((5, 3))},
                    )
                ],
            )
            report_bad = validate_submission(manifest, malformed)
            self.assertFalse(report_bad.ok)
            self.assertTrue(report_bad.malformed_fields)

            invalid_values = {
                "boolean": True,
                "numeric string": "0.0",
                "NaN": float("nan"),
                "Infinity": float("inf"),
            }
            for name, invalid_value in invalid_values.items():
                with self.subTest(name=name):
                    invalid = root / f"invalid_{name.replace(' ', '_')}.jsonl"
                    invalid.write_text(
                        json.dumps(
                            {
                                "sample_id": "S001/mug_grab_a1b2:000000",
                                "left_to_object": [
                                    [invalid_value, 0.0, 0.0]
                                    for _ in range(NUM_HAND_LANDMARKS)
                                ],
                            }
                        )
                        + "\n"
                    )
                    with self.assertRaises(ValueError):
                        validate_submission(manifest, invalid)


if __name__ == "__main__":
    unittest.main()
