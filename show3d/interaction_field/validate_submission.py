# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Validate an interaction-field ``predictions.jsonl`` against a test manifest.

    python -m show3d.interaction_field.validate_submission \\
        --manifest test_manifest_5fps_202607.jsonl --submission predictions.jsonl

Exits non-zero if the submission has a ``sample_id`` not in the manifest or a
field that is not ``(21, 3)``. Missing predictions are reported but allowed;
they lower recall and affect the official ranking.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import validate_submission


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate an interaction-field predictions.jsonl before submitting"
    )
    parser.add_argument(
        "--manifest", type=Path, required=True, help="the test manifest .jsonl"
    )
    parser.add_argument(
        "--submission", type=Path, required=True, help="your predictions.jsonl"
    )
    args = parser.parse_args()

    try:
        report = validate_submission(args.manifest, args.submission)
    except ValueError as error:
        print(f"INVALID: {error}")
        raise SystemExit(1)

    print(f"manifest samples  : {report.num_manifest_samples}")
    print(
        f"predicted         : {report.num_matched_samples} matched "
        f"(left {report.left_predicted}, right {report.right_predicted})"
    )
    print(
        f"missing prediction: {len(report.missing_sample_ids)} "
        "(allowed; lowers recall and affects official ranking)"
    )
    if report.unknown_sample_ids:
        preview = ", ".join(report.unknown_sample_ids[:3])
        print(
            f"UNKNOWN sample_ids: {len(report.unknown_sample_ids)} not in the "
            f"manifest (e.g. {preview})"
        )
    for line in report.malformed_fields[:10]:
        print(f"MALFORMED: {line}")

    if report.ok:
        print("OK: well-formed and ready to upload.")
    else:
        print("INVALID: fix the errors above before submitting.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
