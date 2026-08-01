from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import review_workload  # noqa: E402
import subtitle_frontend  # noqa: E402


class ReviewWorkloadTests(unittest.TestCase):
    def test_groups_common_roots_and_keeps_worst_sample_per_checkpoint(self) -> None:
        workload = review_workload.build_review_workload(
            [
                {
                    "check": "readability",
                    "status": "review",
                    "issue": "english_cps",
                    "segment_id": 7,
                    "start": 10.0,
                    "details": {"cps": 21.0, "target": 20.0},
                },
                {
                    "check": "readability",
                    "status": "review",
                    "issue": "english_cps",
                    "segment_id": 7,
                    "start": 11.0,
                    "details": {"cps": 36.0, "target": 20.0},
                },
                {
                    "check": "readability",
                    "status": "review",
                    "issue": "english_cps",
                    "segment_id": 8,
                    "start": 20.0,
                    "details": {"cps": 25.0, "target": 20.0},
                },
            ]
        )

        self.assertEqual(workload["input_sample_count"], 3)
        self.assertEqual(workload["actionable_sample_count"], 2)
        self.assertEqual(workload["collapsed_sample_count"], 1)
        self.assertEqual(len(workload["groups"]), 1)
        self.assertEqual(workload["groups"][0]["sample_count"], 2)
        self.assertEqual(workload["groups"][0]["severity"], "high")
        self.assertEqual(workload["items"][0]["details"]["cps"], 36.0)

    def test_failures_sort_before_advisory_groups(self) -> None:
        workload = review_workload.build_review_workload(
            [
                {
                    "check": "readability",
                    "status": "review",
                    "issue": "chinese_cps",
                    "segment_id": 1,
                    "details": {"cps": 9.5, "target": 9.0},
                },
                {
                    "check": "timing",
                    "status": "fail",
                    "issue": "timeline_overlap",
                    "segment_id": 2,
                },
            ]
        )

        self.assertEqual(workload["groups"][0]["severity"], "critical")
        self.assertEqual(workload["groups"][0]["issue"], "timeline_overlap")

    def test_non_blocking_sample_is_not_promoted_by_check_level_failure(self) -> None:
        severity, _score = review_workload.review_item_severity(
            {
                "check": "timing",
                "status": "fail",
                "issue": "under_min_duration",
                "segment_id": 3,
            }
        )

        self.assertEqual(severity, "medium")

    def test_recognition_risks_share_one_actionable_root(self) -> None:
        workload = review_workload.build_review_workload(
            [
                {
                    "check": "recognition_confidence",
                    "status": "review",
                    "issue": "low_asr_log_probability, high_asr_compression_ratio",
                    "segment_id": 4,
                },
                {
                    "check": "recognition_confidence",
                    "status": "review",
                    "issue": "low_ocr_confidence",
                    "segment_id": 5,
                },
            ]
        )

        self.assertEqual(len(workload["groups"]), 1)
        self.assertEqual(workload["groups"][0]["issue"], "recognition_confidence")
        self.assertEqual(workload["groups"][0]["sample_count"], 2)

    def test_global_check_uses_stable_root_instead_of_error_sentence(self) -> None:
        workload = review_workload.build_review_workload(
            [
                {
                    "check": "subtitle_sync",
                    "status": "review",
                    "issue": "Subtitle synchronization result requires review: rejected_partial.",
                }
            ]
        )

        self.assertEqual(workload["groups"][0]["issue"], "subtitle_sync")

    def test_checkpoint_info_distinguishes_reported_issues_from_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_dir = root / "Series" / "Movie"
            out_dir.mkdir(parents=True)
            (out_dir / "Movie.segments.checkpoint.json").write_text(
                json.dumps(
                    [
                        {
                            "id": 7,
                            "start": 10.0,
                            "end": 20.0,
                            "en": "Long subtitle",
                            "zh": "较长字幕",
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (out_dir / "Movie.quality-report.json").write_text(
                json.dumps(
                    {
                        "status": "review",
                        "manual_review_pending": True,
                        "summary": {"review_items": 75},
                        "checks": {
                            "readability": {
                                "status": "review",
                                "samples": [
                                    {
                                        "checkpoint_segment_id": 7,
                                        "issue": "english_cps",
                                        "cps": 22.0,
                                        "target": 20.0,
                                    },
                                    {
                                        "checkpoint_segment_id": 7,
                                        "issue": "english_cps",
                                        "cps": 35.0,
                                        "target": 20.0,
                                    },
                                ],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            info = subtitle_frontend.checkpoint_info(root, "Series", "Movie")

        self.assertEqual(info["quality_review_reported_count"], 75)
        self.assertEqual(info["quality_review_input_sample_count"], 2)
        self.assertEqual(info["quality_review_reviewed_sample_count"], 0)
        self.assertEqual(info["quality_review_pending_count"], 1)
        self.assertEqual(info["quality_review_collapsed_sample_count"], 1)
        self.assertTrue(info["quality_review_samples_truncated"])
        self.assertTrue(info["manual_review_pending"])
        self.assertEqual(len(info["quality_review_groups"]), 1)

    def test_checkpoint_info_counts_already_reviewed_report_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_dir = root / "Series" / "Movie"
            out_dir.mkdir(parents=True)
            (out_dir / "Movie.segments.checkpoint.json").write_text(
                json.dumps(
                    [
                        {
                            "id": 4,
                            "start": 5.0,
                            "end": 7.0,
                            "en": "Long subtitle",
                            "zh": "较长字幕",
                            "manual_reviewed_at": "2026-08-01T00:00:00+00:00",
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (out_dir / "Movie.quality-report.json").write_text(
                json.dumps(
                    {
                        "status": "review",
                        "summary": {"review_items": 40},
                        "checks": {
                            "readability": {
                                "status": "review",
                                "samples": [
                                    {
                                        "checkpoint_segment_id": 4,
                                        "issue": "english_cps",
                                    }
                                ],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            info = subtitle_frontend.checkpoint_info(root, "Series", "Movie")

        self.assertEqual(info["quality_review_reported_count"], 40)
        self.assertEqual(info["quality_review_input_sample_count"], 1)
        self.assertEqual(info["quality_review_reviewed_sample_count"], 1)
        self.assertEqual(info["quality_review_pending_count"], 0)
        self.assertEqual(info["quality_review_groups"], [])
        self.assertFalse(info["manual_review_pending"])


if __name__ == "__main__":
    unittest.main()
