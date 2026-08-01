import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import subtitle_frontend  # noqa: E402
from audio_to_subtitle import (  # noqa: E402
    checkpoint_matches_segments,
    prepare_segments_for_output,
)
from subtitle_quality import (  # noqa: E402
    build_layout_policy,
    build_quality_report,
    write_quality_report,
)


class SubtitleQualityTests(unittest.TestCase):
    def test_pixel_width_policy_splits_wide_mobile_subtitle(self) -> None:
        layout = build_layout_policy(
            (608, 1080),
            profile_name="mobile",
            font_name="Arial",
            font_scale=160,
        )
        source = [
            {
                "id": 0,
                "start": 1.0,
                "end": 6.0,
                "text": "MMMM MMMM MMMM MMMM MMMM MMMM",
                "en": "MMMM MMMM MMMM MMMM MMMM MMMM",
                "zh": "\u8fd9\u662f\u4e00\u6761\u7528\u4e8e\u6d4b\u8bd5\u7a84\u753b\u9762\u5bbd\u5ea6\u7684\u5b57\u5e55",
            }
        ]

        output = prepare_segments_for_output(
            source,
            max_words=12,
            max_chars=42,
            max_duration=5.5,
            layout_policy=layout,
        )

        self.assertGreater(len(output), 1)
        for segment in output:
            self.assertEqual(segment["checkpoint_segment_id"], 0)
            bilingual = bool(segment.get("en") and segment.get("zh"))
            self.assertTrue(
                layout.fits(segment.get("en") or "", "en", bilingual=bilingual)
            )
            self.assertTrue(
                layout.fits(segment.get("zh") or "", "zh", bilingual=bilingual)
            )

    def test_quality_report_records_model_hide_override(self) -> None:
        layout = build_layout_policy(
            (1920, 1080),
            profile_name="adaptive",
            font_name="Arial",
            font_scale=100,
        )
        source = [
            {"id": 0, "start": 1.0, "end": 3.0, "text": "Hello"},
        ]
        processed = [
            {
                **source[0],
                "en": "Hello",
                "zh": "\u4f60\u597d",
                "display": True,
                "model_display_requested": False,
                "display_guard": "forced_visible",
                "display_guard_reason": "unique_source_not_covered",
            }
        ]

        report = build_quality_report(
            source,
            processed,
            processed,
            layout,
            max_duration=5.5,
            target_chinese_cps=9.0,
            target_english_cps=20.0,
        )

        self.assertEqual(report["status"], "review")
        self.assertEqual(report["summary"]["model_hide_overrides"], 1)
        self.assertEqual(
            report["checks"]["model_display_guard"]["samples"][0]["id"],
            0,
        )

    def test_quality_report_is_written_atomically(self) -> None:
        layout = build_layout_policy(
            (1920, 1080),
            profile_name="adaptive",
            font_name="Arial",
            font_scale=100,
        )
        segment = {
            "id": 0,
            "start": 1.0,
            "end": 3.0,
            "text": "Hello",
            "en": "Hello",
            "zh": "\u4f60\u597d",
            "display": True,
        }
        report = build_quality_report(
            [segment],
            [segment],
            [segment],
            layout,
            max_duration=5.5,
            target_chinese_cps=9.0,
            target_english_cps=20.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "movie.quality-report.json"
            write_quality_report(report_path, report)
            loaded = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(loaded["status"], "pass")
        self.assertEqual(loaded["layout"]["measurement_method"], "unicode_estimate")

    def test_frontend_loads_and_displays_delivery_quality(self) -> None:
        page = subtitle_frontend.html_page()
        self.assertIn('<b id="qualityState">未生成</b>', page)
        self.assertIn("质检建议复核", page)
        self.assertIn('id="qualityReview"', page)
        self.assertIn('id="qualityViewMode"', page)
        self.assertIn('id="qualitySeverityFilter"', page)
        self.assertIn('id="qualityGroupTable"', page)
        self.assertIn('id="qualityReviewEmpty"', page)
        self.assertIn("人工复核修改已保存到 checkpoint", page)
        self.assertIn("当前没有未复核代表样本", page)
        self.assertIn("/api/checkpoint/update", page)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_dir = root / "Series" / "Movie"
            out_dir.mkdir(parents=True)
            report_path = out_dir / "Movie.quality-report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "status": "review",
                        "summary": {
                            "review_items": 2,
                            "model_hide_overrides": 1,
                        },
                        "checks": {},
                    }
                ),
                encoding="utf-8",
            )

            info = subtitle_frontend.checkpoint_info(
                root,
                "Series",
                "Movie",
            )

        self.assertTrue(info["quality_report_exists"])
        self.assertEqual(info["quality_status"], "review")
        self.assertEqual(info["quality_summary"]["model_hide_overrides"], 1)

    def test_frontend_flattens_quality_samples_with_checkpoint_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_dir = root / "Series" / "Movie"
            out_dir.mkdir(parents=True)
            (out_dir / "Movie.segments.checkpoint.json").write_text(
                json.dumps(
                    [
                        {
                            "id": 7,
                            "start": 12.0,
                            "end": 13.0,
                            "text": "Source",
                            "en": "Corrected",
                            "zh": "\u6821\u5bf9",
                            "display": True,
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
                        "summary": {"review_items": 1},
                        "checks": {
                            "recognition_confidence": {
                                "status": "review",
                                "samples": [
                                    {
                                        "id": 70,
                                        "checkpoint_segment_id": 7,
                                        "start": 12.0,
                                        "issues": ["low_ocr_confidence"],
                                        "metrics": {"ocr_confidence": 0.62},
                                    }
                                ],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            info = subtitle_frontend.checkpoint_info(root, "Series", "Movie")

        review = info["quality_review_items"][0]
        self.assertEqual(review["check"], "recognition_confidence")
        self.assertEqual(review["segment_id"], 7)
        self.assertEqual(review["details"]["id"], 70)
        self.assertEqual(review["segment"]["en"], "Corrected")
        self.assertEqual(review["segment"]["zh"], "\u6821\u5bf9")

    def test_frontend_hides_manually_reviewed_quality_samples(self) -> None:
        report = {
            "checks": {
                "recognition_confidence": {
                    "status": "review",
                    "samples": [
                        {
                            "checkpoint_segment_id": 7,
                            "issues": ["low_ocr_confidence"],
                        },
                        {
                            "checkpoint_segment_id": 8,
                            "issues": ["low_ocr_confidence"],
                        },
                    ],
                }
            }
        }
        checkpoint = [
            {
                "id": 7,
                "start": 1.0,
                "end": 2.0,
                "en": "Reviewed",
                "zh": "\u5df2\u590d\u6838",
                "manual_reviewed_at": "2026-07-31T10:00:00+00:00",
            },
            {
                "id": 8,
                "start": 2.0,
                "end": 3.0,
                "en": "Pending",
                "zh": "\u5f85\u590d\u6838",
            },
        ]

        review_items = subtitle_frontend.flatten_quality_review_items(
            report,
            checkpoint,
        )

        self.assertEqual([item["segment_id"] for item in review_items], [8])

    def test_frontend_updates_one_checkpoint_segment_atomically(self) -> None:
        source_segment = {
            "id": 4,
            "start": 10.0,
            "end": 12.0,
            "text": "Original",
            "source_language": "en",
            "display": True,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_dir = root / "Series" / "Movie"
            out_dir.mkdir(parents=True)
            checkpoint_path = out_dir / "Movie.segments.checkpoint.json"
            checkpoint_path.write_text(
                json.dumps(
                    [
                        {
                            **source_segment,
                            "en": "Before",
                            "zh": "\u4fee\u6539\u524d",
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            result = subtitle_frontend.update_checkpoint_segment(
                {
                    "output_root": str(root),
                    "series_name": "Series",
                    "movie_name": "Movie",
                    "id": 4,
                    "expected_start": 10.0,
                    "start": 10.1,
                    "end": 12.2,
                    "en": "After",
                    "zh": "\u4fee\u6539\u540e",
                    "display": True,
                }
            )
            updated = json.loads(checkpoint_path.read_text(encoding="utf-8"))[0]

        self.assertEqual(updated["en"], "After")
        self.assertEqual(updated["zh"], "\u4fee\u6539\u540e")
        self.assertEqual(updated["source_language"], "en")
        self.assertEqual(updated["manual_source_start"], 10.0)
        self.assertEqual(updated["manual_source_end"], 12.0)
        self.assertEqual(updated["manual_source_text"], "Original")
        self.assertTrue(updated["manual_reviewed_at"])
        self.assertTrue(checkpoint_matches_segments([updated], [source_segment]))
        self.assertTrue(result["regeneration_required"])

    def test_quality_report_flags_language_lane_contamination(self) -> None:
        layout = build_layout_policy(
            (1920, 1080),
            profile_name="adaptive",
            font_name="Arial",
            font_scale=100,
        )
        segment = {
            "id": 0,
            "start": 1.0,
            "end": 3.0,
            "text": "Hello",
            "en": "\u4f60\u597d",
            "zh": "Hello",
            "display": True,
        }

        report = build_quality_report(
            [segment],
            [segment],
            [segment],
            layout,
            max_duration=5.5,
            target_chinese_cps=9.0,
            target_english_cps=20.0,
        )

        completeness = report["checks"]["completeness"]
        self.assertEqual(report["status"], "review")
        self.assertEqual(completeness["invalid_chinese_target_count"], 1)
        self.assertEqual(completeness["english_contains_cjk_count"], 1)

    def test_quality_report_requires_review_when_subtitle_sync_is_rejected(self) -> None:
        layout = build_layout_policy(
            (1920, 1080),
            profile_name="adaptive",
            font_name="Arial",
            font_scale=100,
        )
        segment = {
            "id": 0,
            "start": 1.0,
            "end": 3.0,
            "text": "Hello",
            "en": "Hello",
            "zh": "\u4f60\u597d",
            "timing_origin": "existing_subtitle",
            "display": True,
        }

        report = build_quality_report(
            [segment],
            [segment],
            [segment],
            layout,
            max_duration=5.5,
            target_chinese_cps=9.0,
            target_english_cps=20.0,
            sync_report={
                "mode": "auto",
                "status": "rejected_low_quality",
                "pipeline_quality_passed": False,
                "quality_reasons": ["offset reached the search boundary"],
            },
            source_kind="embedded",
            video_duration_seconds=120.0,
        )

        self.assertEqual(report["status"], "review")
        self.assertEqual(report["summary"]["sync_review_items"], 1)
        self.assertEqual(report["checks"]["subtitle_sync"]["status"], "review")
        self.assertEqual(
            report["checks"]["subtitle_sync"]["reasons"],
            ["offset reached the search boundary"],
        )

    def test_word_timing_supersedes_rejected_existing_subtitle_sync(self) -> None:
        layout = build_layout_policy(
            (1920, 1080),
            profile_name="adaptive",
            font_name="Arial",
            font_scale=100,
        )
        source = {
            "id": 0,
            "start": 1.0,
            "end": 3.0,
            "text": "Hello",
            "en": "Hello",
            "zh": "\u4f60\u597d",
            "timing_origin": "existing_subtitle",
            "display": True,
        }
        processed = {
            **source,
            "word_timing_source": "whisper-words.json",
            "words": [{"word": "Hello", "start": 1.0, "end": 1.4}],
        }

        report = build_quality_report(
            [source],
            [processed],
            [processed],
            layout,
            max_duration=5.5,
            target_chinese_cps=9.0,
            target_english_cps=20.0,
            sync_report={
                "mode": "auto",
                "status": "rejected_low_quality",
                "pipeline_quality_passed": False,
            },
            source_kind="embedded",
            video_duration_seconds=120.0,
        )

        sync_check = report["checks"]["subtitle_sync"]
        self.assertEqual(sync_check["status"], "pass")
        self.assertEqual(sync_check["review_items"], 0)
        self.assertEqual(sync_check["result"], "superseded_by_word_timing")
        self.assertEqual(sync_check["superseded_result"], "rejected_low_quality")

    def test_terminology_check_ignores_targets_absent_from_rendered_text(self) -> None:
        layout = build_layout_policy(
            (1920, 1080),
            profile_name="adaptive",
            font_name="Arial",
            font_scale=100,
        )
        segments = [
            {
                "id": 0,
                "start": 1.0,
                "end": 3.0,
                "text": "lotus position",
                "en": "lotus position",
                "zh": "\u83b2\u82b1\u5f0f",
                "terminology": [
                    {"source": "lotus position", "target": "\u83b2\u5ea7\u5f0f"}
                ],
            },
            {
                "id": 1,
                "start": 3.1,
                "end": 5.1,
                "text": "lotus position",
                "en": "lotus position",
                "zh": "\u83b2\u82b1\u5f0f",
                "terminology": [
                    {"source": "lotus position", "target": "\u83b2\u82b1\u5f0f"}
                ],
            },
        ]

        report = build_quality_report(
            segments,
            segments,
            segments,
            layout,
            max_duration=5.5,
            target_chinese_cps=9.0,
            target_english_cps=20.0,
        )

        self.assertEqual(report["checks"]["terminology"]["status"], "pass")
        self.assertEqual(
            report["checks"]["terminology"]["inconsistency_count"],
            0,
        )

    def test_quality_report_flags_short_events_after_final_timing(self) -> None:
        layout = build_layout_policy(
            (1920, 1080),
            profile_name="adaptive",
            font_name="Arial",
            font_scale=100,
        )
        segment = {
            "id": 0,
            "start": 1.0,
            "end": 1.5,
            "text": "Yes",
            "en": "Yes",
            "zh": "\u662f",
            "display": True,
        }

        report = build_quality_report(
            [segment],
            [segment],
            [segment],
            layout,
            max_duration=5.5,
            target_chinese_cps=9.0,
            target_english_cps=20.0,
        )

        self.assertEqual(report["status"], "review")
        self.assertEqual(report["checks"]["timing"]["under_min_duration_count"], 1)
        self.assertAlmostEqual(
            report["thresholds"]["min_duration_seconds"],
            5 / 6,
        )

    def test_quality_report_flags_likely_sparse_existing_subtitle_track(self) -> None:
        layout = build_layout_policy(
            (1920, 1080),
            profile_name="adaptive",
            font_name="Arial",
            font_scale=100,
        )
        segments = [
            {
                "id": index,
                "start": 60.0 * index,
                "end": 60.0 * index + 2.0,
                "text": f"Sign {index}",
                "en": f"Sign {index}",
                "zh": f"\u6807\u8bc6{index}",
                "timing_origin": "existing_subtitle",
                "display": True,
            }
            for index in range(8)
        ]

        report = build_quality_report(
            segments,
            segments,
            segments,
            layout,
            max_duration=5.5,
            target_chinese_cps=9.0,
            target_english_cps=20.0,
            sync_report={
                "mode": "auto",
                "status": "already_aligned",
                "pipeline_quality_passed": True,
                "applied": False,
            },
            source_kind="existing Chinese embedded subtitles",
            video_duration_seconds=3600.0,
        )

        source_check = report["checks"]["source_completeness"]
        self.assertEqual(report["status"], "review")
        self.assertEqual(source_check["status"], "review")
        self.assertTrue(source_check["likely_sparse"])
        self.assertLess(source_check["timeline_coverage_ratio"], 0.01)

    def test_quality_report_flags_low_asr_and_ocr_confidence(self) -> None:
        layout = build_layout_policy(
            (1920, 1080),
            profile_name="adaptive",
            font_name="Arial",
            font_scale=100,
        )
        source = [
            {
                "id": 0,
                "start": 1.0,
                "end": 3.0,
                "text": "uncertain speech",
                "en": "uncertain speech",
                "zh": "\u4e0d\u786e\u5b9a\u7684\u8bed\u97f3",
                "asr_avg_logprob": -1.2,
                "asr_no_speech_prob": 0.72,
                "words": [
                    {"word": "uncertain", "start": 1.0, "end": 2.0, "probability": 0.2},
                    {"word": "speech", "start": 2.0, "end": 3.0, "probability": 0.3},
                ],
                "display": True,
            },
            {
                "id": 1,
                "start": 3.0,
                "end": 5.0,
                "text": "OCR text",
                "en": "OCR text",
                "zh": "OCR \u6587\u672c",
                "ocr_confidence": 0.61,
                "display": True,
            },
        ]

        report = build_quality_report(
            source,
            source,
            source,
            layout,
            max_duration=5.5,
            target_chinese_cps=9.0,
            target_english_cps=20.0,
        )

        confidence = report["checks"]["recognition_confidence"]
        self.assertEqual(report["status"], "review")
        self.assertEqual(confidence["status"], "review")
        self.assertEqual(confidence["low_confidence_count"], 2)
        self.assertEqual(report["summary"]["low_confidence_events"], 2)

    def test_quality_report_requires_review_when_independent_qa_fails(self) -> None:
        layout = build_layout_policy(
            (1920, 1080),
            profile_name="adaptive",
            font_name="Arial",
            font_scale=100,
        )
        segment = {
            "id": 0,
            "start": 1.0,
            "end": 3.0,
            "text": "Hello",
            "en": "Hello",
            "zh": "\u4f60\u597d",
            "display": True,
        }

        report = build_quality_report(
            [segment],
            [segment],
            [segment],
            layout,
            max_duration=5.5,
            target_chinese_cps=9.0,
            target_english_cps=20.0,
            style_guide={"status": "ready", "terminology": []},
            final_qa_report={"status": "failed", "error": "remote unavailable"},
        )

        self.assertEqual(report["status"], "review")
        self.assertEqual(report["checks"]["movie_style"]["status"], "pass")
        self.assertEqual(report["checks"]["independent_final_qa"]["status"], "review")
        self.assertEqual(report["summary"]["llm_review_items"], 1)

    def test_quality_report_exposes_rejected_final_qa_event(self) -> None:
        layout = build_layout_policy(
            (1920, 1080),
            profile_name="adaptive",
            font_name="Arial",
            font_scale=100,
        )
        segment = {
            "id": 8,
            "start": 4.0,
            "end": 6.0,
            "text": "Hello",
            "en": "Hello",
            "zh": "\u4f60\u597d",
            "display": True,
        }

        report = build_quality_report(
            [segment],
            [segment],
            [segment],
            layout,
            max_duration=5.5,
            target_chinese_cps=9.0,
            target_english_cps=20.0,
            final_qa_report={
                "status": "review",
                "rejected_count": 1,
                "rejected_items": [
                    {
                        "id": 8,
                        "start": 4.0,
                        "reason": "source_language_changed",
                    }
                ],
            },
        )

        check = report["checks"]["independent_final_qa"]
        self.assertEqual(check["status"], "review")
        self.assertEqual(check["samples"][0]["id"], 8)
        self.assertEqual(
            check["samples"][0]["issue"],
            "source_language_changed",
        )

    def test_quality_report_samples_worst_and_timeline_spread(self) -> None:
        layout = build_layout_policy(
            (1920, 1080),
            profile_name="adaptive",
            font_name="Arial",
            font_scale=100,
        )
        segments = [
            {
                "id": index,
                "start": float(index * 2),
                "end": float(index * 2 + 1),
                "text": "a" * (80 if index == 39 else 21),
                "en": "a" * (80 if index == 39 else 21),
                "zh": "好",
                "display": True,
            }
            for index in range(40)
        ]

        report = build_quality_report(
            segments,
            segments,
            segments,
            layout,
            max_duration=5.5,
            target_chinese_cps=9.0,
            target_english_cps=20.0,
        )

        readability = report["checks"]["readability"]
        sampled_ids = [item["id"] for item in readability["samples"]]
        self.assertEqual(readability["sample_count"], 40)
        self.assertEqual(readability["sample_limit"], 20)
        self.assertTrue(readability["samples_truncated"])
        self.assertEqual(len(sampled_ids), 20)
        self.assertIn(39, sampled_ids)
        self.assertTrue(any(10 <= item_id <= 30 for item_id in sampled_ids))

    def test_final_qa_review_count_is_not_capped_by_sample_limit(self) -> None:
        layout = build_layout_policy(
            (1920, 1080),
            profile_name="adaptive",
            font_name="Arial",
            font_scale=100,
        )
        segment = {
            "id": 0,
            "start": 1.0,
            "end": 3.0,
            "text": "Hello",
            "en": "Hello",
            "zh": "你好",
            "display": True,
        }
        rejected_items = [
            {
                "id": index,
                "start": float(index),
                "reason": "invalid_language_or_terminology",
            }
            for index in range(35)
        ]

        report = build_quality_report(
            [segment],
            [segment],
            [segment],
            layout,
            max_duration=5.5,
            target_chinese_cps=9.0,
            target_english_cps=20.0,
            final_qa_report={
                "status": "review",
                "rejected_count": 35,
                "rejected_items": rejected_items,
            },
        )

        check = report["checks"]["independent_final_qa"]
        self.assertEqual(check["review_items"], 35)
        self.assertEqual(check["sample_count"], 35)
        self.assertTrue(check["samples_truncated"])
        self.assertEqual(len(check["samples"]), 20)

    def test_japanese_source_track_is_split_and_not_reported_as_english_contamination(self) -> None:
        layout = build_layout_policy(
            (1920, 1080),
            profile_name="adaptive",
            font_name="Arial",
            font_scale=100,
        )
        japanese = "\u3053\u308c\u306f\u9577\u3044\u65e5\u672c\u8a9e\u5b57\u5e55\u3092\u81ea\u7136\u306b\u5206\u5272\u3059\u308b\u305f\u3081\u306e\u6587\u7ae0\u3067\u3059"
        source = [
            {
                "id": 0,
                "start": 1.0,
                "end": 7.0,
                "text": japanese,
                "en": japanese,
                "zh": "\u8fd9\u662f\u7528\u4e8e\u81ea\u7136\u62c6\u5206\u957f\u65e5\u8bed\u5b57\u5e55\u7684\u53e5\u5b50",
                "source_language": "ja",
            }
        ]

        output = prepare_segments_for_output(
            source,
            max_words=12,
            max_chars=42,
            max_duration=5.5,
            layout_policy=layout,
        )
        report = build_quality_report(
            source,
            source,
            output,
            layout,
            max_duration=5.5,
            target_chinese_cps=9.0,
            target_english_cps=20.0,
        )

        self.assertGreater(len(output), 1)
        self.assertEqual("".join(item["en"] for item in output), japanese)
        self.assertTrue(all(len(item["en"]) <= 16 for item in output))
        self.assertEqual(
            report["checks"]["completeness"]["english_contains_cjk_count"],
            0,
        )
