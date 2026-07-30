import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import subtitle_frontend  # noqa: E402
from audio_to_subtitle import prepare_segments_for_output  # noqa: E402
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
