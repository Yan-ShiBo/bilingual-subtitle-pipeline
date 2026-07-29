import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import font_delivery  # noqa: E402
import pipeline_policy  # noqa: E402
import audio_to_subtitle  # noqa: E402
import subtitle_frontend  # noqa: E402
import subtitle_pipeline  # noqa: E402
import subtitle_sync  # noqa: E402
from frontend_settings import FrontendSettingsStore  # noqa: E402


class SubtitleSyncTests(unittest.TestCase):
    def test_synced_srt_is_mapped_back_by_event_marker_after_reordering(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "reordered.srt"
            path.write_text(
                "2\n00:00:20,000 --> 00:00:21,000\nSUBTITLE_SYNC_EVENT_000001\n\n"
                "1\n00:00:10,000 --> 00:00:11,000\nSUBTITLE_SYNC_EVENT_000000\n",
                encoding="utf-8",
            )

            timing = subtitle_sync._read_timing_srt(path)

        self.assertEqual(timing, [(10.0, 11.0), (20.0, 21.0)])

    def test_auto_sync_applies_verified_piecewise_timing_and_writes_report(self) -> None:
        source = [
            {"id": 0, "start": 10.0, "end": 11.0, "text": "First"},
            {"id": 1, "start": 20.0, "end": 21.0, "text": "Second"},
        ]

        def fake_run(_video, _input, output, **_kwargs):
            subtitle_sync._write_timing_srt(
                output,
                [
                    {"start": 9.5, "end": 10.5},
                    {"start": 20.4, "end": 21.4},
                ],
            )
            return {
                "retval": 0,
                "sync_was_successful": True,
                "offset_seconds": -0.05,
                "framerate_scale_factor": 1.0,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "sync.json"
            with patch.object(subtitle_sync, "_run_ffsubsync", side_effect=fake_run):
                output, report = subtitle_sync.synchronize_subtitle_segments(
                    Path(tmpdir) / "movie.mkv",
                    source,
                    mode="auto",
                    report_path=report_path,
                )
            saved = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(output[0]["start"], 9.5)
        self.assertEqual(output[1]["start"], 20.4)
        self.assertEqual(output[0]["pre_sync_start"], 10.0)
        self.assertTrue(report["applied"])
        self.assertEqual(saved["status"], "aligned")

    def test_detect_mode_reports_candidate_without_modifying_timing(self) -> None:
        source = [{"id": 0, "start": 10.0, "end": 11.0, "text": "Line"}]

        def fake_run(_video, _input, output, **_kwargs):
            subtitle_sync._write_timing_srt(output, [{"start": 9.0, "end": 10.0}])
            return {
                "retval": 0,
                "sync_was_successful": True,
                "offset_seconds": -1.0,
                "framerate_scale_factor": 1.0,
            }

        with patch.object(subtitle_sync, "_run_ffsubsync", side_effect=fake_run):
            output, report = subtitle_sync.synchronize_subtitle_segments(
                Path("movie.mkv"),
                source,
                mode="detect",
            )

        self.assertEqual(output[0]["start"], 10.0)
        self.assertFalse(report["applied"])
        self.assertEqual(report["max_absolute_shift_seconds"], 1.0)

    def test_auto_sync_rejects_piecewise_result_that_reorders_cues(self) -> None:
        source = [
            {"id": 0, "start": 10.0, "end": 11.0, "text": "First"},
            {"id": 1, "start": 20.0, "end": 21.0, "text": "Second"},
        ]

        def fake_run(_video, _input, output, **_kwargs):
            subtitle_sync._write_timing_srt(
                output,
                [
                    {"start": 10.0, "end": 11.0},
                    {"start": 5.0, "end": 6.0},
                ],
            )
            return {
                "retval": 0,
                "sync_was_successful": True,
                "offset_seconds": -7.5,
                "framerate_scale_factor": 1.0,
            }

        with patch.object(subtitle_sync, "_run_ffsubsync", side_effect=fake_run):
            output, report = subtitle_sync.synchronize_subtitle_segments(
                Path("movie.mkv"),
                source,
                mode="auto",
            )

        self.assertEqual(output, source)
        self.assertFalse(report["applied"])
        self.assertEqual(report["status"], "rejected_low_quality")
        self.assertEqual(report["out_of_order_cue_count"], 1)

    def test_auto_sync_falls_back_to_safe_global_offset(self) -> None:
        source = [
            {"id": 0, "start": 10.0, "end": 11.0, "text": "First"},
            {"id": 1, "start": 20.0, "end": 21.0, "text": "Second"},
        ]

        def fake_run(_video, _input, output, *, piecewise, **_kwargs):
            if piecewise:
                timing = [
                    {"start": 10.0, "end": 11.0},
                    {"start": 5.0, "end": 6.0},
                ]
                offset = -7.5
            else:
                timing = [
                    {"start": 10.5, "end": 11.5},
                    {"start": 20.5, "end": 21.5},
                ]
                offset = 0.5
            subtitle_sync._write_timing_srt(output, timing)
            return {
                "retval": 0,
                "sync_was_successful": True,
                "offset_seconds": offset,
                "framerate_scale_factor": 1.0,
            }

        with patch.object(subtitle_sync, "_run_ffsubsync", side_effect=fake_run):
            output, report = subtitle_sync.synchronize_subtitle_segments(
                Path("movie.mkv"),
                source,
                mode="auto",
            )

        self.assertEqual(output[0]["start"], 10.5)
        self.assertTrue(report["applied"])
        self.assertTrue(report["used_global_fallback"])
        self.assertEqual(report["status"], "aligned_global_fallback")


class FontDeliveryTests(unittest.TestCase):
    def test_font_bundle_copies_font_and_attaches_it_to_mks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ass = root / "movie.bilingual.ass"
            ass.write_text("[Script Info]\n", encoding="utf-8")
            font = root / "Custom.ttf"
            font.write_bytes(b"font data")
            commands = []

            def fake_run(command, **_kwargs):
                commands.append(command)
                Path(command[-1]).write_bytes(b"matroska")
                return SimpleNamespace(returncode=0, stderr="")

            with patch.object(font_delivery.subprocess, "run", side_effect=fake_run):
                manifest = font_delivery.package_ass_with_font(
                    ass,
                    font,
                    font_family="Custom Family",
                    ffmpeg_path="ffmpeg",
                )

            self.assertTrue(Path(manifest["matroska_subtitle_bundle"]).is_file())
            self.assertTrue((root / "fonts" / "Custom.ttf").is_file())
            self.assertIn("-attach", commands[0])
            self.assertIn("filename=Custom.ttf", commands[0])


class FrontendSettingsTests(unittest.TestCase):
    def test_form_settings_persist_across_store_instances(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            first = FrontendSettingsStore(path)
            first.save_form({"source": "embedded", "subtitleSync": "auto"})
            second = FrontendSettingsStore(path)

            self.assertEqual(second.load()["form"]["source"], "embedded")
            self.assertEqual(second.load()["form"]["subtitleSync"], "auto")
            self.assertEqual([item.name for item in path.parent.iterdir()], ["settings.json"])

    @unittest.skipUnless(os.name == "nt", "Windows DPAPI test")
    def test_password_is_dpapi_protected_at_rest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "settings.json"
            store = FrontendSettingsStore(path)
            store.save_remote(
                {
                    "host": "10.12.96.203",
                    "user": "csynth",
                    "auth_method": "password",
                    "password": "secret-value",
                    "remember_password": True,
                }
            )

            raw = path.read_text(encoding="utf-8")
            loaded = store.load()

        self.assertNotIn("secret-value", raw)
        self.assertEqual(loaded["remote"]["password"], "secret-value")

    def test_frontend_exposes_sync_font_and_server_settings_controls(self) -> None:
        page = subtitle_frontend.html_page()

        self.assertIn('id="subtitleSync"', page)
        self.assertIn('id="subtitleFontFile"', page)
        self.assertIn("/api/settings/load", page)
        self.assertIn("Windows 凭据保护", page)
        self.assertIn("legacy.auth_method || 'password'", page)

    def test_frontend_accepts_only_loopback_host_headers(self) -> None:
        self.assertTrue(subtitle_frontend.is_loopback_host("127.0.0.1:8765"))
        self.assertTrue(subtitle_frontend.is_loopback_host("localhost:8765"))
        self.assertTrue(subtitle_frontend.is_loopback_host("[::1]:8765"))
        self.assertFalse(subtitle_frontend.is_loopback_host("subtitle.example:8765"))
        self.assertFalse(subtitle_frontend.is_loopback_host("127.0.0.1.example:8765"))

    def test_frontend_marks_checkpoint_from_old_model_for_reprocessing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_dir = root / "Series" / "Movie"
            out_dir.mkdir(parents=True)
            checkpoint = out_dir / "Movie.segments.checkpoint.json"
            checkpoint.write_text(
                json.dumps(
                    [
                        {
                            "id": 0,
                            "start": 1.0,
                            "end": 2.0,
                            "text": "Line",
                            "display": True,
                            "processing_policy_version": pipeline_policy.PROCESSING_POLICY_VERSION,
                            "terminology_policy_version": pipeline_policy.TERMINOLOGY_POLICY_VERSION,
                            "llm_model": "qwen3:14b",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            info = subtitle_frontend.checkpoint_info(
                root,
                "Series",
                "Movie",
                "qwen3:30b",
            )

        self.assertFalse(info["checkpoint_compatible"])
        self.assertEqual(info["checkpoint_model"], "qwen3:14b")


class EntrypointTests(unittest.TestCase):
    def test_legacy_cli_routes_processing_to_canonical_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "movie.mkv"
            video.write_bytes(b"video")
            original_argv = list(sys.argv)
            captured = {}

            def canonical_main():
                captured["argv"] = list(sys.argv)

            try:
                with (
                    patch.object(
                        sys,
                        "argv",
                        [
                            "subtitle_pipeline.py",
                            "--video",
                            str(video),
                            "--output",
                            str(root / "out"),
                            "--model",
                            "qwen3:30b",
                        ],
                    ),
                    patch.object(subtitle_pipeline, "find_ffmpeg", return_value="ffmpeg"),
                    patch.object(audio_to_subtitle, "main", side_effect=canonical_main),
                ):
                    self.assertEqual(subtitle_pipeline.main(), 0)
            finally:
                sys.argv = original_argv

        self.assertTrue(captured["argv"][0].endswith("audio_to_subtitle.py"))
        self.assertIn("--output-root", captured["argv"])
        self.assertEqual(captured["argv"][captured["argv"].index("--llm-model") + 1], "qwen3:30b")


if __name__ == "__main__":
    unittest.main()
