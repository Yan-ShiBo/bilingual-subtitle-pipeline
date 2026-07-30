import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import audio_to_subtitle  # noqa: E402
import subtitle_frontend  # noqa: E402
import subtitle_pipeline  # noqa: E402
from output_paths import find_existing_output_root, resolve_output_root  # noqa: E402
from subtitle_pipeline import PgsImageEvent, SubtitleEvent, ocr_pgs_events, pair_events  # noqa: E402
from audio_to_subtitle import (  # noqa: E402
    ExistingSubtitleTracks,
    apply_display_timing,
    build_source_request_fingerprint,
    checkpoint_matches_segments,
    extend_display_over_hidden_segments,
    find_sidecar_subtitle,
    generate_ass,
    guard_model_hidden_segments,
    load_cached_source_segments,
    merge_existing_subtitle_segments,
    optimize_readability_timing,
    prepare_segments_for_output,
    proofread_existing_chinese_segments,
    readability_stats,
    resolve_timeline_overlaps,
    sanitize_checkpoint_display_timing,
    segments_have_chinese,
    split_segments_for_subtitles,
    source_cache_uses_current_timing_policy,
    source_cache_matches_request,
    suppress_subtitle_track_label_artifacts,
    synchronize_existing_subtitle_tracks,
    translate_and_correct_segments,
)


class DisplayCleanupTests(unittest.TestCase):
    def test_atomic_json_write_retries_transient_windows_lock(self) -> None:
        original_replace = Path.replace
        replace_calls = 0

        def flaky_replace(source, target):
            nonlocal replace_calls
            replace_calls += 1
            if replace_calls < 3:
                raise PermissionError(5, "temporarily locked")
            return original_replace(source, target)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "checkpoint.json"
            with (
                patch.object(Path, "replace", new=flaky_replace),
                patch.object(audio_to_subtitle.time, "sleep") as sleep,
            ):
                audio_to_subtitle.write_json_atomic(
                    output_path,
                    {"status": "running", "reviewed_count": 270},
                )
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            temporary_files = list(Path(tmpdir).glob("*.tmp"))

        self.assertEqual(payload["reviewed_count"], 270)
        self.assertEqual(replace_calls, 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(temporary_files, [])

    def test_output_root_discovers_existing_central_subtitle_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "Movies" / "Film" / "film.mkv"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"video")
            output_root = root / "Movies" / "1 \u5b57\u5e55"
            output_dir = output_root / "Film" / "Film"
            output_dir.mkdir(parents=True)
            (output_dir / "Film.segments.source.json").write_text("[]", encoding="utf-8")

            discovered = find_existing_output_root(video, "Film", "Film")

        self.assertEqual(discovered, output_root)

    def test_output_root_discovers_existing_episode_with_spacing_difference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            media_root = root / "\u7535\u5f71" / "21\u4e16\u7eaa\u6027\u7231\u6307\u5357"
            video = media_root / "[21\u4e16\u7eaa\u6027\u7231\u6307\u5357].A.Girls.Guide.4of8.avi"
            media_root.mkdir(parents=True)
            video.write_bytes(b"video")
            output_dir = (
                media_root
                / "21\u4e16\u7eaa\u6027\u7231\u6307\u5357 A Girl's Guide"
                / "21\u4e16\u7eaa\u6027\u7231\u6307\u5357 A Girl's Guide 4 of 8"
            )
            output_dir.mkdir(parents=True)
            (output_dir / "episode.segments.checkpoint.json").write_text("[]", encoding="utf-8")

            discovered = find_existing_output_root(video, "A Girls Guide", "Episode 4")

        self.assertEqual(discovered, media_root)

    def test_output_root_discovers_standalone_movie_by_high_title_similarity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            movie_root = root / "Movies"
            video = movie_root / "Oppenheimer.2023.2160p.UHD.BluRay.mkv"
            movie_root.mkdir(parents=True)
            video.write_bytes(b"video")
            output_root = movie_root / "1 \u5b57\u5e55"
            output_dir = output_root / "Oppenheimer" / "Oppenheimer"
            output_dir.mkdir(parents=True)
            (output_dir / "Oppenheimer.bilingual.ass").write_text("subtitle", encoding="utf-8")

            discovered = find_existing_output_root(video, "Unknown", "Unknown")

        self.assertEqual(discovered, output_root)

    def test_output_root_respects_manual_directory_but_migrates_old_project_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "Movies" / "Show" / "show.4of8.mkv"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"video")
            existing_dir = video.parent / "Show" / "Show 4 of 8"
            existing_dir.mkdir(parents=True)
            (existing_dir / "Show 4 of 8.segments.source.json").write_text("[]", encoding="utf-8")
            manual_root = root / "manual-output"
            legacy_default = root / "Engineering" / "1 \u5b57\u5e55"

            manual, manual_source = resolve_output_root(
                video,
                "Show",
                "Show 4 of 8",
                requested=manual_root,
                legacy_default=legacy_default,
            )
            migrated, migrated_source = resolve_output_root(
                video,
                "Show",
                "Show 4 of 8",
                requested=legacy_default,
                legacy_default=legacy_default,
            )

        self.assertEqual((manual, manual_source), (manual_root, "manual"))
        self.assertEqual((migrated, migrated_source), (video.parent, "existing"))

    def test_frontend_analysis_recovers_existing_output_from_legacy_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "Show" / "show.4of8.mkv"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"video")
            output_dir = video.parent / "Show" / "Show 4 of 8"
            output_dir.mkdir(parents=True)
            (output_dir / "Show 4 of 8.segments.source.json").write_text("[]", encoding="utf-8")
            with (
                patch.object(
                    subtitle_frontend,
                    "default_names",
                    return_value={
                        "series_name": "Show",
                        "movie_name": "Show 4 of 8",
                        "name_source": "test",
                    },
                ),
                patch.object(subtitle_frontend, "list_sidecar_subtitles", return_value=[]),
                patch.object(subtitle_frontend, "list_embedded_subtitles", return_value=[]),
                patch.object(subtitle_frontend, "list_embedded_audio", return_value=[]),
            ):
                info = subtitle_frontend.analyze_input(
                    {
                        "path": str(video),
                        "output_root": str(subtitle_frontend.LEGACY_DEFAULT_OUTPUT_ROOT),
                    }
                )

        self.assertEqual(info["output_root"], str(video.parent))
        self.assertEqual(info["output_root_source"], "existing")
        self.assertTrue(info["source_segments_exists"])

    def test_frontend_page_migrates_project_relative_output_default(self) -> None:
        page = subtitle_frontend.html_page()

        self.assertIn("自动查找已有输出", page)
        self.assertIn("function migrateOutputRootDefault()", page)
        self.assertIn("subtitleOutputRootMigrationVersion", page)
        self.assertIn("< 2", page)
        self.assertIn("document.getElementById('outputRoot').value = data.output_root || ''", page)
        self.assertIn("已找到已有输出", page)
        self.assertIn("已找到旧任务", page)
        self.assertIn("output_root_source", subtitle_frontend.analyze_input.__code__.co_varnames)

    def test_hidden_duplicate_extends_previous_visible_subtitle(self) -> None:
        segments = [
            {"id": 0, "start": 10.0, "end": 11.0, "text": "hello", "en": "hello", "zh": "hello zh"},
            {"id": 1, "start": 11.0, "end": 12.0, "text": "hello", "en": "hello", "zh": "hello zh", "display": False},
            {"id": 2, "start": 12.0, "end": 13.0, "text": "hello", "en": "hello", "zh": "hello zh", "display": False},
        ]

        extend_display_over_hidden_segments(segments)

        self.assertEqual(segments[0]["display_end"], 13.0)

    def test_cached_checkpoint_timing_is_rebuilt_from_source_anchors(self) -> None:
        cached = [
            {
                "id": 0,
                "start": 10.0,
                "end": 11.0,
                "display_start": 35.0,
                "display_end": 36.0,
                "text": "complete",
                "display": True,
            },
            {
                "id": 1,
                "start": 11.0,
                "end": 12.0,
                "display_start": 36.0,
                "display_end": 37.0,
                "text": "duplicate",
                "display": False,
            },
        ]

        sanitized = sanitize_checkpoint_display_timing(cached)

        self.assertNotIn("display_start", sanitized[0])
        self.assertEqual(sanitized[0]["display_end"], 12.0)
        self.assertNotIn("display_end", sanitized[1])

    def test_apply_display_timing_rejects_model_timing_changes(self) -> None:
        source = {"id": 4, "start": 20.0, "end": 21.0, "text": "fragment"}

        output = apply_display_timing(
            {**source, "en": "complete sentence", "zh": "complete sentence zh"},
            {
                "display": True,
                "display_start": 20.0,
                "display_end": 24.0,
            },
            context_start=18.0,
            context_end=25.0,
        )

        self.assertEqual(output["start"], 20.0)
        self.assertEqual(output["end"], 21.0)
        self.assertNotIn("display_start", output)
        self.assertNotIn("display_end", output)
        self.assertTrue(checkpoint_matches_segments([output], [source]))

    def test_generate_ass_skips_hidden_segments(self) -> None:
        segments = [
            {
                "id": 0,
                "start": 30.0,
                "end": 31.0,
                "display_end": 33.0,
                "text": "complete",
                "en": "complete",
                "zh": "complete zh",
            },
            {
                "id": 1,
                "start": 31.0,
                "end": 32.0,
                "text": "duplicate",
                "en": "duplicate",
                "zh": "duplicate zh",
                "display": False,
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "out.ass"
            generate_ass(segments, out_path, "bilingual")
            body = out_path.read_text(encoding="utf-8")

        self.assertIn("0:00:30.00,0:00:33.00", body)
        self.assertIn("complete", body)
        self.assertNotIn("duplicate", body)

    def test_old_checkpoint_without_display_metadata_is_rejected(self) -> None:
        source = [{"id": 0, "start": 1.0, "end": 2.0, "text": "same text"}]
        old_checkpoint = [{"id": 0, "start": 1.0, "end": 2.0, "text": "same text"}]

        self.assertFalse(checkpoint_matches_segments(old_checkpoint, source))

    def test_checkpoint_rejects_different_model_or_policy(self) -> None:
        source = [{"id": 0, "start": 1.0, "end": 2.0, "text": "same text"}]
        cached = [
            {
                **source[0],
                "display": True,
                "processing_mode": "translate",
                "processing_policy_version": audio_to_subtitle.PROCESSING_POLICY_VERSION,
                "llm_model": "qwen3:14b",
            }
        ]

        self.assertFalse(
            checkpoint_matches_segments(
                cached,
                source,
                expected_mode="translate",
                expected_model="qwen3:30b",
                expected_policy_version=audio_to_subtitle.PROCESSING_POLICY_VERSION,
            )
        )
        cached[0]["llm_model"] = "qwen3:30b"
        cached[0]["processing_policy_version"] -= 1
        self.assertFalse(
            checkpoint_matches_segments(
                cached,
                source,
                expected_mode="translate",
                expected_model="qwen3:30b",
                expected_policy_version=audio_to_subtitle.PROCESSING_POLICY_VERSION,
            )
        )

    def test_checkpoint_rejects_outdated_terminology_policy(self) -> None:
        source = [{"id": 0, "start": 1.0, "end": 2.0, "text": "John"}]
        cached = [
            {
                **source[0],
                "display": True,
                "processing_mode": "translate",
                "processing_policy_version": audio_to_subtitle.PROCESSING_POLICY_VERSION,
                "terminology_policy_version": 0,
                "llm_model": "qwen3:30b",
            }
        ]

        self.assertFalse(
            checkpoint_matches_segments(
                cached,
                source,
                expected_mode="translate",
                expected_model="qwen3:30b",
                expected_policy_version=audio_to_subtitle.PROCESSING_POLICY_VERSION,
                expected_terminology_policy_version=audio_to_subtitle.TERMINOLOGY_POLICY_VERSION,
            )
        )

    def test_translate_segments_accepts_llm_hidden_duplicates(self) -> None:
        source = [
            {"id": 0, "start": 209.03, "end": 211.0, "text": "The female orgasm builds to a"},
            {"id": 1, "start": 211.0, "end": 212.0, "text": "The female orgasm builds to a"},
            {"id": 2, "start": 212.0, "end": 213.0, "text": "point of warmth and ecstasy"},
        ]
        llm_response = """
        [
          {
            "index": 0,
            "corrected_text": "The female orgasm builds to a point of warmth and ecstasy",
            "chinese_translation": "\u5408\u5e76\u540e\u7684\u7ffb\u8bd1",
            "display": true,
            "display_start": 209.03,
            "display_end": 213.0
          },
          {
            "index": 1,
            "corrected_text": "",
            "chinese_translation": "",
            "display": false
          },
          {
            "index": 2,
            "corrected_text": "",
            "chinese_translation": "",
            "display": false
          }
        ]
        """

        with patch.object(audio_to_subtitle, "call_llm", return_value=llm_response):
            output = translate_and_correct_segments(
                source,
                llm_model="qwen3:30b",
                batch_size=3,
                context_lines=0,
                source_language="en",
            )

        self.assertEqual(output[0]["display_end"], 213.0)
        self.assertFalse(output[1]["display"])
        self.assertFalse(output[2]["display"])
        self.assertEqual(output[0]["zh"], "\u5408\u5e76\u540e\u7684\u7ffb\u8bd1")

    def test_model_hide_guard_forces_unique_source_line_visible(self) -> None:
        source = [
            {"id": 0, "start": 1.0, "end": 2.0, "text": "First complete sentence."},
            {"id": 1, "start": 2.1, "end": 3.0, "text": "A different important sentence."},
        ]
        processed = [
            {
                **source[0],
                "en": "First complete sentence.",
                "zh": "\u7b2c\u4e00\u53e5\u5b8c\u6574\u7684\u8bdd\u3002",
                "display": True,
            },
            {
                **source[1],
                "en": "A different important sentence.",
                "zh": "\u53e6\u4e00\u53e5\u91cd\u8981\u7684\u8bdd\u3002",
                "display": False,
                "model_display_requested": False,
            },
        ]

        guarded = guard_model_hidden_segments(source, processed)

        self.assertTrue(guarded[1]["display"])
        self.assertEqual(guarded[1]["display_guard"], "forced_visible")
        self.assertEqual(
            guarded[1]["display_guard_reason"],
            "unique_source_not_covered",
        )

    def test_model_hide_guard_accepts_fragment_covered_by_earlier_line(self) -> None:
        source = [
            {"id": 0, "start": 10.0, "end": 11.0, "text": "A sentence starts here"},
            {"id": 1, "start": 11.0, "end": 12.0, "text": "and finishes here"},
        ]
        processed = [
            {
                **source[0],
                "en": "A sentence starts here and finishes here",
                "zh": "\u4e00\u53e5\u8bdd\u5728\u8fd9\u91cc\u5b8c\u6574\u8bf4\u5b8c",
                "display": True,
            },
            {
                **source[1],
                "en": "and finishes here",
                "zh": "",
                "display": False,
                "model_display_requested": False,
            },
        ]

        guarded = guard_model_hidden_segments(source, processed)

        self.assertFalse(guarded[1]["display"])
        self.assertEqual(guarded[1]["display_guard"], "accepted_hidden")
        self.assertEqual(
            guarded[1]["display_guard_reason"],
            "covered_by_adjacent_visible_line",
        )

    def test_delivery_suppresses_subtitle_track_label_artifact(self) -> None:
        segments = [
            {
                "id": 0,
                "start": 1.0,
                "end": 2.0,
                "text": "(SPEAKING ENGLISH)",
                "en": "(SPEAKING ENGLISH)",
                "zh": "（说英语）",
                "display": True,
            },
            {
                "id": 1,
                "start": 99.0,
                "end": 100.0,
                "text": "(ENGLISH - SDH)",
                "en": "(ENGLISH - SDH)",
                "zh": "（英文对白）",
                "display": True,
            },
        ]

        output, suppressed = suppress_subtitle_track_label_artifacts(segments)

        self.assertTrue(output[0]["display"])
        self.assertFalse(output[1]["display"])
        self.assertEqual(
            output[1]["display_suppression"],
            "subtitle_track_label_artifact",
        )
        self.assertEqual([item["id"] for item in suppressed], [1])
        output[1]["model_display_requested"] = False
        guarded = guard_model_hidden_segments(segments, output)
        self.assertFalse(guarded[1]["display"])

    def test_translation_pipeline_overrides_unsupported_model_hide(self) -> None:
        source = [
            {"id": 0, "start": 1.0, "end": 2.0, "text": "This line is unique."},
        ]
        llm_response = (
            '[{"index":0,"corrected_text":"This line is unique.",'
            '"chinese_translation":"\u8fd9\u53e5\u8bdd\u662f\u552f\u4e00\u7684\u3002","display":false}]'
        )

        with patch.object(audio_to_subtitle, "call_llm", return_value=llm_response):
            output = translate_and_correct_segments(
                source,
                llm_model="qwen3:30b",
                batch_size=1,
                context_lines=0,
                source_language="en",
            )

        self.assertTrue(output[0]["display"])
        self.assertEqual(output[0]["display_guard"], "forced_visible")

    def test_forced_visible_line_retries_until_chinese_is_complete(self) -> None:
        source = [
            {"id": 0, "start": 1.0, "end": 2.0, "text": "This line is unique."},
        ]
        responses = [
            (
                '[{"index":0,"corrected_text":"This line is unique.",'
                '"chinese_translation":"","display":false}]'
            ),
            (
                '[{"index":0,"corrected_text":"This line is unique.",'
                '"chinese_translation":"\u8fd9\u53e5\u8bdd\u662f\u552f\u4e00\u7684\u3002","display":true}]'
            ),
        ]
        prompts = []

        def call_llm(prompt, *_args, **_kwargs):
            prompts.append(prompt)
            return responses.pop(0)

        cp1252_stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
        with (
            patch.object(audio_to_subtitle, "call_llm", side_effect=call_llm),
            patch.object(sys, "stdout", cp1252_stdout),
        ):
            output = translate_and_correct_segments(
                source,
                llm_model="qwen3:30b",
                batch_size=1,
                context_lines=0,
                source_language="en",
            )

        self.assertEqual(len(prompts), 2)
        self.assertIn("MUST set", prompts[1])
        self.assertTrue(output[0]["display"])
        self.assertEqual(output[0]["zh"], "\u8fd9\u53e5\u8bdd\u662f\u552f\u4e00\u7684\u3002")
        self.assertEqual(output[0]["display_guard"], "forced_visible")

    def test_resume_reprocesses_restored_unique_line_with_missing_translation(self) -> None:
        source = [
            {"id": 0, "start": 1.0, "end": 2.0, "text": "This line is unique."},
        ]
        cached = [
            {
                **source[0],
                "en": "This line is unique.",
                "zh": "",
                "display": False,
                "processing_mode": "translate",
                "processing_policy_version": audio_to_subtitle.PROCESSING_POLICY_VERSION,
                "terminology_policy_version": audio_to_subtitle.TERMINOLOGY_POLICY_VERSION,
                "llm_model": "qwen3:30b",
            }
        ]
        response = (
            '[{"index":0,"corrected_text":"This line is unique.",'
            '"chinese_translation":"\u8fd9\u53e5\u8bdd\u662f\u552f\u4e00\u7684\u3002","display":true}]'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"
            checkpoint_path.write_text(
                json.dumps(cached, ensure_ascii=False),
                encoding="utf-8",
            )
            with patch.object(audio_to_subtitle, "call_llm", return_value=response) as call:
                output = translate_and_correct_segments(
                    source,
                    llm_model="qwen3:30b",
                    batch_size=1,
                    context_lines=0,
                    source_language="en",
                    checkpoint_path=checkpoint_path,
                )

        call.assert_called_once()
        self.assertTrue(output[0]["display"])
        self.assertEqual(output[0]["zh"], "\u8fd9\u53e5\u8bdd\u662f\u552f\u4e00\u7684\u3002")

    def test_translation_reuses_approved_name_terminology_across_batches(self) -> None:
        source = [
            {"id": 0, "start": 1.0, "end": 2.0, "text": "John arrived."},
            {"id": 1, "start": 2.0, "end": 3.0, "text": "John sat down."},
        ]
        prompts = []

        def call_llm(prompt, *_args, **_kwargs):
            prompts.append(prompt)
            if len(prompts) == 1:
                return (
                    '[{"index":0,"corrected_text":"John arrived.",'
                    '"chinese_translation":"\u7ea6\u7ff0\u5230\u4e86\u3002","display":true,'
                    '"terminology":[{"source":"John","target":"\u7ea6\u7ff0"}]}]'
                )
            self.assertIn("APPROVED MOVIE-WIDE TERMINOLOGY", prompt)
            self.assertIn("John => \u7ea6\u7ff0", prompt)
            return (
                '[{"index":0,"corrected_text":"John sat down.",'
                '"chinese_translation":"\u7ea6\u7ff0\u5750\u4e0b\u4e86\u3002","display":true,'
                '"terminology":[{"source":"John","target":"\u7ea6\u7ff0"}]}]'
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            terminology_path = Path(tmpdir) / "terminology.json"
            with patch.object(audio_to_subtitle, "call_llm", side_effect=call_llm):
                output = translate_and_correct_segments(
                    source,
                    llm_model="qwen3:30b",
                    batch_size=1,
                    context_lines=0,
                    source_language="en",
                    terminology_path=terminology_path,
                )
            terminology = json.loads(terminology_path.read_text(encoding="utf-8"))

        self.assertEqual(output[1]["zh"], "\u7ea6\u7ff0\u5750\u4e0b\u4e86\u3002")
        self.assertEqual(terminology["entries"], [{"source": "John", "target": "\u7ea6\u7ff0"}])

    def test_proofreading_restores_known_speaker_label_missing_in_chinese(self) -> None:
        source = [
            {
                "id": 0,
                "start": 1.0,
                "end": 3.0,
                "text": "AECH: Okay, Mom, I heard you.",
                "en": "AECH: Okay, Mom, I heard you.",
                "zh": "\u597d\uff0c\u8001\u5988\uff0c\u542c\u5230\u4e86",
                "source_language": "en",
                "source_text_authority": "ocr",
            }
        ]
        style_guide = {
            "status": "ready",
            "terminology": [{"source": "Aech", "target": "\u827e\u533a"}],
        }
        response = (
            '[{"index":0,"corrected_english":"AECH: Okay, Mom, I heard you.",'
            '"corrected_chinese":"\u597d\uff0c\u8001\u5988\uff0c\u542c\u5230\u4e86","display":true,'
            '"terminology":[]}]'
        )

        with (
            patch.object(audio_to_subtitle, "call_llm", return_value=response),
            patch.object(
                audio_to_subtitle,
                "retry_single_segment_llm",
                side_effect=AssertionError("speaker label restoration should be deterministic"),
            ),
        ):
            output = proofread_existing_chinese_segments(
                source,
                llm_model="qwen3.5:122b",
                batch_size=1,
                context_lines=0,
                style_guide=style_guide,
            )

        self.assertEqual(output[0]["zh"], "\u827e\u533a\uff1a\u597d\uff0c\u8001\u5988\uff0c\u542c\u5230\u4e86")

    def test_single_item_retry_rejects_known_name_conflict(self) -> None:
        source = {
            "id": 0,
            "start": 1.0,
            "end": 3.0,
            "text": "John arrived.",
        }
        responses = [
            (
                '[{"index":0,"corrected_text":"John arrived.",'
                '"chinese_translation":"\u5f3a\u5c3c\u5230\u4e86\u3002","display":true,'
                '"terminology":[]}]'
            ),
            (
                '[{"index":0,"corrected_text":"John arrived.",'
                '"chinese_translation":"\u7ea6\u7ff0\u5230\u4e86\u3002","display":true,'
                '"terminology":[]}]'
            ),
        ]
        glossary = {"john": {"source": "John", "target": "\u7ea6\u7ff0"}}

        with patch.object(
            audio_to_subtitle,
            "call_llm",
            side_effect=responses,
        ) as call:
            output = audio_to_subtitle.retry_single_segment_llm(
                source,
                "",
                "",
                None,
                1.0,
                3.0,
                "system",
                "qwen3:30b",
                "English",
                terminology_text="John => \u7ea6\u7ff0",
                terminology_glossary=glossary,
            )

        self.assertEqual(call.call_count, 2)
        self.assertEqual(output["chinese_translation"], "\u7ea6\u7ff0\u5230\u4e86\u3002")

    def test_single_item_retry_reports_exact_missing_full_name(self) -> None:
        source = {
            "id": 557,
            "start": 2292.665,
            "end": 2296.127,
            "text": "No, no. Karen Underwood, as in Ogden Morrow's wife?",
            "en": "No, no. Karen Underwood, as in Ogden Morrow's wife?",
            "zh": "\u4e0d\u4f1a\u5427\uff1b\u51ef\u4f26\u5b89\u5fb7\u4f0d\u6b27\u683c\u987f\u83ab\u6d1b\u7684\u8001\u5a46\uff1f",
        }
        responses = [
            (
                '[{"index":0,"corrected_english":"No, no. Karen Underwood, '
                "as in Ogden Morrow's wife?\","
                '"corrected_chinese":"\u4e0d\u4f1a\u5427\uff0c\u51ef\u4f26\u00b7\u5b89\u5fb7\u4f0d\u5fb7\uff1f'
                '\u83ab\u6d1b\u7684\u8001\u5a46\uff1f","display":true,"terminology":[]}]'
            ),
            (
                '[{"index":0,"corrected_english":"No, no. Karen Underwood, '
                "as in Ogden Morrow's wife?\","
                '"corrected_chinese":"\u4e0d\u4f1a\u5427\uff0c\u51ef\u4f26\u00b7\u5b89\u5fb7\u4f0d\u5fb7\uff1f'
                '\u6b27\u683c\u987f\u00b7\u83ab\u6d1b\u7684\u59bb\u5b50\uff1f","display":true,"terminology":[]}]'
            ),
        ]
        glossary = {
            "karen underwood": {
                "source": "Karen Underwood",
                "target": "\u51ef\u4f26\u00b7\u5b89\u5fb7\u4f0d\u5fb7",
            },
            "morrow": {"source": "Morrow", "target": "\u83ab\u6d1b"},
            "ogden morrow": {
                "source": "Ogden Morrow",
                "target": "\u6b27\u683c\u987f\u00b7\u83ab\u6d1b",
            },
        }
        prompts = []

        def call_llm(prompt, *_args, **_kwargs):
            prompts.append(prompt)
            return responses[len(prompts) - 1]

        with patch.object(
            audio_to_subtitle,
            "call_llm",
            side_effect=call_llm,
        ) as call:
            output = audio_to_subtitle.retry_single_segment_llm(
                source,
                "",
                "",
                None,
                2292.665,
                2296.127,
                "system",
                "qwen3.5:122b",
                "English",
                is_proofread=True,
                terminology_glossary=glossary,
            )

        self.assertEqual(call.call_count, 2)
        self.assertIn("REQUIRED TERMINOLOGY FOR THIS TARGET", prompts[0])
        self.assertIn(
            "Ogden Morrow => \u6b27\u683c\u987f\u00b7\u83ab\u6d1b",
            prompts[0],
        )
        self.assertIn("AUTOMATIC VALIDATION FEEDBACK", prompts[1])
        self.assertIn(
            "Ogden Morrow => \u6b27\u683c\u987f\u00b7\u83ab\u6d1b",
            prompts[1],
        )
        self.assertIn(
            "\u51ef\u4f26\u00b7\u5b89\u5fb7\u4f0d\u5fb7",
            output["corrected_chinese"],
        )
        self.assertIn(
            "\u6b27\u683c\u987f\u00b7\u83ab\u6d1b",
            output["corrected_chinese"],
        )

    def test_language_validation_preserves_existing_short_latin_token(self) -> None:
        self.assertTrue(
            audio_to_subtitle.is_language_valid(
                "Z!",
                "Z!",
                "Z",
                original_translation="Z",
            )
        )
        self.assertFalse(
            audio_to_subtitle.is_language_valid(
                "Z!",
                "Z!",
                "Z",
            )
        )
        self.assertFalse(
            audio_to_subtitle.is_language_valid(
                "Go!",
                "Go!",
                "Go",
                original_translation="\u8d70\uff01",
            )
        )

    def test_proofreading_keeps_existing_short_latin_token(self) -> None:
        source = [
            {
                "id": 0,
                "start": 6069.981,
                "end": 6070.982,
                "text": "Z!",
                "en": "Z!",
                "zh": "Z",
                "source_language": "en",
                "source_text_authority": "ocr",
            }
        ]
        response = (
            '[{"index":0,"corrected_english":"Z!",'
            '"corrected_chinese":"Z","display":true,"terminology":[]}]'
        )

        with (
            patch.object(audio_to_subtitle, "call_llm", return_value=response),
            patch.object(
                audio_to_subtitle,
                "retry_single_segment_llm",
                side_effect=AssertionError("the existing Latin token is valid"),
            ),
        ):
            output = proofread_existing_chinese_segments(
                source,
                llm_model="qwen3.5:122b",
                batch_size=1,
                context_lines=0,
            )

        self.assertEqual(output[0]["en"], "Z!")
        self.assertEqual(output[0]["zh"], "Z")

    def test_movie_style_prepass_is_cached_and_seeds_name_policy(self) -> None:
        source = [
            {"id": 0, "start": 1.0, "end": 2.0, "text": "John arrived."},
            {"id": 1, "start": 2.0, "end": 3.0, "text": "John sat down."},
        ]
        response = json.dumps(
            {
                "register": "Natural conversational Chinese.",
                "name_policy": "Always translate John as \u7ea6\u7ff0.",
                "address_policy": "Use relationship-aware forms of address.",
                "sdh_policy": "Translate English-only SDH cues when Chinese lacks them.",
                "punctuation_policy": "Use concise Chinese subtitle punctuation.",
                "terminology": [{"source": "John", "target": "\u7ea6\u7ff0"}],
            },
            ensure_ascii=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "style.json"
            with patch.object(audio_to_subtitle, "call_llm", return_value=response) as call:
                first = audio_to_subtitle.analyze_movie_style(
                    source,
                    "qwen3:30b",
                    artifact_path,
                )
                second = audio_to_subtitle.analyze_movie_style(
                    source,
                    "qwen3:30b",
                    artifact_path,
                )

        self.assertEqual(call.call_count, 1)
        self.assertEqual(first["terminology"], [{"source": "John", "target": "\u7ea6\u7ff0"}])
        self.assertEqual(second["input_fingerprint"], first["input_fingerprint"])

    def test_movie_terminology_review_only_applies_high_confidence_fixes(self) -> None:
        source = [
            {
                "id": 0,
                "start": 1.0,
                "end": 2.0,
                "text": "Aech arrived.",
                "en": "Aech arrived.",
                "zh": "\u827e\u533a\u6765\u4e86\u3002",
                "source_language": "en",
                "processing_mode": "proofread_existing_chinese",
                "terminology": [{"source": "Aech", "target": "\u827e\u533a"}],
            },
            {
                "id": 1,
                "start": 2.0,
                "end": 3.0,
                "text": "i-R0k.",
                "en": "i-R0k.",
                "zh": "\u6211\u6700\u5e1a\u3002",
                "source_language": "en",
                "processing_mode": "proofread_existing_chinese",
                "terminology": [{"source": "i-R0k", "target": "\u6211\u6700\u5e1a"}],
            },
        ]
        response = json.dumps(
            [
                {
                    "index": 0,
                    "target": "\u827e\u5947",
                    "confidence": 1.0,
                    "decision": "ocr_repair",
                },
                {
                    "index": 1,
                    "target": "\u827e\u6d1b\u514b",
                    "confidence": 1.0,
                    "decision": "retranslate",
                },
            ],
            ensure_ascii=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "terminology-review.json"
            with patch.object(
                audio_to_subtitle,
                "call_llm",
                return_value=response,
            ) as call:
                resolved, report = audio_to_subtitle.review_movie_terminology(
                    source,
                    "qwen3:30b",
                    artifact_path,
                    title="Ready Player One",
                )
                cached, cached_report = (
                    audio_to_subtitle.review_movie_terminology(
                        [
                            {
                                **segment,
                                "zh": f"{segment['zh']}\u3002",
                                "final_qa_policy_version": 4,
                            }
                            for segment in source
                        ],
                        "qwen3:30b",
                        artifact_path,
                        title="Ready Player One",
                    )
                )

        self.assertEqual(
            resolved,
            [
                {"source": "Aech", "target": "\u827e\u5947"},
                {"source": "i-R0k", "target": "\u6211\u6700\u5e1a"},
            ],
        )
        self.assertEqual(cached, resolved)
        self.assertEqual(report["correction_count"], 1)
        self.assertEqual(report["unapplied_count"], 1)
        self.assertEqual(cached_report["status"], "review")
        self.assertEqual(call.call_count, 1)

    def test_final_qa_fingerprint_includes_reviewed_terminology(self) -> None:
        source = [
            {
                "id": 0,
                "start": 1.0,
                "end": 2.0,
                "en": "Aech arrived.",
                "zh": "\u827e\u533a\u6765\u4e86\u3002",
            }
        ]
        first = audio_to_subtitle.final_qa_fingerprint(
            source,
            "qwen3:30b",
            {
                "input_fingerprint": "style",
                "terminology": [{"source": "Aech", "target": "\u827e\u533a"}],
            },
        )
        second = audio_to_subtitle.final_qa_fingerprint(
            source,
            "qwen3:30b",
            {
                "input_fingerprint": "style",
                "terminology": [{"source": "Aech", "target": "\u827e\u5947"}],
            },
        )

        self.assertNotEqual(first, second)

    def test_ocr_terminology_gate_rejects_semantic_retranslations(self) -> None:
        self.assertTrue(
            audio_to_subtitle.is_plausible_ocr_terminology_repair(
                "i-R0k",
                "\u6211\u6700\u5e1a",
                "\u6211\u6700\u5c4c",
            )
        )
        self.assertTrue(
            audio_to_subtitle.is_plausible_ocr_terminology_repair(
                "i-R0k",
                "\u6211\u6700\u5e1a",
                "i-R0k",
            )
        )
        self.assertFalse(
            audio_to_subtitle.is_plausible_ocr_terminology_repair(
                "The Distracted Globe",
                "\u6270\u52a8\u7403",
                "\u5206\u5fc3\u661f\u7403",
            )
        )
        self.assertFalse(
            audio_to_subtitle.is_plausible_ocr_terminology_repair(
                "Toshiro",
                "\u5c0f\u5200",
                "\u654f\u90ce",
            )
        )

    def test_manual_terminology_overrides_replace_reviewed_target(self) -> None:
        resolved = audio_to_subtitle.apply_terminology_overrides(
            [
                {"source": "i-R0k", "target": "\u6211\u6700\u5e1a"},
                {"source": "Aech", "target": "\u827e\u533a"},
            ],
            [{"source": "i-R0k", "target": "\u6211\u6700\u5c4c"}],
        )

        self.assertEqual(
            resolved,
            [
                {"source": "i-R0k", "target": "\u6211\u6700\u5c4c"},
                {"source": "Aech", "target": "\u827e\u533a"},
            ],
        )

    def test_manual_terminology_overrides_remove_context_sensitive_terms(self) -> None:
        resolved = audio_to_subtitle.apply_terminology_overrides(
            [
                {"source": "money", "target": "\u91d1\u5e01"},
                {"source": "Jade Key", "target": "\u7389\u94a5"},
            ],
            [{"source": "money", "remove": True}],
        )

        self.assertEqual(
            resolved,
            [{"source": "Jade Key", "target": "\u7389\u94a5"}],
        )

    def test_load_terminology_overrides_preserves_removals(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "terminology-overrides.json"
            path.write_text(
                json.dumps(
                    {
                        "entries": [
                            {"source": "money", "remove": True},
                            {"source": "i-R0k", "target": "\u6211\u6700\u5c4c"},
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            overrides = audio_to_subtitle.load_terminology_overrides(path)

        self.assertEqual(
            overrides,
            [
                {"source": "money", "remove": True},
                {"source": "i-R0k", "target": "\u6211\u6700\u5c4c"},
            ],
        )

    def test_translation_prompt_uses_movie_wide_style_before_first_batch(self) -> None:
        source = [{"id": 0, "start": 1.0, "end": 2.0, "text": "John arrived."}]
        style_guide = {
            "status": "ready",
            "register": "Concise conversational Chinese.",
            "name_policy": "Always use \u7ea6\u7ff0 for John.",
            "address_policy": "Use stable forms of address.",
            "sdh_policy": "Translate missing English-only cues.",
            "punctuation_policy": "Use concise punctuation.",
            "terminology": [{"source": "John", "target": "\u7ea6\u7ff0"}],
        }
        response = (
            '[{"index":0,"corrected_text":"John arrived.",'
            '"chinese_translation":"\u7ea6\u7ff0\u5230\u4e86\u3002","display":true,'
            '"terminology":[{"source":"John","target":"\u7ea6\u7ff0"}]}]'
        )

        with patch.object(audio_to_subtitle, "call_llm", return_value=response) as call:
            output = translate_and_correct_segments(
                source,
                llm_model="qwen3:30b",
                batch_size=1,
                context_lines=0,
                source_language="en",
                style_guide=style_guide,
            )

        prompt = call.call_args.args[0]
        self.assertIn("MOVIE-WIDE STYLE GUIDE", prompt)
        self.assertIn("Always use \u7ea6\u7ff0 for John", prompt)
        self.assertIn("John => \u7ea6\u7ff0", prompt)
        self.assertEqual(output[0]["zh"], "\u7ea6\u7ff0\u5230\u4e86\u3002")

    def test_independent_final_qa_repairs_name_without_changing_timing(self) -> None:
        source = [
            {
                "id": 0,
                "start": 1.0,
                "end": 3.0,
                "text": "John arrived.",
                "en": "John arrived.",
                "zh": "\u5f3a\u5c3c\u5230\u4e86\u3002",
                "display": True,
            }
        ]
        style_guide = {
            "status": "ready",
            "input_fingerprint": "style",
            "register": "Natural Chinese.",
            "name_policy": "John is always \u7ea6\u7ff0.",
            "address_policy": "Stable.",
            "sdh_policy": "Translate missing cues.",
            "punctuation_policy": "Concise.",
            "terminology": [{"source": "John", "target": "\u7ea6\u7ff0"}],
        }
        response = (
            '[{"index":0,"corrected_english":"John arrived.",'
            '"corrected_chinese":"\u7ea6\u7ff0\u5230\u4e86\u3002","display":true,'
            '"terminology":[{"source":"John","target":"\u7ea6\u7ff0"}]}]'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "final-qa.json"
            with patch.object(audio_to_subtitle, "call_llm", return_value=response):
                output, report = audio_to_subtitle.run_final_subtitle_qa(
                    source,
                    "qwen3:30b",
                    artifact_path,
                    style_guide=style_guide,
                )

        self.assertEqual((output[0]["start"], output[0]["end"]), (1.0, 3.0))
        self.assertEqual(output[0]["zh"], "\u7ea6\u7ff0\u5230\u4e86\u3002")
        self.assertTrue(output[0]["final_qa_changed"])
        self.assertEqual(report["status"], "pass")

    def test_independent_final_qa_cannot_add_unreviewed_terminology(self) -> None:
        source = [
            {
                "id": 0,
                "start": 1.0,
                "end": 3.0,
                "text": "John met Parzival.",
                "en": "John met Parzival.",
                "zh": "\u7ea6\u7ff0\u89c1\u5230\u4e86\u5e15\u897f\u6cd5\u5c14\u3002",
                "display": True,
                "terminology": [{"source": "John", "target": "\u7ea6\u7ff0"}],
            }
        ]
        style_guide = {
            "status": "ready",
            "input_fingerprint": "style",
            "terminology": [{"source": "John", "target": "\u7ea6\u7ff0"}],
        }
        response = json.dumps(
            [
                {
                    "index": 0,
                    "corrected_english": "John met Parzival.",
                    "corrected_chinese": "\u7ea6\u7ff0\u89c1\u5230\u4e86\u5e15\u897f\u6cd5\u5c14\u3002",
                    "display": True,
                    "terminology": [
                        {"source": "John", "target": "\u7ea6\u7ff0"},
                        {"source": "Parzival", "target": "\u5e15\u897f\u6cd5\u5c14"},
                    ],
                }
            ],
            ensure_ascii=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "final-qa.json"
            with patch.object(
                audio_to_subtitle,
                "call_llm",
                return_value=response,
            ):
                output, _report = audio_to_subtitle.run_final_subtitle_qa(
                    source,
                    "qwen3:30b",
                    artifact_path,
                    style_guide=style_guide,
                )

        self.assertEqual(
            output[0]["terminology"],
            [{"source": "John", "target": "\u7ea6\u7ff0"}],
        )

    def test_independent_final_qa_preserves_labels_cues_and_quote_style(self) -> None:
        source = [
            {
                "id": 0,
                "start": 1.0,
                "end": 3.0,
                "text": "HALLIDAY: Why?",
                "en": "HALLIDAY: Why?",
                "zh": "\u54c8\u52d2\u4ee3\uff1a\u4e3a\u4ec0\u4e48\uff1f",
                "source_language": "en",
                "display": True,
            },
            {
                "id": 1,
                "start": 3.0,
                "end": 5.0,
                "text": "Fast. (LAUGHS)",
                "en": "Fast. (LAUGHS)",
                "zh": "\u5feb\u70b9\u3002\uff08\u7b11\uff09",
                "source_language": "en",
                "display": True,
            },
            {
                "id": 2,
                "start": 5.0,
                "end": 7.0,
                "text": "Fine.",
                "en": "Fine.",
                "zh": "\u884c\u5427\u3002",
                "source_language": "en",
                "display": True,
            },
        ]
        response = json.dumps(
            [
                {
                    "index": 0,
                    "corrected_english": "Why?",
                    "corrected_chinese": "\u4e3a\u4ec0\u4e48\uff1f",
                    "display": True,
                    "terminology": [],
                },
                {
                    "index": 1,
                    "corrected_english": "Fast.",
                    "corrected_chinese": "\u5feb\u70b9\u3002",
                    "display": True,
                    "terminology": [],
                },
                {
                    "index": 2,
                    "corrected_english": "Fine.",
                    "corrected_chinese": "\u201c\u884c\u5427\u3002\u201d",
                    "display": True,
                    "terminology": [],
                },
            ],
            ensure_ascii=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "final-qa.json"
            with patch.object(audio_to_subtitle, "call_llm", return_value=response):
                output, report = audio_to_subtitle.run_final_subtitle_qa(
                    source,
                    "qwen3:30b",
                    artifact_path,
                )

        self.assertEqual(output[0]["en"], "HALLIDAY: Why?")
        self.assertEqual(output[0]["zh"], "\u54c8\u52d2\u4ee3\uff1a\u4e3a\u4ec0\u4e48\uff1f")
        self.assertEqual(output[1]["en"], "Fast. (LAUGHS)")
        self.assertEqual(output[1]["zh"], "\u5feb\u70b9\u3002\uff08\u7b11\uff09")
        self.assertEqual(output[2]["zh"], "\u884c\u5427\u3002")
        self.assertEqual(report["changed_count"], 0)

    def test_independent_final_qa_ignores_spurious_source_for_chinese_only(self) -> None:
        source = [
            {
                "id": 0,
                "start": 1.0,
                "end": 3.0,
                "text": "\u81ea\u7531\uff0c\u6211\u4eec\u4fe1\u4ef0\u4e0a\u5e1d",
                "en": "",
                "zh": "\u81ea\u7531\uff0c\u6211\u4eec\u4fe1\u4ef0\u4e0a\u5e1d",
                "source_language": "zh",
                "display": True,
            }
        ]
        response = (
            '[{"index":0,"corrected_english":"Freedom, in God we trust.",'
            '"corrected_chinese":"\u81ea\u7531\uff0c\u6211\u4eec\u4fe1\u4ef0\u4e0a\u5e1d\u3002",'
            '"display":true,"terminology":[]}]'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "final-qa.json"
            with patch.object(audio_to_subtitle, "call_llm", return_value=response):
                output, report = audio_to_subtitle.run_final_subtitle_qa(
                    source,
                    "qwen3:30b",
                    artifact_path,
                )

        self.assertEqual(output[0]["en"], "")
        self.assertEqual(output[0]["zh"], "\u81ea\u7531\uff0c\u6211\u4eec\u4fe1\u4ef0\u4e0a\u5e1d\u3002")
        self.assertEqual(report["status"], "pass")

    def test_independent_final_qa_rejects_translation_of_japanese_source(self) -> None:
        source = [
            {
                "id": 0,
                "start": 1.0,
                "end": 3.0,
                "text": "\u3053\u3093\u306b\u3061\u306f",
                "en": "\u3053\u3093\u306b\u3061\u306f",
                "zh": "\u4f60\u597d",
                "source_language": "ja",
                "display": True,
            }
        ]
        response = (
            '[{"index":0,"corrected_english":"Hello",'
            '"corrected_chinese":"\u4f60\u597d","display":true,"terminology":[]}]'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "final-qa.json"
            with patch.object(
                audio_to_subtitle,
                "call_llm",
                return_value=response,
            ) as call:
                output, report = audio_to_subtitle.run_final_subtitle_qa(
                    source,
                    "qwen3:30b",
                    artifact_path,
                )

        self.assertIn("SOURCE_LANGUAGE=ja", call.call_args.args[0])
        self.assertEqual(output[0]["en"], "\u3053\u3093\u306b\u3061\u306f")
        self.assertEqual(report["status"], "review")
        self.assertEqual(
            report["rejected_items"][0]["reason"],
            "source_language_changed",
        )

    def test_independent_final_qa_preserves_manual_review(self) -> None:
        source = [
            {
                "id": 0,
                "start": 1.1,
                "end": 3.2,
                "text": "John arrived.",
                "en": "John arrived.",
                "zh": "\u7ea6\u7ff0\u5230\u4e86\u3002",
                "display": True,
                "manual_reviewed_at": "2026-07-30T10:00:00+00:00",
            }
        ]
        response = (
            '[{"index":0,"corrected_english":"Jack arrived.",'
            '"corrected_chinese":"\u6770\u514b\u5230\u4e86\u3002","display":true,'
            '"terminology":[]}]'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "final-qa.json"
            with patch.object(audio_to_subtitle, "call_llm", return_value=response):
                output, report = audio_to_subtitle.run_final_subtitle_qa(
                    source,
                    "qwen3:30b",
                    artifact_path,
                )

        self.assertEqual(output[0]["en"], "John arrived.")
        self.assertEqual(output[0]["zh"], "\u7ea6\u7ff0\u5230\u4e86\u3002")
        self.assertTrue(output[0]["final_qa_manual_preserved"])
        self.assertEqual(report["manual_preserved_count"], 1)

    def test_relevant_early_terminology_survives_prompt_limit(self) -> None:
        glossary = {"john": {"source": "John", "target": "\u7ea6\u7ff0"}}
        for index in range(300):
            glossary[f"term-{index}"] = {
                "source": f"Term {index}",
                "target": f"\u672f\u8bed{index}",
            }

        prompt_glossary = audio_to_subtitle.format_terminology_glossary(
            glossary,
            "John returns after a long absence.",
        )

        self.assertIn("John => \u7ea6\u7ff0", prompt_glossary)
        self.assertIn("Term 299 => \u672f\u8bed299", prompt_glossary)

    def test_translation_retries_when_known_name_mapping_is_inconsistent(self) -> None:
        source = [
            {"id": 0, "start": 1.0, "end": 2.0, "text": "John arrived."},
            {"id": 1, "start": 2.0, "end": 3.0, "text": "John sat down."},
        ]
        responses = [
            (
                '[{"index":0,"corrected_text":"John arrived.",'
                '"chinese_translation":"\u7ea6\u7ff0\u5230\u4e86\u3002","display":true,'
                '"terminology":[{"source":"John","target":"\u7ea6\u7ff0"}]}]'
            ),
            (
                '[{"index":0,"corrected_text":"John sat down.",'
                '"chinese_translation":"\u5f3a\u5c3c\u5750\u4e0b\u4e86\u3002","display":true,'
                '"terminology":[{"source":"John","target":"\u5f3a\u5c3c"}]}]'
            ),
        ]
        repaired = {
            "corrected_text": "John sat down.",
            "chinese_translation": "\u7ea6\u7ff0\u5750\u4e0b\u4e86\u3002",
            "display": True,
            "terminology": [{"source": "John", "target": "\u7ea6\u7ff0"}],
        }

        with (
            patch.object(audio_to_subtitle, "call_llm", side_effect=responses),
            patch.object(
                audio_to_subtitle,
                "retry_single_segment_llm",
                return_value=repaired,
            ) as retry,
        ):
            output = translate_and_correct_segments(
                source,
                llm_model="qwen3:30b",
                batch_size=1,
                context_lines=0,
                source_language="en",
            )

        retry.assert_called_once()
        self.assertEqual(output[1]["zh"], "\u7ea6\u7ff0\u5750\u4e0b\u4e86\u3002")

    def test_main_reuses_source_cache_without_audio_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "video.mp4"
            video.write_bytes(b"placeholder")
            output_root = root / "out"
            out_dir = output_root / "Series" / "Episode"
            out_dir.mkdir(parents=True)
            source_cache = out_dir / "Episode.segments.source.json"
            source_cache.write_text(
                json.dumps(
                    [
                        {
                            "id": 0,
                            "start": 1.0,
                            "end": 2.0,
                            "text": "cached line",
                            "source_language": "en",
                        }
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            argv = [
                "audio_to_subtitle.py",
                "--video",
                str(video),
                "--source",
                "audio",
                "--output-root",
                str(output_root),
                "--series-name",
                "Series",
                "--movie-name",
                "Episode",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(audio_to_subtitle, "extract_audio", side_effect=AssertionError("extract_audio called")),
                patch.object(audio_to_subtitle, "transcribe_audio", side_effect=AssertionError("transcribe_audio called")),
                patch.object(audio_to_subtitle, "translate_and_correct_segments", side_effect=lambda segments, **_: segments),
            ):
                audio_to_subtitle.main()

            self.assertTrue((out_dir / "Episode.bilingual.ass").exists())
            upgraded_cache = json.loads(source_cache.read_text(encoding="utf-8"))
            self.assertTrue(upgraded_cache[0]["source_request_fingerprint"])
            source_manifest = json.loads(
                (out_dir / "Episode.source-manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            processing_plan = json.loads(
                (out_dir / "Episode.processing-plan.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(source_manifest["assets"])
            self.assertTrue(processing_plan["execution"]["source_cache_reused"])
            self.assertEqual(processing_plan["execution"]["local_gpu"], [])

    def test_main_reuses_audio_cache_from_previous_sync_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "video.mp4"
            video.write_bytes(b"placeholder")
            output_root = root / "out"
            out_dir = output_root / "Series" / "Episode"
            out_dir.mkdir(parents=True)
            previous_fingerprint = build_source_request_fingerprint(
                video,
                video,
                SimpleNamespace(source="audio"),
                sync_policy_version=5,
            )
            source_cache = out_dir / "Episode.segments.source.json"
            source_cache.write_text(
                json.dumps(
                    [
                        {
                            "id": 0,
                            "start": 1.0,
                            "end": 2.0,
                            "text": "cached line",
                            "source_language": "en",
                            "timing_origin": "audio_asr",
                            "source_cache_version": 3,
                            "source_request_fingerprint": previous_fingerprint,
                        }
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            argv = [
                "audio_to_subtitle.py",
                "--video",
                str(video),
                "--source",
                "audio",
                "--output-root",
                str(output_root),
                "--series-name",
                "Series",
                "--movie-name",
                "Episode",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(audio_to_subtitle, "extract_audio", side_effect=AssertionError("extract_audio called")),
                patch.object(audio_to_subtitle, "transcribe_audio", side_effect=AssertionError("transcribe_audio called")),
                patch.object(audio_to_subtitle, "translate_and_correct_segments", side_effect=lambda segments, **_: segments),
            ):
                audio_to_subtitle.main()

            upgraded_cache = json.loads(source_cache.read_text(encoding="utf-8"))

        self.assertNotEqual(
            upgraded_cache[0]["source_request_fingerprint"],
            previous_fingerprint,
        )
        self.assertEqual(
            upgraded_cache[0]["source_cache_version"],
            audio_to_subtitle.SOURCE_CACHE_VERSION,
        )

    def test_source_cache_fingerprint_changes_with_selected_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "movie.mkv"
            first = root / "movie.en.srt"
            second = root / "movie.zh.srt"
            video.write_bytes(b"video")
            first.write_text("first", encoding="utf-8")
            second.write_text("second", encoding="utf-8")
            args = SimpleNamespace(
                source="sidecar",
                merge_existing_subtitles="no",
                subtitle_stream=None,
                chinese_subtitle_stream=None,
                english_subtitle_stream=None,
                audio_stream=None,
                source_language="en",
                asr_language="source",
                subtitle_ocr_lang="auto",
                device="gpu:0",
                ocr_scale=2.0,
                crop_pad=8,
                fast_ocr=False,
            )

            first_fingerprint = build_source_request_fingerprint(
                video,
                video,
                args,
                sidecar_path=first,
            )
            second_fingerprint = build_source_request_fingerprint(
                video,
                video,
                args,
                sidecar_path=second,
            )
            cached = [
                {
                    "id": 0,
                    "start": 1.0,
                    "end": 2.0,
                    "text": "cached",
                    "source_request_fingerprint": first_fingerprint,
                }
            ]

            self.assertNotEqual(first_fingerprint, second_fingerprint)
            self.assertFalse(source_cache_matches_request(cached, second_fingerprint))

    def test_source_request_accepts_equivalent_auto_and_explicit_sync_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "subtitle-sync.report.json"
            report_path.write_text(
                json.dumps({"audio_stream": 1}),
                encoding="utf-8",
            )

            explicit = audio_to_subtitle.source_request_audio_stream_variants(
                1,
                1,
                "auto",
                report_path,
            )
            automatic = audio_to_subtitle.source_request_audio_stream_variants(
                None,
                1,
                "auto",
                report_path,
            )
            changed = audio_to_subtitle.source_request_audio_stream_variants(
                2,
                2,
                "auto",
                report_path,
            )

        self.assertEqual(explicit, {None, 1})
        self.assertEqual(automatic, {None, 1})
        self.assertEqual(changed, {2})

    def test_auto_source_does_not_select_another_episodes_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "Show.S01E01.mkv"
            video.write_bytes(b"video")
            (root / "Show.S01E02.en.srt").write_text("English", encoding="utf-8")
            (root / "Show.S01E02.zh.srt").write_text("\u4e2d\u6587", encoding="utf-8")

            selected = find_sidecar_subtitle(video, video)
            frontend_items = subtitle_frontend.list_sidecar_subtitles(video, video)

        self.assertIsNone(selected)
        self.assertTrue(frontend_items)
        self.assertFalse(any(item["likely_match"] for item in frontend_items))

    def test_checkpoint_rejects_changed_existing_chinese_track(self) -> None:
        source = [
            {
                "id": 0,
                "start": 1.0,
                "end": 2.0,
                "text": "Hello",
                "en": "Hello",
                "zh": "\u65b0\u4e2d\u6587",
                "source_request_fingerprint": "new-request",
            }
        ]
        cached = [
            {
                **source[0],
                "zh": "\u65e7\u4e2d\u6587",
                "source_request_fingerprint": "old-request",
                "display": True,
                "processing_mode": "proofread_existing_chinese",
            }
        ]

        self.assertFalse(
            checkpoint_matches_segments(
                cached,
                source,
                expected_mode="proofread_existing_chinese",
            )
        )

    def test_incomplete_llm_batch_preserves_last_good_checkpoint(self) -> None:
        source = [
            {"id": 0, "start": 1.0, "end": 2.0, "text": "First line"},
            {"id": 1, "start": 2.0, "end": 3.0, "text": "Second line"},
        ]
        responses = [
            '[{"index":0,"corrected_text":"First line","chinese_translation":"\u7b2c\u4e00\u53e5","display":true}]',
            "[]",
            "[]",
            "[]",
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "checkpoint.json"
            with (
                patch.object(audio_to_subtitle, "call_llm", side_effect=responses),
                patch("time.sleep", return_value=None),
                self.assertRaises(RuntimeError),
            ):
                translate_and_correct_segments(
                    source,
                    llm_model="qwen3:30b",
                    batch_size=1,
                    context_lines=0,
                    source_language="en",
                    checkpoint_path=checkpoint,
                )

            saved = json.loads(checkpoint.read_text(encoding="utf-8"))

        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["zh"], "\u7b2c\u4e00\u53e5")

    def test_complete_partial_batch_checkpoint_is_not_reprocessed(self) -> None:
        source = [{"id": 0, "start": 1.0, "end": 2.0, "text": "Complete"}]
        cached = [
            {
                **source[0],
                "en": "Complete",
                "zh": "\u5df2\u5b8c\u6210",
                "display": True,
                "processing_mode": "translate",
                "processing_policy_version": audio_to_subtitle.PROCESSING_POLICY_VERSION,
                "terminology_policy_version": audio_to_subtitle.TERMINOLOGY_POLICY_VERSION,
                "llm_model": "qwen3:30b",
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "checkpoint.json"
            checkpoint.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")
            with patch.object(
                audio_to_subtitle,
                "call_llm",
                side_effect=AssertionError("complete checkpoint was reprocessed"),
            ):
                output = translate_and_correct_segments(
                    source,
                    llm_model="qwen3:30b",
                    batch_size=5,
                    context_lines=0,
                    source_language="en",
                    checkpoint_path=checkpoint,
                )

        self.assertEqual(output[0]["zh"], "\u5df2\u5b8c\u6210")

    def test_main_uses_output_scoped_temp_audio_and_preserves_sibling_wav(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "video.mp4"
            video.write_bytes(b"placeholder")
            sibling_wav = video.with_suffix(".wav")
            sibling_wav.write_bytes(b"user audio")
            output_root = root / "out"
            captured = {}

            def extract(_video, temp_audio, **_):
                captured["path"] = temp_audio
                temp_audio.write_bytes(b"pipeline temp")

            argv = [
                "audio_to_subtitle.py",
                "--video",
                str(video),
                "--source",
                "audio",
                "--output-root",
                str(output_root),
                "--series-name",
                "Series",
                "--movie-name",
                "Episode",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(audio_to_subtitle, "extract_audio", side_effect=extract),
                patch.object(
                    audio_to_subtitle,
                    "transcribe_audio",
                    return_value=[{"id": 0, "start": 1.0, "end": 2.0, "text": "line", "source_language": "en"}],
                ),
                patch.object(audio_to_subtitle, "translate_and_correct_segments", side_effect=lambda segments, **_: segments),
            ):
                audio_to_subtitle.main()

            self.assertEqual(sibling_wav.read_bytes(), b"user audio")
            self.assertEqual(captured["path"].parent, output_root / "Series" / "Episode")
            self.assertFalse(captured["path"].exists())

    def test_restart_cleanup_preserves_video_sibling_wav(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "movie.mp4"
            video.write_bytes(b"video")
            sibling_wav = video.with_suffix(".wav")
            sibling_wav.write_bytes(b"user audio")
            out_dir = root / "out"
            out_dir.mkdir()
            pipeline_temp = out_dir / ".Episode.subtitle-audio.123.wav"
            pipeline_temp.write_bytes(b"temp")
            source_cache = out_dir / "Episode.segments.source.json"
            source_cache.write_text("[]", encoding="utf-8")
            embedded_dir = out_dir / "embedded"
            embedded_dir.mkdir()
            (embedded_dir / "stream.ass").write_text("cached", encoding="utf-8")

            subtitle_frontend.remove_resume_files(out_dir, "Episode")

            self.assertEqual(sibling_wav.read_bytes(), b"user audio")
            self.assertFalse(pipeline_temp.exists())
            self.assertFalse(source_cache.exists())
            self.assertFalse(embedded_dir.exists())

    def test_reprocess_cleanup_preserves_recognition_caches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            source_cache = out_dir / "Episode.segments.source.json"
            source_cache.write_text('[{"text":"recognized"}]', encoding="utf-8")
            checkpoint = out_dir / "Episode.segments.checkpoint.json"
            checkpoint.write_text('[{"zh":"old"}]', encoding="utf-8")
            output_paths = [
                out_dir / "Episode.en.ass",
                out_dir / "Episode.zh.ass",
                out_dir / "Episode.bilingual.ass",
            ]
            for path in output_paths:
                path.write_text("old output", encoding="utf-8")
            embedded_dir = out_dir / "embedded"
            embedded_dir.mkdir()
            embedded_cache = embedded_dir / "stream.ass"
            embedded_cache.write_text("cached OCR", encoding="utf-8")
            pipeline_temp = out_dir / ".Episode.subtitle-audio.123.wav"
            pipeline_temp.write_bytes(b"cached audio")

            subtitle_frontend.remove_translation_outputs(out_dir, "Episode")

            self.assertTrue(source_cache.exists())
            self.assertTrue(embedded_cache.exists())
            self.assertTrue(pipeline_temp.exists())
            self.assertFalse(checkpoint.exists())
            self.assertTrue(all(not path.exists() for path in output_paths))

    def test_frontend_run_modes_keep_legacy_restart_compatibility(self) -> None:
        self.assertEqual(subtitle_frontend.resolve_run_mode({"run_mode": "resume"}), "resume")
        self.assertEqual(subtitle_frontend.resolve_run_mode({"run_mode": "reprocess"}), "reprocess")
        self.assertEqual(subtitle_frontend.resolve_run_mode({"run_mode": "restart"}), "restart")
        self.assertEqual(subtitle_frontend.resolve_run_mode({"restart": True}), "restart")
        self.assertEqual(subtitle_frontend.resolve_run_mode({}), "resume")

    def test_frontend_reprocess_routes_to_translation_only_cleanup(self) -> None:
        class FakeProcess:
            pid = 54321

            @staticmethod
            def poll():
                return None

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "movie.mp4"
            video.write_bytes(b"video")
            log_path = root / "run.log"
            err_path = root / "run.err.log"
            payload = {
                "path": str(video),
                "output_root": str(root / "out"),
                "series_name": "Series",
                "movie_name": "Episode",
                "source": "audio",
                "run_mode": "reprocess",
            }
            subtitle_frontend.RUNS.clear()
            try:
                with (
                    patch.object(subtitle_frontend, "frontend_log_paths", return_value=(log_path, err_path)),
                    patch.object(subtitle_frontend, "remove_translation_outputs") as translation_cleanup,
                    patch.object(subtitle_frontend, "remove_resume_files") as full_cleanup,
                    patch.object(subtitle_frontend.subprocess, "Popen", return_value=FakeProcess()),
                ):
                    result = subtitle_frontend.start_processing(payload)
            finally:
                subtitle_frontend.RUNS.clear()

        translation_cleanup.assert_called_once()
        full_cleanup.assert_not_called()
        self.assertEqual(result["run_mode"], "reprocess")

    def test_frontend_remote_run_uses_its_own_tunnel_port(self) -> None:
        class FakeProcess:
            pid = 54322

            @staticmethod
            def poll():
                return None

        class FakeTunnel:
            @staticmethod
            def status():
                return {"connected": True, "local_port": 32123}

        captured_env = {}

        def popen(_args, **kwargs):
            captured_env.update(kwargs["env"])
            return FakeProcess()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "movie.mp4"
            video.write_bytes(b"video")
            log_path = root / "run.log"
            err_path = root / "run.err.log"
            payload = {
                "path": str(video),
                "output_root": str(root / "out"),
                "series_name": "Series",
                "movie_name": "Episode",
                "source": "audio",
                "llm_model": "remote:AI Server:qwen3:30b",
            }
            subtitle_frontend.RUNS.clear()
            try:
                with (
                    patch.object(subtitle_frontend, "tunnel_manager", FakeTunnel()),
                    patch.object(subtitle_frontend, "frontend_log_paths", return_value=(log_path, err_path)),
                    patch.object(subtitle_frontend.subprocess, "Popen", side_effect=popen),
                ):
                    subtitle_frontend.start_processing(payload)
            finally:
                subtitle_frontend.RUNS.clear()

        self.assertEqual(captured_env["SUBTITLE_REMOTE_OLLAMA_URL"], "http://127.0.0.1:32123")

    def test_frontend_rejects_remote_run_without_its_own_tunnel(self) -> None:
        class DisconnectedTunnel:
            @staticmethod
            def status():
                return {"connected": False, "local_port": None}

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "movie.mp4"
            video.write_bytes(b"video")
            payload = {
                "path": str(video),
                "output_root": str(root / "out"),
                "series_name": "Series",
                "movie_name": "Episode",
                "source": "audio",
                "llm_model": "remote:AI Server:qwen3:30b",
            }
            with (
                patch.object(subtitle_frontend, "tunnel_manager", DisconnectedTunnel()),
                patch.object(
                    subtitle_frontend.subprocess,
                    "Popen",
                    side_effect=AssertionError("remote process was launched without a tunnel"),
                ),
                self.assertRaisesRegex(RuntimeError, "尚未连接远程服务器"),
            ):
                subtitle_frontend.start_processing(payload)

    def test_remote_llm_uses_frontend_owned_tunnel_url(self) -> None:
        captured = {}

        def post_chat(base_url, payload):
            captured["base_url"] = base_url
            captured["payload"] = payload
            return {"message": {"content": "ok"}}

        with (
            patch.dict(
                audio_to_subtitle.os.environ,
                {"SUBTITLE_REMOTE_OLLAMA_URL": "http://127.0.0.1:32123"},
                clear=False,
            ),
            patch.object(audio_to_subtitle, "post_ollama_chat", side_effect=post_chat),
        ):
            result = audio_to_subtitle.call_llm(
                "prompt",
                model="remote:AI Server:qwen3:30b",
            )

        self.assertEqual(result, "ok")
        self.assertEqual(captured["base_url"], "http://127.0.0.1:32123")
        self.assertEqual(captured["payload"]["model"], "qwen3:30b")

    def test_llm_role_applies_explicit_generation_profile_and_json_schema(self) -> None:
        captured = {}

        def post_chat(base_url, payload):
            captured["base_url"] = base_url
            captured["payload"] = payload
            return {"message": {"content": "[]"}}

        with patch.object(audio_to_subtitle, "post_ollama_chat", side_effect=post_chat):
            audio_to_subtitle.call_llm(
                "proofread",
                model="qwen3:30b",
                role="proofread",
                response_schema=audio_to_subtitle.indexed_subtitle_schema(
                    2,
                    "proofread",
                ),
            )

        request = captured["payload"]
        self.assertEqual(captured["base_url"], "http://localhost:11434")
        self.assertEqual(request["options"]["temperature"], 0.1)
        self.assertEqual(request["options"]["top_p"], 0.8)
        self.assertEqual(request["options"]["seed"], 43)
        self.assertEqual(request["options"]["top_k"], 20)
        self.assertEqual(request["options"]["num_ctx"], 65536)
        self.assertFalse(request["think"])
        self.assertEqual(request["format"]["maxItems"], 2)

    def test_llm_seed_offset_makes_retries_distinct(self) -> None:
        captured = {}

        def post_chat(_base_url, payload):
            captured["payload"] = payload
            return {"message": {"content": "[]"}}

        with patch.object(audio_to_subtitle, "post_ollama_chat", side_effect=post_chat):
            audio_to_subtitle.call_llm(
                "proofread",
                model="qwen3:30b",
                role="proofread",
                seed_offset=2,
            )

        self.assertEqual(captured["payload"]["options"]["seed"], 45)

    def test_final_qa_normalizes_local_index_and_protocol_defaults(self) -> None:
        batch = [
            {"id": 3, "display": True},
            {"id": 4, "display": False},
        ]
        response = [
            {
                "local_index": 0,
                "start_time": 1.0,
                "corrected_english": "Hello",
                "corrected_chinese": "\u4f60\u597d",
            },
            {
                "local_index": 1,
                "end_time": 3.0,
                "corrected_english": "World",
                "corrected_chinese": "\u4e16\u754c",
            },
        ]

        normalized = audio_to_subtitle.normalize_final_qa_batch_response(
            response,
            batch,
        )
        lookup = audio_to_subtitle.validate_indexed_batch_response(
            normalized,
            2,
            "proofread",
        )

        self.assertEqual(lookup[0]["index"], 0)
        self.assertTrue(lookup[0]["display"])
        self.assertEqual(lookup[0]["terminology"], [])
        self.assertEqual(lookup[1]["index"], 1)
        self.assertFalse(lookup[1]["display"])
        self.assertEqual(lookup[1]["terminology"], [])

    def test_final_qa_normalizes_controlled_text_field_aliases(self) -> None:
        batch = [{"id": 3, "display": True}]
        response = [
            {
                "local_index": 0,
                "en": "Hello",
                "zh": "\u4f60\u597d",
            }
        ]

        normalized = audio_to_subtitle.normalize_final_qa_batch_response(
            response,
            batch,
        )
        lookup = audio_to_subtitle.validate_indexed_batch_response(
            normalized,
            1,
            "proofread",
        )

        self.assertEqual(lookup[0]["corrected_english"], "Hello")
        self.assertEqual(lookup[0]["corrected_chinese"], "\u4f60\u597d")

    def test_final_qa_reuses_cache_for_manual_checkpoint_overrides(self) -> None:
        source = [
            {
                "id": 0,
                "start": 1.0,
                "end": 2.0,
                "text": "Hello.",
                "en": "Hello.",
                "zh": "\u4f60\u597d\u3002",
                "display": True,
                "source_language": "en",
            },
            {
                "id": 1,
                "start": 2.1,
                "end": 3.0,
                "text": "World.",
                "en": "World.",
                "zh": "\u4e16\u754c\u3002",
                "display": True,
                "source_language": "en",
            },
        ]
        response = (
            '[{"index":0,"corrected_english":"Hello.",'
            '"corrected_chinese":"\u4f60\u597d\u3002","display":true,"terminology":[]},'
            '{"index":1,"corrected_english":"World.",'
            '"corrected_chinese":"\u4e16\u754c\u3002","display":true,"terminology":[]}]'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "final-qa.json"
            with patch.object(audio_to_subtitle, "call_llm", return_value=response):
                first, _ = audio_to_subtitle.run_final_subtitle_qa(
                    source,
                    "qwen3:30b",
                    artifact_path,
                    batch_size=2,
                )
            manually_reviewed = [dict(item) for item in first]
            manually_reviewed[0].update(
                {
                    "en": "Hello!",
                    "zh": "\u4f60\u597d\uff01",
                    "manual_reviewed_at": "2026-07-31T10:00:00+00:00",
                }
            )
            with patch.object(audio_to_subtitle, "call_llm") as call:
                second, report = audio_to_subtitle.run_final_subtitle_qa(
                    manually_reviewed,
                    "qwen3:30b",
                    artifact_path,
                    batch_size=2,
                )

        call.assert_not_called()
        self.assertEqual(second[0]["en"], "Hello!")
        self.assertEqual(second[0]["zh"], "\u4f60\u597d\uff01")
        self.assertTrue(second[0]["final_qa_manual_preserved"])
        self.assertEqual(report["manual_override_count"], 1)
        self.assertEqual(report["output_fingerprint"], report["input_fingerprint"])

    def test_structured_json_repairs_only_unquoted_schema_strings(self) -> None:
        response_schema = audio_to_subtitle.indexed_subtitle_schema(2, "proofread")
        malformed = """[
  {
    "index": 0,
    "corrected_english": "(GRUNTING)",
    "corrected_chinese": (低吼声),
    "display": true,
    "terminology": []
  },
  {
    "index": 1,
    "corrected_english": "Pizza, guys.",
    "corrected_chinese": 各位，披萨来了。
    "display": true,
    "terminology": []
  }
]"""

        parsed = audio_to_subtitle.parse_json_response(
            malformed,
            response_schema=response_schema,
        )
        validated = audio_to_subtitle.validate_indexed_batch_response(
            parsed,
            2,
            "proofread",
        )

        self.assertEqual(validated[0]["corrected_chinese"], "(低吼声)")
        self.assertEqual(validated[1]["corrected_chinese"], "各位，披萨来了。")
        with self.assertRaises(json.JSONDecodeError):
            audio_to_subtitle.parse_json_response(malformed)

    def test_structured_json_repairs_concatenated_objects_as_array(self) -> None:
        response_schema = audio_to_subtitle.indexed_terminology_schema(2)
        malformed = """{
  "index": 0,
  "target": "\u827e\u5947",
  "confidence": 1.0,
  "decision": "ocr_repair"
}
{
  "index": 1,
  "target": "\u827e\u6d1b\u514b",
  "confidence": 1.0,
  "decision": "ocr_repair"
}"""

        parsed = audio_to_subtitle.parse_json_response(
            malformed,
            response_schema=response_schema,
        )
        validated = audio_to_subtitle.validate_terminology_review_response(
            parsed,
            2,
        )

        self.assertEqual(validated[0]["target"], "\u827e\u5947")
        self.assertEqual(validated[1]["target"], "\u827e\u6d1b\u514b")

    def test_structured_json_does_not_repair_non_string_schema_values(self) -> None:
        response_schema = audio_to_subtitle.indexed_subtitle_schema(1, "proofread")
        malformed = """[
  {
    "index": 0,
    "corrected_english": "",
    "corrected_chinese": 必胜客，
    "display": maybe,
    "terminology": []
  }
]"""

        with self.assertRaises(json.JSONDecodeError):
            audio_to_subtitle.parse_json_response(
                malformed,
                response_schema=response_schema,
            )

    def test_transcribe_audio_unloads_whisper_from_local_gpu(self) -> None:
        unloaded = []

        class FakeRuntimeModel:
            @staticmethod
            def unload_model():
                unloaded.append(True)

        class FakeWhisperModel:
            def __init__(self, *_args, **_kwargs):
                self.model = FakeRuntimeModel()

            @staticmethod
            def transcribe(*_args, **_kwargs):
                segment = SimpleNamespace(
                    start=1.0,
                    end=2.0,
                    text="Recognized line",
                    words=[],
                )
                info = SimpleNamespace(language="en", language_probability=0.99)
                return iter([segment]), info

        with patch.dict(
            sys.modules,
            {"faster_whisper": SimpleNamespace(WhisperModel=FakeWhisperModel)},
        ):
            output = audio_to_subtitle.transcribe_audio(Path("audio.wav"), language="en")

        self.assertEqual(output[0]["text"], "Recognized line")
        self.assertEqual(unloaded, [True])

    def test_transcribe_audio_persists_whisper_confidence_metadata(self) -> None:
        captured_kwargs = {}

        class FakeWhisperModel:
            model = None

            def __init__(self, *_args, **_kwargs):
                pass

            @staticmethod
            def transcribe(*_args, **kwargs):
                captured_kwargs.update(kwargs)
                segment = SimpleNamespace(
                    start=1.0,
                    end=3.0,
                    text="Recognized line",
                    avg_logprob=-0.42,
                    compression_ratio=1.2,
                    no_speech_prob=0.08,
                    temperature=0.0,
                    words=[
                        SimpleNamespace(
                            start=1.0,
                            end=2.0,
                            word="Recognized",
                            probability=0.91,
                        ),
                        SimpleNamespace(
                            start=2.0,
                            end=3.0,
                            word="line",
                            probability=0.82,
                        ),
                    ],
                )
                info = SimpleNamespace(language="en", language_probability=0.97)
                return iter([segment]), info

        with patch.dict(
            sys.modules,
            {"faster_whisper": SimpleNamespace(WhisperModel=FakeWhisperModel)},
        ):
            output = audio_to_subtitle.transcribe_audio(Path("audio.wav"), language="auto")

        self.assertTrue(captured_kwargs["multilingual"])
        self.assertEqual(captured_kwargs["hallucination_silence_threshold"], 2.0)
        self.assertEqual(output[0]["asr_avg_logprob"], -0.42)
        self.assertEqual(output[0]["asr_language_probability"], 0.97)
        self.assertEqual(output[0]["words"][0]["probability"], 0.91)
        self.assertAlmostEqual(output[0]["asr_word_confidence"], 0.865)

    def test_frontend_rejects_duplicate_run_for_same_output(self) -> None:
        class FakeProcess:
            pid = 43210

            @staticmethod
            def poll():
                return None

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "movie.mp4"
            video.write_bytes(b"video")
            payload = {
                "path": str(video),
                "output_root": str(root / "out"),
                "series_name": "Series",
                "movie_name": "Episode",
                "source": "audio",
            }
            log_path = root / "run.log"
            err_path = root / "run.err.log"
            subtitle_frontend.RUNS.clear()
            try:
                with (
                    patch.object(subtitle_frontend, "frontend_log_paths", return_value=(log_path, err_path)),
                    patch.object(subtitle_frontend.subprocess, "Popen", return_value=FakeProcess()),
                ):
                    subtitle_frontend.start_processing(payload)
                    with self.assertRaisesRegex(RuntimeError, "任务已在运行"):
                        subtitle_frontend.start_processing(payload)
            finally:
                subtitle_frontend.RUNS.clear()

    def test_frontend_recovers_active_run_after_registry_reset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "out"
            out_dir = output_root / "Series" / "Episode"
            state_path = subtitle_frontend.run_state_path(out_dir, "Episode")
            subtitle_frontend.write_run_state(
                state_path,
                {
                    "version": 1,
                    "pid": 4321,
                    "series_name": "Series",
                    "movie_name": "Episode",
                    "state_path": str(state_path),
                },
            )
            subtitle_frontend.RUNS.clear()
            try:
                with patch.object(subtitle_frontend, "process_matches_run", return_value=True):
                    status = subtitle_frontend.run_status(
                        {
                            "pid": 0,
                            "output_root": str(output_root),
                            "series_name": "Series",
                            "movie_name": "Episode",
                        }
                    )
            finally:
                subtitle_frontend.RUNS.clear()

        self.assertTrue(status["running"])
        self.assertEqual(status["pid"], 4321)

    def test_frontend_rejects_persisted_duplicate_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "video.mp4"
            video.write_bytes(b"placeholder")
            output_root = root / "out"
            out_dir = output_root / "Series" / "Episode"
            state_path = subtitle_frontend.run_state_path(out_dir, "Episode")
            subtitle_frontend.write_run_state(
                state_path,
                {
                    "version": 1,
                    "pid": 9876,
                    "series_name": "Series",
                    "movie_name": "Episode",
                    "state_path": str(state_path),
                },
            )
            payload = {
                "path": str(video),
                "output_root": str(output_root),
                "series_name": "Series",
                "movie_name": "Episode",
                "source": "audio",
            }
            subtitle_frontend.RUNS.clear()
            try:
                with (
                    patch.object(subtitle_frontend, "process_matches_run", return_value=True),
                    patch.object(
                        subtitle_frontend.subprocess,
                        "Popen",
                        side_effect=AssertionError("duplicate process was launched"),
                    ),
                    self.assertRaisesRegex(RuntimeError, "9876"),
                ):
                    subtitle_frontend.start_processing(payload)
            finally:
                subtitle_frontend.RUNS.clear()

    def test_main_uses_selected_english_embedded_stream_as_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "movie.mkv"
            video.write_bytes(b"video")
            output_root = root / "out"
            captured = {}

            def load_embedded(_video, stream_index, **_):
                captured["stream_index"] = stream_index
                return [{"id": 0, "start": 1.0, "end": 2.0, "text": "English line", "source_language": "en"}]

            argv = [
                "audio_to_subtitle.py",
                "--video",
                str(video),
                "--source",
                "embedded",
                "--subtitle-sync",
                "off",
                "--english-subtitle-stream",
                "7",
                "--output-root",
                str(output_root),
                "--series-name",
                "Series",
                "--movie-name",
                "Episode",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(audio_to_subtitle, "find_existing_embedded_subtitle_tracks", return_value=None),
                patch.object(audio_to_subtitle, "load_embedded_subtitle_events", side_effect=load_embedded),
                patch.object(audio_to_subtitle, "extract_audio", side_effect=AssertionError("extract_audio called")),
                patch.object(audio_to_subtitle, "translate_and_correct_segments", side_effect=lambda segments, **_: segments),
            ):
                audio_to_subtitle.main()

            self.assertEqual(captured["stream_index"], 7)

    def test_main_audio_asr_language_defaults_to_selected_source_language(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "movie.mp4"
            video.write_bytes(b"placeholder")
            output_root = root / "out"
            captured = {}

            def transcribe(_audio_path, language):
                captured["language"] = language
                return [{"id": 0, "start": 1.0, "end": 2.0, "text": "konnichiwa"}]

            argv = [
                "audio_to_subtitle.py",
                "--video",
                str(video),
                "--source",
                "audio",
                "--output-root",
                str(output_root),
                "--series-name",
                "Series",
                "--movie-name",
                "Episode",
                "--source-language",
                "ja",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(audio_to_subtitle, "extract_audio"),
                patch.object(audio_to_subtitle, "transcribe_audio", side_effect=transcribe),
                patch.object(audio_to_subtitle, "translate_and_correct_segments", side_effect=lambda segments, **_: segments),
            ):
                audio_to_subtitle.main()

            self.assertEqual(captured["language"], "ja")

    def test_main_ignores_cached_source_when_requested_language_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "movie.mp4"
            video.write_bytes(b"placeholder")
            output_root = root / "out"
            out_dir = output_root / "Series" / "Episode"
            out_dir.mkdir(parents=True)
            source_cache = out_dir / "Episode.segments.source.json"
            source_cache.write_text(
                json.dumps(
                    [
                        {
                            "id": 0,
                            "start": 1.0,
                            "end": 2.0,
                            "text": "old English recognition",
                            "source_language": "en",
                        }
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            captured = {}

            def transcribe(_audio_path, language):
                captured["language"] = language
                return [{"id": 0, "start": 1.0, "end": 2.0, "text": "konnichiwa", "source_language": "ja"}]

            argv = [
                "audio_to_subtitle.py",
                "--video",
                str(video),
                "--source",
                "audio",
                "--output-root",
                str(output_root),
                "--series-name",
                "Series",
                "--movie-name",
                "Episode",
                "--source-language",
                "ja",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(audio_to_subtitle, "extract_audio"),
                patch.object(audio_to_subtitle, "transcribe_audio", side_effect=transcribe),
                patch.object(audio_to_subtitle, "translate_and_correct_segments", side_effect=lambda segments, **_: segments),
            ):
                audio_to_subtitle.main()

            self.assertEqual(captured["language"], "ja")
            refreshed_cache = json.loads(source_cache.read_text(encoding="utf-8"))
            self.assertEqual(refreshed_cache[0]["source_language"], "ja")

    def test_japanese_source_with_kanji_is_not_treated_as_existing_chinese_subtitle(self) -> None:
        source = [
            {
                "id": 0,
                "start": 1.0,
                "end": 2.0,
                "text": "\u9332\u97f3\u306e\u6280\u8853",
                "source_language": "ja",
            }
        ]

        self.assertFalse(segments_have_chinese(source))

    def test_cached_source_fills_missing_language_from_dominant_language(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source.json"
            source_path.write_text(
                json.dumps(
                    [
                        {
                            "id": 0,
                            "start": 1.0,
                            "end": 2.0,
                            "text": "\u9332\u97f3\u306e\u6280\u8853",
                            "source_language": "ja",
                        },
                        {
                            "id": 1,
                            "start": 2.0,
                            "end": 3.0,
                            "text": "\u697d\u66f2\u306e\u5370\u8c61",
                        },
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            loaded = load_cached_source_segments(source_path)

        self.assertEqual(loaded[1]["source_language"], "ja")

    def test_legacy_merged_subtitle_cache_requires_timing_rebuild(self) -> None:
        legacy = [
            {
                "id": 0,
                "start": 1.0,
                "end": 4.0,
                "text": "line",
                "en": "line",
                "zh": "\u5b57\u5e55",
                "source_language": "existing Chinese embedded subtitle",
            }
        ]
        current = [{**legacy[0], "source_cache_version": audio_to_subtitle.SOURCE_CACHE_VERSION}]

        self.assertFalse(source_cache_uses_current_timing_policy(legacy))
        self.assertTrue(source_cache_uses_current_timing_policy(current))

    def test_split_segments_preserves_source_language(self) -> None:
        source = [
            {
                "id": 0,
                "start": 1.0,
                "end": 10.0,
                "text": "\u30ec\u30b3\u30fc\u30c7\u30a3\u30f3\u30b0 \u30df\u30c3\u30af\u30b9 \u6280\u8853",
                "source_language": "ja",
                "words": [
                    {"start": 1.0, "end": 2.0, "word": "\u30ec\u30b3\u30fc\u30c7\u30a3\u30f3\u30b0"},
                    {"start": 2.0, "end": 3.0, "word": "\u30df\u30c3\u30af\u30b9"},
                    {"start": 3.0, "end": 4.0, "word": "\u6280\u8853"},
                ],
            }
        ]

        pieces = split_segments_for_subtitles(source, max_words=14, max_chars=8, max_duration=3.0)

        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(piece.get("source_language") == "ja" for piece in pieces))

    def test_split_segments_splits_bilingual_text_and_duration_together(self) -> None:
        source = [
            {
                "id": 0,
                "start": 10.0,
                "end": 18.0,
                "text": "This is a deliberately long subtitle that needs multiple readable events.",
                "en": "This is a deliberately long subtitle that needs multiple readable events.",
                "zh": "\u8fd9\u662f\u4e00\u6761\u6545\u610f\u5199\u5f97\u5f88\u957f\u7684\u5b57\u5e55\u9700\u8981\u88ab\u5206\u6210\u591a\u4e2a\u5bb9\u6613\u9605\u8bfb\u7684\u7247\u6bb5",
                "source_language": "existing Chinese embedded subtitle",
            }
        ]

        pieces = split_segments_for_subtitles(source, max_words=6, max_chars=28, max_duration=3.0)

        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(float(piece["end"]) - float(piece["start"]) <= 3.01 for piece in pieces))
        self.assertTrue(all(len(piece["en"]) <= 28 for piece in pieces if piece.get("en")))
        self.assertTrue(all(len(piece["zh"]) <= 28 for piece in pieces if piece.get("zh")))
        self.assertEqual(
            " ".join(piece["en"] for piece in pieces if piece.get("en")),
            source[0]["en"],
        )
        self.assertEqual(
            "".join(piece["zh"] for piece in pieces if piece.get("zh")),
            source[0]["zh"],
        )

    def test_bilingual_split_uses_word_timestamps_instead_of_equal_time(self) -> None:
        words = [
            {"start": 10.0, "end": 10.4, "word": "One"},
            {"start": 10.4, "end": 10.8, "word": "two"},
            {"start": 10.8, "end": 11.2, "word": "three"},
            {"start": 11.2, "end": 11.6, "word": "four"},
            {"start": 15.0, "end": 15.4, "word": "five"},
            {"start": 15.4, "end": 15.8, "word": "six"},
            {"start": 15.8, "end": 16.2, "word": "seven"},
            {"start": 16.2, "end": 16.6, "word": "eight"},
        ]
        source = [
            {
                "id": 0,
                "start": 10.0,
                "end": 18.0,
                "text": "One two three four five six seven eight",
                "en": "One two three four five six seven eight",
                "zh": "\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b",
                "timing_origin": "audio_asr",
                "words": words,
            }
        ]

        pieces = split_segments_for_subtitles(
            source,
            max_words=4,
            max_chars=42,
            max_duration=10.0,
        )

        self.assertEqual(len(pieces), 2)
        self.assertAlmostEqual(pieces[0]["end"], 11.6)
        self.assertAlmostEqual(pieces[1]["start"], 15.0)
        self.assertEqual([word["word"] for word in pieces[1]["words"]], ["five", "six", "seven", "eight"])

    def test_unanchored_existing_cue_is_not_delayed_by_synthetic_split(self) -> None:
        source = [
            {
                "id": 0,
                "start": 10.0,
                "end": 18.0,
                "text": "This cue has no word timestamps and must retain its source timing.",
                "en": "This cue has no word timestamps and must retain its source timing.",
                "zh": "\u8fd9\u6761\u5b57\u5e55\u6ca1\u6709\u8bcd\u7ea7\u65f6\u95f4\u952e\u5e76\u5fc5\u987b\u4fdd\u7559\u539f\u65f6\u95f4",
                "timing_origin": "existing_subtitle",
            }
        ]

        pieces = split_segments_for_subtitles(
            source,
            max_words=4,
            max_chars=20,
            max_duration=3.0,
        )

        self.assertEqual(len(pieces), 1)
        self.assertEqual((pieces[0]["start"], pieces[0]["end"]), (10.0, 18.0))
        self.assertTrue(pieces[0]["unanchored_split_avoided"])

    def test_bilingual_output_uses_independent_one_line_limits(self) -> None:
        source = [
            {
                "id": 0,
                "start": 0.0,
                "end": 6.0,
                "text": "This deliberately long English subtitle must be split into readable professional cues.",
                "en": "This deliberately long English subtitle must be split into readable professional cues.",
                "zh": "这是一条故意写得很长而且必须拆成专业可读片段的中文字幕。",
            }
        ]

        pieces = split_segments_for_subtitles(
            source,
            max_words=12,
            max_chars=56,
            max_duration=5.5,
        )

        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(len(piece["en"]) <= 42 for piece in pieces if piece.get("en")))
        self.assertTrue(all(len(piece["zh"]) <= 16 for piece in pieces if piece.get("zh")))
        self.assertEqual(" ".join(piece["en"] for piece in pieces), source[0]["en"])
        self.assertEqual("".join(piece["zh"] for piece in pieces), source[0]["zh"])

    def test_balanced_bilingual_split_prefers_punctuation_boundaries(self) -> None:
        source = [
            {
                "id": 0,
                "start": 0.0,
                "end": 4.0,
                "text": "We came home and waited, because the storm grew worse.",
                "en": "We came home and waited, because the storm grew worse.",
                "zh": "我们回到家里等待，因为暴风雨越来越猛烈。",
            }
        ]

        pieces = split_segments_for_subtitles(
            source,
            max_words=12,
            max_chars=42,
            max_duration=5.5,
        )

        self.assertGreaterEqual(len(pieces), 2)
        self.assertTrue(pieces[0]["en"].endswith(","))
        self.assertTrue(pieces[0]["zh"].endswith("，"))

    def test_reading_budget_caps_each_bilingual_language_to_one_line(self) -> None:
        chinese_budget, english_budget = audio_to_subtitle.reading_budgets(
            {"id": 0, "start": 0.0, "end": 10.0, "text": "Long subtitle"},
        )

        self.assertEqual(chinese_budget, 16)
        self.assertEqual(english_budget, 42)

    def test_word_timed_split_checks_limits_before_adding_next_word(self) -> None:
        words = [
            {"start": 0.0, "end": 1.0, "word": "Alpha"},
            {"start": 1.0, "end": 2.0, "word": "bravo"},
            {"start": 2.0, "end": 3.0, "word": "charlie"},
            {"start": 3.0, "end": 6.0, "word": "delta."},
        ]
        segments = [
            {
                "id": 0,
                "start": 0.0,
                "end": 6.0,
                "text": "Alpha bravo charlie delta.",
                "words": words,
            }
        ]

        pieces = split_segments_for_subtitles(
            segments,
            max_words=3,
            max_chars=18,
            max_duration=4.0,
        )

        self.assertEqual(" ".join(piece["text"] for piece in pieces), "Alpha bravo charlie delta.")
        self.assertTrue(all(len(piece["text"]) <= 18 for piece in pieces))
        self.assertTrue(all(len(piece["text"].split()) <= 3 for piece in pieces))
        self.assertTrue(all(piece["end"] - piece["start"] <= 4.0 for piece in pieces))

    def test_output_suppresses_recent_long_exact_duplicate_missed_by_model(self) -> None:
        repeated = "The same hallucinated sentence keeps repeating again."
        segments = [
            {"id": 0, "start": 10.0, "end": 11.0, "text": repeated, "en": repeated, "zh": "\u540c\u4e00\u53e5\u8bdd\u3002", "display": True},
            {"id": 1, "start": 11.0, "end": 12.0, "text": repeated, "en": repeated, "zh": "\u540c\u4e00\u53e5\u8bdd\u3002", "display": True},
        ]

        output = prepare_segments_for_output(segments, max_words=12, max_chars=56, max_duration=5.5)

        self.assertEqual(len(output), 2)
        self.assertEqual(output[0]["start"], 10.0)
        self.assertLessEqual(output[-1]["end"], 12.5)
        self.assertEqual(" ".join(item["en"] for item in output), repeated)
        self.assertTrue(all(len(item["en"]) <= 42 for item in output))

    def test_output_caps_single_word_with_abnormal_long_timing(self) -> None:
        segments = [
            {
                "id": 0,
                "start": 10.0,
                "end": 19.0,
                "text": "Hello",
                "words": [{"start": 10.0, "end": 19.0, "word": "Hello"}],
                "display": True,
            }
        ]

        output = prepare_segments_for_output(segments, max_words=12, max_chars=56, max_duration=5.5)

        self.assertEqual(len(output), 1)
        self.assertEqual(output[0]["start"], 10.0)
        self.assertEqual(output[0]["end"], 15.5)

    def test_output_extends_into_safe_gap_for_reading_speed(self) -> None:
        segments = [
            {
                "id": 0,
                "start": 0.0,
                "end": 1.0,
                "text": "A line",
                "en": "A line",
                "zh": "这是十二个字的中文字幕",
                "display": True,
            },
            {
                "id": 1,
                "start": 2.0,
                "end": 3.0,
                "text": "Next",
                "en": "Next",
                "zh": "下一句",
                "display": True,
            },
        ]

        output = prepare_segments_for_output(
            segments,
            max_words=12,
            max_chars=56,
            max_duration=5.5,
        )

        self.assertGreater(output[0]["end"], 1.0)
        self.assertLess(output[0]["end"], output[1]["start"])
        self.assertLessEqual(output[0]["end"], 1.5)
        self.assertLessEqual(readability_stats(output)["chinese_max_cps"], 9.0)

    def test_reading_extension_never_overlaps_next_subtitle(self) -> None:
        segments = [
            {"id": 0, "start": 0.0, "end": 1.0, "text": "很长很长的中文字幕"},
            {"id": 1, "start": 1.05, "end": 2.0, "text": "下一句"},
        ]

        output = optimize_readability_timing(segments, max_duration=5.5)

        self.assertEqual(output[0]["end"], 1.0)
        self.assertEqual(output[1]["start"], 1.05)

    def test_final_output_is_sorted_and_removes_stacked_overlaps(self) -> None:
        segments = [
            {"id": 0, "start": 2.0, "end": 3.0, "text": "Later"},
            {"id": 1, "start": 0.0, "end": 2.0, "text": "Earlier"},
            {"id": 2, "start": 1.0, "end": 2.5, "text": "Overlapping"},
        ]

        output = prepare_segments_for_output(
            segments,
            max_words=12,
            max_chars=56,
            max_duration=5.5,
        )

        self.assertEqual([item["text"] for item in output], ["Earlier", "Overlapping", "Later"])
        self.assertTrue(
            all(output[index]["end"] <= output[index + 1]["start"] for index in range(len(output) - 1))
        )

    def test_tiny_overlap_fragment_is_dropped_instead_of_flashing(self) -> None:
        segments = [
            {"id": 0, "start": 0.0, "end": 2.0, "text": "fragment"},
            {"id": 1, "start": 0.3, "end": 2.5, "text": "Complete sentence"},
        ]

        output = resolve_timeline_overlaps(segments)

        self.assertEqual(len(output), 1)
        self.assertEqual(output[0]["text"], "Complete sentence")

    def test_distinct_existing_subtitles_can_overlap_without_data_loss(self) -> None:
        segments = [
            {
                "id": 0,
                "start": 1.0,
                "end": 3.0,
                "text": "First speaker",
                "preserve_distinct_overlap": True,
            },
            {
                "id": 1,
                "start": 1.2,
                "end": 2.8,
                "text": "Second speaker",
                "preserve_distinct_overlap": True,
            },
        ]

        output = resolve_timeline_overlaps(segments)

        self.assertEqual([item["text"] for item in output], ["First speaker", "Second speaker"])
        self.assertEqual(output[0]["end"], 3.0)
        self.assertEqual(output[1]["start"], 1.2)

    def test_prompt_contains_duration_aware_readability_budgets(self) -> None:
        segment = {"id": 0, "start": 0.0, "end": 1.0, "text": "A readable line"}

        text = audio_to_subtitle.format_prompt_segment(
            0,
            segment,
            next_start=2.0,
            include_readability_limits=True,
        )

        self.assertIn("ZH_MAX=13", text)
        self.assertIn("SOURCE_MAX=30", text)

    def test_prompt_budget_is_capped_before_overlapping_next_subtitle(self) -> None:
        segment = {"id": 0, "start": 0.0, "end": 2.0, "text": "An overlapping line"}

        text = audio_to_subtitle.format_prompt_segment(
            0,
            segment,
            next_start=1.0,
            include_readability_limits=True,
        )

        self.assertIn("ZH_MAX=8", text)
        self.assertIn("SOURCE_MAX=18", text)

    def test_output_keeps_short_or_distant_repeated_dialogue(self) -> None:
        segments = [
            {"id": 0, "start": 0.0, "end": 1.0, "text": "No.", "en": "No.", "zh": "\u4e0d\u3002", "display": True},
            {"id": 1, "start": 1.0, "end": 2.0, "text": "No.", "en": "No.", "zh": "\u4e0d\u3002", "display": True},
            {
                "id": 2,
                "start": 20.0,
                "end": 21.0,
                "text": "A deliberate repeated sentence.",
                "en": "A deliberate repeated sentence.",
                "zh": "\u4e00\u53e5\u6709\u610f\u91cd\u590d\u7684\u53f0\u8bcd\u3002",
                "display": True,
            },
            {
                "id": 3,
                "start": 30.0,
                "end": 31.0,
                "text": "A deliberate repeated sentence.",
                "en": "A deliberate repeated sentence.",
                "zh": "\u4e00\u53e5\u6709\u610f\u91cd\u590d\u7684\u53f0\u8bcd\u3002",
                "display": True,
            },
        ]

        output = prepare_segments_for_output(segments, max_words=12, max_chars=56, max_duration=5.5)

        self.assertEqual(len(output), 4)

    def test_merge_existing_subtitles_uses_spoken_track_timing(self) -> None:
        en_segments = [
            {"id": 0, "start": 30.0, "end": 32.0, "text": "The spoken line ends here."}
        ]
        zh_segments = [
            {"id": 0, "start": 30.3, "end": 35.5, "text": "\u4e2d\u6587\u5b57\u5e55\u7684\u65f6\u95f4\u8f74\u660e\u663e\u504f\u665a"}
        ]

        merged = merge_existing_subtitle_segments(
            en_segments,
            zh_segments,
            "existing Chinese embedded subtitle",
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["start"], 30.0)
        self.assertEqual(merged[0]["end"], 32.0)

    def test_existing_subtitle_tracks_are_synchronized_independently_before_merge(self) -> None:
        tracks = ExistingSubtitleTracks(
            english=[{"id": 0, "start": 30.0, "end": 32.0, "text": "Spoken line"}],
            chinese=[{"id": 0, "start": 30.3, "end": 35.5, "text": "中文字幕"}],
            source_label="existing Chinese embedded subtitle",
        )
        calls = []

        def synchronize(_video, segments, **kwargs):
            calls.append(kwargs)
            shift = -20.0 if segments[0]["text"] == "Spoken line" else -20.2
            output = [
                {
                    **segment,
                    "start": segment["start"] + shift,
                    "end": segment["end"] + shift,
                }
                for segment in segments
            ]
            return output, {
                "status": "aligned",
                "pipeline_quality_passed": True,
                "applied": True,
            }

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.object(audio_to_subtitle, "synchronize_subtitle_segments", side_effect=synchronize),
        ):
            synchronized, report = synchronize_existing_subtitle_tracks(
                Path(tmpdir) / "movie.mkv",
                tracks,
                mode="auto",
                report_path=Path(tmpdir) / "sync.json",
                audio_stream=4,
            )

        merged = merge_existing_subtitle_segments(
            synchronized.english,
            synchronized.chinese,
            synchronized.source_label,
        )
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call["audio_stream"] == 4 for call in calls))
        self.assertEqual(merged[0]["start"], 10.0)
        self.assertEqual(merged[0]["end"], 12.0)
        self.assertEqual(report["status"], "aligned")
        self.assertTrue(report["applied"])

    def test_existing_subtitle_sync_reverts_both_tracks_when_one_fails_quality(self) -> None:
        tracks = ExistingSubtitleTracks(
            english=[{"id": 0, "start": 30.0, "end": 32.0, "text": "Spoken line"}],
            chinese=[{"id": 0, "start": 30.3, "end": 35.5, "text": "中文字幕"}],
            source_label="existing Chinese sidecar subtitle",
        )

        def synchronize(_video, segments, **_kwargs):
            if segments[0]["text"] == "Spoken line":
                return [
                    {**segments[0], "start": 10.0, "end": 12.0}
                ], {
                    "status": "aligned",
                    "pipeline_quality_passed": True,
                    "applied": True,
                }
            return [dict(segments[0])], {
                "status": "rejected_low_quality",
                "pipeline_quality_passed": False,
                "applied": False,
            }

        with patch.object(
            audio_to_subtitle,
            "synchronize_subtitle_segments",
            side_effect=synchronize,
        ):
            synchronized, report = synchronize_existing_subtitle_tracks(
                Path("movie.mkv"),
                tracks,
                mode="auto",
                audio_stream=2,
            )

        self.assertEqual(synchronized.english, tracks.english)
        self.assertEqual(synchronized.chinese, tracks.chinese)
        self.assertEqual(report["status"], "rejected_partial")
        self.assertFalse(report["applied"])
        self.assertTrue(report["reverted_all_tracks"])

    def test_pair_events_rejects_nonoverlapping_subtitles_with_large_gap(self) -> None:
        pairs = pair_events(
            [SubtitleEvent(10.0, 11.0, "spoken")],
            [SubtitleEvent(12.0, 13.0, "\u8fdf\u5230\u5b57\u5e55")],
        )

        self.assertEqual(pairs[0], (SubtitleEvent(10.0, 11.0, "spoken"), None))
        self.assertEqual(pairs[1], (None, SubtitleEvent(12.0, 13.0, "\u8fdf\u5230\u5b57\u5e55")))

    def test_pair_events_groups_two_english_events_with_one_chinese_event(self) -> None:
        pairs = pair_events(
            [
                SubtitleEvent(10.0, 11.0, "First sentence."),
                SubtitleEvent(11.0, 12.0, "Second sentence."),
            ],
            [SubtitleEvent(10.0, 12.0, "\u7b2c\u4e00\u53e5\u3002\u7b2c\u4e8c\u53e5\u3002")],
        )

        self.assertEqual(
            pairs,
            [
                (
                    SubtitleEvent(10.0, 12.0, "First sentence. Second sentence."),
                    SubtitleEvent(10.0, 12.0, "\u7b2c\u4e00\u53e5\u3002\u7b2c\u4e8c\u53e5\u3002"),
                )
            ],
        )

    def test_pair_events_groups_one_english_event_with_two_chinese_events(self) -> None:
        pairs = pair_events(
            [SubtitleEvent(20.0, 22.0, "A complete sentence.")],
            [
                SubtitleEvent(20.0, 21.0, "\u4e00\u4e2a"),
                SubtitleEvent(21.0, 22.0, "\u5b8c\u6574\u53e5\u5b50\u3002"),
            ],
        )

        self.assertEqual(
            pairs,
            [
                (
                    SubtitleEvent(20.0, 22.0, "A complete sentence."),
                    SubtitleEvent(20.0, 22.0, "\u4e00\u4e2a\u5b8c\u6574\u53e5\u5b50\u3002"),
                )
            ],
        )

    def test_pair_events_does_not_chain_normal_adjacent_shifted_subtitles(self) -> None:
        pairs = pair_events(
            [
                SubtitleEvent(0.0, 2.0, "First."),
                SubtitleEvent(2.0, 4.0, "Second."),
            ],
            [
                SubtitleEvent(0.2, 2.2, "\u7b2c\u4e00\u53e5\u3002"),
                SubtitleEvent(2.2, 4.2, "\u7b2c\u4e8c\u53e5\u3002"),
            ],
        )

        self.assertEqual(len(pairs), 2)
        self.assertEqual([pair[0].text for pair in pairs if pair[0]], ["First.", "Second."])
        self.assertEqual([pair[1].text for pair in pairs if pair[1]], ["\u7b2c\u4e00\u53e5\u3002", "\u7b2c\u4e8c\u53e5\u3002"])

    def test_pair_events_does_not_transitively_merge_a_shifted_sequence(self) -> None:
        pairs = pair_events(
            [
                SubtitleEvent(0.0, 2.0, "First."),
                SubtitleEvent(2.0, 4.0, "Second."),
                SubtitleEvent(4.0, 6.0, "Third."),
            ],
            [
                SubtitleEvent(1.0, 3.0, "\u7b2c\u4e00\u53e5\u3002"),
                SubtitleEvent(3.0, 5.0, "\u7b2c\u4e8c\u53e5\u3002"),
                SubtitleEvent(5.0, 7.0, "\u7b2c\u4e09\u53e5\u3002"),
            ],
        )

        self.assertEqual(len(pairs), 3)
        self.assertEqual([pair[0].text for pair in pairs if pair[0]], ["First.", "Second.", "Third."])
        self.assertEqual(
            [pair[1].text for pair in pairs if pair[1]],
            ["\u7b2c\u4e00\u53e5\u3002", "\u7b2c\u4e8c\u53e5\u3002", "\u7b2c\u4e09\u53e5\u3002"],
        )

    def test_extracted_subtitle_cache_is_bound_to_video_stream_and_codec(self) -> None:
        calls: list[tuple[int, str]] = []

        def fake_extract(_video, stream_index, _ffmpeg, out_path, codec):
            calls.append((stream_index, codec))
            out_path.write_text("subtitle", encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "movie.mkv"
            video.write_bytes(b"video-v1")
            extracted = root / "stream_03.srt"
            with patch.object(subtitle_pipeline, "extract_subtitle", side_effect=fake_extract):
                subtitle_pipeline.ensure_extracted_subtitle(video, 3, "ffmpeg", extracted, "srt")
                subtitle_pipeline.ensure_extracted_subtitle(video, 3, "ffmpeg", extracted, "srt")
                video.write_bytes(b"video-v2-with-different-size")
                subtitle_pipeline.ensure_extracted_subtitle(video, 3, "ffmpeg", extracted, "srt")

        self.assertEqual(calls, [(3, "srt"), (3, "srt")])

    def test_ocr_cache_resumes_only_when_image_and_config_fingerprints_match(self) -> None:
        class FakeOcrEngine:
            recognized: list[Path] = []

            def __init__(self, **_kwargs):
                pass

            @staticmethod
            def assert_gpu() -> None:
                pass

            @classmethod
            def recognize(cls, image_path: Path) -> str:
                cls.recognized.append(image_path)
                return "second"

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first_image = root / "first.png"
            second_image = root / "second.png"
            first_image.write_bytes(b"first")
            second_image.write_bytes(b"second")
            cache_path = root / "ocr.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "version": 2,
                        "config": {"lang": "en", "device": "gpu:0", "prefer_accuracy": True},
                        "images": [
                            {"start": 1.0, "end": 2.0, "image_hash": "hash-first"},
                            {"start": 2.0, "end": 3.0, "image_hash": "hash-second"},
                        ],
                        "events": [{"start": 1.0, "end": 2.0, "text": "first"}],
                    }
                ),
                encoding="utf-8",
            )
            image_events = [
                PgsImageEvent(1.0, 2.0, first_image, "hash-first"),
                PgsImageEvent(2.0, 3.0, second_image, "hash-second"),
            ]

            with patch.object(subtitle_pipeline, "PaddleOcrEngine", FakeOcrEngine):
                result = ocr_pgs_events(image_events, "en", "gpu:0", cache_path, True)

        self.assertEqual([event.text for event in result], ["first", "second"])
        self.assertEqual(FakeOcrEngine.recognized, [second_image])

    def test_ocr_observation_accepts_array_like_scores(self) -> None:
        class AmbiguousScores(list):
            def __bool__(self):
                raise AssertionError("array-like scores must not be coerced to bool")

        observation = subtitle_pipeline.extract_ocr_observation(
            {
                "rec_texts": ["recognized"],
                "rec_scores": AmbiguousScores([0.64]),
                "rec_polys": [
                    [[0, 0], [100, 0], [100, 20], [0, 20]],
                ],
            },
            "en",
        )

        self.assertEqual(observation.text, "recognized")
        self.assertEqual(observation.confidence, 0.64)
        self.assertEqual(observation.line_confidences, [0.64])

    def test_ocr_observation_persists_confidence_in_cache(self) -> None:
        class FakeOcrEngine:
            def __init__(self, **_kwargs):
                pass

            @staticmethod
            def assert_gpu() -> None:
                pass

            @staticmethod
            def recognize(_image_path: Path):
                return subtitle_pipeline.OcrObservation(
                    text="recognized",
                    confidence=0.64,
                    line_confidences=[0.64],
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image = root / "subtitle.png"
            image.write_bytes(b"image")
            cache_path = root / "ocr.json"
            image_events = [PgsImageEvent(1.0, 2.0, image, "hash")]

            with patch.object(subtitle_pipeline, "PaddleOcrEngine", FakeOcrEngine):
                result = ocr_pgs_events(image_events, "en", "gpu:0", cache_path, True)

            cached = json.loads(cache_path.read_text(encoding="utf-8"))

        self.assertEqual(result[0].confidence, 0.64)
        self.assertEqual(result[0].metadata["line_confidences"], [0.64])
        self.assertEqual(cached["events"][0]["confidence"], 0.64)

    def test_ocr_cache_rejects_same_length_with_different_image_hash(self) -> None:
        class FakeOcrEngine:
            recognized: list[Path] = []

            def __init__(self, **_kwargs):
                pass

            @staticmethod
            def assert_gpu() -> None:
                pass

            @classmethod
            def recognize(cls, image_path: Path) -> str:
                cls.recognized.append(image_path)
                return image_path.stem

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image = root / "changed.png"
            image.write_bytes(b"changed")
            cache_path = root / "ocr.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "version": 2,
                        "config": {"lang": "en", "device": "gpu:0", "prefer_accuracy": True},
                        "images": [{"start": 1.0, "end": 2.0, "image_hash": "old-hash"}],
                        "events": [{"start": 1.0, "end": 2.0, "text": "stale"}],
                    }
                ),
                encoding="utf-8",
            )
            image_events = [PgsImageEvent(1.0, 2.0, image, "new-hash")]

            with patch.object(subtitle_pipeline, "PaddleOcrEngine", FakeOcrEngine):
                result = ocr_pgs_events(image_events, "en", "gpu:0", cache_path, True)

        self.assertEqual([event.text for event in result], ["changed"])
        self.assertEqual(FakeOcrEngine.recognized, [image])

    def test_validated_ocr_cache_rejects_all_empty_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image = root / "empty.png"
            image.write_bytes(b"empty")
            cache_path = root / "ocr.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "version": 2,
                        "config": {"lang": "en", "device": "gpu:0", "prefer_accuracy": True},
                        "images": [{"start": 1.0, "end": 2.0, "image_hash": "empty-hash"}],
                        "events": [{"start": 1.0, "end": 2.0, "text": ""}],
                    }
                ),
                encoding="utf-8",
            )
            image_events = [PgsImageEvent(1.0, 2.0, image, "empty-hash")]

            with self.assertRaisesRegex(RuntimeError, "OCR produced no text"):
                ocr_pgs_events(image_events, "en", "gpu:0", cache_path, True)

    def test_japanese_translate_rejects_old_unversioned_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.json"
            source = [
                {
                    "id": 0,
                    "start": 1.0,
                    "end": 2.0,
                    "text": "\u9332\u97f3\u306e\u6280\u8853",
                    "source_language": "ja",
                }
            ]
            checkpoint_path.write_text(
                json.dumps(
                    [
                        {
                            **source[0],
                            "en": "",
                            "zh": "\u5f55\u97f3\u7684\u6280\u672f",
                            "display": True,
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            llm_response = """
            [
              {
                "index": 0,
                "corrected_text": "\u9332\u97f3\u306e\u6280\u8853",
                "chinese_translation": "\u5f55\u97f3\u7684\u6280\u672f",
                "display": true
              }
            ]
            """

            with patch.object(audio_to_subtitle, "call_llm", return_value=llm_response):
                output = translate_and_correct_segments(
                    source,
                    llm_model="qwen3:30b",
                    batch_size=1,
                    context_lines=0,
                    source_language="ja",
                    checkpoint_path=checkpoint_path,
                )

        self.assertEqual(output[0]["en"], "\u9332\u97f3\u306e\u6280\u8853")
        self.assertEqual(output[0]["processing_mode"], "translate")

    def test_japanese_translation_retries_source_language_change(self) -> None:
        source = [
            {
                "id": 0,
                "start": 1.0,
                "end": 2.0,
                "text": "\u9332\u97f3\u306e\u6280\u8853",
                "source_language": "ja",
            }
        ]
        translated_source = """
        [
          {
            "index": 0,
            "corrected_text": "Recording technology",
            "chinese_translation": "\u5f55\u97f3\u6280\u672f",
            "display": true,
            "terminology": []
          }
        ]
        """
        repaired = {
            "corrected_text": "\u9332\u97f3\u306e\u6280\u8853",
            "chinese_translation": "\u5f55\u97f3\u6280\u672f",
            "display": True,
            "terminology": [],
        }

        with (
            patch.object(
                audio_to_subtitle,
                "call_llm",
                return_value=translated_source,
            ),
            patch.object(
                audio_to_subtitle,
                "retry_single_segment_llm",
                return_value=repaired,
            ) as retry,
        ):
            output = translate_and_correct_segments(
                source,
                llm_model="qwen3:30b",
                batch_size=1,
                context_lines=0,
                source_language="ja",
            )

        retry.assert_called_once()
        self.assertEqual(output[0]["en"], "\u9332\u97f3\u306e\u6280\u8853")
        self.assertEqual(output[0]["zh"], "\u5f55\u97f3\u6280\u672f")

    def test_frontend_defaults_audio_recognition_to_follow_source_language(self) -> None:
        frontend_source = (ROOT / "src" / "subtitle_frontend.py").read_text(encoding="utf-8")

        self.assertIn('<option value="source" selected>', frontend_source)
        self.assertNotIn('<option value="en" selected>', frontend_source)
        self.assertIn("resolveAsrLanguageForPayload()", frontend_source)

    def test_main_uses_chinese_sidecar_without_translation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "movie.mp4"
            video.write_bytes(b"placeholder")
            en_srt = root / "movie.en.srt"
            zh_srt = root / "movie.zh.srt"
            en_srt.write_text(
                "1\n00:00:01,000 --> 00:00:02,000\nHelo world\n\n",
                encoding="utf-8",
            )
            zh_srt.write_text(
                "1\n00:00:01,000 --> 00:00:02,000\n\u7e41\u9ad4\u5b57\u5e55\n\n",
                encoding="utf-8",
            )
            output_root = root / "out"
            out_dir = output_root / "Series" / "Episode"

            def proofread(segments, **_):
                return [
                    {
                        **segments[0],
                        "text": "Hello world",
                        "en": "Hello world",
                        "zh": "\u7e41\u4f53\u5b57\u5e55",
                        "display": True,
                    }
                ]

            argv = [
                "audio_to_subtitle.py",
                "--video",
                str(video),
                "--source",
                "auto",
                "--subtitle-sync",
                "off",
                "--output-root",
                str(output_root),
                "--series-name",
                "Series",
                "--movie-name",
                "Episode",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(audio_to_subtitle, "extract_audio", side_effect=AssertionError("extract_audio called")),
                patch.object(audio_to_subtitle, "transcribe_audio", side_effect=AssertionError("transcribe_audio called")),
                patch.object(audio_to_subtitle, "translate_and_correct_segments", side_effect=AssertionError("translate called")),
                patch.object(audio_to_subtitle, "proofread_existing_chinese_segments", side_effect=proofread),
            ):
                audio_to_subtitle.main()

            body = (out_dir / "Episode.bilingual.ass").read_text(encoding="utf-8")
            source_cache = json.loads((out_dir / "Episode.segments.source.json").read_text(encoding="utf-8"))
            quality_report = json.loads(
                (out_dir / "Episode.quality-report.json").read_text(encoding="utf-8")
            )
            self.assertIn("Hello world", body)
            self.assertIn("\u7e41\u4f53\u5b57\u5e55", body)
            self.assertEqual(source_cache[0]["zh"], "\u7e41\u4f53\u5b57\u5e55")
            self.assertEqual(quality_report["status"], "review")
            self.assertEqual(
                quality_report["checks"]["subtitle_sync"]["result"],
                "disabled",
            )
            self.assertEqual(
                quality_report["artifacts"]["bilingual_ass"],
                str(out_dir / "Episode.bilingual.ass"),
            )

    def test_existing_chinese_proofreading_corrects_english_and_hides_duplicates(self) -> None:
        source = [
            {
                "id": 0,
                "start": 1.0,
                "end": 2.0,
                "text": "Helo world",
                "en": "Helo world",
                "zh": "\u4f60\u597d\u4e16\u754c",
            },
            {
                "id": 1,
                "start": 2.0,
                "end": 3.0,
                "text": "Helo world",
                "en": "Helo world",
                "zh": "\u4f60\u597d\u4e16\u754c",
            },
        ]
        llm_response = """
        [
          {
            "index": 0,
            "corrected_english": "Hello world",
            "corrected_chinese": "\u4f60\u597d\u4e16\u754c",
            "display": true,
            "display_start": 1.0,
            "display_end": 3.0
          },
          {
            "index": 1,
            "corrected_english": "",
            "corrected_chinese": "",
            "display": false
          }
        ]
        """

        with patch.object(audio_to_subtitle, "call_llm", return_value=llm_response):
            output = proofread_existing_chinese_segments(
                source,
                llm_model="qwen3:30b",
                batch_size=2,
                context_lines=0,
            )

        self.assertEqual(output[0]["en"], "Hello world")
        self.assertEqual(output[0]["zh"], "\u4f60\u597d\u4e16\u754c")
        self.assertEqual(output[0]["display_end"], 3.0)
        self.assertFalse(output[1]["display"])

    def test_existing_chinese_proofreading_fills_missing_chinese_cues_only(self) -> None:
        source = [
            {
                "id": 0,
                "start": 4.0,
                "end": 5.0,
                "text": "(MUSIC PLAYING)",
                "en": "(MUSIC PLAYING)",
                "zh": "",
            }
        ]

        def call_llm(prompt, *_args, **_kwargs):
            self.assertIn(
                "Translate source-track information into corrected_chinese only when the original Chinese is empty",
                prompt,
            )
            self.assertIn("SDH/non-speech cues", prompt)
            self.assertIn("Keep names and recurring terminology consistent", prompt)
            return """
            [
              {
                "index": 0,
                "corrected_english": "(MUSIC PLAYING)",
                "corrected_chinese": "\u64ad\u653e\u97f3\u4e50",
                "display": true
              }
            ]
            """

        with patch.object(audio_to_subtitle, "call_llm", side_effect=call_llm):
            output = proofread_existing_chinese_segments(
                source,
                llm_model="qwen3:30b",
                batch_size=1,
                context_lines=0,
            )

        self.assertEqual(output[0]["zh"], "\u64ad\u653e\u97f3\u4e50")
        self.assertEqual(output[0]["en"], "(MUSIC PLAYING)")

    def test_existing_chinese_proofreading_repairs_unquoted_model_strings(self) -> None:
        source = [
            {
                "id": 0,
                "start": 4.0,
                "end": 5.0,
                "text": "(GRUNTING)",
                "en": "(GRUNTING)",
                "zh": "",
            },
            {
                "id": 1,
                "start": 5.0,
                "end": 6.0,
                "text": "Pizza, guys.",
                "en": "Pizza, guys.",
                "zh": "\u5404\u4f4d\u2019\u62ab\u8428\u6765\u4e86",
            },
        ]
        malformed_response = """[
          {
            "index": 0,
            "corrected_english": "(GRUNTING)",
            "corrected_chinese": (\u4f4e\u543c\u58f0),
            "display": true,
            "terminology": []
          },
          {
            "index": 1,
            "corrected_english": "Pizza, guys.",
            "corrected_chinese": \u5404\u4f4d\uff0c\u62ab\u8428\u6765\u4e86\u3002",
            "display": true,
            "terminology": []
          }
        ]"""

        with patch.object(
            audio_to_subtitle,
            "call_llm",
            return_value=malformed_response,
        ):
            output = proofread_existing_chinese_segments(
                source,
                llm_model="qwen3.5:122b",
                batch_size=2,
                context_lines=0,
            )

        self.assertEqual(output[0]["zh"], "(\u4f4e\u543c\u58f0)")
        self.assertEqual(output[1]["zh"], "\u5404\u4f4d\uff0c\u62ab\u8428\u6765\u4e86\u3002")

    def test_existing_chinese_proofreading_never_keeps_chinese_in_english_track(self) -> None:
        source = [
            {
                "id": 0,
                "start": 6.0,
                "end": 7.0,
                "text": "\u4fc4\u4ea5\u4fc4\u5dde \u54e5\u4f26\u5e03\u5e02 2045\u5e74",
                "en": "\u4fc4\u4ea5\u4fc4\u5dde \u54e5\u4f26\u5e03\u5e02 2045\u5e74",
                "zh": "\u4fc4\u4ea5\u4fc4\u5dde \u54e5\u4f26\u5e03\u5e02 2045\u5e74",
            }
        ]
        llm_response = """
        [
          {
            "index": 0,
            "corrected_english": "\u4fc4\u4ea5\u4fc4\u5dde \u54e5\u4f26\u5e03\u5e02 2045\u5e74",
            "corrected_chinese": "\u4fc4\u4ea5\u4fc4\u5dde \u54e5\u4f26\u5e03\u5e02 2045\u5e74",
            "display": true
          }
        ]
        """

        def call_llm(prompt, *_args, **_kwargs):
            self.assertIn("EN: - | ZH: \u4fc4\u4ea5\u4fc4\u5dde \u54e5\u4f26\u5e03\u5e02 2045\u5e74", prompt)
            return llm_response

        with patch.object(audio_to_subtitle, "call_llm", side_effect=call_llm):
            output = proofread_existing_chinese_segments(
                source,
                llm_model="qwen3:30b",
                batch_size=1,
                context_lines=0,
            )

        self.assertEqual(output[0]["en"], "")
        self.assertEqual(output[0]["zh"], "\u4fc4\u4ea5\u4fc4\u5dde \u54e5\u4f26\u5e03\u5e02 2045\u5e74")

    def test_existing_chinese_proofreading_preserves_japanese_source(self) -> None:
        source = [
            {
                "id": 0,
                "start": 6.0,
                "end": 8.0,
                "text": "\u3053\u3093\u306b\u3061\u306f",
                "en": "\u3053\u3093\u306b\u3061\u306f",
                "zh": "\u4f60\u597d",
                "source_language": "ja",
            }
        ]
        translated_source = """
        [
          {
            "index": 0,
            "corrected_english": "Hello",
            "corrected_chinese": "\u4f60\u597d",
            "display": true,
            "terminology": []
          }
        ]
        """
        repaired = {
            "corrected_english": "\u3053\u3093\u306b\u3061\u306f",
            "corrected_chinese": "\u4f60\u597d",
            "display": True,
            "terminology": [],
        }

        with (
            patch.object(
                audio_to_subtitle,
                "call_llm",
                return_value=translated_source,
            ) as call,
            patch.object(
                audio_to_subtitle,
                "retry_single_segment_llm",
                return_value=repaired,
            ) as retry,
        ):
            output = proofread_existing_chinese_segments(
                source,
                llm_model="qwen3:30b",
                batch_size=1,
                context_lines=0,
            )

        self.assertIn("SOURCE_LANGUAGE=ja", call.call_args.args[0])
        retry.assert_called_once()
        self.assertEqual(output[0]["en"], "\u3053\u3093\u306b\u3061\u306f")
        self.assertEqual(output[0]["zh"], "\u4f60\u597d")

    def test_generate_ass_does_not_write_chinese_in_english_layer(self) -> None:
        segments = [
            {
                "id": 0,
                "start": 8.0,
                "end": 9.0,
                "text": "\u62d6\u8f66\u5c4b\u56ed\u533a",
                "en": "\u62d6\u8f66\u5c4b\u56ed\u533a",
                "zh": "\u62d6\u8f66\u5c4b\u56ed\u533a",
                "display": True,
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            bilingual_path = Path(tmpdir) / "out.bilingual.ass"
            english_path = Path(tmpdir) / "out.en.ass"
            generate_ass(segments, bilingual_path, "bilingual")
            generate_ass(segments, english_path, "en")
            bilingual_body = bilingual_path.read_text(encoding="utf-8")
            english_body = english_path.read_text(encoding="utf-8")

        self.assertIn("\u62d6\u8f66\u5c4b\u56ed\u533a", bilingual_body)
        self.assertNotIn("{\\rEnglish}\\N\u62d6\u8f66\u5c4b\u56ed\u533a", bilingual_body)
        self.assertNotIn("Dialogue:", english_body)

    def test_generate_ass_does_not_write_japanese_kana_in_unknown_english_layer(self) -> None:
        segments = [
            {
                "id": 0,
                "start": 8.0,
                "end": 9.0,
                "text": "\u306e",
                "en": "\u306e",
                "zh": "",
                "display": True,
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            english_path = Path(tmpdir) / "out.en.ass"
            generate_ass(segments, english_path, "en")
            english_body = english_path.read_text(encoding="utf-8")

        self.assertNotIn("Dialogue:", english_body)

    def test_generate_ass_keeps_non_english_source_layer_when_language_is_known(self) -> None:
        segments = [
            {
                "id": 0,
                "start": 8.0,
                "end": 9.0,
                "text": "\u9332\u97f3\u306e\u6280\u8853",
                "en": "\u9332\u97f3\u306e\u6280\u8853",
                "zh": "\u5f55\u97f3\u7684\u6280\u672f",
                "source_language": "ja",
                "display": True,
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            bilingual_path = Path(tmpdir) / "out.bilingual.ass"
            source_path = Path(tmpdir) / "out.source.ass"
            generate_ass(segments, bilingual_path, "bilingual")
            generate_ass(segments, source_path, "en")
            bilingual_body = bilingual_path.read_text(encoding="utf-8")
            source_body = source_path.read_text(encoding="utf-8")

        self.assertIn("\u9332\u97f3\u306e\u6280\u8853", bilingual_body)
        self.assertIn("\u9332\u97f3\u306e\u6280\u8853", source_body)

    def test_frontend_preview_does_not_fallback_empty_english_to_source_text(self) -> None:
        frontend_source = (ROOT / "src" / "subtitle_frontend.py").read_text(encoding="utf-8")

        self.assertIn("<th>源文/英文</th>", frontend_source)
        self.assertIn("function previewEnglishText(item)", frontend_source)
        self.assertIn("function sourceLanguageAllowsCjkPreview(item)", frontend_source)
        self.assertIn("\\u3040-\\u30ff", frontend_source)
        self.assertIn("!sourceLanguageAllowsCjkPreview(item) && hasCjk(value)", frontend_source)
        self.assertIn("previewEnglishText(item)", frontend_source)
        self.assertNotIn("item.en || item.text", frontend_source)

    def test_frontend_preview_uses_source_timing_not_legacy_model_timing(self) -> None:
        summary = subtitle_frontend.summarize_segment(
            {
                "id": 1,
                "start": 10.0,
                "end": 11.0,
                "display_start": 35.0,
                "display_end": 36.0,
                "text": "line",
            }
        )

        self.assertEqual(summary["start"], 10.0)
        self.assertEqual(summary["end"], 11.0)

    def test_frontend_uses_progressive_disclosure_for_source_options(self) -> None:
        page = subtitle_frontend.html_page()

        self.assertIn('id="existingSubtitleOptions"', page)
        self.assertIn('id="audioRecognitionOptions"', page)
        self.assertIn('id="advancedSettings"', page)
        self.assertIn("function updateWorkflowVisibility()", page)
        self.assertIn("source.addEventListener('change', () => {", page)
        self.assertIn(
            "invalidateProcessingPlanPreview();\n    updateWorkflowVisibility();",
            page,
        )
        self.assertIn(
            "subtitleSync').addEventListener('change', () => {",
            page,
        )
        self.assertIn("subtitle_file: explicitSource && usesSidecar && !mergeExisting", page)
        self.assertIn("chinese_subtitle_stream: explicitSource && usesEmbedded && mergeExisting", page)
        self.assertIn("setHidden('sidecarOptions', !usesSidecar || automaticSource)", page)
        self.assertIn("const automaticPlanNeedsOcr = automaticSource", page)
        self.assertIn("'embeddedOptions',", page)
        self.assertIn("automaticSource || !usesEmbedded || merge", page)
        self.assertIn("const automaticPlanUsesExisting = automaticSource", page)
        self.assertIn("const mayUseExisting = usesExisting || (source === 'auto' && mode === 'auto')", page)
        self.assertIn("usesExisting && subtitleSync !== 'off'", page)
        self.assertIn("audio_stream: usesSelectedAudioTrack ?", page)
        self.assertIn("用于字幕同步的音轨", page)

    def test_frontend_reports_failed_run_with_last_active_stage(self) -> None:
        stage, stage_index, outcome = subtitle_frontend.infer_stage_info(
            "Processing segments: 8/20\n",
            "Traceback (most recent call last):\nRuntimeError: remote model unavailable\n",
            False,
        )

        self.assertEqual(stage, "翻译与校对失败")
        self.assertEqual(stage_index, 2)
        self.assertEqual(outcome, "failed")
        self.assertEqual(
            subtitle_frontend.summarize_error(
                "Traceback (most recent call last):\n"
                "RuntimeError: remote model unavailable\n"
                "OCR progress: 40%\n"
            ),
            "RuntimeError: remote model unavailable",
        )

    def test_frontend_user_stop_marker_survives_status_refresh(self) -> None:
        stage, stage_index, outcome = subtitle_frontend.infer_stage_info(
            f"Processing segments: 8/20\n{subtitle_frontend.USER_STOP_MARKER}\n",
            "warning emitted before termination\n",
            False,
        )

        self.assertEqual(stage, "翻译与校对已终止")
        self.assertEqual(stage_index, 2)
        self.assertEqual(outcome, "stopped")

    def test_frontend_failure_message_is_visible_and_actionable(self) -> None:
        page = subtitle_frontend.html_page()

        self.assertIn('id="runMessage"', page)
        self.assertIn("data.outcome === 'failed'", page)
        self.assertIn("data.checkpoint_exists", page)
        self.assertIn("继续任务", page)

    def test_frontend_exposes_three_cache_aware_run_modes(self) -> None:
        page = subtitle_frontend.html_page()

        self.assertIn('value="resume" checked>继续任务', page)
        self.assertIn('value="reprocess">重新校对（保留识别）', page)
        self.assertIn('value="restart">完全重建', page)
        self.assertIn("base.run_mode = document.querySelector", page)

    def test_frontend_launcher_does_not_open_an_unrelated_service(self) -> None:
        frontend_source = (ROOT / "src" / "subtitle_frontend.py").read_text(encoding="utf-8")
        launcher = (ROOT / "scripts" / "start_frontend.ps1").read_text(encoding="utf-8")

        self.assertIn('parsed.path == "/api/health"', frontend_source)
        self.assertIn('"app": "bilingual-subtitle-pipeline"', frontend_source)
        self.assertIn('"source_sha256": FRONTEND_SOURCE_SHA256', frontend_source)
        self.assertIn("def frontend_source_fingerprint()", frontend_source)
        self.assertIn('APP_DIR.glob("*.py")', frontend_source)
        self.assertIn('Invoke-RestMethod -Uri "$url/api/health"', launcher)
        self.assertIn("foreach ($port in 8765..8775)", launcher)
        self.assertIn("function Get-FrontendSourceSha256", launcher)
        self.assertIn('-Filter "*.py"', launcher)
        self.assertIn("$expectedSourceSha256", launcher)
        self.assertIn("$health.source_sha256", launcher)
        self.assertIn("Skipping stale subtitle frontend", launcher)


if __name__ == "__main__":
    unittest.main()
