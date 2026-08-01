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
from subtitle_pipeline import StreamInfo  # noqa: E402
from subtitle_sources import (  # noqa: E402
    build_audio_asset,
    build_embedded_asset,
    build_processing_plan,
    build_sidecar_asset,
    classify_role,
)


class SubtitleSourcePlanningTests(unittest.TestCase):
    def test_role_detection_uses_container_dispositions(self) -> None:
        self.assertEqual(classify_role("English", {"hearing_impaired": 1}), "sdh")
        self.assertEqual(classify_role("English", {"forced": 1}), "forced")
        self.assertEqual(classify_role("Director Commentary"), "commentary")
        self.assertEqual(
            classify_role("English", {"visual_impaired": 1}),
            "commentary",
        )
        self.assertEqual(classify_role("中文听障字幕"), "sdh")
        self.assertEqual(classify_role("中文强制字幕"), "forced")
        self.assertEqual(classify_role("导演评论"), "commentary")

    def test_planner_prefers_authored_sdh_source_and_normal_chinese(self) -> None:
        video = Path("movie.mkv")
        chinese = build_embedded_asset(
            video,
            StreamInfo(2, "subtitle", "ass", "chi", "Simplified Chinese Full"),
        )
        english = build_embedded_asset(
            video,
            StreamInfo(3, "subtitle", "ass", "eng", "English Full"),
        )
        english_sdh = build_embedded_asset(
            video,
            StreamInfo(
                4,
                "subtitle",
                "ass",
                "eng",
                "English SDH",
                disposition={"hearing_impaired": 1},
            ),
        )

        plan = build_processing_plan([chinese, english, english_sdh])

        self.assertEqual(plan["lanes"]["chinese"], chinese.asset_id)
        self.assertEqual(plan["lanes"]["source"], english_sdh.asset_id)
        self.assertEqual(plan["route"], "proofread_existing_chinese")
        self.assertIn("translate_missing_chinese_only", plan["operations"])
        self.assertEqual(plan["compute"]["local_gpu"], [])

    def test_existing_traditional_chinese_skips_asr_and_uses_t2s(self) -> None:
        video = Path("movie.mkv")
        chinese = build_embedded_asset(
            video,
            StreamInfo(2, "subtitle", "ass", "chi", "Traditional Chinese"),
        )
        audio = build_audio_asset(
            video,
            StreamInfo(1, "audio", "aac", "eng", "English"),
        )

        plan = build_processing_plan([chinese, audio])

        self.assertEqual(plan["route"], "proofread_existing_chinese")
        self.assertIn("traditional_to_simplified", plan["operations"])
        self.assertNotIn("transcribe_audio", plan["operations"])
        self.assertEqual(plan["compute"]["local_gpu"], [])
        self.assertEqual(plan["lanes"]["audio_reference"], audio.asset_id)

    def test_matching_authored_embedded_chinese_beats_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "film.mkv"
            sidecar_path = root / "film.zh.srt"
            video.write_bytes(b"video")
            sidecar_path.write_text(
                "1\n00:00:01,000 --> 00:00:02,000\n你好\n",
                encoding="utf-8",
            )
            sidecar = build_sidecar_asset(
                video,
                sidecar_path,
                likely_match=True,
                score=50,
            )
            embedded = build_embedded_asset(
                video,
                StreamInfo(2, "subtitle", "ass", "chi", "Simplified Chinese"),
            )

            plan = build_processing_plan([sidecar, embedded])

        self.assertEqual(plan["lanes"]["chinese"], embedded.asset_id)

    def test_bitmap_source_uses_ocr_but_not_speech_recognition(self) -> None:
        video = Path("movie.mkv")
        bitmap = build_embedded_asset(
            video,
            StreamInfo(5, "subtitle", "hdmv_pgs_subtitle", "eng", "English"),
        )
        unsupported_bitmap = build_embedded_asset(
            video,
            StreamInfo(6, "subtitle", "dvb_subtitle", "eng", "English"),
        )
        unsupported_teletext = build_embedded_asset(
            video,
            StreamInfo(7, "subtitle", "dvb_teletext", "eng", "English"),
        )
        audio = build_audio_asset(video, stream_index=1, language="en")

        plan = build_processing_plan([bitmap, audio])

        self.assertEqual(plan["lanes"]["source"], bitmap.asset_id)
        self.assertIn("extract_bitmap_subtitles", plan["operations"])
        self.assertEqual(plan["compute"]["local_gpu"], ["subtitle_ocr"])
        self.assertNotIn("transcribe_audio", plan["operations"])
        self.assertEqual(unsupported_bitmap.representation, "unsupported")
        self.assertEqual(unsupported_teletext.representation, "unsupported")
        self.assertFalse(
            StreamInfo(
                7,
                "subtitle",
                "dvb_teletext",
                "eng",
                "English",
            ).is_text_subtitle
        )

    def test_audio_fallback_avoids_commentary_and_prefers_default_language(self) -> None:
        video = Path("movie.mkv")
        commentary = build_audio_asset(
            video,
            StreamInfo(
                0,
                "audio",
                "aac",
                "eng",
                "Director Commentary",
                disposition={"default": 1},
            ),
        )
        japanese = build_audio_asset(
            video,
            StreamInfo(1, "audio", "aac", "jpn", "Japanese"),
        )
        english = build_audio_asset(
            video,
            StreamInfo(
                2,
                "audio",
                "aac",
                "eng",
                "English",
                disposition={"default": 1},
            ),
        )

        plan = build_processing_plan(
            [commentary, japanese, english],
            preferred_source_language="en",
        )

        self.assertEqual(plan["lanes"]["source"], english.asset_id)
        self.assertEqual(plan["lanes"]["audio_reference"], english.asset_id)
        self.assertEqual(plan["compute"]["local_gpu"], ["speech_recognition"])

    def test_commentary_only_audio_is_not_a_source_or_sync_reference(self) -> None:
        video = Path("movie.mkv")
        commentary = build_audio_asset(
            video,
            StreamInfo(
                0,
                "audio",
                "aac",
                "eng",
                "Director Commentary",
                disposition={"default": 1},
            ),
        )

        plan = build_processing_plan([commentary])

        self.assertIsNone(plan["lanes"]["source"])
        self.assertIsNone(plan["lanes"]["audio_reference"])
        self.assertEqual(plan["operations"], [])
        self.assertEqual(plan["compute"]["remote_llm"], [])
        self.assertTrue(
            any("不能自动用于对白识别" in warning for warning in plan["warnings"])
        )
        with self.assertRaisesRegex(RuntimeError, "commentary or audio-description"):
            audio_to_subtitle.require_safe_automatic_audio_selection(
                [commentary],
                explicit_stream=None,
                selected_asset=None,
            )

    def test_forced_track_is_supplementary_not_primary(self) -> None:
        video = Path("movie.mkv")
        chinese = build_embedded_asset(
            video,
            StreamInfo(1, "subtitle", "ass", "chi", "Simplified Chinese"),
        )
        english = build_embedded_asset(
            video,
            StreamInfo(2, "subtitle", "ass", "eng", "English Full"),
        )
        forced = build_embedded_asset(
            video,
            StreamInfo(
                3,
                "subtitle",
                "ass",
                "eng",
                "English Forced",
                disposition={"forced": 1},
            ),
        )

        plan = build_processing_plan([chinese, english, forced])
        no_merge_plan = build_processing_plan(
            [chinese, english, forced],
            merge_existing=False,
        )
        forced_only_plan = build_processing_plan([forced])

        self.assertEqual(plan["lanes"]["source"], english.asset_id)
        self.assertEqual(plan["lanes"]["supplementary"], [forced.asset_id])
        self.assertEqual(no_merge_plan["lanes"]["supplementary"], [])
        self.assertEqual(forced_only_plan["route"], "no_safe_source")
        self.assertEqual(forced_only_plan["lanes"]["supplementary"], [])

    def test_supplementary_track_matches_source_language_and_uses_one_best_asset(self) -> None:
        video = Path("movie.mkv")
        chinese = build_embedded_asset(
            video,
            StreamInfo(1, "subtitle", "ass", "chi", "Simplified Chinese"),
        )
        japanese = build_embedded_asset(
            video,
            StreamInfo(2, "subtitle", "ass", "jpn", "Japanese Full"),
        )
        japanese_forced = build_embedded_asset(
            video,
            StreamInfo(
                3,
                "subtitle",
                "ass",
                "jpn",
                "Japanese Forced",
                disposition={"forced": 1},
            ),
        )
        japanese_forced_ocr = build_embedded_asset(
            video,
            StreamInfo(
                4,
                "subtitle",
                "hdmv_pgs_subtitle",
                "jpn",
                "Japanese Forced PGS",
                disposition={"forced": 1},
            ),
        )
        english_forced = build_embedded_asset(
            video,
            StreamInfo(
                5,
                "subtitle",
                "ass",
                "eng",
                "English Forced",
                disposition={"forced": 1},
            ),
        )

        plan = build_processing_plan(
            [
                chinese,
                japanese,
                japanese_forced,
                japanese_forced_ocr,
                english_forced,
            ]
        )

        self.assertEqual(plan["lanes"]["source"], japanese.asset_id)
        self.assertEqual(
            plan["lanes"]["supplementary"],
            [japanese_forced.asset_id],
        )
        self.assertNotIn("subtitle_ocr", plan["compute"]["local_gpu"])

    def test_single_source_route_loads_planned_forced_cues(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "film.mkv"
            source_path = root / "film.en.srt"
            forced_path = root / "film.en.forced.srt"
            video.write_bytes(b"video")
            source_path.write_text(
                "1\n00:00:01,000 --> 00:00:02,000\nHello\n",
                encoding="utf-8",
            )
            forced_path.write_text(
                "1\n00:00:03,000 --> 00:00:04,000\n[ITALIAN]\n",
                encoding="utf-8",
            )
            source_asset = build_sidecar_asset(
                video,
                source_path,
                language="en",
                likely_match=True,
                score=100,
            )
            forced_asset = build_sidecar_asset(
                video,
                forced_path,
                language="en",
                likely_match=True,
                score=50,
            )
            primary = audio_to_subtitle.parse_subtitle_file(source_path)
            audio_to_subtitle.attach_asset_to_segments(
                primary,
                source_asset,
                lane="source",
            )
            plan = build_processing_plan([source_asset, forced_asset])

            merged, loaded_assets = (
                audio_to_subtitle.merge_planned_supplementary_segments(
                    primary,
                    [source_asset, forced_asset],
                    plan,
                    video,
                    root / "out",
                    SimpleNamespace(source_language="en"),
                )
            )

        self.assertEqual([segment["text"] for segment in merged], ["Hello", "[ITALIAN]"])
        self.assertTrue(merged[1]["supplementary_source"])
        self.assertEqual(
            [asset.asset_id for asset in loaded_assets],
            [forced_asset.asset_id],
        )

    def test_ffprobe_json_preserves_dispositions_and_tags(self) -> None:
        payload = {
            "streams": [
                {
                    "index": 7,
                    "codec_type": "subtitle",
                    "codec_name": "subrip",
                    "tags": {"language": "eng", "title": "English SDH"},
                    "disposition": {
                        "default": 1,
                        "forced": 0,
                        "hearing_impaired": 1,
                    },
                    "nb_frames": "2400",
                }
            ]
        }
        result = SimpleNamespace(returncode=0, stdout=json.dumps(payload))

        with patch.object(subtitle_pipeline, "run", return_value=result):
            streams = subtitle_pipeline.probe_streams_json(
                Path("movie.mkv"),
                "ffprobe",
            )

        self.assertEqual(len(streams), 1)
        self.assertTrue(streams[0].is_default)
        self.assertTrue(streams[0].is_hearing_impaired)
        self.assertEqual(streams[0].metadata["nb_frames"], "2400")

    def test_automatic_sidecar_selection_prefers_sdh_for_missing_cues(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "film.mkv"
            video.write_bytes(b"video")
            (root / "film.zh.srt").write_text(
                "1\n00:00:01,000 --> 00:00:02,000\n你好\n",
                encoding="utf-8",
            )
            (root / "film.en.srt").write_text(
                "1\n00:00:01,000 --> 00:00:02,000\nHello\n",
                encoding="utf-8",
            )
            (root / "film.en.sdh.srt").write_text(
                "1\n00:00:01,000 --> 00:00:02,000\n(MUSIC PLAYING)\n",
                encoding="utf-8",
            )

            tracks = audio_to_subtitle.find_existing_sidecar_subtitle_tracks(
                video,
                root,
            )

        self.assertIsNotNone(tracks)
        assert tracks is not None
        self.assertEqual(tracks.english_asset.role, "sdh")
        self.assertEqual(tracks.english[0]["text"], "(MUSIC PLAYING)")

    def test_explicit_ocr_sidecar_cleans_trailing_zero_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "film.mkv"
            english = root / "film.en.srt"
            chinese = root / "film.zh.srt"
            video.write_bytes(b"video")
            english.write_text(
                "1\n00:00:01,000 --> 00:00:02,000\nHello 0\n\n"
                "2\n00:00:02,000 --> 00:00:03,000\n0\n",
                encoding="utf-8",
            )
            chinese.write_text(
                "1\n00:00:01,000 --> 00:00:02,000\n\u4f60\u597d\n",
                encoding="utf-8",
            )

            tracks = audio_to_subtitle.find_existing_sidecar_subtitle_tracks(
                video,
                root,
                explicit_zh_path=chinese,
                explicit_en_path=english,
                english_text_authority="ocr",
            )
            assert tracks is not None
            merged = audio_to_subtitle.merge_existing_subtitle_segments(
                tracks.english,
                tracks.chinese,
                tracks.source_label,
                english_asset=tracks.english_asset,
                chinese_asset=tracks.chinese_asset,
            )

        self.assertEqual(tracks.english_asset.text_authority, "ocr")
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["en"], "Hello")
        self.assertEqual(merged[0]["zh"], "\u4f60\u597d")

    def test_cross_origin_merge_adds_sidecar_source_to_embedded_chinese(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "film.mkv"
            source_path = root / "film.en.sdh.srt"
            video.write_bytes(b"video")
            source_path.write_text(
                "1\n00:00:01,000 --> 00:00:02,000\n(MUSIC PLAYING)\n",
                encoding="utf-8",
            )
            chinese_asset = build_embedded_asset(
                video,
                StreamInfo(2, "subtitle", "ass", "chi", "Simplified Chinese"),
            )
            source_asset = build_sidecar_asset(
                video,
                source_path,
                language="en",
                likely_match=True,
                score=50,
            )
            source_segments = [
                {
                    "id": 0,
                    "start": 1.0,
                    "end": 2.0,
                    "text": "(MUSIC PLAYING)",
                    "source_language": "en",
                }
            ]
            tracks = audio_to_subtitle.ExistingSubtitleTracks(
                english=None,
                chinese=[
                    {
                        "id": 0,
                        "start": 1.0,
                        "end": 2.0,
                        "text": "你好",
                    }
                ],
                source_label="embedded Chinese",
                chinese_asset=chinese_asset,
            )
            args = SimpleNamespace(subtitle_ocr_lang="auto")

            with patch.object(
                audio_to_subtitle,
                "load_best_sidecar_source_track",
                return_value=(source_segments, source_asset),
            ):
                merged = audio_to_subtitle.add_cross_origin_source_track(
                    tracks,
                    video,
                    root,
                    root / "out",
                    args,
                )

        self.assertEqual(merged.english, source_segments)
        self.assertEqual(merged.english_asset.asset_id, source_asset.asset_id)
        self.assertIn("sidecar source subtitle", merged.source_label)

    def test_mixed_origin_plan_stays_in_auto_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "film.mkv"
            source_path = root / "film.en.sdh.srt"
            video.write_bytes(b"video")
            source_path.write_text(
                "1\n00:00:01,000 --> 00:00:02,000\n(MUSIC PLAYING)\n",
                encoding="utf-8",
            )
            chinese = build_embedded_asset(
                video,
                StreamInfo(2, "subtitle", "ass", "chi", "Simplified Chinese"),
            )
            source = build_sidecar_asset(
                video,
                source_path,
                language="en",
                likely_match=True,
                score=50,
            )

            plan = build_processing_plan([chinese, source])

        self.assertEqual(plan["recommended_source"], "auto")
        self.assertEqual(plan["lanes"]["chinese"], chinese.asset_id)
        self.assertEqual(plan["lanes"]["source"], source.asset_id)

    def test_supplementary_cue_is_merged_without_duplicate_dialogue(self) -> None:
        video = Path("film.mkv")
        primary_asset = build_embedded_asset(
            video,
            StreamInfo(2, "subtitle", "ass", "eng", "English Full"),
        )
        supplementary_asset = build_embedded_asset(
            video,
            StreamInfo(
                3,
                "subtitle",
                "hdmv_pgs_subtitle",
                "eng",
                "English Forced",
                disposition={"forced": 1},
            ),
        )
        primary = [
            {
                "id": 0,
                "start": 1.0,
                "end": 2.0,
                "text": "Hello",
                "source_language": "en",
            }
        ]
        duplicate_and_cue = [
            {
                "id": 0,
                "start": 1.0,
                "end": 2.0,
                "text": "Hello",
                "source_language": "en",
            },
            {
                "id": 1,
                "start": 2.1,
                "end": 3.0,
                "text": "(MUSIC PLAYING)",
                "source_language": "en",
                "ocr_confidence": 0.68,
            },
        ]
        audio_to_subtitle.attach_asset_to_segments(
            primary,
            primary_asset,
            lane="source",
        )
        audio_to_subtitle.attach_asset_to_segments(
            duplicate_and_cue,
            supplementary_asset,
            lane="source",
        )

        combined = audio_to_subtitle.merge_supplementary_source_segments(
            primary,
            duplicate_and_cue,
        )
        merged = audio_to_subtitle.merge_existing_subtitle_segments(
            combined,
            [
                {
                    "id": 0,
                    "start": 1.0,
                    "end": 2.0,
                    "text": "你好",
                }
            ],
            "mixed existing subtitles",
            english_asset=primary_asset,
            supplementary_assets=(supplementary_asset,),
        )

        self.assertEqual(
            [segment["en"] for segment in merged],
            ["Hello", "(MUSIC PLAYING)"],
        )
        cue = merged[1]
        self.assertEqual(cue["zh"], "")
        self.assertEqual(cue["source_text_authority"], "ocr")
        self.assertTrue(cue["supplementary_source"])
        self.assertEqual(cue["ocr_confidence"], 0.68)
        dialogue_asset_ids = {
            asset["asset_id"]
            for asset in merged[0]["source_assets"]
        }
        cue_asset_ids = {
            asset["asset_id"]
            for asset in cue["source_assets"]
        }
        self.assertEqual(dialogue_asset_ids, {primary_asset.asset_id})
        self.assertEqual(cue_asset_ids, {supplementary_asset.asset_id})
        self.assertNotIn("supplementary_asset_id", merged[0])
        self.assertEqual(
            cue["supplementary_asset_id"],
            supplementary_asset.asset_id,
        )

    def test_supplementary_dialogue_does_not_replace_paired_primary_text(self) -> None:
        pairs = subtitle_pipeline.pair_events(
            [
                subtitle_pipeline.SubtitleEvent(1.0, 2.0, "Hello"),
                subtitle_pipeline.SubtitleEvent(
                    1.0,
                    2.0,
                    "Bonjour",
                    metadata={"supplementary_source": True},
                ),
            ],
            [subtitle_pipeline.SubtitleEvent(1.0, 2.0, "你好")],
        )

        primary_pair = next(pair for pair in pairs if pair[1] is not None)
        supplementary_pair = next(
            pair
            for pair in pairs
            if pair[0] is not None and pair[0].text == "Bonjour"
        )
        self.assertEqual(primary_pair[0].text, "Hello")
        self.assertIsNone(supplementary_pair[1])

    def test_existing_chinese_sdh_cue_is_paired_instead_of_retranslated(self) -> None:
        pairs = subtitle_pipeline.pair_events(
            [
                subtitle_pipeline.SubtitleEvent(1.0, 2.0, "Come on."),
                subtitle_pipeline.SubtitleEvent(
                    1.0,
                    2.0,
                    "(MUSIC PLAYING)",
                    metadata={"supplementary_source": True},
                ),
            ],
            [
                subtitle_pipeline.SubtitleEvent(1.0, 2.0, "快点。"),
                subtitle_pipeline.SubtitleEvent(1.0, 2.0, "（播放音乐）"),
            ],
        )

        paired_text = {
            pair[0].text: pair[1].text if pair[1] is not None else None
            for pair in pairs
            if pair[0] is not None
        }
        self.assertEqual(paired_text["Come on."], "快点。")
        self.assertEqual(paired_text["(MUSIC PLAYING)"], "（播放音乐）")
        self.assertFalse(subtitle_pipeline.is_sdh_sound_cue("I like music."))
        self.assertTrue(
            subtitle_pipeline.is_sdh_sound_cue("(MUSIC PLAYING)")
        )

    def test_authored_source_cannot_be_rewritten_or_hidden(self) -> None:
        source = {
            "id": 0,
            "start": 1.0,
            "end": 2.0,
            "text": "Dr. Smith is here.",
            "source_text_authority": "authored",
            "chinese_text_authority": "ocr",
        }
        proposed = audio_to_subtitle.preserve_authored_source(
            source,
            source["text"],
            "Doctor Shi is here.",
        )
        guarded = audio_to_subtitle.guard_model_hidden_segments(
            [source],
            [
                {
                    **source,
                    "en": proposed,
                    "zh": "史医生来了。",
                    "display": False,
                    "model_display_requested": False,
                }
            ],
        )

        self.assertEqual(proposed, "Dr. Smith is here.")
        self.assertTrue(guarded[0]["display"])
        self.assertEqual(
            guarded[0]["display_guard_reason"],
            "authored_source_preserved",
        )

    def test_ocr_source_may_be_corrected(self) -> None:
        source = {
            "text": "He1lo world",
            "source_text_authority": "ocr",
        }
        corrected = audio_to_subtitle.preserve_authored_source(
            source,
            source["text"],
            "Hello world",
        )
        self.assertEqual(corrected, "Hello world")

    def test_checkpoint_rejects_changed_processing_plan(self) -> None:
        current = [
            {
                "id": 0,
                "start": 1.0,
                "end": 2.0,
                "text": "Hello",
                "processing_plan_fingerprint": "plan-new",
            }
        ]
        cached = [
            {
                **current[0],
                "display": True,
                "processing_plan_fingerprint": "plan-old",
            }
        ]

        self.assertFalse(
            audio_to_subtitle.checkpoint_matches_segments(
                cached,
                current,
                expected_plan_fingerprint="plan-new",
            )
        )

    def test_existing_source_request_fingerprint_binds_inventory_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            video = Path(tmpdir) / "film.mkv"
            video.write_bytes(b"video")
            args = SimpleNamespace(source="auto")

            first = audio_to_subtitle.build_source_request_fingerprint(
                video,
                video,
                args,
                processing_plan_fingerprint="plan-a",
            )
            second = audio_to_subtitle.build_source_request_fingerprint(
                video,
                video,
                args,
                processing_plan_fingerprint="plan-b",
            )

        self.assertNotEqual(first, second)

    def test_frontend_exposes_source_plan_and_conditional_ocr_control(self) -> None:
        page = subtitle_frontend.html_page()

        self.assertIn('id="sourcePlan"', page)
        self.assertIn("function renderProcessingPlan(plan)", page)
        self.assertIn('id="subtitleOcrLangGroup"', page)
        self.assertIn("embeddedSelectionNeedsOcr()", page)
        self.assertIn("['同步音轨', sourceAssetLabel(assets.get(lane.audio_reference))]", page)
        self.assertIn("const SOURCE_SELECTION_IDS", page)
        self.assertIn("clearSourceSelections();\n    clearLastAnalysis();", page)
        persisted_fields = page.split("const INPUT_IDS = [", 1)[1].split("];", 1)[0]
        self.assertNotIn("'audioStream'", persisted_fields)
        self.assertNotIn("'subtitleStream'", persisted_fields)
        self.assertIn("评论/解说（不用于对白）", page)
        self.assertIn("`不支持编码 ${item.codec || ''}`", page)
        self.assertIn("!option.disabled && prefer(option.value)", page)
        self.assertIn("function invalidateProcessingPlanPreview()", page)
        self.assertIn(
            "no_safe_source: '没有可安全自动使用的完整对白来源'",
            page,
        )

    def test_frontend_does_not_auto_select_unrelated_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "film.mkv"
            unrelated_path = root / "different-film.en.srt"
            video.write_bytes(b"video")
            unrelated_path.write_text(
                "1\n00:00:01,000 --> 00:00:02,000\nHello\n",
                encoding="utf-8",
            )
            unrelated = build_sidecar_asset(video, unrelated_path)
            audio = build_audio_asset(video, stream_index=1, language="en")
            sidecar_item = {
                "path": str(unrelated_path),
                "likely_match": False,
                "asset": unrelated.to_dict(),
            }
            audio_item = {
                "index": 1,
                "asset": audio.to_dict(),
            }

            with (
                patch.object(
                    subtitle_frontend,
                    "default_names",
                    return_value={
                        "series_name": "Series",
                        "movie_name": "Film",
                        "name_source": "test",
                    },
                ),
                patch.object(
                    subtitle_frontend,
                    "resolve_request_output_root",
                    return_value=(root / "out", "default"),
                ),
                patch.object(
                    subtitle_frontend,
                    "checkpoint_info",
                    return_value={},
                ),
                patch.object(
                    subtitle_frontend,
                    "list_sidecar_subtitles",
                    return_value=[sidecar_item],
                ),
                patch.object(
                    subtitle_frontend,
                    "list_embedded_subtitles",
                    return_value=[],
                ),
                patch.object(
                    subtitle_frontend,
                    "list_embedded_audio",
                    return_value=[audio_item],
                ),
            ):
                info = subtitle_frontend.analyze_input(
                    {
                        "path": str(video),
                        "merge_existing_subtitles": True,
                    }
                )

        self.assertEqual(info["auto_source"], "audio")
        self.assertEqual(len(info["source_assets"]), 2)

    def test_frontend_plan_honors_explicit_source_and_stream_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "film.mkv"
            sidecar_path = root / "film.en.srt"
            video.write_bytes(b"video")
            sidecar_path.write_text(
                "1\n00:00:01,000 --> 00:00:02,000\nHello\n",
                encoding="utf-8",
            )
            sidecar = build_sidecar_asset(
                video,
                sidecar_path,
                likely_match=True,
                score=100,
            )
            embedded = build_embedded_asset(
                video,
                StreamInfo(3, "subtitle", "ass", "eng", "English Full"),
            )
            other_embedded = build_embedded_asset(
                video,
                StreamInfo(4, "subtitle", "ass", "jpn", "Japanese Full"),
            )
            chinese = build_embedded_asset(
                video,
                StreamInfo(5, "subtitle", "ass", "chi", "Chinese Full"),
            )
            audio = build_audio_asset(
                video,
                StreamInfo(2, "audio", "aac", "eng", "English"),
            )
            selected = subtitle_frontend.select_assets_for_request(
                [
                    ({"likely_match": True}, sidecar),
                    ({}, embedded),
                    ({}, other_embedded),
                    ({}, chinese),
                    ({}, audio),
                ],
                {
                    "source": "embedded",
                    "merge_existing_subtitles": False,
                    "subtitle_stream": "3",
                    "audio_stream": "2",
                    "subtitle_sync": "auto",
                },
            )
            plan = build_processing_plan(
                selected,
                allow_audio_source=False,
            )
            partial_lane_selection = subtitle_frontend.select_assets_for_request(
                [
                    ({}, embedded),
                    ({}, other_embedded),
                    ({}, chinese),
                    ({}, audio),
                ],
                {
                    "source": "embedded",
                    "merge_existing_subtitles": True,
                    "chinese_subtitle_stream": "5",
                    "audio_stream": "2",
                    "subtitle_sync": "auto",
                },
            )

        self.assertEqual(
            {asset.asset_id for asset in selected},
            {embedded.asset_id, audio.asset_id},
        )
        self.assertEqual(plan["lanes"]["source"], embedded.asset_id)
        self.assertEqual(plan["lanes"]["audio_reference"], audio.asset_id)
        self.assertEqual(
            {asset.asset_id for asset in partial_lane_selection},
            {
                embedded.asset_id,
                other_embedded.asset_id,
                chinese.asset_id,
                audio.asset_id,
            },
        )

    def test_main_auto_single_track_prefers_existing_chinese(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "film.mkv"
            video.write_bytes(b"video")
            (root / "film.en.srt").write_text(
                "1\n00:00:01,000 --> 00:00:02,000\nHello\n",
                encoding="utf-8",
            )
            (root / "film.zh.srt").write_text(
                "1\n00:00:01,000 --> 00:00:02,000\n你好\n",
                encoding="utf-8",
            )
            output_root = root / "out"
            captured = {}

            def proofread(segments, **_kwargs):
                captured["text"] = segments[0]["text"]
                return [
                    {
                        **segments[0],
                        "en": "",
                        "zh": "你好",
                        "display": True,
                    }
                ]

            argv = [
                "audio_to_subtitle.py",
                "--video",
                str(video),
                "--source",
                "auto",
                "--merge-existing-subtitles",
                "no",
                "--subtitle-sync",
                "off",
                "--output-root",
                str(output_root),
                "--series-name",
                "Series",
                "--movie-name",
                "Film",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    audio_to_subtitle,
                    "extract_audio",
                    side_effect=AssertionError("extract_audio called"),
                ),
                patch.object(
                    audio_to_subtitle,
                    "transcribe_audio",
                    side_effect=AssertionError("transcribe_audio called"),
                ),
                patch.object(
                    audio_to_subtitle,
                    "translate_and_correct_segments",
                    side_effect=AssertionError("translation called"),
                ),
                patch.object(
                    audio_to_subtitle,
                    "proofread_existing_chinese_segments",
                    side_effect=proofread,
                ),
            ):
                audio_to_subtitle.main()

        self.assertEqual(captured["text"], "你好")

    def test_main_auto_audio_uses_planned_non_commentary_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "film.mkv"
            video.write_bytes(b"video")
            output_root = root / "out"
            commentary = build_audio_asset(
                video,
                StreamInfo(
                    0,
                    "audio",
                    "aac",
                    "eng",
                    "Director Commentary",
                    disposition={"default": 1},
                ),
            )
            english = build_audio_asset(
                video,
                StreamInfo(
                    2,
                    "audio",
                    "aac",
                    "eng",
                    "English",
                    disposition={"default": 1},
                ),
            )
            captured = {}

            def extract_audio(_video, _output, audio_stream=None):
                captured["audio_stream"] = audio_stream

            argv = [
                "audio_to_subtitle.py",
                "--video",
                str(video),
                "--source",
                "auto",
                "--output-root",
                str(output_root),
                "--series-name",
                "Series",
                "--movie-name",
                "Film",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    audio_to_subtitle,
                    "discover_automatic_source_assets",
                    return_value=[commentary, english],
                ),
                patch.object(
                    audio_to_subtitle,
                    "extract_audio",
                    side_effect=extract_audio,
                ),
                patch.object(
                    audio_to_subtitle,
                    "transcribe_audio",
                    return_value=[
                        {
                            "id": 0,
                            "start": 1.0,
                            "end": 2.0,
                            "text": "Hello",
                            "source_language": "en",
                        }
                    ],
                ),
                patch.object(
                    audio_to_subtitle,
                    "translate_and_correct_segments",
                    side_effect=lambda segments, **_kwargs: segments,
                ),
            ):
                audio_to_subtitle.main()

        self.assertEqual(captured["audio_stream"], 2)

    def test_main_auto_refuses_commentary_only_audio_for_asr(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "film.mkv"
            video.write_bytes(b"video")
            commentary = build_audio_asset(
                video,
                StreamInfo(
                    0,
                    "audio",
                    "aac",
                    "eng",
                    "Director Commentary",
                    disposition={"default": 1},
                ),
            )
            argv = [
                "audio_to_subtitle.py",
                "--video",
                str(video),
                "--source",
                "auto",
                "--output-root",
                str(root / "out"),
                "--series-name",
                "Series",
                "--movie-name",
                "Film",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    audio_to_subtitle,
                    "discover_automatic_source_assets",
                    return_value=[commentary],
                ),
                patch.object(
                    audio_to_subtitle,
                    "find_existing_sidecar_subtitle_tracks",
                    return_value=None,
                ),
                patch.object(
                    audio_to_subtitle,
                    "find_existing_embedded_subtitle_tracks",
                    return_value=None,
                ),
                patch.object(
                    audio_to_subtitle,
                    "load_embedded_subtitle_events",
                    side_effect=RuntimeError("no embedded subtitle"),
                ),
                patch.object(
                    audio_to_subtitle,
                    "extract_audio",
                    side_effect=AssertionError("unsafe audio extraction"),
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "commentary or audio-description",
                ):
                    audio_to_subtitle.main()


if __name__ == "__main__":
    unittest.main()
