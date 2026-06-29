import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import audio_to_subtitle  # noqa: E402
from audio_to_subtitle import (  # noqa: E402
    apply_display_timing,
    checkpoint_matches_segments,
    extend_display_over_hidden_segments,
    generate_ass,
    proofread_existing_chinese_segments,
    translate_and_correct_segments,
)


class DisplayCleanupTests(unittest.TestCase):
    def test_hidden_duplicate_extends_previous_visible_subtitle(self) -> None:
        segments = [
            {"id": 0, "start": 10.0, "end": 11.0, "text": "hello", "en": "hello", "zh": "hello zh"},
            {"id": 1, "start": 11.0, "end": 12.0, "text": "hello", "en": "hello", "zh": "hello zh", "display": False},
            {"id": 2, "start": 12.0, "end": 13.0, "text": "hello", "en": "hello", "zh": "hello zh", "display": False},
        ]

        extend_display_over_hidden_segments(segments)

        self.assertEqual(segments[0]["display_end"], 13.0)

    def test_apply_display_timing_preserves_source_anchors(self) -> None:
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
        self.assertEqual(output["display_end"], 24.0)
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
            self.assertIn("Hello world", body)
            self.assertIn("\u7e41\u4f53\u5b57\u5e55", body)
            self.assertEqual(source_cache[0]["zh"], "\u7e41\u4f53\u5b57\u5e55")

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
            self.assertIn("Translate from English into corrected_chinese only when the original Chinese is empty", prompt)
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

    def test_frontend_preview_does_not_fallback_empty_english_to_source_text(self) -> None:
        frontend_source = (ROOT / "src" / "subtitle_frontend.py").read_text(encoding="utf-8")

        self.assertIn("function previewEnglishText(item)", frontend_source)
        self.assertIn("hasCjk(value) ? '' : value", frontend_source)
        self.assertIn("previewEnglishText(item)", frontend_source)
        self.assertNotIn("item.en || item.text", frontend_source)


if __name__ == "__main__":
    unittest.main()
