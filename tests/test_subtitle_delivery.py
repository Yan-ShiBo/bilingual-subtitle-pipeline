import tempfile
import unittest
import shutil
from pathlib import Path
from unittest.mock import patch

from PIL import Image


import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import subtitle_delivery  # noqa: E402


class SubtitleDeliveryTests(unittest.TestCase):
    def test_delivery_writes_single_and_bilingual_srt(self) -> None:
        segments = [
            {
                "id": 0,
                "start": 1.234,
                "end": 3.456,
                "en": "Hello",
                "zh": "\u4f60\u597d",
                "display": True,
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = subtitle_delivery.write_delivery_srts(
                segments,
                Path(tmpdir),
                "Movie",
            )
            bilingual = Path(paths["bilingual_srt"]).read_text(encoding="utf-8")
            english = Path(paths["english_srt"]).read_text(encoding="utf-8")

        self.assertIn("00:00:01,234 --> 00:00:03,456", bilingual)
        self.assertIn("\u4f60\u597d\nHello", bilingual)
        self.assertNotIn("\u4f60\u597d", english)

    def test_delivery_does_not_duplicate_chinese_only_source_as_english(self) -> None:
        segments = [
            {
                "id": 1,
                "start": 4.0,
                "end": 6.0,
                "text": "\u6e90\u7e41\u4f53\u4e2d\u6587",
                "en": "",
                "zh": "\u7b80\u4f53\u4e2d\u6587",
                "display": True,
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = subtitle_delivery.write_delivery_srts(
                segments,
                Path(tmpdir),
                "Movie",
            )
            bilingual = Path(paths["bilingual_srt"]).read_text(encoding="utf-8")
            english = Path(paths["english_srt"]).read_text(encoding="utf-8")

        self.assertEqual(english, "")
        self.assertIn("\u7b80\u4f53\u4e2d\u6587", bilingual)
        self.assertNotIn("\u6e90\u7e41\u4f53\u4e2d\u6587", bilingual)
        self.assertEqual(bilingual.count("\u7b80\u4f53\u4e2d\u6587"), 1)

    def test_ass_render_validation_checks_real_pixels(self) -> None:
        segments = [
            {
                "id": 2,
                "checkpoint_segment_id": 9,
                "start": 10.0,
                "end": 12.0,
                "en": "Hello",
                "zh": "\u4f60\u597d",
            }
        ]
        calls = []

        def fake_run(command, **kwargs):
            calls.append((command, kwargs))
            preview = Path(command[-1])
            if not preview.is_absolute():
                preview = Path(kwargs["cwd"]) / preview
            image = Image.new("RGB", (320, 180), (16, 18, 22))
            for x in range(100, 140):
                for y in range(130, 150):
                    image.putpixel((x, y), (255, 255, 255))
            image.save(preview)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ass_path = root / "Movie.ass"
            ass_path.write_text("[Script Info]\n", encoding="utf-8")
            preview_path = root / "preview.png"
            with patch.object(subtitle_delivery.subprocess, "run", side_effect=fake_run):
                report = subtitle_delivery.validate_ass_rendering(
                    ass_path,
                    segments,
                    preview_path,
                    play_resolution=(320, 180),
                    ffmpeg_path="ffmpeg",
                )

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["checkpoint_segment_id"], 9)
        self.assertGreater(report["changed_pixels"], 32)
        command, kwargs = calls[0]
        video_filter = command[command.index("-vf") + 1]
        self.assertTrue(video_filter.startswith("settb=AVTB,setpts="))
        self.assertRegex(
            video_filter,
            r"ass=filename='subtitle-render-[^']+\.ass'",
        )
        self.assertNotIn(str(ass_path.resolve()), video_filter)
        self.assertEqual(Path(kwargs["cwd"]), root.resolve())

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required")
    def test_ass_render_validation_preserves_fractional_sample_time(self) -> None:
        segments = [
            {
                "id": 0,
                "checkpoint_segment_id": 0,
                "start": 0.1,
                "end": 1.899,
                "en": "Tonight",
                "zh": "\u4eca\u665a",
            }
        ]
        ass = """[Script Info]
ScriptType: v4.00+
PlayResX: 320
PlayResY: 180

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,24,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,12,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.10,0:00:01.90,Default,,0,0,0,,Tonight
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ass_path = root / "Movie.ass"
            ass_path.write_text(ass, encoding="utf-8")
            preview_path = root / "preview.png"
            report = subtitle_delivery.validate_ass_rendering(
                ass_path,
                segments,
                preview_path,
                play_resolution=(320, 180),
            )

        self.assertEqual(report["status"], "pass")
        self.assertGreater(report["changed_pixels"], 32)


if __name__ == "__main__":
    unittest.main()
