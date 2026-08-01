import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import subtitle_frontend  # noqa: E402
from ass_styles import (  # noqa: E402
    ass_style_header,
    calculate_play_resolution,
    probe_video_play_resolution,
)
from audio_to_subtitle import generate_ass  # noqa: E402
from subtitle_pipeline import SubtitleEvent, write_bilingual_ass  # noqa: E402


class AssStyleTests(unittest.TestCase):
    def test_play_resolution_uses_display_aspect_ratio(self) -> None:
        self.assertEqual(
            calculate_play_resolution(640, 352, display_aspect_ratio=20 / 11),
            (1964, 1080),
        )
        self.assertEqual(
            calculate_play_resolution(720, 576, sample_aspect_ratio=4 / 3),
            (1800, 1080),
        )

    def test_play_resolution_honors_rotation(self) -> None:
        self.assertEqual(
            calculate_play_resolution(1920, 1080, rotation=90),
            (608, 1080),
        )

    def test_probe_video_resolution_reads_sar_dar_and_rotation(self) -> None:
        payload = {
            "streams": [
                {
                    "width": 1920,
                    "height": 1080,
                    "sample_aspect_ratio": "1:1",
                    "display_aspect_ratio": "16:9",
                    "side_data_list": [{"rotation": -90}],
                }
            ]
        }

        with patch("ass_styles.subprocess.run") as run_mock:
            run_mock.return_value.stdout = json.dumps(payload)
            result = probe_video_play_resolution(Path("movie.mp4"), ffprobe_path="ffprobe")

        self.assertEqual(result, (608, 1080))
        command = run_mock.call_args.args[0]
        self.assertIn("display_aspect_ratio", command[command.index("-show_entries") + 1])

    def test_profiles_scale_font_and_safe_margins(self) -> None:
        adaptive = ass_style_header(1920, 1080)
        mobile = ass_style_header(
            1920,
            1080,
            profile_name="mobile",
            font_name="Noto Sans CJK SC",
            font_scale=125,
        )

        self.assertIn("Style: Chinese,Arial,52,", adaptive)
        self.assertIn("Style: Source,Arial,38,", adaptive)
        self.assertIn("Style: Chinese,Noto Sans CJK SC,75,", mobile)
        self.assertIn("Style: Source,Noto Sans CJK SC,54,", mobile)
        self.assertIn("ScaledBorderAndShadow: yes", mobile)
        self.assertIn("PlayResX: 1920", mobile)

    def test_font_name_is_sanitized_for_ass_csv(self) -> None:
        header = ass_style_header(font_name="Custom,\nFont")

        self.assertIn("Style: Chinese,Custom Font,", header)
        self.assertNotIn("Style: Chinese,Custom,", header)

    def test_generate_ass_uses_role_styles_without_leading_blank_line(self) -> None:
        segments = [
            {
                "id": 0,
                "start": 1.0,
                "end": 2.0,
                "text": "[MUSIC]",
                "en": "[MUSIC]",
                "zh": "",
                "display": True,
            },
            {
                "id": 1,
                "start": 2.0,
                "end": 3.0,
                "text": "Hello",
                "en": "Hello",
                "zh": "\u4f60\u597d",
                "display": True,
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "sample.ass"
            generate_ass(
                segments,
                output,
                "bilingual",
                play_resolution=(1964, 1080),
                style_profile="mobile",
            )
            body = output.read_text(encoding="utf-8")

        self.assertIn("PlayResX: 1964", body)
        self.assertIn("Dialogue: 0,0:00:01.00,0:00:02.00,SourceOnly,,0,0,0,,[MUSIC]", body)
        self.assertNotIn("\\N[MUSIC]", body)
        self.assertIn("{\\rChinese}\u4f60\u597d\\N{\\rSource}Hello", body)

    def test_legacy_pipeline_uses_shared_ass_roles(self) -> None:
        pairs = [
            (
                SubtitleEvent(1.0, 2.0, "Hello"),
                SubtitleEvent(1.0, 2.0, "\u4f60\u597d"),
            )
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "sample.ass"
            write_bilingual_ass(pairs, output, style_profile="compact")
            body = output.read_text(encoding="utf-8-sig")

        self.assertIn("Style: Chinese,Arial,46,", body)
        self.assertIn("{\\rChinese}\u4f60\u597d\\N{\\rSource}Hello", body)
        self.assertNotIn("{\\fs34}", body)

    def test_frontend_exposes_layered_ass_appearance_options(self) -> None:
        page = subtitle_frontend.html_page()

        self.assertIn('<details class="advanced" id="subtitleAppearance">', page)
        self.assertIn('<option value="adaptive" selected>自适应画面</option>', page)
        self.assertIn('<option value="mobile">手机大字</option>', page)
        self.assertIn("外置 ASS 不会自动携带字体文件", page)
        self.assertIn("subtitle_style_profile:", page)
        self.assertIn("subtitle_font_name:", page)
        self.assertIn("subtitle_font_scale:", page)
        self.assertIn("subtitle_font_file:", page)

    def test_frontend_passes_ass_appearance_to_child_process(self) -> None:
        class FakeProcess:
            pid = 43199

            @staticmethod
            def poll():
                return None

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "movie.mp4"
            video.write_bytes(b"video")
            captured = {}

            def popen(args, **kwargs):
                captured["args"] = args
                return FakeProcess()

            payload = {
                "path": str(video),
                "output_root": str(root / "out"),
                "series_name": "Series",
                "movie_name": "Episode",
                "source": "audio",
                "subtitle_style_profile": "mobile",
                "subtitle_font_name": "Noto Sans CJK SC",
                "subtitle_font_scale": 125,
                "subtitle_font_file": str(root / "NotoSansCJKSC.ttf"),
                "subtitle_sync": "detect",
            }
            subtitle_frontend.RUNS.clear()
            try:
                with (
                    patch.object(
                        subtitle_frontend,
                        "frontend_log_paths",
                        return_value=(root / "run.log", root / "run.err.log"),
                    ),
                    patch.object(subtitle_frontend.subprocess, "Popen", side_effect=popen),
                ):
                    subtitle_frontend.start_processing(payload)
            finally:
                subtitle_frontend.RUNS.clear()

        args = captured["args"]
        self.assertEqual(args[args.index("--subtitle-style-profile") + 1], "mobile")
        self.assertEqual(args[args.index("--subtitle-font-name") + 1], "Noto Sans CJK SC")
        self.assertEqual(args[args.index("--subtitle-font-scale") + 1], "125")
        self.assertEqual(args[args.index("--subtitle-font-file") + 1], str(root / "NotoSansCJKSC.ttf"))
        self.assertEqual(args[args.index("--subtitle-sync") + 1], "detect")


if __name__ == "__main__":
    unittest.main()
