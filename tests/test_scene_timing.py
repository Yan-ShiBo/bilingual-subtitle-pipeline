import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scene_timing import align_subtitles_to_scene_cuts, parse_scene_times  # noqa: E402


class SceneTimingTests(unittest.TestCase):
    def test_parse_scene_times_deduplicates_showinfo_output(self) -> None:
        output = """
        [Parsed_showinfo_3] n: 1 pts: 600 pts_time:10.000
        [Parsed_showinfo_3] n: 2 pts: 780 pts_time:13.000
        [Parsed_showinfo_3] n: 2 pts: 780 pts_time:13.000
        """

        self.assertEqual(parse_scene_times(output), [10.0, 13.0])

    def test_scene_alignment_moves_edges_without_delaying_dialogue(self) -> None:
        segments = [
            {
                "id": 0,
                "start": 10.12,
                "end": 12.82,
                "text": "Line",
                "words": [
                    {"start": 10.12, "end": 12.7, "word": "Line"},
                ],
            }
        ]

        aligned = align_subtitles_to_scene_cuts(segments, [10.0, 13.0])

        self.assertEqual(aligned[0]["start"], 10.0)
        self.assertEqual(aligned[0]["end"], 13.0)
        self.assertIn("start_to_previous_cut", aligned[0]["scene_timing_adjustments"])
        self.assertIn("end_to_next_cut", aligned[0]["scene_timing_adjustments"])

    def test_scene_alignment_never_cuts_off_a_word_anchor(self) -> None:
        segments = [
            {
                "id": 0,
                "start": 20.0,
                "end": 22.15,
                "text": "Line",
                "words": [
                    {"start": 20.0, "end": 22.1, "word": "Line"},
                ],
            }
        ]

        aligned = align_subtitles_to_scene_cuts(segments, [22.0])

        self.assertGreaterEqual(aligned[0]["end"], 22.1)


if __name__ == "__main__":
    unittest.main()
