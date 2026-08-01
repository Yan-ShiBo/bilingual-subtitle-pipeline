from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import run_estimator  # noqa: E402
import task_artifacts  # noqa: E402


class RunEstimatorTests(unittest.TestCase):
    def test_fresh_two_round_review_counts_each_model_stage(self) -> None:
        estimate = run_estimator.estimate_run_workload(
            total_segments=280,
            completed_segments=0,
            video_duration_seconds=2_100,
            batch_size=5,
            autonomous_review_rounds=2,
            llm_model="remote:AI Server:qwen3.5:122b",
            run_mode="resume",
            operations=["translate_source"],
        )

        self.assertTrue(estimate["available"])
        self.assertEqual(estimate["calls"]["primary"], 56)
        self.assertEqual(estimate["calls"]["final_review"], 28)
        self.assertEqual(estimate["calls"]["autonomous_review"], 56)
        self.assertEqual(estimate["total_calls"], 142)
        self.assertGreater(estimate["max_minutes"], estimate["min_minutes"])

    def test_completed_cached_resume_needs_no_more_model_calls(self) -> None:
        estimate = run_estimator.estimate_run_workload(
            total_segments=280,
            completed_segments=280,
            batch_size=5,
            autonomous_review_rounds=2,
            run_mode="resume",
            artifact_cache={
                "style_guide_exists": True,
                "terminology_review_exists": True,
                "final_qa_exists": True,
                "autonomous_rounds_completed": 2,
            },
            checkpoint_compatible=True,
        )

        self.assertEqual(estimate["total_calls"], 0)
        self.assertEqual(estimate["min_minutes"], 2)
        self.assertEqual(estimate["max_minutes"], 8)

    def test_duration_provides_a_clearly_marked_rough_segment_basis(self) -> None:
        estimate = run_estimator.estimate_run_workload(
            video_duration_seconds=3_400,
            batch_size=10,
        )

        self.assertEqual(estimate["segment_basis"], "duration")
        self.assertEqual(estimate["segment_count"], 1_000)
        self.assertIn("segment_count_estimated_from_duration", estimate["notes"])


class TaskArtifactTests(unittest.TestCase):
    def _create_task(self, root: Path, series: str, movie: str) -> Path:
        out_dir = root / series / movie
        out_dir.mkdir(parents=True)
        bilingual = out_dir / f"{movie}.bilingual.ass"
        bilingual.write_text("subtitle", encoding="utf-8")
        report = out_dir / f"{movie}.quality-report.json"
        report.write_text(
            json.dumps(
                {
                    "status": "review",
                    "summary": {"review_items": 3},
                    "artifacts": {"bilingual_ass": str(bilingual)},
                }
            ),
            encoding="utf-8",
        )
        return out_dir

    def test_artifact_manifest_ignores_paths_outside_task_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_dir = self._create_task(root, "Series", "Movie")
            outside = root / "outside.ass"
            outside.write_text("private", encoding="utf-8")
            report = out_dir / "Movie.quality-report.json"
            payload = json.loads(report.read_text(encoding="utf-8"))
            payload["artifacts"]["english_ass"] = str(outside)
            report.write_text(json.dumps(payload), encoding="utf-8")

            artifacts = task_artifacts.collect_task_artifacts(out_dir, "Movie")

        self.assertEqual([item["key"] for item in artifacts], ["bilingual_ass", "quality_report"])

    def test_recent_tasks_are_recovered_from_real_output_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._create_task(root, "Series", "Episode 1")

            tasks = task_artifacts.discover_recent_tasks(root)

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["series_name"], "Series")
        self.assertEqual(tasks[0]["movie_name"], "Episode 1")
        self.assertEqual(tasks[0]["status"], "review")
        self.assertEqual(tasks[0]["review_items"], 3)
        self.assertEqual(tasks[0]["artifact_count"], 2)
        self.assertNotIn("artifacts", tasks[0])

    def test_output_path_cannot_escape_selected_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaises(ValueError):
                task_artifacts.task_output_dir(root, "..", "Movie")
            with self.assertRaises(ValueError):
                task_artifacts.task_output_dir(root, "Series", "../Movie")

    def test_empty_directory_is_not_treated_as_a_known_task_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "Series" / "Unrelated").mkdir(parents=True)

            with self.assertRaises(FileNotFoundError):
                task_artifacts.resolve_known_target(
                    root,
                    "Series",
                    "Unrelated",
                    "output_dir",
                )

    def test_file_manager_only_receives_resolved_known_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_dir = self._create_task(root, "Series", "Movie")
            target = task_artifacts.resolve_known_target(
                root,
                "Series",
                "Movie",
                "bilingual_ass",
            )
            with patch.object(task_artifacts.subprocess, "Popen") as popen:
                task_artifacts.open_in_file_manager(target)

        self.assertEqual(target, out_dir / "Movie.bilingual.ass")
        popen.assert_called_once()


if __name__ == "__main__":
    unittest.main()
