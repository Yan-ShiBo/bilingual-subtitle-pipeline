from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import remote_ollama_bridge  # noqa: E402
import review_evidence  # noqa: E402
import subtitle_frontend  # noqa: E402
import subtitle_queue  # noqa: E402
import subtitle_queue_worker  # noqa: E402


class PersistentQueueTests(unittest.TestCase):
    def test_episode_discovery_uses_natural_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in ("Episode 10.mkv", "Episode 2.mkv", "Episode 1.mkv"):
                (root / name).write_bytes(b"video")
            (root / "notes.txt").write_text("ignore", encoding="utf-8")

            episodes = subtitle_queue.discover_episode_files(root)

        self.assertEqual(
            [item["name"] for item in episodes],
            ["Episode 1.mkv", "Episode 2.mkv", "Episode 10.mkv"],
        )

    def test_queue_persists_deduplicates_and_retries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "Episode 1.mkv"
            video.write_bytes(b"video")
            store = subtitle_queue.SubtitleQueueStore(
                root / "queue.json",
                root / "queue.lock",
            )

            added = store.enqueue(
                [str(video), str(video)],
                {
                    "path": "old.mkv",
                    "movie_name": "Old",
                    "series_name": "Series",
                    "subtitle_file": "old.srt",
                    "llm_model": "remote:AI:qwen3:30b",
                },
            )
            self.assertEqual(len(added["added_ids"]), 1)
            item = store.snapshot()["items"][0]
            self.assertEqual(item["payload"]["path"], str(video.resolve()))
            self.assertEqual(item["payload"]["movie_name"], "")
            self.assertEqual(item["payload"]["subtitle_file"], "")
            self.assertEqual(item["payload"]["run_mode"], "resume")

            store.set_paused(False)
            claimed = store.claim_next(4321)
            self.assertIsNotNone(claimed)
            self.assertEqual(store.snapshot()["items"][0]["status"], "running")
            store.update_item(claimed["id"], status="failed", error="network")
            store.action("retry", claimed["id"])

            reloaded = subtitle_queue.SubtitleQueueStore(
                root / "queue.json",
                root / "queue.lock",
            ).snapshot()
            self.assertFalse(reloaded["paused"])
            self.assertEqual(reloaded["items"][0]["status"], "queued")
            self.assertEqual(reloaded["items"][0]["error"], "")

    def test_worker_prepares_per_episode_asset_selection(self) -> None:
        analysis = {
            "output_root": "C:/output",
            "series_name": "Detected Series",
            "movie_name": "Episode 02",
            "processing_plan": {
                "lanes": {
                    "chinese": "zh",
                    "source": "en",
                    "audio_reference": "audio",
                },
                "selected_assets": [
                    {
                        "asset_id": "zh",
                        "origin": "embedded",
                        "stream_index": 4,
                    },
                    {
                        "asset_id": "en",
                        "origin": "sidecar",
                        "path": "C:/show/Episode 02.en.srt",
                    },
                    {
                        "asset_id": "audio",
                        "origin": "audio",
                        "stream_index": 2,
                    },
                ],
            },
        }
        item = {
            "path": "C:/show/Episode 02.mkv",
            "payload": {
                "series_name": "My Series",
                "merge_existing_subtitles": True,
            },
        }
        with patch.object(subtitle_frontend, "analyze_input", return_value=analysis):
            payload, returned_analysis = subtitle_queue_worker.prepare_payload(item)

        self.assertIs(returned_analysis, analysis)
        self.assertEqual(payload["series_name"], "My Series")
        self.assertEqual(payload["movie_name"], "Episode 02")
        self.assertEqual(payload["chinese_subtitle_stream"], 4)
        self.assertEqual(payload["english_subtitle_file"], "C:/show/Episode 02.en.srt")
        self.assertEqual(payload["audio_stream"], 2)

    def test_worker_distinguishes_reported_issues_from_review_samples(self) -> None:
        with patch.object(subtitle_queue_worker, "queue_store") as store:
            subtitle_queue_worker._finish_item(
                "episode-1",
                {
                    "outcome": "success",
                    "quality_status": "review",
                    "quality_review_reported_count": 195,
                    "quality_review_pending_count": 19,
                    "quality_review_items": [{"check": "readability"}],
                },
            )

        updates = store.update_item.call_args.kwargs
        self.assertEqual(updates["quality_reported_count"], 195)
        self.assertEqual(updates["review_pending_count"], 19)
        self.assertEqual(
            updates["stage"],
            "已完成，质检发现 195 项（19 个代表样本待复核）",
        )

    def test_worker_reports_when_all_saved_samples_were_reviewed(self) -> None:
        with patch.object(subtitle_queue_worker, "queue_store") as store:
            subtitle_queue_worker._finish_item(
                "episode-8",
                {
                    "outcome": "success",
                    "quality_status": "review",
                    "quality_review_reported_count": 174,
                    "quality_review_reviewed_sample_count": 25,
                    "quality_review_pending_count": 0,
                },
            )

        updates = store.update_item.call_args.kwargs
        self.assertEqual(updates["quality_reviewed_sample_count"], 25)
        self.assertEqual(
            updates["stage"],
            "已完成，质检 174 项（25 个代表样本已复核）",
        )


class RemoteBridgeTests(unittest.TestCase):
    def test_remote_fingerprint_records_availability_not_password(self) -> None:
        base = {
            "host": "ai-server",
            "port": 22,
            "user": "csynth",
            "auth_method": "password",
            "remember_password": True,
        }
        first = remote_ollama_bridge.remote_config_fingerprint({**base, "password": "one"})
        second = remote_ollama_bridge.remote_config_fingerprint({**base, "password": "two"})
        missing = remote_ollama_bridge.remote_config_fingerprint({**base, "password": ""})

        self.assertEqual(first, second)
        self.assertNotEqual(first, missing)
        self.assertNotIn("one", first)

    def test_frontend_remote_restart_requires_key_or_saved_password(self) -> None:
        self.assertTrue(
            subtitle_frontend.remote_settings_support_restart(
                {"host": "ai-server", "auth_method": "key"}
            )
        )
        self.assertTrue(
            subtitle_frontend.remote_settings_support_restart(
                {
                    "host": "ai-server",
                    "auth_method": "password",
                    "remember_password": True,
                    "password": "secret",
                }
            )
        )
        self.assertFalse(
            subtitle_frontend.remote_settings_support_restart(
                {
                    "host": "ai-server",
                    "auth_method": "password",
                    "remember_password": False,
                    "password": "secret",
                }
            )
        )

    def test_bridge_monitor_retries_after_initial_connection_failure(self) -> None:
        class FakeTunnel:
            def __init__(self) -> None:
                self.attempts = 0
                self.connected = False
                self.last_error = ""

            def connect(self, *_args: object) -> bool:
                self.attempts += 1
                self.connected = self.attempts >= 2
                self.last_error = "temporary network failure" if not self.connected else ""
                return self.connected

            def disconnect(self) -> None:
                self.connected = False

            def fetch_models(self) -> list[str]:
                return ["qwen3:30b"] if self.connected else []

            def status(self) -> dict[str, object]:
                return {
                    "connected": self.connected,
                    "local_port": 43210 if self.connected else None,
                    "error": self.last_error,
                }

        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = remote_ollama_bridge.BridgeRuntime(
                11436,
                Path(tmpdir) / "status.json",
            )
            fake_tunnel = FakeTunnel()
            runtime.tunnel = fake_tunnel
            with patch.object(
                remote_ollama_bridge.settings_store,
                "load",
                return_value={
                    "remote": {
                        "host": "ai-server",
                        "port": 22,
                        "user": "csynth",
                        "auth_method": "key",
                    }
                },
            ):
                runtime.start()
                deadline = time.monotonic() + 2.5
                while fake_tunnel.attempts < 2 and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertGreaterEqual(fake_tunnel.attempts, 2)
                self.assertTrue(runtime.status()["connected"])
                self.assertEqual(runtime.models, ["qwen3:30b"])
                runtime.stop()


class ReviewEvidenceTests(unittest.TestCase):
    def test_generates_cached_evidence_and_neighbor_context(self) -> None:
        segments = [
            {"id": index, "start": index * 2.0, "end": index * 2.0 + 1.0, "en": f"en {index}", "zh": f"zh {index}"}
            for index in range(5)
        ]

        def fake_ffmpeg(_command: list[str], expected: Path) -> None:
            expected.write_bytes(b"media")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "movie.mkv"
            video.write_bytes(b"video")
            evidence_root = root / "evidence"
            with (
                patch.object(review_evidence, "find_ffmpeg", return_value="ffmpeg"),
                patch.object(review_evidence, "_run_ffmpeg", side_effect=fake_ffmpeg) as runner,
            ):
                first = review_evidence.generate_review_evidence(
                    video,
                    segments,
                    2,
                    evidence_root,
                    audio_stream=3,
                )
                second = review_evidence.generate_review_evidence(
                    video,
                    segments,
                    2,
                    evidence_root,
                    audio_stream=3,
                )

            self.assertEqual(runner.call_count, 3)
            self.assertEqual(first["cache_key"], second["cache_key"])
            self.assertEqual(len(first["neighbors"]), 5)
            self.assertTrue(first["neighbors"][2]["is_current"])
            self.assertEqual(first["event_offset"], 4.0)
            self.assertTrue(Path(first["frame_path"]).is_file())
            manifest = json.loads(
                (Path(first["frame_path"]).parent / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["audio_stream"], 3)

    def test_unknown_checkpoint_segment_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "movie.mkv"
            video.write_bytes(b"video")
            with self.assertRaises(KeyError):
                review_evidence.generate_review_evidence(
                    video,
                    [{"id": 1, "start": 0, "end": 1}],
                    99,
                    root / "evidence",
                    ffmpeg_path="ffmpeg",
                )

    def test_review_media_registry_does_not_accept_paths_as_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            media = Path(tmpdir) / "frame.jpg"
            media.write_bytes(b"image")
            url = subtitle_frontend.register_review_media(media)
            token = url.rsplit("/", 1)[-1]

            self.assertEqual(subtitle_frontend.review_media_path(token), media.resolve())
            self.assertIsNone(subtitle_frontend.review_media_path(str(media)))


if __name__ == "__main__":
    unittest.main()
