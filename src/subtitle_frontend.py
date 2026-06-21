import argparse
import json
import os
import re
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MOVIE_ROOT = PROJECT_ROOT.parent
RUNTIME_DIR = PROJECT_ROOT / "runtime"
LOG_DIR = RUNTIME_DIR / "logs"
DEFAULT_OUTPUT_ROOT = MOVIE_ROOT / ("1 " + "\u5b57\u5e55")
VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".m2ts", ".ts", ".mov", ".wmv"}
RUNS: Dict[int, subprocess.Popen] = {}


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def clean_name(name: str) -> str:
    name = Path(name).stem
    name = re.sub(r"\[[^\]]+\]", " ", name)
    name = re.sub(r"\(([^)]*)\)", r" \1 ", name)
    name = name.replace(".", " ").replace("_", " ")
    name = re.sub(r"\b(1080p|2160p|BluRay|REMUX|WEB-DL|HDR10|DV|HEVC|AVC|DTS|Atmos|TrueHD|x265|x264)\b", " ", name, flags=re.I)
    name = normalize_space(name)
    return name or "Unknown"


def safe_file_part(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return value or "subtitle"


def frontend_log_paths(series_name: str, movie_name: str) -> tuple[Path, Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    base = f"{safe_file_part(series_name)}__{safe_file_part(movie_name)}"
    return LOG_DIR / f"{base}.stdout.log", LOG_DIR / f"{base}.stderr.log"


def find_main_video_in_folder(folder_path: Path) -> Path:
    stream_dir = folder_path / "BDMV" / "STREAM"
    search_roots = [stream_dir] if stream_dir.exists() else [folder_path]
    candidates: List[Path] = []

    for root in search_roots:
        for candidate in root.rglob("*"):
            if candidate.is_file() and candidate.suffix.lower() in VIDEO_EXTENSIONS:
                candidates.append(candidate)

    if not candidates and stream_dir.exists():
        for candidate in folder_path.rglob("*"):
            if candidate.is_file() and candidate.suffix.lower() in VIDEO_EXTENSIONS:
                candidates.append(candidate)

    if not candidates:
        raise FileNotFoundError(f"No supported video file found in {folder_path}")

    return max(candidates, key=lambda path: path.stat().st_size)


def resolve_video_path(input_path: Path) -> Path:
    if input_path.is_dir():
        return find_main_video_in_folder(input_path)
    if input_path.is_file() and input_path.suffix.lower() in VIDEO_EXTENSIONS:
        return input_path
    raise FileNotFoundError(f"Unsupported video input: {input_path}")


def default_names(input_path: Path, video_path: Path) -> Dict[str, str]:
    if input_path.is_dir():
        series_name = clean_name(input_path.name)
        movie_name = clean_name(video_path.name)
    else:
        movie_name = clean_name(video_path.name)
        series_name = movie_name
    return {"series_name": series_name, "movie_name": movie_name}


def output_dir(output_root: Path, series_name: str, movie_name: str) -> Path:
    return output_root / series_name / movie_name


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_info(output_root: Path, series_name: str, movie_name: str) -> Dict[str, Any]:
    out_dir = output_dir(output_root, series_name, movie_name)
    checkpoint = out_dir / f"{movie_name}.segments.checkpoint.json"
    source = out_dir / f"{movie_name}.segments.source.json"
    info: Dict[str, Any] = {
        "output_dir": str(out_dir),
        "checkpoint_path": str(checkpoint),
        "source_segments_path": str(source),
        "checkpoint_exists": checkpoint.exists(),
        "source_segments_exists": source.exists(),
        "completed_count": 0,
        "total_count": None,
        "last_item": None,
        "preview": [],
    }

    if source.exists():
        try:
            source_items = read_json(source)
            if isinstance(source_items, list):
                info["total_count"] = len(source_items)
        except Exception as exc:
            info["source_error"] = str(exc)

    if checkpoint.exists():
        try:
            items = read_json(checkpoint)
            if isinstance(items, list):
                info["completed_count"] = len(items)
                if items:
                    info["last_item"] = summarize_segment(items[-1])
                    info["preview"] = [summarize_segment(item) for item in items[-10:]]
        except Exception as exc:
            info["checkpoint_error"] = str(exc)

    return info


def summarize_segment(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": item.get("id"),
        "start": item.get("start"),
        "end": item.get("end"),
        "text": item.get("text"),
        "en": item.get("en"),
        "zh": item.get("zh"),
    }


def analyze_input(payload: Dict[str, Any]) -> Dict[str, Any]:
    selected = Path(payload.get("path", "")).expanduser()
    if not selected.exists():
        raise FileNotFoundError(f"Path not found: {selected}")

    video = resolve_video_path(selected)
    output_root = Path(payload.get("output_root") or DEFAULT_OUTPUT_ROOT)
    names = default_names(selected, video)
    series_name = payload.get("series_name") or names["series_name"]
    movie_name = payload.get("movie_name") or names["movie_name"]

    info = checkpoint_info(output_root, series_name, movie_name)
    info.update(
        {
            "selected_path": str(selected),
            "selected_is_folder": selected.is_dir(),
            "video_path": str(video),
            "video_size_gb": round(video.stat().st_size / 1024 / 1024 / 1024, 2),
            "output_root": str(output_root),
            "series_name": series_name,
            "movie_name": movie_name,
            "default_series_name": names["series_name"],
            "default_movie_name": names["movie_name"],
        }
    )
    return info


def choose_file() -> str:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="选择视频文件",
        filetypes=[
            ("Video files", "*.mkv *.mp4 *.avi *.m2ts *.ts *.mov *.wmv"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return path


def choose_folder() -> str:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askdirectory(title="选择视频或蓝光文件夹")
    root.destroy()
    return path


def remove_resume_files(out_dir: Path, movie_name: str) -> None:
    names = [
        f"{movie_name}.segments.checkpoint.json",
        f"{movie_name}.segments.source.json",
        f"{movie_name}.en.ass",
        f"{movie_name}.zh.ass",
        f"{movie_name}.bilingual.ass",
    ]
    for name in names:
        path = out_dir / name
        if path.exists():
            path.unlink()


def start_processing(payload: Dict[str, Any]) -> Dict[str, Any]:
    selected = Path(payload["path"]).expanduser()
    output_root = Path(payload.get("output_root") or DEFAULT_OUTPUT_ROOT)
    series_name = payload["series_name"]
    movie_name = payload["movie_name"]
    restart = bool(payload.get("restart"))
    source_mode = payload.get("source_mode") or "audio"
    batch_size = int(payload.get("batch_size") or 5)
    context_lines = int(payload.get("context_lines") or 30)
    max_words = int(payload.get("max_words") or 14)
    max_chars = int(payload.get("max_chars") or 82)
    max_duration = float(payload.get("max_duration") or 6.0)

    out_dir = output_dir(output_root, series_name, movie_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    if restart:
        remove_resume_files(out_dir, movie_name)

    log_path, err_path = frontend_log_paths(series_name, movie_name)
    for path in (log_path, err_path):
        if path.exists():
            path.unlink()

    args = [
        sys.executable,
        str(APP_DIR / "audio_to_subtitle.py"),
        "--video",
        str(selected),
        "--source",
        source_mode,
        "--output-root",
        str(output_root),
        "--series-name",
        series_name,
        "--movie-name",
        movie_name,
        "--llm-model",
        payload.get("llm_model") or "qwen3:14b",
        "--batch-size",
        str(batch_size),
        "--context-lines",
        str(context_lines),
        "--max-words",
        str(max_words),
        "--max-chars",
        str(max_chars),
        "--max-duration",
        str(max_duration),
    ]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    stdout = log_path.open("w", encoding="utf-8")
    stderr = err_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(args, cwd=str(PROJECT_ROOT), stdout=stdout, stderr=stderr, env=env)
    finally:
        stdout.close()
        stderr.close()
    RUNS[process.pid] = process

    return {
        "pid": process.pid,
        "stdout_log": str(log_path),
        "stderr_log": str(err_path),
        "output_dir": str(out_dir),
        "command": " ".join(args),
    }


def tail_text(path: Path, max_chars: int = 8000) -> str:
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="replace")
    return data[-max_chars:]


def run_status(payload: Dict[str, Any]) -> Dict[str, Any]:
    pid = int(payload.get("pid") or 0)
    output_root = Path(payload.get("output_root") or DEFAULT_OUTPUT_ROOT)
    series_name = payload["series_name"]
    movie_name = payload["movie_name"]
    out_dir = output_dir(output_root, series_name, movie_name)
    process = RUNS.get(pid)
    running = process.poll() is None if process else process_is_running(pid)
    log_path, err_path = frontend_log_paths(series_name, movie_name)

    info = checkpoint_info(output_root, series_name, movie_name)
    info.update(
        {
            "pid": pid,
            "running": running,
            "stdout_tail": tail_text(log_path),
            "stderr_tail": tail_text(err_path),
            "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    return info


def process_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def html_page() -> str:
    return r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>字幕识别翻译控制台</title>
  <style>
    :root { color-scheme: light; font-family: "Microsoft YaHei", system-ui, sans-serif; }
    body { margin: 0; background: #f6f7f9; color: #20242a; }
    main { max-width: 1180px; margin: 0 auto; padding: 24px; }
    h1 { font-size: 24px; margin: 0 0 18px; font-weight: 650; }
    h2 { font-size: 17px; margin: 0 0 12px; }
    section { background: white; border: 1px solid #dde1e7; border-radius: 8px; padding: 18px; margin-bottom: 16px; }
    label { display: block; font-size: 13px; color: #4f5a66; margin-bottom: 6px; }
    input, select { width: 100%; box-sizing: border-box; height: 38px; border: 1px solid #c8ced8; border-radius: 6px; padding: 0 10px; background: white; color: #1f242b; }
    button { height: 38px; border: 1px solid #1f6feb; background: #1f6feb; color: white; border-radius: 6px; padding: 0 14px; cursor: pointer; }
    button.secondary { background: white; color: #1f6feb; }
    button.danger { background: #b42318; border-color: #b42318; }
    button:disabled { opacity: .55; cursor: not-allowed; }
    .grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 12px; align-items: end; }
    .span-2 { grid-column: span 2; }
    .span-3 { grid-column: span 3; }
    .span-4 { grid-column: span 4; }
    .span-6 { grid-column: span 6; }
    .span-8 { grid-column: span 8; }
    .span-12 { grid-column: span 12; }
    .stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
    .stat { border: 1px solid #dde1e7; border-radius: 8px; padding: 12px; background: #fafbfc; min-height: 60px; }
    .stat b { display: block; font-size: 20px; margin-top: 4px; }
    .muted { color: #66717f; font-size: 13px; }
    pre { white-space: pre-wrap; word-break: break-word; background: #111827; color: #e5e7eb; border-radius: 8px; padding: 12px; max-height: 320px; overflow: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { border-bottom: 1px solid #e5e7eb; text-align: left; padding: 8px; vertical-align: top; }
    th { color: #4f5a66; font-weight: 600; background: #fafbfc; }
    .inline { display: flex; gap: 8px; align-items: center; }
    .radio { display: flex; gap: 16px; height: 38px; align-items: center; }
    .radio label { margin: 0; color: #20242a; }
    .radio input { width: auto; height: auto; margin-right: 6px; }
    @media (max-width: 800px) {
      main { padding: 14px; }
      .grid { grid-template-columns: 1fr; }
      .span-2, .span-3, .span-4, .span-6, .span-8, .span-12 { grid-column: span 1; }
      .stats { grid-template-columns: 1fr 1fr; }
    }
  </style>
</head>
<body>
<main>
  <h1>字幕识别翻译控制台</h1>

  <section>
    <h2>输入</h2>
    <div class="grid">
      <div class="span-8">
        <label>视频文件或蓝光文件夹</label>
        <input id="path" placeholder="例如 E:\4杜比HDR\电影\解除好友2：暗网的磁力 ...">
      </div>
      <div class="span-2"><button class="secondary" onclick="selectFile()">选择视频</button></div>
      <div class="span-2"><button class="secondary" onclick="selectFolder()">选择文件夹</button></div>
      <div class="span-4">
        <label>输出根目录</label>
        <input id="outputRoot">
      </div>
      <div class="span-4">
        <label>系列名</label>
        <input id="seriesName">
      </div>
      <div class="span-4">
        <label>片名/集名</label>
        <input id="movieName">
      </div>
      <div class="span-2">
        <label>字幕来源</label>
        <select id="sourceMode">
          <option value="audio" selected>Whisper 音频识别</option>
          <option value="auto">自动：优先英文字幕</option>
          <option value="srt">只使用同名 SRT</option>
        </select>
      </div>
      <div class="span-2">
        <label>每组字幕单元</label>
        <input id="batchSize" type="number" value="5" min="1" max="20">
      </div>
      <div class="span-2">
        <label>前后参考单元</label>
        <input id="contextLines" type="number" value="30" min="0" max="100">
      </div>
      <div class="span-2">
        <label>最大词数</label>
        <input id="maxWords" type="number" value="14" min="4" max="40">
      </div>
      <div class="span-2">
        <label>最大字符</label>
        <input id="maxChars" type="number" value="82" min="20" max="160">
      </div>
      <div class="span-2">
        <label>最长秒数</label>
        <input id="maxDuration" type="number" value="6" min="1" max="20" step="0.5">
      </div>
      <div class="span-2"><button onclick="analyze()">分析</button></div>
    </div>
  </section>

  <section>
    <h2>状态</h2>
    <div class="stats">
      <div class="stat"><span class="muted">主视频</span><b id="videoSize">-</b></div>
      <div class="stat"><span class="muted">总字幕单元</span><b id="totalCount">未知</b></div>
      <div class="stat"><span class="muted">已完成</span><b id="completedCount">0</b></div>
      <div class="stat"><span class="muted">进程</span><b id="runningState">未运行</b></div>
    </div>
    <p class="muted" id="paths"></p>
    <div class="inline">
      <div class="radio">
        <label><input type="radio" name="runMode" value="resume" checked>从中断继续</label>
        <label><input type="radio" name="runMode" value="restart">从头开始</label>
      </div>
      <button onclick="startRun()">运行程序</button>
      <button class="secondary" onclick="refreshStatus()">刷新状态</button>
    </div>
  </section>

  <section>
    <h2>Checkpoint 预览</h2>
    <table>
      <thead><tr><th>ID</th><th>时间</th><th>英文</th><th>中文</th></tr></thead>
      <tbody id="preview"></tbody>
    </table>
  </section>

  <section>
    <h2>日志</h2>
    <pre id="log"></pre>
  </section>
</main>

<script>
const state = { pid: 0, lastAnalysis: null };
document.getElementById('outputRoot').value = String.raw`__DEFAULT_OUTPUT_ROOT__`;

async function api(path, body = {}) {
  const res = await fetch(path, { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || res.statusText);
  return data;
}

async function selectFile() {
  const data = await api('/api/select-file');
  if (data.path) document.getElementById('path').value = data.path;
}

async function selectFolder() {
  const data = await api('/api/select-folder');
  if (data.path) document.getElementById('path').value = data.path;
}

function payload() {
  return {
    path: document.getElementById('path').value,
    output_root: document.getElementById('outputRoot').value,
    series_name: document.getElementById('seriesName').value,
    movie_name: document.getElementById('movieName').value,
    source_mode: document.getElementById('sourceMode').value,
    batch_size: Number(document.getElementById('batchSize').value || 5),
    context_lines: Number(document.getElementById('contextLines').value || 30),
    max_words: Number(document.getElementById('maxWords').value || 14),
    max_chars: Number(document.getElementById('maxChars').value || 82),
    max_duration: Number(document.getElementById('maxDuration').value || 6)
  };
}

async function analyze() {
  const data = await api('/api/analyze', payload());
  state.lastAnalysis = data;
  document.getElementById('seriesName').value = data.series_name;
  document.getElementById('movieName').value = data.movie_name;
  render(data);
}

async function startRun() {
  const base = payload();
  base.restart = document.querySelector('input[name="runMode"]:checked').value === 'restart';
  const data = await api('/api/start', base);
  state.pid = data.pid;
  document.getElementById('log').textContent = `已启动 PID ${data.pid}\n${data.command}`;
  setTimeout(refreshStatus, 1500);
}

async function refreshStatus() {
  const base = payload();
  base.pid = state.pid;
  const data = await api('/api/status', base);
  render(data);
  document.getElementById('log').textContent = [data.stdout_tail || '', data.stderr_tail || ''].filter(Boolean).join('\n\n--- stderr ---\n');
}

function render(data) {
  document.getElementById('videoSize').textContent = data.video_size_gb ? `${data.video_size_gb} GB` : '-';
  document.getElementById('totalCount').textContent = data.total_count ?? '未知';
  document.getElementById('completedCount').textContent = data.completed_count ?? 0;
  document.getElementById('runningState').textContent = data.running ? '运行中' : '未运行';
  document.getElementById('paths').textContent = [
    data.video_path ? `主视频：${data.video_path}` : '',
    data.output_dir ? `输出：${data.output_dir}` : '',
    data.checkpoint_path ? `checkpoint：${data.checkpoint_path}` : ''
  ].filter(Boolean).join('  |  ');

  const rows = (data.preview || []).map(item => {
    const time = `${item.start ?? ''} -> ${item.end ?? ''}`;
    return `<tr><td>${escapeHtml(item.id)}</td><td>${escapeHtml(time)}</td><td>${escapeHtml(item.en || item.text || '')}</td><td>${escapeHtml(item.zh || '')}</td></tr>`;
  }).join('');
  document.getElementById('preview').innerHTML = rows || '<tr><td colspan="4" class="muted">暂无 checkpoint 内容</td></tr>';
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}

setInterval(() => { if (state.pid) refreshStatus().catch(() => {}); }, 5000);
</script>
</body>
</html>""".replace("__DEFAULT_OUTPUT_ROOT__", str(DEFAULT_OUTPUT_ROOT).replace("\\", "\\\\"))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_text(html_page(), "text/html; charset=utf-8")
        else:
            self.send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:
        try:
            if self.path == "/api/select-file":
                self.send_json({"path": choose_file()})
            elif self.path == "/api/select-folder":
                self.send_json({"path": choose_folder()})
            elif self.path == "/api/analyze":
                self.send_json(analyze_input(self.read_json_body()))
            elif self.path == "/api/start":
                self.send_json(start_processing(self.read_json_body()))
            elif self.path == "/api/status":
                self.send_json(run_status(self.read_json_body()))
            else:
                self.send_json({"error": "Not found"}, status=404)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=500)

    def read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def send_text(self, content: str, content_type: str) -> None:
        data = content.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Local web UI for subtitle OCR/ASR translation.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Subtitle frontend running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
