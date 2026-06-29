# 问题与解决记录

本文记录字幕流水线最近修过的关键问题、根因、解决方式和使用注意。

## 1. 选择远程服务器翻译时，本地 GPU 仍然满载

### 现象

前端已经连接远程 Ollama，并在下拉框中选择了远程模型，但本机 `nvidia-smi` 仍显示本地 `llama-server.exe` 在跑，`ollama ps` 也显示本地 `qwen3:14b` 常驻 GPU。

### 根因

前端下拉框有远程模型选项，但启动任务时没有把 `llmModel` 写入请求 payload。后端拿不到 `llm_model` 后回退到默认 `qwen3:14b`，于是请求仍打到本机 `http://localhost:11434`。

另外，`ollama ps` 中的 `100% GPU` 表示模型完全驻留在 GPU，并不等于瞬时算力占用。真正是否在计算要看 `nvidia-smi` 的 GPU-Util 或 `nvidia-smi pmon`。

### 修复

- 前端 payload 增加 `llm_model`。
- 片名分析也使用当前选择的模型。
- 远程模型列表刷新后恢复用户保存的远程模型选择。
- `remote:<name>:<model>` 会被后端路由到本地 SSH 隧道端口 `127.0.0.1:11435`，再转发到远程 `11434`。

### 使用注意

如果只想释放本地 Ollama 显存，可以运行：

```powershell
ollama stop qwen3:14b
```

## 2. 中断后 checkpoint 可能损坏

### 现象

任务被终止时，`*.segments.checkpoint.json` 或其它 JSON 缓存偶尔只写了一半，之后前端状态或继续运行读 JSON 会失败。

### 根因

旧代码直接 `write_text(json.dumps(...))` 写目标文件。进程在写入中间被杀掉时，目标文件会留下半截 JSON。

### 修复

JSON 缓存改成原子写：

1. 先写同目录临时文件。
2. 写完后用 replace 替换目标文件。

覆盖范围包括：

- `*.segments.source.json`
- `*.segments.checkpoint.json`
- OCR cache
- image events cache
- Ollama translation cache

## 3. Whisper/ASR 产生重复字幕，翻译阶段无法修复

### 现象

音频识别出的字幕有连续重复或重叠片段，例如同一句话被识别成多条相似字幕，前端 checkpoint 预览和最终 ASS 都会重复显示。

### 根因

重复已经存在于 ASR 之后的 `segments.source.json`。旧翻译提示词又明确要求：

- 每个输入行必须对应一个输出对象。
- 不允许合并。
- 不允许省略。

这导致即使用更强的 `qwen3:30b`，模型也只能逐条翻译重复内容，不能真正清洗结构。

### 修复

翻译阶段仍保持“一条输入一条 JSON 输出”用于 checkpoint 对齐，但增加显示控制字段：

```json
{
  "index": 0,
  "corrected_text": "complete sentence",
  "chinese_translation": "完整译文",
  "display": true,
  "display_start": 209.03,
  "display_end": 213.00
}
```

规则变为：

- 重复或重叠的 ASR 片段可以设置为 `"display": false`。
- 保留的字幕可以用 `display_start/display_end` 覆盖整段真实发音时间。
- checkpoint 仍保留所有原始片段，保证续跑能校验原始时间轴。
- 生成 ASS 时只输出 `display != false` 的片段。
- 前端 checkpoint 预览也只显示可见片段。

旧 checkpoint 没有 `display` 元数据，会被自动忽略并重新翻译，避免继续复用旧重复结果。

## 4. “从中断处继续”仍然重新抽取音频和 Whisper 识别

### 现象

指定输出目录里已经有：

```text
<片名>.segments.source.json
<片名>.segments.checkpoint.json
```

但点击“从中断处继续”后，程序仍然重新抽取 `.wav` 并重新跑 faster-whisper。

### 根因

前端的“从中断处继续”只是不删除缓存文件；但 `audio_to_subtitle.py` 主流程没有读取已有 `*.segments.source.json` 的逻辑。它每次都会重新走来源选择：

1. sidecar
2. embedded
3. audio extraction
4. Whisper transcription

然后再覆盖写入 `segments.source.json`。

### 修复

主流程现在会优先读取：

```text
<输出目录>\<片名>.segments.source.json
```

只要这个缓存存在且可读，就直接复用已切分好的源字幕事件，不再抽取音频、不再跑 Whisper。

只有以下情况才会重新抽取/识别：

- 用户选择“从头开始”，前端会先删除 `segments.source.json`。
- `segments.source.json` 不存在。
- `segments.source.json` 损坏或结构不合法。

这次修复还增加了测试，确保已有 source cache 时 `extract_audio()` 和 `transcribe_audio()` 不会被调用。

## 5. “从中断处继续”和“从头开始”的边界

### 从中断处继续

保留：

- `*.segments.source.json`
- `*.segments.checkpoint.json`
- 已有输出 ASS
- embedded/OCR cache

行为：

- 优先复用 source cache，跳过抽取/识别。
- 如果 checkpoint 与当前 source cache 匹配，并且包含新格式 `display` 元数据，则继续翻译。
- 如果 checkpoint 是旧格式或损坏，则重新翻译，但仍不重新抽取音频或跑 Whisper。

### 从头开始

前端会删除：

- `*.segments.source.json`
- `*.segments.checkpoint.json`
- `*.en.ass`
- `*.zh.ass`
- `*.bilingual.ass`
- `embedded/`
- 同名临时 `.wav`

行为：

- 重新分析来源。
- 必要时重新抽取音频和 Whisper 识别。
- 重新 OCR 或重新翻译。

## 6. 验证命令

常规静态检查：

```powershell
python -m py_compile src\audio_to_subtitle.py src\subtitle_frontend.py src\subtitle_pipeline.py src\ssh_tunnel.py tests\test_audio_display_cleanup.py
```

单元测试：

```powershell
python -m unittest discover -s tests
```

前端可用性：

```powershell
Invoke-WebRequest -UseBasicParsing -Uri http://127.0.0.1:8765/ -TimeoutSec 3
```

## 7. 已有中文字幕时不应重新翻译

### 现象

电影目录或视频内已经带有中文字幕时，旧流程仍可能选择英文字幕、内嵌字幕或音频识别结果，然后继续执行英译中。这样会浪费远程模型时间，也可能把原本较好的官方中文字幕改坏。

### 根因

旧的 `audio_to_subtitle.py` 源选择逻辑一次只选一个字幕来源：

1. 找到一个 sidecar 字幕就直接解析。
2. 没找到再尝试内嵌字幕。
3. 再失败才抽音频并跑 Whisper。

它没有先判断“是否已经存在中文字幕”，也没有把英文字幕和中文字幕按时间轴合并成双语源。因此即使目录里同时有 `*.en.srt` 和 `*.zh.srt`，也可能进入翻译路径。

### 修复

- 源选择阶段先扫描同目录 sidecar 字幕，识别中文和英文字幕。
- sidecar 未命中时，再扫描内嵌字幕流，优先选择中文流，并可同时选择英文流作为参考。
- 英文和中文字幕按时间重叠配对合并为统一的 source cache。
- 中文字幕统一繁体转简体。
- 已经存在中文时跳过翻译，改走“双语校对”：
  - 英文也会校对，因为它可能来自 OCR。
  - 中文会校对 OCR/字幕识别错误，并保持简体中文。
  - 重复、重叠、碎片行仍可通过 `display=false` 隐藏，保留 checkpoint 可续跑。

## 8. 远程服务器密码保存

### 现象

远程服务器连接信息可以保存主机、端口、用户名和显示名称，但每次重新打开页面都要重新输入密码。

### 修复

远程连接弹窗增加“保存密码到本机浏览器”选项。勾选后，密码会和远程服务器配置一起保存到当前浏览器的 `localStorage`，下次打开弹窗会自动填入。

### 注意

该密码保存是本机浏览器明文保存，方便个人本机使用；如果是共享电脑，不要勾选保存密码。
