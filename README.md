# 双语字幕生成流水线

本项目用于在 Windows 本机生成中英双语 ASS 字幕，支持视频文件、蓝光文件夹、MKV 内封字幕、PGS/OCR、Whisper 语音识别和 Ollama 翻译。

## 目录结构

```text
.
├─ src/                 # Python 主程序代码
│  ├─ subtitle_frontend.py
│  ├─ audio_to_subtitle.py
│  └─ subtitle_pipeline.py
├─ scripts/             # 安装、示例运行脚本
├─ runtime/
│  ├─ logs/             # 前端和字幕任务日志，可删除
│  └─ scratch/          # 测试输出、临时缓存，可删除
├─ start_frontend.bat   # 双击启动本地前端
├─ requirements-gpu.txt
└─ README.md
```

`runtime/logs/` 和 `runtime/scratch/` 里的运行产物都不提交到 Git，可以在没有任务运行时清理。

## 前端启动

最简单方式：

```text
双击 start_frontend.bat
```

脚本会启动本地前端并打开：

```text
http://127.0.0.1:8765
```

手动启动：

```powershell
cd /d "E:\4杜比HDR\电影\字幕抽取识别"
python .\src\subtitle_frontend.py --host 127.0.0.1 --port 8765
```

前端流程：

1. 选择视频文件或蓝光文件夹。
2. 点“分析”，查看主视频、checkpoint、已完成句数。
3. 选择“从中断继续”或“从头开始”。
4. 点“运行程序”。
5. 在页面中查看日志和 checkpoint 预览。

## 核心流程

### 有内封字幕的 MKV

使用 `src/subtitle_pipeline.py`：

- 先用 `ffprobe` 检测字幕轨。
- 文本字幕优先直接提取。
- PGS 或图片字幕才走 OCR。
- 如果有英文和简体中文字幕，直接生成双语 ASS。
- 如果有繁体中文字幕，先转简体。
- 如果只有英文字幕，进入统一的纠错翻译流程。

### 没有字幕的视频或蓝光文件夹

使用 `src/audio_to_subtitle.py`：

- 默认 `--source audio`，忽略旁边旧 SRT，使用 Whisper 识别英文语音。
- `--source auto`：优先同名英文 SRT，没有 SRT 再使用 Whisper。
- `--source srt`：只使用同名英文 SRT 或 `--srt` 指定的英文 SRT。
- 支持把 `--video` 指向蓝光文件夹；如果存在 `BDMV\STREAM`，优先选择其中体积最大的 `.m2ts` 主视频。

## 纠错翻译规则

无论来源是“英文字幕”还是“Whisper 语音识别”，英译中都使用同一套流程：

- 先把英文切成字幕单元。
- 长句、词很多的句子、持续时间太长的句子，会拆成半句或词组单元。
- 默认每组处理 `5` 个字幕单元。
- 每组参考前 `30` 个和后 `30` 个字幕单元。
- 参考上下文只用于理解人物、代词、术语和上下文关系。
- 组内先纠正英文，再翻译成自然的简体中文。
- 每个输入字幕单元必须对应一个输出对象。
- 不允许合并两句。
- 不允许把一句拆成多个输出对象。
- 不允许输出上下文内容。

## 时间标签规则

时间标签不交给 LLM 纠错或翻译。

程序只把字幕文本发给 LLM，不把 `start` / `end` 时间码放进 prompt。LLM 返回后，程序把原始时间码复制回输出字幕，并校验：

- 输出条数必须等于输入条数。
- 每条输出的 `start` 必须等于原始 `start`。
- 每条输出的 `end` 必须等于原始 `end`。
- 如果时间码被改变，程序直接报错。

另外，程序会对 Whisper 偶发的“短文本异常长时间”做兜底截断，避免一两个词挂十几秒。

## Checkpoint

翻译过程中会写 checkpoint：

```text
<输出目录>\<片名>.segments.checkpoint.json
```

重新运行同一任务时会先校验 checkpoint 是否和当前字幕切分一致。一致才继续，不一致会忽略旧 checkpoint，避免时间轴错配。

如果存在：

```text
<输出目录>\<片名>.segments.source.json
```

前端会显示总字幕单元数。

## 示例 1：Ready Player One MKV

在项目根目录运行：

```powershell
$root = Split-Path (Get-Location) -Parent
$outRoot = Get-ChildItem -LiteralPath $root -Directory | Where-Object { $_.Name -like '1 *' } | Select-Object -First 1
$video = Join-Path $root 'Ready.Player.One.2018.Eng.Fre.Ger.Ita.Por.Spa.Cze.Hun.Pol.Rus.Tha.Tur.Jpn.2160p.BluRay.Remux.DV.HDR.HEVC.Atmos-SGF.mkv'

python .\src\subtitle_pipeline.py `
  --video "$video" `
  --output "$($outRoot.FullName)" `
  --model qwen3:14b `
  --batch-size 5 `
  --context-lines 30
```

## 示例 2：头脑特工队2蓝光文件夹

```powershell
$root = Split-Path (Get-Location) -Parent
$outRoot = Get-ChildItem -LiteralPath $root -Directory | Where-Object { $_.Name -like '1 *' } | Select-Object -First 1
$folder = Join-Path $root '头脑特工队2 [国粤英多音轨+特效中文字幕].2024.USA.BluRay.Remux.UHD.HDR10.2160p.Atmos.TrueHD7.1-DreamHD'

python .\src\audio_to_subtitle.py `
  --video "$folder" `
  --source auto `
  --output-root "$($outRoot.FullName)" `
  --series-name "Inside Out 2" `
  --movie-name "Inside Out 2 2024" `
  --llm-model qwen3:14b `
  --batch-size 5 `
  --context-lines 30
```

也可以直接在前端选择这个文件夹。

## 依赖

- Python
- ffmpeg 或 `imageio-ffmpeg`
- `faster-whisper`
- CUDA 可用的 PyTorch / CTranslate2 环境
- PaddleOCR / pgsrip / OpenCC，用于 PGS/OCR 路径
- Ollama
- Ollama 模型：`qwen3:14b`

安装辅助脚本在 `scripts/` 目录中。

## 质量检查

生成后建议检查：

- ASS 事件数量是否合理。
- 是否存在异常长字幕段。
- 是否存在异常大空档。
- 中文和英文是否一一对应。
- 时间轴是否整体同步。

脚本会打印基础时间统计，例如最大单条时长、超过 6 秒或 8 秒的数量、超过 5 秒的空档数量。
