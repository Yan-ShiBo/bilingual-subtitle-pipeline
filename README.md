# 字幕抽取、识别、纠错与双语 ASS 生成方案

这个目录放的是本地字幕处理工具，目标是在 Windows 本机上为电影或剧集生成中英双语 ASS 字幕。

## 目标

- 对有内封字幕轨的 MKV，优先提取现成字幕轨。
- 如果已有英文和简体中文字幕，直接生成双语 ASS。
- 如果已有英文和繁体中文字幕，先转成简体中文，再生成双语 ASS。
- 如果只有英文字幕，使用 Ollama `qwen3:14b` 翻译成简体中文，再生成双语 ASS。
- 如果视频本身没有字幕，使用 Whisper 在 RTX 4070 Ti Super 上进行英文语音识别，再纠错、翻译、生成双语 ASS。

## 主要脚本

### `subtitle_pipeline.py`

用于处理 MKV 内封字幕轨、PGS 字幕和 OCR 场景。

设计原则：

- 先用 `ffprobe` 检测字幕轨语言和编码。
- 文本字幕优先直接提取，避免 OCR。
- PGS 或图片字幕才走 OCR。
- 中文字幕统一输出简体。
- 输出英文、中文、双语 ASS。

### `audio_to_subtitle.py`

用于处理没有字幕的视频。

当前默认策略：

- 默认忽略同目录旧 SRT，使用音频识别：`--source audio`。
- 如果指定 `--source auto`，会优先使用同目录英文 SRT；没有 SRT 时再使用 Whisper。
- 如果指定 `--source srt`，只使用同目录英文 SRT 或 `--srt` 指定的英文 SRT。
- 使用 `faster-whisper large-v3`。
- GPU 模式：`device="cuda"`，`compute_type="float16"`。
- 启用 `word_timestamps=True`，尽量获得词级时间戳。
- 启用 VAD，减少长静音导致的错误切段。
- 对 Whisper 输出做二次字幕单元切分。
- 之后调用 Ollama `qwen3:14b` 做英文纠错和中文翻译。

如果只有英文字幕，英译中的处理流程和英文语音识别后的处理流程一致：

1. 先把英文字幕切成字幕单元。
2. 长句或词很多的句子拆成半句或词组单元。
3. 每组 5 个字幕单元。
4. 参考前后各 30 个字幕单元。
5. 组内先纠正英文，再翻译成简体中文。
6. 时间标签不进入 LLM，不纠错、不翻译，只由程序原样复制并校验。

`--video` 参数既可以传视频文件，也可以传蓝光文件夹。传入文件夹时，程序会递归查找视频文件；如果存在 `BDMV\STREAM`，优先从这里选择体积最大的 `.m2ts` 作为主视频。

### `subtitle_frontend.py`

本地前端页面，用于选择视频或蓝光文件夹、查看 checkpoint、选择继续或从头开始，并启动处理程序。

最简单启动方式：

```text
双击 start_frontend.bat
```

脚本会自动打开：

```text
http://127.0.0.1:8765
```

如果前端已经在运行，脚本会直接打开浏览器。

启动：

```powershell
python .\subtitle_frontend.py --host 127.0.0.1 --port 8765
```

打开：

```text
http://127.0.0.1:8765
```

前端功能：

- 选择单个视频文件。
- 选择蓝光视频文件夹。
- 自动解析蓝光文件夹主视频。
- 设置输出根目录、系列名、片名/集名。
- 选择字幕来源：Whisper 音频识别、自动优先英文字幕、只使用同名 SRT。
- 判断是否存在 checkpoint。
- 显示已完成字幕单元数。
- 如果存在 `segments.source.json`，显示总字幕单元数。
- 显示最近 checkpoint 内容。
- 选择从中断继续或从头开始。
- 启动后台处理并查看日志。

## 字幕单元切分规则

LLM 不直接决定时间轴。程序先把 Whisper 输出切成稳定的字幕单元，再交给 LLM 处理。

一个字幕单元通常是一句英文；如果句子很长、词很多、持续时间太长，则半句或一组词也算一个字幕单元。

默认限制：

- 每个字幕单元最多约 `14` 个英文词。
- 每个字幕单元最多约 `82` 个英文字符。
- 每个字幕单元最长约 `6` 秒。
- 有词级时间戳时，优先按词级时间戳切分。
- 没有词级时间戳时，按文本长度和时间比例切分。

## 纠错与翻译规则

当前效率和质量折中方案：

- 每组处理 `5` 个字幕单元。
- 每组参考前 `30` 个字幕单元和后 `30` 个字幕单元。
- 参考上下文只用于理解人物、代词、术语和上下文关系。
- LLM 只输出当前 5 个字幕单元的结果。
- 每个输入字幕单元必须对应一个输出对象。
- 不允许合并两句。
- 不允许把一句拆成多个输出对象。
- 不允许输出上下文内容。
- 先纠正英文识别错误，再翻译成自然的简体中文。

## 时间标签规则

时间标签不交给 LLM 纠错或翻译。

程序只把字幕文本发给 LLM，不把 `start` / `end` 时间码放进 prompt。LLM 返回后，程序把原始时间码原样复制回输出字幕。

每批处理完成后会做时间码校验：

- 输出条数必须等于输入条数。
- 每条输出的 `start` 必须等于原始 `start`。
- 每条输出的 `end` 必须等于原始 `end`。
- 如果时间码被改变，程序直接报错，不继续生成错误字幕。

## 断点续跑

翻译过程中会写 checkpoint：

```powershell
<输出目录>\<片名>.segments.checkpoint.json
```

中断后重新运行同一条命令，会从已完成的字幕单元继续。

注意：checkpoint 保存的是已经翻译完成的字幕单元；Whisper 转录本身目前会重新执行。

## 示例命令

在 `E:\4杜比HDR\电影\字幕抽取识别` 下运行：

### 示例 1：单个 MKV 文件

```powershell
$root = Split-Path (Get-Location) -Parent
$outRoot = Get-ChildItem -LiteralPath $root -Directory | Where-Object { $_.Name -like '1 *' } | Select-Object -First 1
$video = Join-Path $root 'Ready.Player.One.2018.Eng.Fre.Ger.Ita.Por.Spa.Cze.Hun.Pol.Rus.Tha.Tur.Jpn.2160p.BluRay.Remux.DV.HDR.HEVC.Atmos-SGF.mkv'

python .\subtitle_pipeline.py `
  --video "$video" `
  --output "$($outRoot.FullName)" `
  --model qwen3:14b `
  --batch-size 5 `
  --context-lines 30
```

这个示例优先处理 MKV 内封字幕；如果只有英文字幕，英译中会进入同一套 5 单元分组、前后 30 单元参考、时间码只校验不交给 LLM 的流程。

### 示例 2：蓝光视频文件夹

```powershell
$root = Split-Path (Get-Location) -Parent
$outRoot = Get-ChildItem -LiteralPath $root -Directory | Where-Object { $_.Name -like '1 *' } | Select-Object -First 1
$folder = Join-Path $root '头脑特工队2 [国粤英多音轨+特效中文字幕].2024.USA.BluRay.Remux.UHD.HDR10.2160p.Atmos.TrueHD7.1-DreamHD'

python .\audio_to_subtitle.py `
  --video "$folder" `
  --source auto `
  --output-root "$($outRoot.FullName)" `
  --series-name "Inside Out 2" `
  --movie-name "Inside Out 2 2024" `
  --llm-model qwen3:14b `
  --batch-size 5 `
  --context-lines 30
```

这个示例会从蓝光文件夹中自动选择主视频；如果旁边有英文 SRT，`--source auto` 会优先使用英文 SRT，否则使用 Whisper 音频识别。

也可以直接打开前端：

```text
http://127.0.0.1:8765
```

然后选择这个蓝光文件夹。

## 本机依赖

- Python
- ffmpeg 或 `imageio-ffmpeg`
- `faster-whisper`
- CUDA 可用的 PyTorch / CTranslate2 环境
- Ollama
- Ollama 模型：`qwen3:14b`

## 质量检查

生成后需要检查：

- ASS 事件数量是否合理。
- 是否存在异常长字幕段。
- 是否存在异常大空档。
- 中文和英文是否一一对应。
- 时间轴是否整体同步。

脚本会打印基础时间统计，例如最大单条时长、超过 6 秒或 8 秒的数量、超过 5 秒的空档数量。
