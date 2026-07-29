# 双语字幕生成流水线

本项目用于在 Windows 本地生成双语 ASS 字幕，支持普通视频文件、蓝光文件夹、已有字幕文件、MKV/M2TS 内封字幕、PGS 图像字幕 OCR、Whisper 语音识别，以及 Ollama `qwen3:14b` 纠错翻译。

## 界面预览

![控制台自动模式](docs/images/frontend-current.png)

![已有字幕双轨选择](docs/images/frontend-sidecar-merge.png)

## 目录结构

```text
.
├─ src/                  # Python 主程序代码
│  ├─ subtitle_frontend.py
│  ├─ audio_to_subtitle.py
│  └─ subtitle_pipeline.py
├─ docs/                 # 项目文档及界面截图
│  └─ images/
├─ scripts/              # 安装和示例运行脚本
├─ runtime/
│  ├─ logs/              # 前端任务日志，可删除
│  └─ scratch/           # 测试输出和临时缓存，可删除
├─ start_frontend.bat    # 双击启动本地前端
├─ requirements-gpu.txt
└─ README.md
```

`runtime/logs/` 和 `runtime/scratch/` 里的运行产物不提交到 Git。没有任务运行时可以清理。

## 前端启动

最简单方式：

```text
双击 start_frontend.bat
```

启动后打开：

```text
http://127.0.0.1:8765
```

启动器会先检查 `/api/health`，确认端口上运行的确实是本项目。若 `8765` 已被其他程序占用，会自动选择 `8766` 到 `8775` 中的第一个空闲端口，避免打开另一个项目的旧页面。

手动启动：

```powershell
cd /d "E:\6_Engineering_Projects\bilingual-subtitle-pipeline"
python .\src\subtitle_frontend.py --host 127.0.0.1 --port 8765
```

前端流程：

1. 选择视频文件或蓝光文件夹。
2. 选择字幕来源：自动、已有字幕文件、视频内封字幕或 Whisper 音频识别。
3. 页面只展开当前来源需要的选项；中英双轨选择、OCR 和音频识别设置不会同时平铺。
4. 点击“分析视频”，页面会显示主视频、已有/内封字幕、自动来源判断、checkpoint 和已完成数量；自动模式随后展开命中的来源选项。
5. 按需选择具体字幕文件、字幕轨道和语言。批量、上下文与显示长度限制在“高级设置”中。
6. 选择“从中断继续”或“从头开始”，点击“运行程序”。
7. 需要中断时点击“终止运行”，前端会终止当前任务进程树。

单个视频文件在“分析”时会优先调用本地 Ollama `qwen3:14b` 识别干净的系列名和片名，例如把发布组、分辨率、编码、音轨、HDR/DV 等标签从文件名中剔除。Ollama 不可用时会退回本地规则识别。

## 字幕来源优先级

默认 `--source auto` 使用以下优先级：

1. 已有字幕文件：优先查找视频同目录或所选文件夹中与当前视频文件名匹配的 `.srt`、`.ass`、`.ssa`、`.vtt`。目录中存在多个其他影片/剧集的字幕时不会任意选择。
2. 视频内封字幕：读取 MKV/M2TS 等视频内部字幕轨。文本字幕直接抽取；PGS 等图像字幕使用 OCR。
3. 音频识别：前两类都没有时，使用 faster-whisper `large-v3` 从音频生成源语言字幕。

也可以手动指定：

```powershell
--source auto       # 自动：已有字幕 -> 内封字幕 -> 音频识别
--source sidecar    # 只使用已有字幕文件
--source embedded   # 只使用视频内封字幕
--source audio      # 只使用 Whisper 音频识别
```

相关参数：

```powershell
--subtitle-file "E:\path\movie.en.srt"  # 指定已有字幕文件
--subtitle-stream 3                     # 指定内封字幕轨，例如 ffprobe 的 0:3
--source-language en                    # 源字幕语言，auto/en/ja/ko/fr/de/es/zh 等
--asr-language source                   # Whisper 识别语言；source 表示跟随 --source-language，auto 表示自动检测
--subtitle-ocr-lang en                  # 图像字幕 OCR 语言，auto/en/ch/chinese_cht/japan/korean 等
```

注意：源字幕和音频不一定是英文。只有英文字幕时，英译中会走和“英文语音识别后再英译中”一致的 5 句分组纠错翻译流程；其他源语言也会先纠错源文，再翻译成简体中文。

## 纠错翻译规则

无论来源是已有字幕、内封字幕 OCR，还是 Whisper 语音识别，都会进入同一套纠错翻译流程：

- 长句、词很多的句子、持续时间太长的句子，会先拆成半句或词组级字幕单元。
- 默认每组处理 `5` 个字幕单元。
- 每组参考前 `30` 个和后 `30` 个字幕单元。
- 上下文只用于理解人物、代词、术语和语义连续性，不会输出到结果。
- 后续批次还会看到前面已经确认的源文/中文结果，用于保持人名译法和术语一致。
- 组内先纠正源文，再翻译成自然的简体中文。
- 每个输入字幕单元必须对应且只能对应一个输出对象；模型缺行、重复索引或返回可见空译文时会中断当前批次，不会把错误结果写入 checkpoint。
- 重复或滚动片段可以保留最早完整项并把后续冗余项设为 `display=false`。
- 模型完成后会再次按默认 12 词、56 个英文字符、28 个中文字符和 5.5 秒的上限准备 ASS 显示事件。

## 时间标签规则

时间标签不交给 LLM 处理。

程序会把时间码作为上下文参考，但不允许 LLM 返回或修改时间。LLM 返回后，程序继续使用源字幕时间锚，并校验：

- 输出条数必须等于输入条数。
- 每条输出的 `start` 必须等于原始 `start`。
- 每条输出的 `end` 必须等于原始 `end`。
- 如果 checkpoint 与当前字幕切分不匹配，会忽略旧 checkpoint，避免时间轴错配。

Whisper 偶发的“短文本异常长时间”会再经过兜底切分，避免一两个词挂十几秒。

已有中英文字幕合并时，以英文对白轨为时间锚；没有重叠且间隔超过 0.75 秒的中英文事件不会强行配对。

## Checkpoint

翻译过程会写 checkpoint：

```text
<输出目录>\<片名>.segments.checkpoint.json
```

源字幕切分会写：

```text
<输出目录>\<片名>.segments.source.json
```

重新运行同一任务时会先校验 checkpoint 是否和当前字幕切分及来源请求一致。一致才继续，不一致会忽略旧 checkpoint。

旧的纯 ASR/单语 source cache 继续直接复用。旧的双语合并 source cache 因缺少新版时间锚元数据会重建一次；sidecar 只重新解析字幕文件。新版 embedded/OCR 缓存会按视频、轨道、渲染配置和图片哈希复用；旧的无指纹 OCR 缓存会重建一次。

新版 source cache 带有来源请求指纹，覆盖视频、字幕文件及修改时间、字幕/音频轨、识别语言和 OCR 配置。更换字幕文件或轨道后不会继续使用上一轮缓存。旧纯 ASR 缓存和 checkpoint 会在一次成功续跑后原位升级，不需要重新抽取音频。

音频抽取的临时 WAV 写在当前字幕输出目录，使用 `.subtitle-audio.<PID>.wav` 专用名称；程序不会创建、覆盖或删除视频旁边的同名 WAV。

前端启动的任务还会写入隐藏的 `.subtitle-run.json` 状态文件。前端意外关闭或重启后可以重新发现原进程，避免重复启动，并仍可通过网页安全终止该任务。

前端状态会区分运行中、完成、失败、中断和用户终止。失败时显示最后阶段和错误摘要；只有实际存在 checkpoint 时才提示从中断继续。

## 示例 1：Ready Player One MKV

```powershell
$root = Split-Path (Get-Location) -Parent
$outRoot = Get-ChildItem -LiteralPath $root -Directory | Where-Object { $_.Name -like '1 *' } | Select-Object -First 1
$video = Join-Path $root 'Ready.Player.One.2018.Eng.Fre.Ger.Ita.Por.Spa.Cze.Hun.Pol.Rus.Tha.Tur.Jpn.2160p.BluRay.Remux.DV.HDR.HEVC.Atmos-SGF.mkv'

python .\src\audio_to_subtitle.py `
  --video "$video" `
  --source auto `
  --output-root "$($outRoot.FullName)" `
  --series-name "Ready Player One" `
  --movie-name "Ready Player One 2018" `
  --llm-model qwen3:14b `
  --batch-size 5 `
  --context-lines 30
```

## 示例 2：头脑特工队2 蓝光文件夹

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

也可以直接在前端选择这个文件夹，先点击“分析”，再从页面中选择字幕来源、字幕轨道和语言。

## 输出

默认输出到：

```text
E:\4杜比HDR\电影\1 字幕\<系列名>\<片名>\
```

主要输出：

```text
<片名>.en.ass          # 源文字幕层，文件名保留 en 是为了兼容旧流程
<片名>.zh.ass          # 简体中文字幕
<片名>.bilingual.ass   # 双语字幕
```

## 依赖

- Python
- ffmpeg 或 `imageio-ffmpeg`
- `faster-whisper`
- `psutil`，用于前端重启后验证并恢复运行任务
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
- 源文和中文是否一一对应。
- 时间轴是否整体同步。

脚本会打印基础时间统计，例如最大单条时长、超过 6 秒或 8 秒的数量、超过 5 秒的空档数量。

## 更新日志

* OCR/字幕抽取缓存增加输入和配置指纹，并支持从部分 OCR 进度继续；旧无指纹缓存会安全重建一次。
* 中英文事件合并支持一对多和多对一断句，同时避免正常相邻字幕因轻微轨道偏移被连锁合并。
* 前端新增稳定的失败、中断和用户终止状态，显示最后失败阶段与错误摘要，并按真实 checkpoint 存在性给出恢复提示。
* 完善了 LLM 输出契约和语言校验：批次缺行、重复索引、可见空译文或中文轨仍是外文时会重试；连续失败则中断并保留最近 checkpoint，不再用原外文污染中文字幕。
* 修复了前端控制台在最后“生成合并双语字幕”阶段没有状态更新的问题，现在会正确显示“合并中英文字幕中”。
* 修复了合并已有中英文字幕时，若英文行缺失（如中文单方面翻译了屏幕 UI 文本），ASS 字幕文件中对应位置会强行打出字面量 `-` 的问题。现在会保持干净空白。
* 翻译提示词中新增了自动翻译括号或方括号内大写场景提示音（如 `(SIGHS)`、`[MUSIC]`）的规则。
* 增强了 LLM 翻译过程中的网络容错与中断恢复机制：遇到网络或大模型服务连接异常时会自动重试 3 次，若依然失败则主动抛出错误中断翻译，保留最近的 checkpoint，用户排除网络故障后重新运行即可无缝断点续传。
