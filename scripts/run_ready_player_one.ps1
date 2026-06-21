$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$project = Split-Path -Parent $here
$movieRoot = Split-Path -Parent $project
$video = Join-Path $movieRoot "Ready.Player.One.2018.Eng.Fre.Ger.Ita.Por.Spa.Cze.Hun.Pol.Rus.Tha.Tur.Jpn.2160p.BluRay.Remux.DV.HDR.HEVC.Atmos-SGF.mkv"
$out = Join-Path $movieRoot "1 字幕"

python (Join-Path $project "src\subtitle_pipeline.py") `
  --video $video `
  --output $out `
  --device gpu:0
