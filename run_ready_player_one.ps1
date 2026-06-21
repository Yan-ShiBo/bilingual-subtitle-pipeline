$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$video = Join-Path (Split-Path -Parent $here) "Ready.Player.One.2018.Eng.Fre.Ger.Ita.Por.Spa.Cze.Hun.Pol.Rus.Tha.Tur.Jpn.2160p.BluRay.Remux.DV.HDR.HEVC.Atmos-SGF.mkv"
$out = Join-Path $here "outputs"

python (Join-Path $here "subtitle_pipeline.py") `
  --video $video `
  --output $out `
  --device gpu:0

