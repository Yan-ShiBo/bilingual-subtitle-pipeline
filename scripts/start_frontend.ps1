$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$appName = "bilingual-subtitle-pipeline"
$selectedPort = $null

function Get-FrontendSourceSha256 {
    $stream = [System.IO.MemoryStream]::new()
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        $sourceRoot = Join-Path $projectRoot "src"
        $sourceFiles = Get-ChildItem -LiteralPath $sourceRoot -Filter "*.py" -File | Sort-Object Name
        foreach ($sourceFile in $sourceFiles) {
            $nameBytes = [System.Text.Encoding]::UTF8.GetBytes($sourceFile.Name)
            $stream.Write($nameBytes, 0, $nameBytes.Length)
            $stream.WriteByte(0)
            $contentBytes = [System.IO.File]::ReadAllBytes($sourceFile.FullName)
            $stream.Write($contentBytes, 0, $contentBytes.Length)
            $stream.WriteByte(0)
        }
        $hash = $sha256.ComputeHash($stream.ToArray())
        return ([System.BitConverter]::ToString($hash)).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $sha256.Dispose()
        $stream.Dispose()
    }
}

$expectedSourceSha256 = Get-FrontendSourceSha256

function Test-TcpPort {
    param(
        [string]$HostName,
        [int]$Port
    )

    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $task = $client.ConnectAsync($HostName, $Port)
        if (-not $task.Wait(300)) {
            return $false
        }
        return $client.Connected
    }
    catch {
        return $false
    }
    finally {
        $client.Dispose()
    }
}

foreach ($port in 8765..8799) {
    $url = "http://127.0.0.1:$port"
    if (Test-TcpPort -HostName "127.0.0.1" -Port $port) {
        try {
            $health = Invoke-RestMethod -Uri "$url/api/health" -TimeoutSec 1
            if (
                $health.app -eq $appName -and
                "$($health.source_sha256)".ToLowerInvariant() -eq $expectedSourceSha256
            ) {
                Start-Process $url
                Write-Host "Frontend is already running at $url"
                exit 0
            }
            if ($health.app -eq $appName) {
                Write-Host "Skipping stale subtitle frontend at $url"
            }
        }
        catch {
        }
        continue
    }

    $selectedPort = $port
    break
}

if ($null -eq $selectedPort) {
    throw "No free frontend port found in range 8765-8799."
}

$frontendUrl = "http://127.0.0.1:$selectedPort/"
Write-Host "Starting subtitle frontend at $frontendUrl"
Start-Process $frontendUrl

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
& python (Join-Path $projectRoot "src\subtitle_frontend.py") --host 127.0.0.1 --port $selectedPort
exit $LASTEXITCODE
