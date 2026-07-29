$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$appName = "bilingual-subtitle-pipeline"
$selectedPort = $null

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

foreach ($port in 8765..8775) {
    $url = "http://127.0.0.1:$port"
    if (Test-TcpPort -HostName "127.0.0.1" -Port $port) {
        try {
            $health = Invoke-RestMethod -Uri "$url/api/health" -TimeoutSec 1
            if ($health.app -eq $appName) {
                Start-Process $url
                Write-Host "Frontend is already running at $url"
                exit 0
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
    throw "No free frontend port found in range 8765-8775."
}

$frontendUrl = "http://127.0.0.1:$selectedPort/"
Write-Host "Starting subtitle frontend at $frontendUrl"
Start-Process $frontendUrl

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
& python (Join-Path $projectRoot "src\subtitle_frontend.py") --host 127.0.0.1 --port $selectedPort
exit $LASTEXITCODE
