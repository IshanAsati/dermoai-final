$ErrorActionPreference = "Stop"

$pidFile = Join-Path $PSScriptRoot "server.pid"
if (-not (Test-Path -LiteralPath $pidFile)) {
    Write-Output "DermoAI tester is not running."
    exit 0
}

$serverProcessId = [int](Get-Content -LiteralPath $pidFile -Raw)
$process = Get-Process -Id $serverProcessId -ErrorAction SilentlyContinue
if ($null -eq $process) {
    Remove-Item -LiteralPath $pidFile
    Write-Output "Removed a stale server record."
    exit 0
}

if ($process.ProcessName -notmatch "^python") {
    throw "Refusing to stop process $serverProcessId because it is not a Python process."
}

Stop-Process -Id $serverProcessId
Remove-Item -LiteralPath $pidFile -ErrorAction SilentlyContinue
Write-Output "DermoAI tester stopped."
