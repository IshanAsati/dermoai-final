$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\Environment.ps1"
if (-not $env:DERMOAI_CONTROL_TOKEN) { throw 'Set DERMOAI_CONTROL_TOKEN, or dot-source scripts\Set-Credentials.ps1 first.' }
& $Python -m dermoai.cli --config config.json serve --port 9090
