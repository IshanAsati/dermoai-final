$ErrorActionPreference = 'Stop'
$LabelStudio = Join-Path $env:LOCALAPPDATA 'LabelStudioRuntime\venv312\Scripts\label-studio.exe'
if (-not (Test-Path -LiteralPath $LabelStudio)) { throw 'Existing Label Studio environment was not found.' }
$env:ML_BLOCK_LOCAL_IP = 'false'
$env:SSRF_PROTECTION_ENABLED = 'true'
& $LabelStudio start --internal-host 127.0.0.1 --port 8080 --no-browser
