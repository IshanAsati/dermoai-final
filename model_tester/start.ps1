$ErrorActionPreference = "Stop"

$app = Join-Path $PSScriptRoot "app.py"
$repository = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$workspacePython = Join-Path (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path "work\dermoai-venv\Scripts\python.exe"
$localPython = Join-Path $repository ".venv\Scripts\python.exe"

if (Test-Path -LiteralPath $localPython) {
    $python = $localPython
} elseif (Test-Path -LiteralPath $workspacePython) {
    $python = $workspacePython
} else {
    $python = (Get-Command python -ErrorAction Stop).Source
}

& $python $app --host 127.0.0.1 --port 7861
