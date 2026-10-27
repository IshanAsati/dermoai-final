$ProjectRoot = Split-Path $PSScriptRoot -Parent
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = [IO.Path]::GetFullPath((Join-Path $ProjectRoot '..\..\work\dermoai-venv\Scripts\python.exe'))
}
if (-not (Test-Path -LiteralPath $Python)) { throw 'Run scripts\Install.ps1 first.' }
$env:MPLCONFIGDIR = Join-Path $ProjectRoot 'runtime\cache\matplotlib'
$env:HF_HOME = Join-Path $ProjectRoot 'runtime\cache\huggingface'
$env:TORCH_HOME = Join-Path $ProjectRoot 'runtime\cache\torch'
$controlPath = Join-Path $ProjectRoot 'runtime\control-token.dpapi'
if (-not $env:DERMOAI_CONTROL_TOKEN) {
    if (Test-Path -LiteralPath $controlPath) {
        $protectedControl = Get-Content -LiteralPath $controlPath | ConvertTo-SecureString
        $env:DERMOAI_CONTROL_TOKEN = [Net.NetworkCredential]::new('', $protectedControl).Password
    } else {
        $env:DERMOAI_CONTROL_TOKEN = [guid]::NewGuid().ToString('N') + [guid]::NewGuid().ToString('N')
        New-Item -ItemType Directory -Path (Split-Path $controlPath -Parent) -Force | Out-Null
        $env:DERMOAI_CONTROL_TOKEN | ConvertTo-SecureString -AsPlainText -Force | ConvertFrom-SecureString | Set-Content -LiteralPath $controlPath
    }
}
Set-Location -LiteralPath $ProjectRoot
