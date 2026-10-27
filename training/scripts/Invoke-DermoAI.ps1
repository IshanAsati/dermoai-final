. "$PSScriptRoot\Environment.ps1"
& $Python -m dermoai.cli @args
if ($LASTEXITCODE -ne 0) { throw 'DermoAI command failed. See the error above.' }
