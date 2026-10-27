param([switch]$Approve)
$ErrorActionPreference = 'Stop'
if (-not $Approve) { throw 'Training has not started. Use -Approve only when ready to launch the exploratory experiment.' }
. "$PSScriptRoot\Environment.ps1"
& $Python -m dermoai.cli --config training-config.json train --approve
if ($LASTEXITCODE -ne 0) { throw 'Training could not be started.' }
