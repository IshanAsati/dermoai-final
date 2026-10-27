param(
    [string]$RunId = '',
    [int]$RefreshSeconds = 3
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path $PSScriptRoot -Parent
if (-not $RunId) {
    $latest = Get-ChildItem -LiteralPath (Join-Path $ProjectRoot 'runtime') -Directory |
        Where-Object { $_.Name -match '^\d{8}-\d{6}-[a-f0-9]+$' -and (Test-Path -LiteralPath (Join-Path $_.FullName 'status.json')) } |
        Sort-Object Name -Descending | Select-Object -First 1
    if (-not $latest) { throw 'No training run found.' }
    $RunId = $latest.Name
}
$StatusPath = Join-Path $ProjectRoot "runtime\$RunId\status.json"
$LogPath = Join-Path $ProjectRoot "runtime\$RunId\worker.log"

Write-Host "DermoAI live training monitor"
Write-Host "Run: $RunId"
Write-Host "Press Ctrl+C to close this monitor; training continues in the background."
Write-Host ""

while ($true) {
    if (-not (Test-Path -LiteralPath $StatusPath)) {
        Write-Host "Waiting for status file..."
        Start-Sleep -Seconds $RefreshSeconds
        continue
    }

    try {
        $fileStream = [IO.File]::Open($StatusPath, [IO.FileMode]::Open, [IO.FileAccess]::Read, ([IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete))
        $reader = [IO.StreamReader]::new($fileStream)
        try {
            $status = $reader.ReadToEnd() | ConvertFrom-Json
        } finally {
            $reader.Dispose()
            $fileStream.Dispose()
        }
    } catch {
        Start-Sleep -Milliseconds 500
        continue
    }

    $epoch = if ($null -ne $status.epoch_in_progress) { $status.epoch_in_progress } else { $status.epoch }
    $batch = if ($null -ne $status.batch) { $status.batch } else { 0 }
    $batches = if ($null -ne $status.batches) { $status.batches } else { 0 }
    $batchPercent = if ($batches -gt 0) { [math]::Round(100 * $batch / $batches, 1) } else { 0 }
    $bestValue = if ($null -ne $status.best_selection_score) { $status.best_selection_score } else { $status.best_macro_f1 }
    $bestScore = if ($null -ne $bestValue -and $bestValue -ge 0) { '{0:P2}' -f $bestValue } else { 'pending' }
    $gpu = if ($null -ne $status.gpu_allocated_mb) { '{0:N0} MB' -f $status.gpu_allocated_mb } else { 'loading' }
    $elapsed = if ($null -ne $status.elapsed_seconds) { [timespan]::FromSeconds($status.elapsed_seconds).ToString('hh\:mm\:ss') } else { '00:00:00' }
    $accuracy = if ($null -ne $status.validation_metrics.accuracy) { '{0:P2}' -f $status.validation_metrics.accuracy } else { 'pending' }
    $macroF1 = if ($null -ne $status.validation_metrics.macro_f1) { '{0:P2}' -f $status.validation_metrics.macro_f1 } else { 'pending' }
    $line = "State: $($status.state) $($status.phase) | Epoch: $epoch/$($status.epochs) | Batch: $batch/$batches ($batchPercent%) | Val acc: $accuracy | F1: $macroF1 | Best selection score: $bestScore | GPU: $gpu | Elapsed: $elapsed"
    Write-Host ("`r" + $line.PadRight(180)) -NoNewline

    if ($status.state -notin @('starting', 'running')) {
        Write-Host ""
        if ($status.state -eq 'completed') {
            Write-Host "Training completed. Best epoch: $($status.best_epoch)"
            Write-Host "Best checkpoint: $($status.best_checkpoint)"
        } else {
            Write-Host "Training ended with state: $($status.state)"
            if ($status.error) { Write-Host "Error: $($status.error)" }
            if (Test-Path -LiteralPath $LogPath) { Write-Host "Log: $LogPath" }
        }
        break
    }

    Start-Sleep -Seconds $RefreshSeconds
}
