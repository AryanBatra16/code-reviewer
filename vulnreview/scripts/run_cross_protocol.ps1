#!/usr/bin/env pwsh
# run_cross_protocol.ps1
# Runs ONLY the missing cross-protocol evaluations (P2 and P3)
# B4_no_sev P1 seed 2 is already complete, skipped.
# 
# Order:
#   1. B3 P2 (seeds 0,1,2)
#   2. B4 P2 (seeds 0,1,2)
#   3. B3 P3 (seeds 0,1,2)
#   4. B4 P3 (seeds 0,1,2)
#   5. evaluate --summarize
#
# Incremental: train.py skips already-completed seeds automatically.

$ErrorActionPreference = "Continue"
$WorkDir = "c:\Users\batra\Desktop\research paper\vulnreview"
Set-Location $WorkDir
$env:PYTHONPATH = "src"

$LogFile = "logs\cross_protocol_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
New-Item -ItemType Directory -Path logs -Force | Out-Null

function Log { param([string]$msg, [string]$color="White")
    $ts = "[$(Get-Date -Format 'HH:mm:ss')]"
    Write-Host "$ts $msg" -ForegroundColor $color
    Add-Content -Path $LogFile -Value "$ts $msg"
}

function Run-Training {
    param([string]$Protocol, [string]$Model, [string]$Ablation = "none")
    $tag = "${Model}_${Ablation}_${Protocol}"
    $csvPath = "results\train_results_${Model}_${Ablation}_${Protocol}.csv"
    
    # Check how many seeds already done
    if (Test-Path $csvPath) {
        $rows = (Import-Csv $csvPath | Measure-Object).Count
        if ($rows -ge 3) {
            Log "SKIP: $tag already has $rows/3 seeds complete." "Green"
            return
        }
        Log "RESUME: $tag has $rows/3 seeds, running remainder..." "Yellow"
    } else {
        Log "START: $tag (no CSV found)" "Cyan"
    }
    
    Log "==============================" "Cyan"
    Log "TRAINING: $tag  $(Get-Date -Format 'HH:mm:ss')" "Cyan"
    Log "==============================" "Cyan"
    
    python -m vulnreview.train --protocol $Protocol --model $Model --ablation $Ablation --seeds 0 1 2 2>&1 | Tee-Object -FilePath $LogFile -Append
    
    if ($LASTEXITCODE -ne 0) {
        Log "ERROR: $tag exited $LASTEXITCODE" "Red"
    } else {
        Log "DONE: $tag  $(Get-Date -Format 'HH:mm:ss')" "Green"
    }
}

Log "=== VulnReview Cross-Protocol Training Pipeline ===" "Yellow"
Log "Started: $(Get-Date)" "Yellow"
Log "Log: $LogFile" "Yellow"
Log ""

Run-Training -Protocol P2 -Model B3
Run-Training -Protocol P2 -Model B4
Run-Training -Protocol P3 -Model B3
Run-Training -Protocol P3 -Model B4

Log ""
Log "=== All training complete! Running evaluation summary... ===" "Yellow"
python -m vulnreview.evaluate --summarize 2>&1 | Tee-Object -FilePath $LogFile -Append
Log "evaluate done (exit $LASTEXITCODE)" "Yellow"

Log ""
Log "=== ALL DONE: $(Get-Date) ===" "Green"
Log "Check results/results_summary.md for final numbers." "Green"
