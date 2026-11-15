#!/usr/bin/env pwsh
# run_remaining_training.ps1
# Runs all remaining model training after B3 P1 seeds complete.
# Usage: .\scripts\run_remaining_training.ps1
#
# Order of execution:
#   1. B4 full  P1 seeds 0,1,2    (overwrites best_B4_P1.pt with full training)
#   2. B4 no_cwe P1 seeds 0,1,2   (ablation)
#   3. B4 no_sev P1 seeds 0,1,2   (ablation)
#   4. B3       P2 seeds 0,1,2
#   5. B4 full  P2 seeds 0,1,2
#   6. B3       P3 seeds 0,1,2
#   7. B4 full  P3 seeds 0,1,2

$ErrorActionPreference = "Stop"
$WorkDir = "c:\Users\batra\Desktop\research paper\vulnreview"
Set-Location $WorkDir
$env:PYTHONPATH = "src"

function Run-Training {
    param(
        [string]$Protocol,
        [string]$Model,
        [string]$Ablation = "none"
    )
    $tag = "$Model-$Ablation-$Protocol"
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "STARTING: $tag  $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    
    python -m vulnreview.train --protocol $Protocol --model $Model --ablation $Ablation --seeds 0 1 2
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: $tag failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "DONE: $tag  $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Green
}

Write-Host "=== VulnReview Remaining Training Pipeline ===" -ForegroundColor Yellow
Write-Host "Started: $(Get-Date)" -ForegroundColor Yellow
Write-Host ""

# --- B4 P1 (most critical - overwrites checkpoint) ---
Run-Training -Protocol P1 -Model B4 -Ablation none

# --- Ablations on P1 ---
Run-Training -Protocol P1 -Model B4 -Ablation no_cwe
Run-Training -Protocol P1 -Model B4 -Ablation no_sev

# --- Cross-protocol B3/B4 ---
Run-Training -Protocol P2 -Model B3
Run-Training -Protocol P2 -Model B4

Run-Training -Protocol P3 -Model B3
Run-Training -Protocol P3 -Model B4

Write-Host ""
Write-Host "=== All training complete! Running evaluation summary... ===" -ForegroundColor Yellow
python -m vulnreview.evaluate --summarize
if ($LASTEXITCODE -ne 0) {
    Write-Host "Note: evaluate exited $LASTEXITCODE (may be non-critical, check results_summary.md)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Running Integrated Gradients explanation (300 samples) ===" -ForegroundColor Yellow
python -m vulnreview.explain --protocol P1 --max-samples 300

Write-Host ""
Write-Host "=== ALL DONE: $(Get-Date) ===" -ForegroundColor Green
Write-Host "Check results/results_summary.md for final numbers." -ForegroundColor Green
