# run_overnight_final.ps1
# One overnight run that:
#   1) Trains seed45 (3rd clean seed, closes task #2 -- multi-seed)
#   2) Trains true_nostyle_nocontrastive (the missing factorial cell, closes task #5)
#   3) Saves raw per-boundary predictions (.npz) for seed44, seed45, and
#      seed43 (if its checkpoint is found) using save_test_predictions.py
#   4) Runs compute_significance.py pairwise on the Hard tier (seed43 vs 44,
#      43 vs 45, 44 vs 45) so you get real bootstrap CIs + p-values -- task #8
#   5) Runs ensemble_seeds.py across whichever seed prediction files exist

$logDir = "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$Seed43Checkpoint = "checkpoints/curriculum_228_2025only_contrastive_seed43.pt"
$Seed43Threshold  = 0.50

$Seed44Checkpoint = "checkpoints/curriculum_228_2025only_contrastive_seed44.pt"
$Seed44Threshold  = 0.52

function Run-TrainAndEval {
    param(
        [string]$Name,
        [string]$TrainArgs,
        [string]$Checkpoint,
        [string]$EvalArgs
    )
    Write-Host "`n=== TRAIN: $Name ==="
    $logFile = "$logDir\$Name.log"
    $fullTrainCmd = "python -u scripts/train_deberta_fusion.py $TrainArgs --save-checkpoint $Checkpoint"
    Invoke-Expression $fullTrainCmd 2>&1 | Tee-Object -FilePath $logFile

    Write-Host "--- cooldown: waiting 30s for GPU memory to release ---"
    Start-Sleep -Seconds 30

    $thresholdMatch = Select-String -Path $logFile -Pattern "Threshold:\s+([\d.]+)" | Select-Object -Last 1
    if ($thresholdMatch -match "Threshold:\s+([\d.]+)") {
        $threshold = $matches[1]
    } else {
        Write-Host "!!! Could not parse threshold for $Name, defaulting to 0.5 !!!"
        $threshold = "0.5"
    }

    Write-Host "=== EVAL (official test, one-time): $Name  threshold=$threshold ==="
    $evalLog = "$logDir\${Name}_eval.log"
    $fullEvalCmd = "python -u scripts/eval_official_test_2025.py --checkpoint $Checkpoint --threshold $threshold $EvalArgs"
    Invoke-Expression $fullEvalCmd 2>&1 | Tee-Object -FilePath $evalLog

    Write-Host "--- cooldown: waiting 30s for GPU memory to release ---"
    Start-Sleep -Seconds 30

    return $threshold
}

function Save-Predictions {
    param(
        [string]$Name,
        [string]$Checkpoint,
        [string]$Threshold,
        [string]$ArchFlags
    )
    if (-not (Test-Path $Checkpoint)) {
        Write-Host "`n!!! Skipping predictions for $Name -- checkpoint not found: $Checkpoint !!!"
        return $null
    }
    Write-Host "`n=== SAVE PREDICTIONS: $Name ==="
    $outPath = "checkpoints/$Name/test_predictions.npz"
    $cmd = "python -u scripts/save_test_predictions.py --checkpoint $Checkpoint --threshold $Threshold $ArchFlags --out $outPath"
    Invoke-Expression $cmd 2>&1 | Tee-Object -FilePath "$logDir\${Name}_predictions.log"
    Start-Sleep -Seconds 30
    return $outPath
}

$baseTrain = "--finetune --curriculum --year 2025 --attention-fusion --bilstm --epochs 12 --no-char-ngrams --curriculum-phases 2,2,8"
$baseEval  = "--attention-fusion --bilstm --no-char-ngrams"

# ============================================================
# 1) seed45 -- closes task #2 (3-seed)
# ============================================================
$seed45Threshold = Run-TrainAndEval -Name "seed45" `
    -TrainArgs "$baseTrain --seed 45 --contrastive --contrastive-weight 0.3 --temperature 0.1" `
    -Checkpoint "checkpoints/curriculum_228_2025only_contrastive_seed45.pt" `
    -EvalArgs "$baseEval --contrastive"

# ============================================================
# 2) true_nostyle_nocontrastive -- closes task #5 (factorial ablation)
# ============================================================
Run-TrainAndEval -Name "true_nostyle_nocontrastive" `
    -TrainArgs "$baseTrain --seed 42 --no-style" `
    -Checkpoint "checkpoints/true_nostyle_nocontrastive_seed42.pt" `
    -EvalArgs "--attention-fusion --bilstm --no-style" | Out-Null

# ============================================================
# 3) Save raw per-boundary predictions for significance testing / ensembling
# ============================================================
$seed43Preds = Save-Predictions -Name "seed43" -Checkpoint $Seed43Checkpoint `
    -Threshold $Seed43Threshold -ArchFlags "--attention-fusion --bilstm --no-char-ngrams --contrastive"

$seed44Preds = Save-Predictions -Name "seed44" -Checkpoint $Seed44Checkpoint `
    -Threshold $Seed44Threshold -ArchFlags "--attention-fusion --bilstm --no-char-ngrams --contrastive"

$seed45Preds = Save-Predictions -Name "seed45" -Checkpoint "checkpoints/curriculum_228_2025only_contrastive_seed45.pt" `
    -Threshold $seed45Threshold -ArchFlags "--attention-fusion --bilstm --no-char-ngrams --contrastive"

# ============================================================
# 4) Significance testing (Hard tier) -- task #8
# ============================================================
Write-Host "`n=== SIGNIFICANCE TESTING (Hard tier) ==="
if ($seed43Preds -and $seed44Preds) {
    Write-Host "`n--- seed43 vs seed44 ---"
    Invoke-Expression "python scripts/compute_significance.py --a $seed43Preds --b $seed44Preds --tier hard --a-name seed43 --b-name seed44" 2>&1 | Tee-Object -FilePath "$logDir\sig_seed43_vs_seed44.log"
}
if ($seed43Preds -and $seed45Preds) {
    Write-Host "`n--- seed43 vs seed45 ---"
    Invoke-Expression "python scripts/compute_significance.py --a $seed43Preds --b $seed45Preds --tier hard --a-name seed43 --b-name seed45" 2>&1 | Tee-Object -FilePath "$logDir\sig_seed43_vs_seed45.log"
}
if ($seed44Preds -and $seed45Preds) {
    Write-Host "`n--- seed44 vs seed45 ---"
    Invoke-Expression "python scripts/compute_significance.py --a $seed44Preds --b $seed45Preds --tier hard --a-name seed44 --b-name seed45" 2>&1 | Tee-Object -FilePath "$logDir\sig_seed44_vs_seed45.log"
}

# ============================================================
# 5) Multi-seed ensemble -- whichever prediction files actually exist
# ============================================================
$ensembleInputs = @($seed43Preds, $seed44Preds, $seed45Preds) | Where-Object { $_ -ne $null }
if ($ensembleInputs.Count -ge 2) {
    Write-Host "`n=== MULTI-SEED ENSEMBLE ($($ensembleInputs.Count) seeds) ==="
    $predArgs = ($ensembleInputs -join " ")
    Invoke-Expression "python scripts/ensemble_seeds.py --predictions $predArgs" 2>&1 | Tee-Object -FilePath "$logDir\ensemble_seeds.log"
} else {
    Write-Host "`n!!! Fewer than 2 prediction files available, skipping ensemble step !!!"
}

Write-Host "`n=== ALL OVERNIGHT RUNS COMPLETE ==="
Write-Host "Check logs\ for: seed45.log, true_nostyle_nocontrastive.log, seed43/44/45_predictions.log,"
Write-Host "sig_seed43_vs_seed44.log, sig_seed43_vs_seed45.log, sig_seed44_vs_seed45.log, ensemble_seeds.log"