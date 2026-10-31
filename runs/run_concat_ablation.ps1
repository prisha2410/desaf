# Isolated gate ablation: identical to the seed43/44/45 headline config, minus --attention-fusion.
# Style stays ON (default), so the model fuses with raw concat [h ; delta] instead of the gate.
# Run from D:\sass-tad with .venv active:   .\run_concat_ablation.ps1

$seeds = @(44, 45)    # seed43 already done

$logDir = "logs\concat_ablation"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$trainArgs = @("--finetune", "--curriculum", "--year", "2025", "--bilstm", "--epochs", "12",
               "--no-char-ngrams", "--curriculum-phases", "2,2,8",
               "--contrastive", "--contrastive-weight", "0.3", "--temperature", "0.1")
$archArgs  = @("--bilstm", "--no-char-ngrams", "--contrastive")

foreach ($seed in $seeds) {
    $name     = "concat_seed$seed"
    $ckpt     = "checkpoints/ablation_concat_contrastive_seed$seed.pt"
    $trainLog = "$logDir\${name}_train.log"
    $evalLog  = "$logDir\${name}_eval.log"
    $predLog  = "$logDir\${name}_predictions.log"
    $predOut  = "checkpoints/$name/test_predictions.npz"

    if (Test-Path $ckpt) {
        Write-Host "!!! $ckpt already exists. Not overwriting. Delete or rename it first. !!!"
        continue
    }

    Write-Host "`n=== TRAIN: $name ==="
    python -u scripts/train_deberta_fusion.py @trainArgs --seed $seed --save-checkpoint $ckpt 2>&1 |
        Tee-Object -FilePath $trainLog
    # Python can exit non-zero on Windows after finishing cleanly, so trust the saved checkpoint instead.
    if (-not (Test-Path $ckpt)) { Write-Host "!!! No checkpoint written for $name, see $trainLog !!!"; continue }
    if ($LASTEXITCODE -ne 0) { Write-Host "(note: python exit code $LASTEXITCODE, but checkpoint exists, continuing)" }
    Start-Sleep -Seconds 30

    $match = Select-String -Path $trainLog -Pattern "Threshold:\s+([\d.]+)" | Select-Object -Last 1
    if (-not $match) { Write-Host "!!! No threshold found in $trainLog, skipping eval for $name. !!!"; continue }
    $thr = $match.Matches[0].Groups[1].Value
    Write-Host "Validation-tuned threshold for ${name}: $thr"

    Write-Host "`n=== EVAL (official test, one-time): $name ==="
    python -u scripts/eval_official_test_2025.py --checkpoint $ckpt --threshold $thr @archArgs 2>&1 |
        Tee-Object -FilePath $evalLog
    if ($LASTEXITCODE -ne 0) { Write-Host "(note: eval exit code $LASTEXITCODE for $name, check $evalLog)" }
    Start-Sleep -Seconds 30

    Write-Host "`n=== SAVE PREDICTIONS: $name ==="
    python -u scripts/save_test_predictions.py --checkpoint $ckpt --threshold $thr @archArgs --out $predOut 2>&1 |
        Tee-Object -FilePath $predLog
    if (-not (Test-Path $predOut)) { Write-Host "!!! Predictions not saved for $name, see $predLog !!!" }
    Start-Sleep -Seconds 30
}

Write-Host "`nDone. Paste the eval log(s) from $logDir here."
