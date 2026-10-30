# run_everything_overnight_v2.ps1
$logDir = "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

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
}

$baseTrain = "--finetune --curriculum --year 2025 --attention-fusion --bilstm --epochs 12 --no-char-ngrams"
$baseEval  = "--attention-fusion --bilstm --no-char-ngrams"

Run-TrainAndEval -Name "seed43" `
    -TrainArgs "$baseTrain --curriculum-phases 2,2,8 --seed 43 --contrastive --contrastive-weight 0.3 --temperature 0.1" `
    -Checkpoint "checkpoints/curriculum_228_2025only_contrastive_seed43.pt" `
    -EvalArgs "$baseEval --contrastive"

Run-TrainAndEval -Name "seed44" `
    -TrainArgs "$baseTrain --curriculum-phases 2,2,8 --seed 44 --contrastive --contrastive-weight 0.3 --temperature 0.1" `
    -Checkpoint "checkpoints/curriculum_228_2025only_contrastive_seed44.pt" `
    -EvalArgs "$baseEval --contrastive"

Run-TrainAndEval -Name "schedule_444" `
    -TrainArgs "$baseTrain --curriculum-phases 4,4,4 --seed 42 --contrastive --contrastive-weight 0.3 --temperature 0.1" `
    -Checkpoint "checkpoints/curriculum_444_2025only_contrastive_seed42.pt" `
    -EvalArgs "$baseEval --contrastive"

Run-TrainAndEval -Name "schedule_633" `
    -TrainArgs "$baseTrain --curriculum-phases 6,3,3 --seed 42 --contrastive --contrastive-weight 0.3 --temperature 0.1" `
    -Checkpoint "checkpoints/curriculum_633_2025only_contrastive_seed42.pt" `
    -EvalArgs "$baseEval --contrastive"

Run-TrainAndEval -Name "factorial_nofusion_contrastive" `
    -TrainArgs "$baseTrain --curriculum-phases 2,2,8 --seed 42 --no-style --contrastive --contrastive-weight 0.3 --temperature 0.1" `
    -Checkpoint "checkpoints/factorial_nofusion_contrastive_seed42.pt" `
    -EvalArgs "--contrastive --no-style"

Run-TrainAndEval -Name "factorial_nofusion_nocontrastive" `
    -TrainArgs "$baseTrain --curriculum-phases 2,2,8 --seed 42" `
    -Checkpoint "checkpoints/factorial_nofusion_nocontrastive_seed42.pt" `
    -EvalArgs "--no-style"

Run-TrainAndEval -Name "film_rdrop_combined" `
    -TrainArgs "$baseTrain --curriculum-phases 2,2,8 --seed 42 --film-gate --rdrop --rdrop-weight 0.3" `
    -Checkpoint "checkpoints/film_rdrop_combined_seed42.pt" `
    -EvalArgs "--attention-fusion --bilstm --no-char-ngrams"

Write-Host "`n=== Threshold-sensitivity robustness check on existing seed-42 checkpoints (fixed threshold 0.5) ==="
python -u scripts/eval_official_test_2025.py --checkpoint checkpoints/curriculum_228_2025only_193dim_seed42.pt --threshold 0.5 --attention-fusion --bilstm 2>&1 | Tee-Object -FilePath "$logDir\plain_seed42_thresh05_eval.log"
python -u scripts/eval_official_test_2025.py --checkpoint checkpoints/curriculum_228_2025only_contrastive.pt --threshold 0.5 --attention-fusion --bilstm --contrastive 2>&1 | Tee-Object -FilePath "$logDir\contrastive_seed42_thresh05_eval.log"

Write-Host "`n=== ALL RUNS AND EVALUATIONS COMPLETE ==="
Write-Host "Check the logs/ folder for every training log and every official test-set eval result."