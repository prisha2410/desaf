# rerun_broken_evals.ps1
# Re-runs ONLY the three eval calls that failed in run_remaining_v3.ps1
# due to missing/invalid architecture flags. Uses Invoke-Expression like
# the original script so multi-flag strings split correctly.
# (threshold-sensitivity block already succeeded -- not repeated here)

$logDir = "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$baseEval = "--attention-fusion --bilstm --no-char-ngrams"

Write-Host "`n=== EVAL: factorial_nofusion_contrastive (fixed flags) ==="
$cmd1 = "python -u scripts/eval_official_test_2025.py --checkpoint checkpoints/factorial_nofusion_contrastive_seed42.pt --threshold 0.60 $baseEval --contrastive --no-style"
Invoke-Expression $cmd1 2>&1 | Tee-Object -FilePath "$logDir\factorial_nofusion_contrastive_eval_fixed.log"

Write-Host "--- cooldown: waiting 30s for GPU memory to release ---"
Start-Sleep -Seconds 30

Write-Host "`n=== EVAL: factorial_nofusion_nocontrastive (fixed flags) ==="
$cmd2 = "python -u scripts/eval_official_test_2025.py --checkpoint checkpoints/factorial_nofusion_nocontrastive_seed42.pt --threshold 0.66 $baseEval --no-style"
Invoke-Expression $cmd2 2>&1 | Tee-Object -FilePath "$logDir\factorial_nofusion_nocontrastive_eval_fixed.log"

Write-Host "--- cooldown: waiting 30s for GPU memory to release ---"
Start-Sleep -Seconds 30

Write-Host "`n=== EVAL: film_rdrop_combined (fixed flags, --film-gate dropped: not a valid eval-script flag) ==="
$cmd3 = "python -u scripts/eval_official_test_2025.py --checkpoint checkpoints/film_rdrop_combined_seed42.pt --threshold 0.70 $baseEval"
Invoke-Expression $cmd3 2>&1 | Tee-Object -FilePath "$logDir\film_rdrop_combined_eval_fixed.log"

Write-Host "`n=== ALL FIXED EVALS COMPLETE ==="
