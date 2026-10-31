# Removes files the DeSAF paper does not use. Run from D:\sass-tad with all work committed.
# Nothing is lost: everything stays in git history, and the tag "before-cleanup" marks the old state.
# The removals are only STAGED; review them with `git status`, then commit and push yourself.
#
#   .\cleanup_repo.ps1                 keeps the Hard-ensemble scripts
#   .\cleanup_repo.ps1 -DropEnsemble   also removes them (do this if the paper no longer reports
#                                      the single-seed 0.7350 ensemble)
param([switch]$DropEnsemble)

if (-not (Test-Path .git)) { Write-Host "Run this from D:\sass-tad (the repo root)."; exit 1 }
$dirty = git status --porcelain
if ($dirty) {
    Write-Host "You have uncommitted changes. Commit (and push) them first, then rerun this script:"
    git status --short
    exit 1
}

git tag -f before-cleanup | Out-Null
Write-Host "Restore point: tag 'before-cleanup' = $(git rev-parse --short HEAD)`n"

$remove = @(
    # old SASS-TAD project and the discarded pre-pan2025_full (2024+2025 combined) pipeline
    "archive", "archive_pre_pan2025full", "archive_pre_pan2025full.ps1",
    "sass_tad.egg-info", "RESULTS_SUMMARY.md", "hard_tier_feature_probe.py",
    # scripts no paper result depends on
    "scripts/analyze_stylometric_by_length.py", "scripts/check_datasets.py",
    "scripts/check_label_alignment.py", "scripts/eval_finetune_joint.py", "scripts/finetune_joint.py",
    "scripts/finetune_hard.py", "scripts/eval_pan2025.py", "scripts/eval_pan2025_test.py",
    "scripts/eval_per_tier_threshold.py", "scripts/precompute_official_test.py",
    "scripts/precompute_topics.py", "scripts/run_baselines.py", "scripts/train_sspc_baseline.py",
    "scripts/save_results.py",
    # superseded by scripts/c50_multiauthor_test.py
    "scripts/analysis_c50_crosscorpus.py", "scripts/analysis_c50_overlap_binned.py",
    # library code nothing imports any more
    "src/models/baselines.py", "src/models/sspc_baseline.py", "src/training",
    "src/utils/sbert_cache.py", "src/features/ner_masker.py", "src/explainability"
)
if ($DropEnsemble) {
    $remove += @("scripts/ensemble_by_tier.py", "scripts/ensemble_hard_tier.py",
                 "scripts/ensemble_hard_tier_v2.py", "scripts/ensemble_hard_tier_v3.py",
                 "scripts/ensemble_hard_tier_v4.py", "scripts/ensemble_seeds.py")
}

$count = 0
foreach ($p in $remove) {
    if (git ls-files -- $p) {
        git rm -r -q -- $p
        $count++
        Write-Host "  removed  $p"
    }
    if (Test-Path $p) {
        $left = (Get-ChildItem -Recurse -File -Force $p -ErrorAction SilentlyContinue | Measure-Object).Count
        Write-Host "    note: $p still holds $left local file(s) git never tracked (left untouched; delete by hand if unneeded)"
    }
}

# the run scripts that produced paper results go into runs/ (run them from the repo root as .\runs\<name>.ps1)
New-Item -ItemType Directory -Force -Path runs | Out-Null
foreach ($r in "run_concat_ablation.ps1", "run_remaining_v3.ps1", "run_overnight_final.ps1", "rerun_broken_evals.ps1") {
    if (git ls-files -- $r) { git mv $r "runs/$r"; Write-Host "  moved    $r -> runs/$r" }
}

$ignore = Get-Content .gitignore -ErrorAction SilentlyContinue
if ($ignore -notcontains "*.egg-info/") { Add-Content .gitignore "*.egg-info/"; git add .gitignore }

Write-Host "`n$count paths removed (staged, not committed). Review with:  git status"
Write-Host "Then:  git commit -m `"Remove code the paper does not use`"  ;  git push origin main  ;  git push -f origin before-cleanup"
Write-Host "To undo before committing:  git reset --hard before-cleanup"
