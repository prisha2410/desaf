# archive_pre_pan2025full.ps1
# Moves (does NOT delete) files belonging to the old 2024+2025-combined
# feature-cache pipeline into archive_pre_pan2025full/, so the live repo
# only shows the pan2025_full-only pipeline actually used in the paper.
# Everything here is reversible -- nothing is deleted.

$archiveRoot    = "archive_pre_pan2025full"
$archiveScripts = "$archiveRoot\scripts"
$archiveData    = "$archiveRoot\data"

New-Item -ItemType Directory -Force -Path $archiveScripts | Out-Null
New-Item -ItemType Directory -Force -Path $archiveData | Out-Null

# --- Scripts that load from the old 2024+2025 feature cache ---
$staleScripts = @(
    "scripts\test_eval.py",
    "scripts\threshold_sweep.py",
    "scripts\tier_routing_check.py",
    "scripts\hard_tier_probe_v2.py",
    "scripts\hard_tier_probe_v3.py",
    "scripts\hard_tier_probe_v4.py"
)

# length_artifact_check*, run_everything_over*, run_tierb_overnight* have
# truncated names in the file tree -- wildcard-match them to catch the
# real filenames without guessing exactly.
$staleWildcards = @(
    "scripts\length_artifact_check*.py",
    "run_everything_over*.ps1",
    "run_tierb_overnight*.ps1"
)

Write-Host "=== Moving known stale scripts ==="
foreach ($f in $staleScripts) {
    if (Test-Path $f) {
        Move-Item -Path $f -Destination $archiveScripts -Force
        Write-Host "  moved: $f"
    } else {
        Write-Host "  (not found, skipped): $f"
    }
}

Write-Host "`n=== Moving wildcard-matched stale scripts ==="
foreach ($pattern in $staleWildcards) {
    $matches = Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue
    foreach ($m in $matches) {
        Move-Item -Path $m.FullName -Destination $archiveScripts -Force
        Write-Host "  moved: $($m.FullName)"
    }
}

# --- The old 2024+2025 feature cache itself ---
Write-Host "`n=== Moving old feature cache (data/processed/features_finetuned) ==="
if (Test-Path "data\processed\features_finetuned") {
    Move-Item -Path "data\processed\features_finetuned" -Destination $archiveData -Force
    Write-Host "  moved: data\processed\features_finetuned"
} else {
    Write-Host "  (not found, skipped): data\processed\features_finetuned"
}

Write-Host "`n=== Done. Everything moved into $archiveRoot\ -- nothing deleted. ==="
Write-Host "NOTE: RESULTS_SUMMARY.md was NOT touched automatically -- it mixes"
Write-Host "the old (2024+2025) analysis and the current (pan2025_full-only) one"
Write-Host "in a single file. Split that by hand: keep only the second section"
Write-Host "(starting '# DeSAF: Results Summary' / 'All results are on the"
Write-Host "official PAN 2025 splits (pan2025_full)') as the live file, and move"
Write-Host "the first 'FINAL' section into $archiveRoot\RESULTS_SUMMARY_OLD_2024plus2025.md"
