$common = "--finetune --curriculum --curriculum-phases 2,2,8 --year 2025 --attention-fusion --bilstm --epochs 12 --seed 42"

Write-Host "=== Tier B Run 1/5: XBM-backed contrastive ==="
python scripts/train_deberta_fusion.py $common.Split(" ") --contrastive --xbm --save-checkpoint checkpoints/tierb_xbm_contrastive.pt

Write-Host "=== Tier B Run 2/5: R-Drop ==="
python scripts/train_deberta_fusion.py $common.Split(" ") --rdrop --save-checkpoint checkpoints/tierb_rdrop.pt

Write-Host "=== Tier B Run 3/5: Self-adjusting Dice loss ==="
python scripts/train_deberta_fusion.py $common.Split(" ") --dice-loss --save-checkpoint checkpoints/tierb_dice.pt

Write-Host "=== Tier B Run 4/5: FiLM fusion ==="
python scripts/train_deberta_fusion.py $common.Split(" ") --film-gate --save-checkpoint checkpoints/tierb_film.pt

Write-Host "=== Tier B Run 5/5: Transformer-over-BiLSTM ==="
python scripts/train_deberta_fusion.py $common.Split(" ") --transformer-seq --save-checkpoint checkpoints/tierb_transformer.pt

Write-Host "=== All Tier B runs complete ==="