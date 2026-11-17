# When Style Clues Point the Wrong Way

Code and verified numbers for our study of how document assembly shapes style change detection on PAN 2025.

The repository holds two things: DeSAF, a detector that finds where one author stops writing and another starts, sentence by sentence; and the analysis of why the hardest level of the PAN 2025 benchmark behaves the way it does. Every number in the paper can be traced to a script here and a check in `REAL_VERIFIED_NUMBERS.md`.

## The task in plain words

Each PAN 2025 document is a run of sentences taken from Reddit and written by several people, joined into one text. For every pair of neighbouring sentences you answer one question: did the author change here or not?

The documents come in three levels:

- **Easy:** the authors write about different things, so a change of topic usually means a change of author.
- **Medium:** the topics are closer.
- **Hard:** the organisers describe it like this: "All sentences in a document pertain to a single topic and are stylistically similar."

The score is the official PAN 2025 measure: two-class macro-F1, pooled over all boundaries of a level. We use only PAN 2025 data for training, so our numbers can be set directly against the PAN 2025 results table.

## What we found

1. **DeSAF is level with the best PAN 2025 systems.** Over three training seeds it scores 0.953 on Easy, 0.824 on Medium and 0.826 on Hard. The two best teams scored 0.958 / 0.823 / 0.830 and 0.955 / 0.825 / 0.829. Every gap is smaller than our own test-sampling uncertainty.

2. **On Hard, the classic writing-style clues point the wrong way, and the way the documents were put together can cause this.** Stylometry rests on one idea: two sentences by the same person look more alike than sentences by two different people. On Hard it is the reverse. Where the author changes, the two sentences share more content words, are closer in length and share more character patterns than two neighbouring sentences by the same author. We took the same kind of text and joined it ourselves. Joined at random, the clues point the normal way. Joined so that each new author starts with a sentence close in topic and style to the previous one, the clues flip.

3. **Telling the model which level a document comes from helps, but only if the level is right.** A small head that gets the true level as an extra input beats the same kind of head trained without it on all 27 of our runs. A wrong level hurts more than giving no level at all. A classifier that guesses the level from the text was right for 85% of test documents, and that was not enough: with its guesses the score fell below plain DeSAF. So DeSAF without any level information is our main system, and the level flag tells us what knowing the level is worth.

A few things we checked keep these findings honest:

- **Repeated bot lines.** About a fifth of Easy boundaries and a quarter of Medium boundaries sit next to one of 42 lines that repeat across the test documents, such as subreddit rules and bot messages. DeSAF is almost perfect there. When we replace those lines with one neutral sentence, its score on those boundaries falls from 0.985-0.996 to 0.50-0.68, and its Easy and Medium scores drop by 0.12 to 0.19. Part of that drop is forced by the test itself, and the test cannot tell whether DeSAF learned these exact lines or is simply thrown by the replacement. Hard changes little (about 0.02), which is why we lead the comparison with the winners on Hard.
- **Where DeSAF goes wrong on Hard.** We predicted that DeSAF would miss more Hard changes where the two sentences look most alike. That prediction narrowly failed (one seed fell short of the bar we set). What we do see, as a description rather than a tested claim, is that among the most look-alike joins, which hold 72% of Hard changes, DeSAF raises more false alarms on same-author pairs. Its recall there is no lower than at average joins.

## How DeSAF works

```
two neighbouring sentences
        |
        +---------------------------+
        |                           |
  DeBERTa-v3-small reads       193 writing-style features
  the pair together            (function words, punctuation,
  (top 2 layers trained)        readability, word and sentence length,
        |                       parts of speech); we keep the
        |                       difference between the two sentences
        |                           |
        +------------+--------------+
                     |
          combine the two (a learned gate)
                     |
          BiLSTM over all boundaries of the document
                     |
          one probability per boundary: author change or not
```

Training uses the PAN 2025 training split only. The model first trains on Easy documents (2 epochs), then on Easy and Medium (2 epochs), then on all three levels (8 epochs), with an extra contrastive loss. The decision threshold is chosen on the validation split.

The learned gate is not what makes DeSAF work. A plain concatenation of the two inputs scores the same (no difference survives correction for multiple tests over three seeds). We keep the gate because the headline runs used it.

The **tier-aware head** is a second, small model. It takes the frozen outputs of a trained DeSAF encoder plus a three-way flag for the level (Easy, Medium or Hard). It is trained on the validation split. In the main results the flag is the true level taken from the dataset folder, which is why we call it an oracle experiment.

## Results on the official PAN 2025 test set

| System | Easy | Medium | Hard |
|---|---|---|---|
| wqd (1st in PAN 2025) | 0.958 | 0.823 | 0.830 |
| xxsu-team (2nd) | 0.955 | 0.825 | 0.829 |
| stylospies | 0.959 | 0.786 | 0.791 |
| better-call-claude | 0.929 | 0.815 | 0.731 |
| **DeSAF, mean of 3 seeds** | **0.953** | **0.824** | **0.826** |
| 95% interval for that mean (test sampling) | 0.947-0.959 | 0.814-0.834 | 0.817-0.835 |
| DeSAF + tier-aware head, true level given (9 runs) | 0.969 | 0.833 | 0.844 |
| Same kind of head trained without the level (9 runs) | 0.958 | 0.818 | 0.820 |

PAN team scores are copied from the PAN 2025 overview paper. Ours were computed by us on the released test labels after the competition, so this is a fair comparison on the same data and measure, but it is not a blind shared-task submission. The interval covers test sampling only. Across seeds our scores range 0.948-0.956 (Easy), 0.822-0.826 (Medium) and 0.819-0.830 (Hard).

## Why Hard is hard

We measure each clue with an AUC: the chance that an author change looks more different than a same-author boundary. 0.5 means the clue tells you nothing. Above 0.5 is the normal direction. Below 0.5 means changes look more alike than same-author pairs.

| Clue | PAN 2025 Hard (validation) | Same text, joined at random | Same text, joined on look-alike sentences |
|---|---|---|---|
| shared content words | 0.277 | 0.654 | 0.358 |
| shared character patterns | 0.259 | 0.658 | 0.394 |
| function words | 0.411 | 0.557 | 0.394 |

The two right-hand columns use author runs cut from PAN 2025 test documents and joined again by us. The Hard numbers replicate on the test split (0.283, 0.261, 0.423). We see the same flip on news articles from a different source (Reuters C50).

What this shows: the way documents are joined is enough to turn the classic clues around. What it does not show: that PAN built Hard exactly this way. Our recipes reproduce the direction of the clues, not their full pattern. We fixed five criteria for "same pattern" before running two extra recipes. No recipe met more than two of the five, on either text source.

Two more pieces fit the picture. The extra similarity sits right at the join: one or two sentences further back it is close to chance (0.283 at the join, 0.482 one and two sentences back). And PAN 2026 Hard, built with a similar kind of filter on different text, is not reversed. This fits the idea that the direction is set by how a benchmark is built, though two editions are not enough to prove it.

Can the five clues alone find author changes? We trained a small tree model on just the five clues, one model per level. Under the official measure a random guess scores about 0.5. On Hard the clue model reaches 0.696, but only because it reads the clues backwards. Forced to read them the normal way it is at chance (0.497), and forced to read them reversed it keeps 0.677. A clue model trained on Easy ranks Hard boundaries far worse than chance (ROC-AUC 0.221), and one trained on Hard does the same on Easy (0.203). Trained on all levels without being told the level, the clue model drops to 0.615 on Hard. DeSAF (0.819 to 0.830 on Hard) is well above all of these. Where a boundary sits in the document also carries some signal (ROC-AUC 0.59 to 0.63), although we had predicted it would not.

## The level flag

Giving the true level raises the average score from 0.865 (the same kind of head trained without the level) to 0.882. It changes how the model ranks boundaries inside a level, not only where it puts the threshold.

A wrong flag is costly. A Hard document labelled as Easy scores 0.65-0.67, far below the 0.82 of the head that gets no flag. If levels were guessed with random mistakes, the guess would have to be right for about 81-85% of documents just to match plain DeSAF, and for about 91-92% to keep half of the gain over plain DeSAF.

We then trained a level classifier (DeBERTa-v3-small reading the start and end of each document) on the PAN 2025 training documents. It guessed the level right for 85% of test documents, below the 88-95% we had predicted. Fed its guesses, the tier-aware head scores 0.863: below plain DeSAF (0.868) and no better than the head trained without the level (0.865). Its mistakes cost more than random mistakes at the same rate would, and we do not yet know why. So on PAN 2025, with this classifier, the level flag is not a usable improvement. A PAN 2026 system (ULILLE) reports 98.7% level accuracy on that year's data; whether that is reachable on PAN 2025 is untested.

## What we do not claim

- That PAN built Hard with our recipe. We show the recipe is enough to cause the flip.
- That stylometry fails in general. On randomly joined text the clues point the normal way, and so do most of them on PAN 2026 Hard and on Easy (on Easy, function words are the exception).
- That the oracle tier result is a deployable system. Real text does not come with a level label, and our learned level classifier did not recover the gain.
- That level routing cannot work. We tried one classifier on one dataset.
- That the PAN winners rely on the repeated lines too. Their predictions are not public, so we cannot test it.
- Anything from a single seed. Our three seeds differ by up to 0.011 on Hard, so every claim about the model rests on all three.

## Repository layout

```
src/
  features/      the 193 writing-style features
  models/        deberta_fusion.py: the DeSAF model
  utils/         data_loader.py: reads the official PAN 2025 splits
  evaluation/    metrics.py
scripts/         everything we ran (grouped below)
runs/            run_concat_ablation.ps1: the gate vs plain concatenation runs
logs/            console output of the runs reported in the paper
docs/            how to get the data
tests/           unit tests for the data loader and the style features (python -m pytest -q tests)
FINDINGS.md      the paper's story, claim by claim, with the numbers to cite
REAL_VERIFIED_NUMBERS.md   the full lab notebook: every checked number and its caveats
```

### Scripts by purpose

**Train, predict, score**

| Script | What it does |
|---|---|
| `train_deberta_fusion.py` | trains DeSAF |
| `eval_official_test_2025.py` | scores a checkpoint on the official test split |
| `save_test_predictions.py` | saves per-boundary predictions for later analysis |
| `rescore_official.py` | re-scores saved predictions with the official PAN measure |
| `significance_official.py`, `compute_significance.py` | paired bootstrap and McNemar tests |
| `bootstrap_ci_check.py` | confidence intervals for the scores and the construction results |
| `dataset_stats_table.py` | counts documents, boundaries and changes |

**Level flag**

| Script | What it does |
|---|---|
| `cache_stage2_features.py` | stores the frozen encoder outputs for the second-stage heads |
| `gate_lab.py` | trains the second-stage heads, with and without the level flag |
| `tier_mechanism.py` | where the flag's gain comes from, and what wrong flags cost |
| `tier_ranking_check.py` | whether the flag changes ranking or only the threshold |
| `predict_tier.py`, `tier_router.py` | classifiers that guess the level from the text |

**Clue direction and the Hard reversal**

| Script | What it does |
|---|---|
| `pan_feature_direction.py`, `feature_direction_by_tier.py` | which way each clue points, per level |
| `pan_direction_followup.py`, `pan_direction_checks.py` | the same with length and topic held fixed |
| `pan_extra_checks.py` | the lag test, clue profiles and repeated lines |
| `length_split_check.py` | splits the length effect into placement and closeness in length |
| `clue_baselines.py` | models that see only the five clues, or only the position, per level and without the level |

**Joining text ourselves**

| Script | What it does |
|---|---|
| `construction_experiment.py` | builds documents from the same text with different joining recipes |
| `construction_profile_check.py` | checks each recipe against PAN Hard's pattern |
| `c50_multiauthor_test.py` | the same idea on Reuters C50 news text |
| `c50_rescore_official.py` | scores DeSAF on the C50 documents with the official measure |
| `desaf_on_constructions.py` | runs DeSAF on the built documents |

**Checks on DeSAF itself**

| Script | What it does |
|---|---|
| `boilerplate_mask_test.py` | replaces the repeated lines and re-scores |
| `hard_error_profile.py` | where DeSAF makes its mistakes on Hard |

## Reproducing the results

Setup (Python 3.10 or newer, one GPU; we used an RTX 3050 with 4 GB):

```bash
git clone https://github.com/prisha2410/desaf.git
cd desaf
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Put the data where `docs/data_acquisition.md` says. Then, from the repository root:

```bash
# 1. train one seed (we used 43, 44 and 45)
python scripts/train_deberta_fusion.py --finetune --curriculum --year 2025 --attention-fusion --bilstm --epochs 12 --no-char-ngrams --curriculum-phases 2,2,8 --contrastive --contrastive-weight 0.3 --temperature 0.1 --seed 43 --save-checkpoint checkpoints/curriculum_228_2025only_contrastive_seed43.pt

# 2. save test predictions (threshold from the training log; 0.50 / 0.52 / 0.64 for seeds 43 / 44 / 45)
python scripts/save_test_predictions.py --checkpoint checkpoints/curriculum_228_2025only_contrastive_seed43.pt --threshold 0.50 --attention-fusion --bilstm --no-char-ngrams --contrastive --out checkpoints/seed43/test_predictions.npz

# 3. score with the official measure
python scripts/rescore_official.py

# 4. tier-aware head: cache encoder outputs, then train the heads
python scripts/cache_stage2_features.py --checkpoint checkpoints/curriculum_228_2025only_contrastive_seed43.pt --threshold 0.50 --name seed43_gate --attention-fusion --bilstm --no-char-ngrams --contrastive
python scripts/gate_lab.py --name seed43_gate --final concat_tier
```

Each analysis script explains its inputs, outputs and any prediction we fixed before running it in the docstring at the top of the file.

## Where the numbers live

- `FINDINGS.md` is the short version: what the paper says, the numbers it cites, and the sentences we are not allowed to write.
- `REAL_VERIFIED_NUMBERS.md` is the full record. Every number in it was checked against the console output of a real run. If a number is not in that file, treat it as unverified.

## Data

PAN 2025 style change detection (Zenodo, access through TIRA), PAN 2026 for one comparison, and the Reuters C50 corpus (UCI) for the cross-domain check. The data is not in this repository. See `docs/data_acquisition.md`.

## References

- Zangerle, Mayerl, Potthast and Stein (2025). Overview of the Multi-Author Writing Style Analysis Task at PAN 2025. CLEF 2025 Working Notes, CEUR-WS Vol. 4038.
- Lin et al. (2025). Team wqd at Style Change Detection in Multi-Author Writing. PAN@CLEF 2025.
- Lin, Liu, Ye and Han (2025). SCL-DeBERTa: Multi-Author Writing Style Change Detection Enhanced by Supervised Contrastive Learning. PAN@CLEF 2025 (team xxsu-team).
- Schmidt, Roemisch, Halchynska, Gorovaia and Yamshchikov (2025). Team "better_call_claude": Style Change Detection using a Sequential Sentence Pair Classifier. arXiv:2508.00675.
- He, Gao and Chen (2021). DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training. ICLR.
- Koshorek et al. (2018). Text Segmentation as a Supervised Learning Task. NAACL.
- Stamatatos (2009). A survey of modern authorship attribution methods. JASIST.
