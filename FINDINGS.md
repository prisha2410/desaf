# What the paper says

Working title: **When Style Clues Point the Wrong Way: How Document Assembly Shapes Style Change Detection.** DeSAF is the name of our detector inside the paper, not the title.

This file is the plan for the paper. For each claim it gives the sentence we write, the numbers we cite, where those numbers are recorded, and the sentences we are not allowed to write.

Every number here comes from `REAL_VERIFIED_NUMBERS.md` (the section is given in brackets). If the two files ever disagree, `REAL_VERIFIED_NUMBERS.md` wins.

All scores use the official PAN 2025 measure: two-class macro-F1, pooled over all boundaries of a level. AUC below means the chance that an author change looks more different than a same-author boundary; 0.5 is no signal, below 0.5 means changes look more alike.

## How we write the paper

Five things a good paper needs, and what each means for this one:

| Quality | What we do |
|---|---|
| Novelty | Lead with what is new: the reversal of classic clues on PAN 2025 Hard, and the joining experiment showing that assembly is enough to cause it. The detector ties the winners and is not the main novelty. Level routing is prior work (ULILLE, PAN 2026), and the encoder, BiLSTM and classifier pattern goes back to Koshorek et al. 2018. Say both plainly. |
| Clear structure | Problem, data, DeSAF, results, why Hard is hard, the level flag, limits. Each finding gets one claim sentence, one table or figure, and its caveat right next to it. |
| Replicability | Give the joining recipes exactly, the feature list, seeds, thresholds, bootstrap settings and the tests fixed before each run. Release the code and name the script behind every number. |
| Grounded in literature | Cite only papers we have read in full. Anything from the snippet-level literature check gets read before it is cited, or dropped. Use official PAN numbers, not self-reported ones. |
| Readability and precision | Short sentences, plain words, terms defined once (AUC, macro-F1, tier, junction). No em dashes. No claim stronger than its test. Failed predictions are reported, not hidden. |

## The story in three sentences

1. We built a detector, DeSAF, that is level with the best PAN 2025 systems.
2. On the Hard level, the classic writing-style clues point the wrong way, and we show that the way the documents were joined is enough to cause this.
3. Because each level was joined differently, telling the model which level it is reading helps, but only if that level is right, and our learned level classifier was not accurate enough to keep the gain.

One-line pitch: "PAN's Hard documents are joined from look-alike pieces, which makes the classic writing-style clues point the wrong way. We show this by joining the same kind of text ourselves, and we build a detector that is as good as the best systems in the competition."

## Finding 1: DeSAF is level with the best PAN 2025 systems

**We write:** "Tier-free DeSAF is level with the two best PAN 2025 systems within our test-sampling uncertainty."

**Numbers** [section 11, (N)]

| | Easy | Medium | Hard |
|---|---|---|---|
| DeSAF, mean of seeds 43/44/45 | 0.9530 | 0.8242 | 0.8261 |
| 95% interval (document bootstrap) | 0.9472-0.9586 | 0.8137-0.8343 | 0.8173-0.8352 |
| wqd / xxsu-team | 0.958 / 0.955 | 0.823 / 0.825 | 0.830 / 0.829 |
| better-call-claude | 0.929 | 0.815 | 0.731 |

- Gaps to wqd and xxsu-team: -0.005 to +0.001, each smaller than the interval half-width (0.006 / 0.010 / 0.009).
- wqd's Easy score (0.958) sits at the top edge of our interval and above all three seeds (0.948-0.956).
- Seed ranges: Easy 0.948-0.956, Medium 0.822-0.826, Hard 0.819-0.830.
- Against better-call-claude, the closest design: +0.024 Easy, +0.009 Medium, +0.095 Hard.
- The learned gate does not matter: plain concatenation scores the same (average +0.0013, paired t p 0.18; 0 of 9 seed-by-level tests significant after Holm) [4b, 11].

**Say with it:** our test scores were computed after the competition on the released labels, so this is a comparison on the same data and measure, not a blind shared-task ranking. The interval covers test sampling, not training randomness. The winners' predictions are not public, so no paired test is possible.

**Do not write:** "DeSAF beats the state of the art"; "6.5 points behind" (that compared different measures); any positive-class F1 (0.9286, 0.7198, 0.7350 and similar) next to the leaderboard.

## Finding 2: On Hard the clues reverse, and the joining is enough to cause it

### 2a. The reversal [(K), (G')]

**We write:** "At PAN 2025 Hard author changes, the two sentences are more alike than two neighbouring sentences by the same author."

| Clue (Hard) | Validation | Test |
|---|---|---|
| shared content words | 0.277 | 0.283 |
| closeness in length (abs. log length ratio) | 0.287 | 0.281 |
| character trigrams | 0.259 | 0.261 |
| function words | 0.411 | 0.423 |
| punctuation | 0.419 | 0.440 |

The pattern replicates on the held-out test split: of 123 cells with a confidence interval on both splits, none is significant in opposite directions and 114 keep the same class. Small sub-cells (per-bin rows, quotation groups) do not replicate, so we do not cite them.

### 2b. Joining text ourselves [(J), (L), (N)]

**We write:** "The sign of classical stylometric evidence at author changes is not a property of the text alone; the way splice points are chosen can reverse it."

- Random joining gives the normal direction on every clue. Reddit (PAN 2025 training text): content 0.653, trigrams 0.649. Reuters C50: 0.764, 0.716.
- Starting each new author at the sentence most similar on one clue reverses that clue: content 0.226 / 0.160, length 0.045 / 0.018, function-word family 0.268 / 0.155 (Reddit / C50).
- Topic matching alone (same_topic) reverses nothing.
- topic_style (on-topic sources, joined on content and style similarity) reverses content (0.347 / 0.314), function words (0.388 / 0.302) and trigrams (0.381 / 0.305). Built from PAN 2025 test text instead: 0.358, 0.394, 0.394 [(L)].
- Hard-tier text joined at random shows no reversal on any of 13 measures [(L)]. So the reversal is not in the Hard sentences themselves.

**Say with it:** this is a sufficiency demonstration. It shows joining can cause the reversal. It does not show that PAN built Hard this way.

### 2c. The recipes match the direction, not the full pattern [(J), (L), (O)]

**We write:** "Each recipe meets some of the criteria we fixed in advance, but none meets more than two of the five, and none reproduces PAN Hard's full pattern."

- topic_style matches the direction on most clues, but its length and trigram reversals are weaker than Hard's, its function-word reversal is stronger, and Hard's punctuation reversal is missing on Reddit text.
- Selecting on length (topic_length, topic_rawstyle) gives a length reversal stronger than Hard's on both sources and, on Reddit text only, Hard's pattern where the punctuation reversal disappears at matched length.
- Selecting on the standardised style vector (topic_style) gives a trigram reversal that survives length and topic matching on both sources; selecting on the unscaled vector does not.
- No recipe meets the length-led function-word criterion (C3) on either source.
- We did not search further, because a recipe tuned until it matches would no longer be an independent test.
- **Do not write:** "reproduces (part of) PAN Hard's profile", "PAN built Hard by selecting on topic, length and style", "a combined recipe would reproduce it".

### 2d. The extra similarity sits at the join [(L)]

**We write:** "The extra similarity sits at the junction pair, as a filter on that pair would produce. This argues against an echo of the previous author's whole run at equal strength."

- Content AUC at lag 1 / 2 / 3: 0.283 / 0.482 / 0.482 (PAN 2025 Hard validation, 5,905 boundaries).
- Known splice recipes show the same drop after lag 1; topic matching without a splice is not reversed at any lag.
- **Do not write** that this rules out replies that answer mainly the previous sentence. The test cannot separate the two.

### 2e. Length [(N)]

**We write:** "Hard changes join sentences of similar length beyond placement: holding the shorter sentence's length fixed removes only about a quarter to a third of the length reversal."

- Hard changes sit between longer sentences (placement AUC 0.718 validation / 0.704 test). Holding placement fixed: 0.356 / 0.338 (still reversed).

### 2f. PAN 2026 as a contrast [(K), (L)]

**We write:** "PAN 2026 Hard, described with a similar topical and stylistic filter, is not reversed."

- Character trigrams are the one clue whose intervals sit on opposite sides of 0.5 under all three joint matchings (2025: 0.333, 0.427, 0.401; 2026: 0.624, 0.604, 0.621).
- Content overlap at changes vs same-author pairs: ratio 2.39 in 2025 Hard, 0.84 in 2026 Hard.
- **Say with it:** the two Hard levels also differ in topic structure, sentence length and size. This fits "the sign depends on where cross-author similarity sits relative to within-author similarity" but cannot establish it.

### 2g. DeSAF on the built documents [(N)]

Numbers only, no mechanism sentence (that was the rule fixed before the run).

- C50: every recipe is below random (drops 0.033-0.079, all intervals exclude 0); the two topic-matched recipes drop most.
- Reddit test text: all four splice recipes are below random (-0.019 to -0.028) and are not separated from one another.
- **Do not write:** "a style splice costs about half as much".

### 2h. Can the clues alone find changes? [(Q)]

**We write:** "On Hard the clue model reaches 0.696, and the clues help only when read in the reversed direction: forced to read all five the usual way, the model is at chance (0.497); forced to read them reversed, it keeps 90% of its gain over 0.5 (0.677). A logistic clue model fitted on Easy ranks Hard boundaries below chance (ROC-AUC 0.221), and one fitted on Hard does the same on Easy (0.203). One tree model pooled over all levels, without the level, loses 0.081 on Hard (0.615) and ranks Hard boundaries worse (ROC-AUC 0.685), consistent with clues that point opposite ways across levels."

- Model: gradient-boosted trees on the five clues, one per level; threshold chosen on validation with the official measure. A random guess at the change rate scores about 0.5 under that measure (0.489 / 0.492 / 0.505 in the run), above predict-0 (0.439 / 0.440 / 0.453), so gains are measured from 0.5.
- Share of the gain over 0.5 on Hard: with the level, roughly half to two-thirds of the true-level head's (0.570); without it, a minority of DeSAF's (0.353).
- DeSAF is 0.123 to 0.134 above the per-level clue model on Hard on every seed (paired intervals exclude 0), although only the clue model gets the level.
- The reversed-direction model kept 90.3% of the full clue model's gain; the bar was 90%, so it passed only narrowly.
- Relative position alone reaches ROC-AUC 0.633 / 0.618 / 0.586, but it partly encodes document length (on Hard, document length alone gives 0.614).
- The cross-level transfer numbers come from logistic regression, not from the tree model; say so.
- **Do not write:** "position carries little signal" (the test failed), "DeSAF learns the clues", "the clues explain DeSAF", "DeSAF does not know the level" (say "is not given the level").

## Finding 3: The level flag helps, but only when it is right

### 3a. The oracle gain [section 11, (N)]

**We write:** "Given the true level, the head gains +0.017 on average over a head of the same design trained without the flag, on every one of 27 runs. The flag changes the ranking within a level, not only the operating point."

- Tier head 0.9688 / 0.8331 / 0.8440 (avg 0.8820) vs the same design without the flag 0.9576 / 0.8175 / 0.8200 (avg 0.8650). Over tier-free DeSAF the gain is +0.014.
- Better in 27 of 27 runs, significant after Holm in 26 of 27.
- With per-level thresholds for both heads, the flag keeps 96-99% of its gain. It also raises within-level ROC-AUC, which no threshold can change: Hard +0.018 to +0.023 on test.
- Lead with this controlled comparison (+0.0169), not with the gain over the original DeSAF (+0.0142), because the original never trained on validation.

- **Is the level an oracle?** PAN releases each level separately and PAN 2024 systems used level-specific models, so the level is available in the shared task (not for real text). The true-level head is 0.010 / 0.008 / 0.014 above the best published score per level (0.959 / 0.825 / 0.830). **Do not write** "new state of the art": the gaps are about the size of the test-sampling intervals, the head also trains on validation, and no paired test is possible.

### 3b. Where the gain comes from [(T)]

- A head without the stylometric difference vector gains about as much ([CLS]-only +0.0191 vs paper head +0.0181).
- **Do not write:** "the flag works by switching the direction of the stylometric clues". The data do not support it.

### 3c. Wrong flags [(T)]

**We write:** "A wrong flag is worse than no flag."

- All 18 wrong-flag cells are below the no-flag head. Worst case: a Hard document read as Easy, 0.650-0.668.
- A wrong flag also damages the ranking, not only the threshold [(P)]: in all 18 cells within-level ROC-AUC falls below the same heads with the true flag (by 0.006-0.176) and below a separately trained head without the flag (by 0.004-0.158; no CIs). Worst: Hard read as Easy, 0.778-0.801 vs 0.953-0.956.
- With random mistakes, a level guess must be right for about 81-85% of documents just to match tier-free DeSAF, and about 91-92% (interpolated) to keep half of the gain over it [(P) corrects the earlier 92-95%].
- Our simple level classifiers (0.76-0.86 accuracy) gave no gain; with soft predicted levels only 9 of 27 runs were better, and the 2 significant differences favoured the original.

### 3d. A learned router [(P)]

**We write:** "With a learned router (85% test accuracy) in place of the true tier, the tier-aware head scores 0.863 (9 heads, 0.860-0.865). That is below tier-free DeSAF (0.868; every head is below its encoder's original) and no better than a separately trained head without the tier input (0.865). Its errors cost more than uniformly random errors at that accuracy. ULILLE reports 0.9867 router accuracy on PAN 2026 validation; whether that is reachable on PAN 2025 is untested."

- Router: DeBERTa-v3-small, start and end of each document (512 tokens), trained on the 12,600 PAN 2025 training documents. Test accuracy 0.854 (validation 0.834); most common error Medium read as Hard (155 of 393).
- Fixed before the run: predicted accuracy 0.88-0.95 (failed); decision "does not recover the gain" (0.8629 is below the 0.8678 bar).
- It is no more accurate than our earlier feature-based level classifiers (0.853-0.863).
- The kind of confusion does not explain the extra cost; which documents it misroutes is untested.
- **Do not write:** "our best router", "tier routing does not work", "PAN 2025 tiers cannot be identified", "its Medium-as-Hard errors explain the loss".

### 3e. The flag does not specifically fix look-alike joins [(O)]

Post hoc, no confidence intervals, so this is a description only. On Hard, the flag's within-third ranking gain is smallest in the most look-alike third of joins on all three encoders (+0.006 to +0.013 vs +0.015 to +0.020 in the least look-alike third), and its recall gain there comes with more false positives. The pre-registered sentence about this ("the flag helps where similarity misleads") was not written because it depended on a test that failed.

**Do not write:** "the flag fixes look-alike joins", "the flag helps where similarity misleads", "the gain at look-alike joins is a threshold artefact".

**Prior work to cite:** routing by level is not new. ULILLE (PAN 2026) used a document-level router with 0.9867 validation accuracy on PAN 2026. Present our result as "what the flag does and what a guess must achieve", not as a new idea.

## Checks that limit the claims

### Repeated lines (boilerplate) [(L), (O)]

- 42 lines repeat at least 10 times in the test documents (subreddit rules, bot messages). All 42 also appear in the training text.
- Boundaries next to them: 20.8% of Easy, 26.5% of Medium, 4.2% of Hard. DeSAF is near-perfect there (0.996 Easy, 0.985-0.986 Medium).
- **We write:** "On boundaries where exactly one side is a repeated line (12.0% of Easy and 9.8% of Medium boundaries), replacing the 42 repeated lines with one neutral placeholder sentence lowers DeSAF's macro-F1 from 0.990-0.992 to 0.507-0.550 (Easy) and from 0.936-0.942 to 0.254-0.394 (Medium). Over all boundaries touching a repeated line it falls from 0.996 to 0.643-0.683 (Easy) and from 0.985-0.986 to 0.505-0.635 (Medium), and the Easy and Medium scores fall by 0.12 and 0.14-0.19. Part of this is forced by the design: 57 (Easy) and 100 (Medium) changes between two different repeated lines become identical placeholders that no model can detect."
- Hard: 0.819-0.830 falls to 0.798-0.808.
- The rule fixed before the run gives "in between", so we draw no conclusion about whether DeSAF relies on having seen these strings. Say in the paper that the rule was ambiguous about its comparison baseline and that we chose the reading that does not trigger the claim.
- **Do not write:** "DeSAF memorised the boilerplate", "the winners rely on it too", or a comparison of the masked scores with the 0.83 / 0.90 rule over all nearby boundaries.
- Caveats: one placeholder sentence, not checked against the training text; masking also changes the document context the model reads; thresholds were not re-tuned.
- The official scores are unchanged, but whether the winners are as sensitive to these lines is untested. So lead the parity claim with Hard, where these lines touch only 4.2% of boundaries.

### Where DeSAF errs on Hard [(O)]

- **We write:** "The pre-registered test that DeSAF misses more Hard changes at the most alike junctions (recall at least 0.10 lower on every encoder) failed narrowly (gaps +0.092 / +0.126 / +0.145). That gap comes from the least-alike third, which holds only 100 of 1,911 Hard changes; recall in the middle and most-alike thirds is the same. DeSAF's within-third ROC-AUC on Hard is lowest among the most alike third of junctions (0.881-0.884, vs 0.943-0.951 and 0.970-0.981), which holds 72% of Hard changes. The weaker separation there shows mainly as more false positives on look-alike same-author pairs, not as lower recall." Label it post hoc; no confidence intervals were computed.
- **Do not write:** "DeSAF misses Hard changes where the junction is most alike", "look-alike joins explain most of Hard's errors".

## Tests fixed in advance that did not come out as predicted

Report these. They make the rest more believable.

- DeSAF on Reddit constructions: random AUC at least 0.90 (got 0.809); topic-matched recipes lowest (no) [(M)].
- Length split: P1 (0.356, just under 0.36) and P3 failed [(N)].
- Correction (2026-11-19): the prediction for the extra recipes was "no full match", and it HELD [(O)]; do not list it as failed. What failed is the generic expectation in construction_experiment.py that clues a recipe does not select on stay usual or at chance (Table III: e.g. content splice reverses trigrams, length splice reverses Reddit punctuation, topic_length reverses punctuation and readability).
- Error profile: P1 missed by one encoder (0.092 vs 0.10) [(O)].
- Recipes: the critic's prediction for topic_rawstyle (meets C1, C2, C5) was partly wrong, and the expectation that cues a recipe does not select on stay normal failed for topic_length (punctuation and readability also reversed) [(O)].
- Boilerplate masking: neither pre-registered sentence triggered [(O)].
- Router: predicted test accuracy 0.88-0.95, got 0.854; the router-fed head does not recover the gain [(P)].
- Clue baselines: relative position alone was predicted to reach ROC-AUC at most 0.55 on every level; it reached 0.59-0.63 [(Q)].

## Limitations to state

1. Test scores are post hoc on released labels, not a blind submission.
2. Three encoder seeds. Architecture claims rest on all three.
3. The tier-aware head uses the true level (oracle) and is trained on the validation split. One learned router on one dataset did not recover its gain.
4. The joining experiments show sufficiency, not how PAN built Hard.
5. 2025 vs 2026 is a descriptive contrast between two editions that differ in many ways.
6. Part of the Easy and Medium scores sits next to 42 repeated lines.

## Numbers that must not appear in the paper

- Positive-class F1 values set against the leaderboard (0.9286 / 0.7198 / 0.7198, 0.7944, 0.7350 ensemble).
- "6.5 points behind", "~6.5pt".
- PAN 2024 as training data (we train on PAN 2025 only).
- "The flag flips the stylometric clues", "about half", "the rest is closeness in length".
- better_call_claude's self-reported 0.923 / 0.828 / 0.724. Use the official 0.929 / 0.815 / 0.731.
