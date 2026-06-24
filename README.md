# 🏀 TakeMeter — Measuring Discourse Quality on r/nba

TakeMeter is a fine-tuned text classifier that sorts r/nba comments into **four kinds of
"take"** — `analysis`, `hot_take`, `reaction`, and `banter`. It fine-tunes
`distilbert-base-uncased` on 568 hand-curated real comments and benchmarks it against a
zero-shot Llama-3.3-70B baseline.

The interesting result isn't a leaderboard win — it's a clean, honest finding about **what a
small fine-tuned model actually learns** versus what the labels intended.

> Design notes and decision rules live in [`planning.md`](planning.md); this README is the
> standalone report.

## TL;DR results (held-out test set, n=86)

| Metric | Fine-tuned DistilBERT | Groq Llama-3.3-70B (zero-shot) |
|---|---|---|
| Accuracy | 0.581 | **0.674** |
| **Macro-F1** | 0.580 | **0.664** |
| Weighted-F1 | 0.586 | **0.698** |

The 66M-parameter fine-tuned model **trails the 70B zero-shot baseline by ~0.08 macro-F1** — but
it **beats the baseline on `reaction`** (F1 0.53 vs 0.39), and its failures are systematic and
diagnosable. The headline lesson: fine-tuning on ~400 examples learned the *surface* of a take
("does this sound like a forceful opinion?") but not the *pragmatic intent* the labels encode
("what is the speaker trying to do — reason, assert, feel, or joke?").

## Demo

A ~7:40 screen-recorded walkthrough: the Gradio app classifying live r/nba comments — three
correct predictions and one instructive failure — followed by a tour of the evaluation report.
Narration follows [`demo/demo_script.md`](demo/demo_script.md).

<!-- INLINE PLAYER: drag demo/takemeter_demo.mp4 into a GitHub Issue or PR comment, wait for the
     upload to finish, then paste the resulting https://github.com/user-attachments/assets/...
     URL on its own line directly below this comment. GitHub renders that bare URL as a player. -->

▶️ **Watch:** [`demo/takemeter_demo.mp4`](demo/takemeter_demo.mp4) (8.3 MB, H.264/AAC) — click to
play in GitHub's file viewer; or add the inline `user-attachments` embed above for an in-README
player.

---

## 1. Community

**[r/nba](https://reddit.com/r/nba).** Its entire culture is *opinions about basketball*, and the
community itself constantly sorts those opinions by quality: a sourced film breakdown, a drive-by
"Luka is top-5 ever," a game-thread freak-out, and a one-line joke about the refs all sit in the
same thread and are received very differently. That makes the quality distinction **native to how
the community talks**, the discourse is **varied enough to be learnable**, and real comments are
**freely available** via the Arctic Shift API. (Full reasoning in `planning.md §1`.)

## 2. Label Taxonomy

Four labels, assigned by a comment's **dominant communicative intent**:

| Label | Definition | Example 1 | Example 2 |
|---|---|---|---|
| **analysis** | Structured argument backed by specific, verifiable evidence **and a stated mechanism** (a "why"), not just a stat next to a verdict. | *"Giannis at 28% from three is why teams wall off the paint and dare him to shoot — the spacing collapses."* | *"No, there was more spacing league-wide in prime Harden's era due to the 3-point focus that started with the 2014 Spurs / 2015 Warriors."* |
| **hot_take** | Bold, confident opinion / ranking / prediction asserted with little or no support; any stat is decorative. | *"Luka is already a top-5 player of all time, no debate."* | *"The Lakers are making the Finals this year, book it."* |
| **reaction** | Immediate emotional response to a moment — a feeling, no argument, no constructed joke. | *"NOOO not again, I can't watch this team blow another 20-point lead 😭"* | *"LETS GOOO that buzzer beater took ten years off my life"* |
| **banter** | Primary purpose is humor — joke, pun, meme, sarcasm, playful trash-talk. | *"Refs really said 'and 1' for breathing on him."* | *"Embiid load-managing this comment section too, didn't even show up."* |

The hardest boundary is **`banter` ↔ `reaction`** (a sarcastic vent carries both a joke and a real
feeling). The decision rule — the **construction test** — labels `banter` if *any* deliberately
constructed comedic device is present, even alongside genuine frustration. The `hot_take` ↔
`analysis` rule requires `analysis` to show a **mechanism**, not just a stat. Full rules +
worked edge cases: `planning.md §3`.

## 3. Dataset

- **Source.** 4,153 real r/nba comments pulled from the **Arctic Shift API** (the public Pushshift
  successor) across 12 anchor dates spanning the 2023–24, 2024–25, and 2025–26 seasons, so the
  data isn't from one news cycle. Filtered to remove deleted/bot/quote/link-only comments and
  anything under 5 or over 80 words → a 570-comment length-stratified candidate set.
  ([`src/collect_data.py`](src/collect_data.py))
- **Labeling process.** Each candidate was labeled by **two independent LLM annotator passes**
  applying the `planning.md` rules; the **68 disagreements were resolved by an adjudicator pass**.
  Every label is therefore a reviewed consensus or adjudicated decision. Inter-annotator
  agreement was **Cohen's κ = 0.842** (88.4% raw, n=568) — "almost perfect." All AI assistance in
  labeling is disclosed in [§10](#10-ai-usage).
- **Final dataset: 568 labeled comments** ([`data/takemeter_nba.csv`](data/takemeter_nba.csv)).

  | Label | Count | Share |
  |---|---|---|
  | hot_take | 174 | 30.6% |
  | banter | 155 | 27.3% |
  | analysis | 149 | 26.2% |
  | reaction | 90 | 15.8% |

  No class exceeds the 70% ceiling. `reaction` is the limiting class at 15.8% — a deliberate trade
  of perfect balance for ~58% more training data (a balanced 360-example set left the small model
  data-starved; expanding to 568 lifted test macro-F1 from 0.54 → 0.58). Split 70/15/15 with a
  fixed seed → train 397 / val 85 / test 86.

### Three genuinely difficult examples (and the decisions)

1. **`hot_take` vs `analysis`** — *"The biggest reason you won the series (by far) is Kawhi being
   injured. The second biggest reason is Paul George turning into Pandemic P…"* → **`hot_take`**.
   It *looks* analytical (it ranks "reasons") but cites no verifiable evidence and states no
   mechanism — it asserts a confident subjective ranking. (This is the case the mechanism rule
   was sharpened for.)
2. **`reaction` vs `hot_take`** — *"It's just unnecessary and frankly disrespectful… it's a fun
   contest and the man is getting 1 minute of TV…"* → **`reaction`**. An emotional response bound
   to the just-happened commentary, no standalone claim.
3. **`analysis` vs `reaction`** — *"He got a pass and had no idea what to do with the ball… he was
   a liability by the 2nd half."* → **`reaction`**. Play-by-play recollection *feels* like
   evidence, but anecdote tied to specific moments isn't verifiable stats/scheme with a mechanism.

## 4. Fine-tuning approach

- **Base model:** `distilbert-base-uncased` (66M params), HuggingFace.
- **Platform:** local — Python 3.12 + PyTorch on Apple-Silicon MPS (CPU works too), ~2.5 min/run —
  **and reproduced on a Google Colab T4 GPU** (the rubric's intended environment) via the `colab`
  CLI (`src/colab_pipeline.py`). The Colab run produced test **macro-F1 0.616** (vs 0.580 local;
  the ~0.04 gap is GPU-vs-MPS run-to-run nondeterminism) with the **same dominant `analysis` →
  `hot_take` failure** — confirming the findings aren't a hardware artifact. Colab-native artifacts
  are in [`colab/`](colab/).
- **Setup:** 4-class sequence classification, max_len 128, AdamW, warmup 0.1, weight decay 0.01,
  seed 42. ([`src/train.py`](src/train.py))
- **Key hyperparameter decision — epochs + early stopping (not the default fixed 3).** With only
  397 training examples on a subjective task, a fixed epoch count is a guess. I trained up to 16
  epochs with **early stopping on validation macro-F1 (patience 4) and load-best-model**, then ran
  a **5-config sweep** over learning rate / batch / class-weighting ([`src/sweep.sh`](src/sweep.sh)).
  - Validation macro-F1 by epoch climbed non-monotonically: **0.27 → 0.43 → 0.48 → 0.46 → 0.56 →
    0.57 → 0.61 (best, epoch 7)**. The default 3 epochs would have stopped at **0.48** — early
    stopping bought **+0.13 macro-F1**.
  - Best config: **lr 4e-5, batch 16, 16 max epochs**. **Class weighting *hurt*** (val 0.56 vs
    0.61) despite the mild imbalance, so it was dropped — a decision driven by observation, not
    default. (`outputs/sweep_results.json`, `outputs/train_log.json`)

## 5. Baseline (zero-shot Groq)

- **Model:** `llama-3.3-70b-versatile` via Groq, temperature 0, on the **same 86-example test
  set**. ([`src/baseline_groq.py`](src/baseline_groq.py))
- **Prompt:** a system prompt containing the *exact same* four label definitions and boundary
  decision rules used for annotation (from [`src/taxonomy.py`](src/taxonomy.py)), instructing the
  model to "Respond with ONLY the single label word." Predictions were parsed to one of the four
  labels; **2 of 86 were unparseable** (2.3%, well under the 10% threshold) and counted as wrong.
- Results were written to `outputs/baseline_results.json` and merged into the evaluation report.

---

## 6. Evaluation Report

### 6.1 Overall + per-class (both models)

| Class | FT precision | FT recall | **FT F1** | Baseline F1 |
|---|---|---|---|---|
| analysis | 0.85 | 0.48 | **0.61** | 0.85 |
| hot_take | 0.49 | 0.69 | **0.57** | 0.71 |
| reaction | 0.47 | 0.62 | **0.53** | 0.39 |
| banter | 0.68 | 0.54 | **0.60** | 0.71 |
| **macro avg** | | | **0.58** | **0.66** |
| **accuracy** | | | **0.581** | **0.674** |

Two things stand out: the fine-tuned model has **high `analysis` precision (0.85) but low recall
(0.48)** — it only flags the most obvious breakdowns — and **`reaction` is the one class where
fine-tuning beats the baseline** (0.53 vs 0.39, because the 70B model over-predicts reaction,
precision 0.30).

### 6.2 Confusion matrix — fine-tuned model (rows = true, cols = predicted)

| true \ pred | analysis | hot_take | reaction | banter |
|---|---|---|---|---|
| **analysis** | 11 | **11** | 1 | 0 |
| **hot_take** | 2 | 18 | 2 | 4 |
| **reaction** | 0 | 3 | 8 | 2 |
| **banter** | 0 | 5 | 6 | 13 |

(Also committed as [`outputs/confusion_matrix.png`](outputs/confusion_matrix.png).) The dominant
off-diagonal cell is **`analysis` → `hot_take` (11 of 23)**. `hot_take` is a **sink**: 19 of 36
total errors land there (11 from analysis, 5 from banter, 3 from reaction).

### 6.3 Three wrong predictions, analyzed

1. **`analysis` → `hot_take`** (conf 0.79) — *"His defensive box plus minus is gassed by his
   rebounds n assist."* This is genuine `analysis`: it names a *mechanism* — DBPM is inflated
   because the box-score proxy rewards rebounds/assists. The model calls it `hot_take`. Crucially
   this comment **contains stats** and is still pushed out of analysis, which **refutes the easy
   "it just keys on stat-words" hypothesis** — the model keys on *assertive tone*, not evidence.
2. **`banter` → `hot_take`** (conf **0.986**) — *"At this point, he's worse than a traffic cone."*
   Comedic hyperbole — clearly `banter`. The model is **98.6% confident it's a hot_take** because
   the *surface* is a dismissive evaluative statement about a player. It has no representation of
   the *humor frame*, and (see calibration) it is most confident exactly where it's wrong.
3. **`hot_take` → `analysis`** (conf 0.97) — *"The biggest reason you won the series is Kawhi being
   injured. The second biggest is Paul George turning into Pandemic P…"* The **reverse** error: a
   confident, *unsupported* ranking gets labeled `analysis` because it's long and structured. The
   model reads **structure/length as reasoning** — it cannot tell a ranked list of asserted causes
   from an argued one. Together, #1 and #3 show the model has no notion of *mechanism*: it sorts on
   how forceful/structured a take sounds, not on whether it is actually supported.

### 6.4 Sample classifications (fine-tuned model)

| Comment (truncated) | Predicted | Conf | True | ✓ |
|---|---|---|---|---|
| *"…more spacing league-wide in prime Harden's era due to the 3-pt focus that started with the 2014 Spurs…"* | analysis | 0.99 | analysis | ✅ |
| *"I would rather Brunson take the offseason off, he already did FIBA last year"* | hot_take | 0.99 | hot_take | ✅ |
| *"I can't wait for the tell-all book on this. Or the netflix/ESPN 30/30"* | banter | 0.47 | banter | ✅ |
| *"At this point, he's worse than a traffic cone."* | hot_take | 0.99 | banter | ❌ |
| *"I just cannot comprehend how this isn't a 'windup'…"* | reaction | 0.43 | analysis | ❌ |

**Why the first is reasonable:** it states an explicit causal mechanism (the 3-point revolution
→ more spacing) with concrete actors (2014 Spurs, 2015 Warriors). That's the prototypical
`analysis` the model learned best — and it's confidently, correctly classified.

### 6.5 Confidence calibration (stretch)

Is a 90%-confident prediction more reliable than a 60%-confident one? **Only at the extremes — the
model is broadly overconfident, ECE = 0.317.** ([`outputs/calibration.png`](outputs/calibration.png))

| Confidence bin | n | Accuracy | Mean conf |
|---|---|---|---|
| 0.25–0.50 | 4 | 0.50 | 0.47 |
| 0.50–0.60 | 3 | 0.67 | 0.54 |
| 0.60–0.70 | 2 | 0.50 | 0.63 |
| 0.70–0.80 | 10 | **0.30** | 0.76 |
| 0.80–0.90 | 9 | **0.33** | 0.86 |
| 0.90–1.00 | 58 | 0.67 | 0.97 |

Calibration is **inverted in the 0.70–0.90 band**: a 0.75–0.86 score signals an *error* more often
than a coin flip, while the model dumps 58 of 86 predictions into the 0.90+ bin (including 12
errors at ≥0.95). Confidence is not a usable correctness signal at any single threshold.

### 6.6 Systematic error patterns (stretch)

Surfaced by a multi-lens AI analysis and then **verified against the data** (see [§10](#10-ai-usage)
and `outputs/failure_analysis.json`). The verification pass rejected several plausible-but-wrong
patterns (e.g., a "stat-words drive the label" hypothesis — refuted; a misattributed-direction
pattern — dropped). The patterns that survived:

- **`hot_take` is a default attractor.** 19 of 36 errors land on `hot_take`; whenever a comment
  contains *any* evaluative claim, the model resolves uncertainty by confidently asserting
  `hot_take` instead of representing how the claim is supported.
- **`analysis` collapses into `hot_take`, overconfidently.** 11 of 12 analysis errors; 7 fire at
  ≥0.94. Hedges ("would like to validate"), citations ("looked it up"), and stats do not move it.
- **`banter` has the weakest containment.** Jokes scatter to `reaction` (6) and `hot_take` (5);
  **deadpan/roleplay jokes with no overt humor markers** (the credit-card "phishing" bit) are read
  literally as sincere `reaction`. The model has no ironic-frame feature.
- **The error is not explained by length or stat-vocabulary** — the misread analyses span 46–326
  chars and most contain no numbers. The operative surface cue is *assertive tone*.

---

## 7. Reflection — what the model captured vs. what I intended

The labels were designed to capture **pragmatic intent**: to *argue* (analysis), to *assert a
staked opinion* (hot_take), to *express a feeling* (reaction), or to *amuse* (banter). The model
did **not** learn intent. It learned a single surface axis — **"evaluative assertiveness"** — and
collapses everything toward the `hot_take` pole on it.

The decisive evidence that this is surface-feature learning, not reasoning:
- A **hedged, sourced argument** ("…in terms of regular-season win %, 53.7% was his worst season")
  is read as a spicy assertion *purely because a judgment is present* — the model encodes *that* a
  claim exists, never *how* it's supported. So it nails `analysis` **precision** (0.85: when it
  commits to analysis it's right) but tanks **recall** (0.48: it misses anything that doesn't look
  like a prototypical essay).
- On the affective side, a deadpan joke is taken as a sincere `reaction` because it has no
  humor-marker feature, and an emotional outburst with a value word is taken as a staked `hot_take`.
- It is **most confident (≥0.97) exactly where it is wrong** (analysis→hot_take), and the failures
  are **not** explained by length or stat-words.

Most pointed of all: the boundary I made the *most* rigorous in label design — requiring `analysis`
to show a **mechanism**, not just a stat — is the boundary the model fails **hardest** on, because
"is this claim actually supported by reasoning?" is precisely the inference a 66M-parameter encoder
can't make from 397 examples, while a 70B model with world knowledge can (its `analysis` precision
is a perfect 1.00). The gap between intended and learned is the gap between **meaning** and
**surface** — and it's also why fine-tuning *helped* on `reaction` (a more lexical/tonal class) but
*hurt* relative to the LLM on `analysis` (a reasoning class).

### Was it "good enough"? (success criteria from `planning.md §6`)

| Criterion | Target | Result | Met? |
|---|---|---|---|
| Macro-F1 | ≥ 0.65 | 0.58 | ❌ |
| Beat baseline macro-F1 | by ≥ 0.10 | −0.08 (lost) | ❌ |
| No dead class (min F1) | ≥ 0.45 | 0.53 | ✅ |
| `analysis` F1 | ≥ 0.60 | 0.61 | ✅ |

Two of four. The "beat a 70B zero-shot by 0.10" bar was, in hindsight, optimistic for a 66M model
on 400 examples of a pragmatic task — and recognizing *why* is the real result: this is a task
where model scale and world knowledge matter more than a small supervised signal.

## 8. Stretch features

- **Inter-annotator reliability** — two independent annotation passes, **Cohen's κ = 0.842** (§3).
- **Confidence calibration** — reliability diagram + ECE 0.317; inverted mid-band (§6.5).
- **Error-pattern analysis** — multi-lens, data-verified systematic patterns (§6.6).
- **Deployed interface** — Gradio app: paste a comment → label + confidence bars
  ([`src/app.py`](src/app.py); `python src/app.py`).

## 9. Spec reflection

- **One way the spec helped:** `planning.md` forced me to run the **label stress-test before
  annotating**. Generating boundary comments and double-annotating them exposed that the original
  `banter`↔`reaction` rule ("humor incidental") gave no answer for hybrid joke-and-vent comments. I
  rewrote it into the concrete **construction test** *before* labeling 568 examples — so the whole
  dataset was annotated under the sharpened rule instead of being re-litigated afterward.
- **One way implementation diverged:** the spec planned a **balanced ~280** dataset. In practice
  the balanced set left DistilBERT data-starved (test macro-F1 0.54), so I expanded to **all 568
  consensus labels** (accepting `reaction` at 16%) to give the small model more signal, which
  lifted it to 0.58. I traded a clean 25%-each story for empirical performance — and documented the
  trade rather than hiding it.

## 10. AI usage

This project used AI tools at three points; in each I directed the tool and then reviewed/overrode
its output.

1. **Label stress-testing (before annotation).** I directed an LLM to *generate boundary comments
   engineered to break* the taxonomy, then had two independent passes classify them. It produced
   18 hybrids that exposed the weak `banter`↔`reaction` seam (89% agreement; the disagreement was
   a joke-and-vent hybrid). **I overrode the original "humor incidental" rule** and wrote the
   construction test. (`outputs/label_stress_test.json`)
2. **Annotation assistance (disclosed).** The 570 candidates were **pre-labeled by an LLM**, then
   **independently re-labeled by a second LLM pass**, with disagreements sent to an adjudicator —
   i.e., labels are consensus/adjudicated, not a single unreviewed pass (κ = 0.842). I designed the
   two-pass-plus-adjudication process specifically so no label shipped unreviewed, and I set the
   selection policy to keep adjudicated hard cases at their natural rate rather than dropping them.
3. **Failure-pattern analysis (after evaluation).** I directed four LLM "lenses" to propose error
   patterns, then a verification pass to check each against the actual predictions. **I kept the
   verifier's rejections** — it caught a wrong "stat-words drive the label" pattern and a
   misattributed-direction pattern, and flagged a bad denominator (it wrote "58 of 89"; the test
   set is 86). The §6.6/§7 write-up uses only the verified patterns. (`outputs/failure_analysis.json`)

---

## Reproduce

```bash
# 0. environment (Python 3.12)
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate

# 1. data  (re-pull is optional; data/takemeter_nba.csv is committed)
python src/collect_data.py            # pull + filter real r/nba comments (writes data/raw_pool.csv, candidates.csv)
#   labeling is done by the workflow in the repo history; the labeled CSV is committed.

# 2. split, baseline, train, evaluate
python src/prepare_split.py           # 70/15/15, fixed seed
echo "GROQ_API_KEY=gsk_..." > .env      # gitignored; get a key at console.groq.com
python src/baseline_groq.py           # zero-shot Llama-3.3-70B on the test set
python src/train.py --lr 4e-5 --epochs 16 --batch 16 --patience 4
python src/evaluate.py                # metrics + confusion_matrix.png + evaluation_results.json
python src/calibration_plot.py        # reliability diagram

# 3. try it
python src/app.py                     # Gradio interface at http://127.0.0.1:7860
```

**Reproduce the fine-tune on a Colab T4 GPU** (the rubric's intended environment), via the
[`google-colab-cli`](https://pypi.org/project/google-colab-cli/):

```bash
colab new -s takemeter --gpu T4
for f in train val test; do colab upload -s takemeter data/$f.csv /content/$f.csv; done
colab upload -s takemeter outputs/baseline_results.json /content/baseline_results.json
colab exec  -s takemeter -f src/colab_pipeline.py --timeout 600
colab download -s takemeter /content/colab_evaluation_results.json colab/colab_evaluation_results.json
colab download -s takemeter /content/confusion_matrix.png          colab/confusion_matrix.png
colab stop -s takemeter
```

### Repository map

| Path | What |
|---|---|
| `planning.md` | Design spec: taxonomy, decision rules, stress-test, metrics, success criteria |
| `data/takemeter_nba.csv` | The 568-example labeled dataset (the deliverable) |
| `data/all_labeled.csv`, `data/difficult_examples.json` | All 568 consensus labels; adjudicated hard cases |
| `src/` | `collect_data`, `prepare_split`, `train`, `baseline_groq`, `evaluate`, `app`, `taxonomy`, `predict`, `colab_pipeline` |
| `outputs/evaluation_results.json` | Full metrics for both models, confusion matrix, wrong preds, calibration |
| `outputs/confusion_matrix.png`, `outputs/calibration.png` | Figures |
| `outputs/label_stress_test.json`, `outputs/failure_analysis.json` | AI-assisted stress-test + verified error patterns |
| `colab/` | Colab T4-GPU reproduction artifacts (`colab_evaluation_results.json`, `confusion_matrix.png`) |
| `demo/demo_script.md` | Demo narration + macOS recording + ffmpeg/GitHub guide |
| `demo/takemeter_demo.mp4` | The demo screen-recording, down-converted for GitHub (8.3 MB, H.264/AAC) |

> The fine-tuned model weights (~270 MB) exceed GitHub's 100 MB limit and are gitignored; run
> `src/train.py` to regenerate `model/` (≈2.5 min on CPU/MPS).
