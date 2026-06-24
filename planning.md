# TakeMeter — Planning & Spec

A fine-tuned text classifier that measures **discourse quality** in **r/nba** by sorting
comments into four kinds of "take." This document is the design thinking *before and during*
the project: label definitions, edge-case rules, data plan, evaluation metrics, success
criteria, and the AI tool plan. The polished reader-facing report lives in `README.md`.

> **Status:** `planning.md` written before data collection (Milestones 1–2). The "Difficult
> examples encountered during annotation" subsection (§3.3) and any label-definition tweaks
> from stress-testing are filled in as the project progresses.

---

## 1. Community

**Choice: [r/nba](https://reddit.com/r/nba).**

r/nba is a high-volume, text-heavy community whose entire culture is *opinions about
basketball* — and the community itself constantly, explicitly sorts those opinions by quality.
A well-sourced film breakdown, a drive-by "Luka is top-5 ever," a game-thread freak-out, and a
one-line joke about the refs all coexist in the same comment section and are received very
differently by the community. That makes the quality distinction **native to how people there
already talk**, not an external rubric I'm imposing.

Why it's a good fit for a classification task specifically:

- **Varied discourse quality.** The same thread contains rigorous analysis, lazy hot takes,
  raw emotional reactions, and pure jokes. That variety is exactly what a classifier needs —
  if every comment were the same kind of take, there'd be nothing to learn.
- **Grounded, recognizable boundaries.** A regular can tell "this person actually watched the
  game and is making an argument" from "this person is just yelling." The labels operationalize
  a distinction the community already feels.
- **Abundant public data.** Real comments are freely available via the Arctic Shift API (the
  public Pushshift successor), so I can collect across the full quality range without scraping
  hassle or private content.

---

## 2. Label Taxonomy (4 classes)

The labels measure **what kind of contribution a comment makes**, ordered roughly from most to
least substantive. They are designed to be mutually exclusive (one label per comment) and
exhaustive enough to cover >90% of real r/nba comments without an "other" bucket.

### `analysis`
> A comment that makes a **structured argument backed by specific, verifiable evidence** —
> statistics, tactical/schematic observation, historical comparison, or rules/CBA knowledge —
> where the *reasoning*, not the opinion, is the point.

The test: if you stripped the opinion framing, would specific, checkable evidence remain that
genuinely supports a claim **through stated reasoning — a mechanism, a "why," a causal link** —
rather than just a stat sitting next to a verdict? If yes → `analysis`. A real, verifiable stat
deployed only to prop up a subjective superlative or ranking ("best ever," "top-3," "MVP lock")
with no mechanism is `hot_take`, not analysis. *(This mechanism requirement was added after
label stress-testing — see §3.5.)*

- *"Giannis shooting 28% from three is why teams just wall off the paint and dare him to shoot — the spacing collapses and Dame can't operate in a phone booth."*
- *"People forget the '04 Pistons had four All-Stars in their primes and a historically elite defensive rating. That title wasn't a fluke; it was the best defense beating the best offense."*

### `hot_take`
> A comment that asserts a **bold, confident opinion, ranking, or prediction with little or no
> supporting evidence**; any statistic it cites is decorative or cherry-picked rather than part
> of a genuine argument.

The claim might even be true — but the comment *asserts* rather than *argues*.

- *"Luka is already a top-5 player of all time, no debate."*
- *"The Lakers are making the Finals this year, book it."*

### `reaction`
> A comment that is an **immediate emotional response to a specific event or moment** — a play,
> a result, a trade, an injury — expressing a feeling with little to no argument.

The point is the *feeling in the moment*, not a claim or a joke.

- *"NOOO not again, I can't watch this team blow another 20-point lead 😭"*
- *"LETS GOOO that buzzer beater just took ten years off my life"*

### `banter`
> A comment whose **primary purpose is humor** — a joke, pun, meme, sarcastic one-liner, or
> playful trash-talk — rather than sincere analysis, opinion, or emotion.

The point is to be funny. Sincerity is incidental or absent.

- *"Embiid load-managing this comment section too, didn't even show up."*
- *"Refs really said 'and 1' for breathing on him."*

### Mutual exclusivity check
Each label is defined by the comment's **dominant communicative intent**: to *argue*
(analysis), to *assert* (hot_take), to *feel* (reaction), or to *amuse* (banter). When a comment
does two things, the decision rules in §3 pick the dominant one. Spot-checking ~30 raw comments
during label design, >90% sorted cleanly into exactly one class, which is the bar for
"exhaustive enough without an 'other' bucket."

---

## 3. Hard Edge Cases

Some comments genuinely sit on a boundary. That's not a flaw in the taxonomy — it's where the
taxonomy needs an explicit **decision rule** so two readers (or annotation passes) agree.

### 3.1 Primary hard case (the named one): `banter` ↔ `reaction`

A sarcastic vent after a tough loss is the single hardest case, because it carries *both* a
genuine feeling and a comedic delivery.

> *"I'm going to bed. Wake me when this franchise hires a GM who can read a box score 💀"*

Could be `reaction` (despair/anger after a loss) or `banter` (a sarcastic bit with a punchline).

**Decision rule (the "construction test," sharpened after stress-testing — see §3.5):** A
comment can carry *both* a real feeling and a joke at once — that 50/50 feeling is **not** a
reason to hesitate. Decide by structure, not vibe:
- If the comment contains **any deliberately constructed comedic device** — a
  metaphor/simile-for-laughs, a setup-and-payoff bit, a meme/reference, sustained sarcasm, or a
  self-deprecating absurd hypothetical — label `banter`, **even when genuine frustration is also
  present**. The effort spent crafting the joke is the dominant communicative choice.
- Reserve `reaction` for venting with **no comedic device**: a raw exclamation, profanity,
  all-caps despair, a bare "we are so done."
- **Demote emojis/tone.** Distress emojis (😭🫠) and an upset tone do *not* by themselves make a
  comment `reaction` when the surrounding text is a constructed joke — there they're part of the
  comedic delivery (bathos). An emoji on *otherwise bare* venting supports `reaction`.
- **Don't treat co-occurring emotion as ambiguity.** A comment isn't borderline just because
  it's funny *and* heartfelt; apply the construction test and commit.

The named example above has a constructed punchline ("a GM who can read a box score") → `banter`.
Worked hybrid: *"Down 30 in the third and I'm still watching like an idiot because hope is a
disease and I refuse the cure. love this franchise actually"* → `banter`, because "hope is a
disease and I refuse the cure" is a crafted ironic aphorism and "love this franchise actually" is
sarcastic reversal — dominant intent is humor despite the real despair underneath.

### 3.2 Secondary hard case: `hot_take` ↔ `analysis` (the rubric's canonical case)

> *"LeBron is overrated — his playoff win rate against top-seeded opponents is below .500."*

A bold claim that also cites a stat.

**Decision rule:** Label `analysis` only if removing the opinion framing leaves **specific,
verifiable evidence that genuinely supports the claim**. If the stat is vague, cherry-picked, or
decorative — present just to *sound* credible rather than to *reason* — label `hot_take`. The
example uses one selected stat for accusatory effect rather than as part of an argument →
`hot_take`.

**Sharpened after stress-testing (§3.5):** a *real, verifiable* stat is still not enough on its
own — `analysis` requires the stat to do **argumentative work through a stated mechanism**. A
genuine stat used only to prop up a subjective ranking/superlative (*"led the league in scoring
AND won a chip, so he's top-3"*) is `hot_take`; the same stat tied to a causal explanation
(*"his sub-50 TS% is **why** the offense stalls when he's the screener"*) is `analysis`.

### 3.3 Tertiary hard case: `reaction` ↔ `hot_take`

In-the-moment hype that doubles as a claim:

> *"WE'RE WINNING THE CHIP THIS YEAR!!!"* (posted seconds after a regular-season buzzer-beater)

**Decision rule:** An **emotional exclamation tied to the event that just happened** →
`reaction`. A **standalone confident prediction/ranking stated as a claim**, not bound to a
specific moment → `hot_take`. The example is an emotional eruption tied to the buzzer-beater →
`reaction`. The same words in a calm pre-season thread ("We're winning the chip this year.")
would be `hot_take`.

### 3.4 Difficult examples encountered during annotation

Three real comments where the two independent annotators **disagreed** and the adjudicator had to
apply a decision rule (all are in `data/all_labeled.csv`; 48 such cases are in
`data/difficult_examples.json`):

1. **`hot_take` vs `analysis`** — *"The biggest reason you won the series (by far) is Kawhi being
   injured. The second biggest reason is Paul George turning into Pandemic P. I don't think
   Westbrook is in the top 5 when it comes to 'reasons you won the series.'"*
   → **`hot_take`.** It *looks* analytical — it ranks "reasons" — but it cites no verifiable
   evidence and states no mechanism; it asserts a confident subjective ranking. This is exactly
   the case the §3.2 **mechanism rule** was sharpened for: structure that mimics analysis without
   doing the argumentative work.

2. **`reaction` vs `hot_take`** — *"It's just unnecessary and frankly disrespectful. This isn't
   the time and place to say people are not good enough… It's a fun contest and the man is getting
   1 minute of TV…"*
   → **`reaction`.** An emotional "this is disrespectful / over the top" response **bound to the
   just-happened commentary**, with no standalone ranking or prediction (Rule 3). The heat reads
   as a feeling in the moment, not an asserted claim.

3. **`analysis` vs `reaction`** (anecdote ≠ evidence) — *"He got a pass and had no idea what to do
   with the ball, and a few plays later he completely missed a pass to him. He should have been
   carrying the team but he was a liability by the 2nd half."*
   → **`reaction`.** Play-by-play recollection *feels* like evidence, but anecdotal "I saw him do
   X" tied to specific moments is not verifiable stats/scheme with a stated mechanism, so it fails
   the `analysis` test and lands as event-bound venting (Rule 2/3).

**Inter-annotator reliability (stretch, computed here):** across all 568 double-labeled comments,
the two independent annotators reached **Cohen's κ = 0.842** (88.4% raw agreement) — "almost
perfect" agreement, which both validates the taxonomy and means these adjudicated cases are the
genuine ~12% tail of boundary difficulty (they appear in the final dataset at their natural rate).

### 3.5 Label stress-test (AI Tool Plan item #1, executed before annotation)

Before annotating, I ran an adversarial stress-test: three LLM "generators" each produced 6
r/nba comments engineered to sit *right on* a boundary (banter↔reaction, hot_take↔analysis,
reaction↔hot_take), then **two independent LLM annotators** classified all 18 under the rules,
and a diagnosis pass found the weakest seam. Full artifact: `outputs/label_stress_test.json`.

**Result: 16/18 inter-annotator agreement (89%)** — the taxonomy holds — with the disagreements
concentrated exactly where expected:

| Boundary | Agreement | Verdict |
|---|---|---|
| `reaction` ↔ `hot_take` | 6/6 | clean — emotional-exclamation-vs-standalone-claim rule works |
| `hot_take` ↔ `analysis` | 5/6 | one miss → added the **mechanism requirement** (§2, §3.2) |
| `banter` ↔ `reaction` | 5/6 | weakest → replaced "humor incidental" with the **construction test** (§3.1) |

**Definition changes this drove** (made *before* committing to the full annotation, which is the
whole point of stress-testing):
1. `analysis` now requires a **stated mechanism**, not just a stat next to a verdict (§2, §3.2).
2. `banter` ↔ `reaction` now uses the **construction test** + **emoji-demotion** + an explicit
   "co-occurring emotion ≠ ambiguity" instruction, with a worked hybrid example (§3.1).

These exact, sharpened rules become the annotation guide in Milestone 3 and the Groq baseline
prompt in Milestone 4.

---

## 4. Data Collection Plan

- **Source.** Real r/nba comments via the **Arctic Shift API** (`arctic-shift.photon-reddit.com`),
  the public Pushshift successor. Public comments only; no private channels or authenticated
  content.
- **Raw pool.** Pull a large pool (~2,000–4,000 comments) across multiple time windows and
  thread types (game threads skew `reaction`/`banter`; post-game and "[Highlight]"/"[Discussion]"
  threads surface more `analysis`). Filter out: `[deleted]`/`[removed]`, AutoModerator/bot
  comments, comments shorter than ~4 words or longer than ~90 words, link-only/quote-only
  comments, and non-English.
- **Target & balance.** Label a large pool, then keep **all consensus-labeled comments** for
  maximum training signal. Final dataset = **568 examples**: `hot_take` 174 (31%), `banter` 155
  (27%), `analysis` 149 (26%), `reaction` 90 (16%). `reaction` is the limiting class. Hard rule:
  **no class exceeds 70%** (max is 31% ✓); the ≥20%-per-class *aim* is met for three classes,
  with `reaction` at 16% — a deliberate trade of perfect balance for ~58% more training data,
  decided after a first experiment showed the small DistilBERT was data-starved on a balanced
  360-example set. Hard/adjudicated cases are kept at their natural ~12% rate, not filtered out.
- **Underrepresentation handling.** After the first labeling pass, count per class. If any class
  is under ~20%, mine additional candidates *specifically for that class* (e.g., keyword/length
  filters, analysis-heavy threads) before proceeding to training — never pad with low-quality
  or mislabeled examples just to hit a number.
- **Why 568 and not the 200 minimum.** The split is 70/15/15, so 200 → a 30-example test set
  (~7–8 per class), very noisy for per-class metrics, and starves a small fine-tuned model of
  training signal. 568 → an 86-example test set and 397 training examples — more trustworthy
  per-class F1 and a fairer shot for DistilBERT against a strong 70B baseline. (Empirically,
  expanding from 360→568 lifted test macro-F1 from 0.54→0.58 — see README.)
- **File.** One labeled CSV, `data/takemeter_nba.csv`, columns: `text`, `label`, `notes`
  (difficult-case reasoning), `pre_labeled` (provenance flag), `source_id` (Reddit comment id
  for traceability). The notebook splits 70/15/15 automatically; the local pipeline mirrors the
  same split with a fixed seed.

---

## 5. Evaluation Metrics

Accuracy alone is not enough for a 4-class, mildly subjective task — it can look healthy while
one class has quietly collapsed. The metric suite:

- **Primary: macro-F1.** Averages F1 across all four classes *equally*, so a model that lets a
  minority class (likely `banter` or `analysis`) die gets penalized regardless of how well it
  does on the majority. That collapse is the exact failure mode I'm worried about, so macro-F1
  is the headline number both models are judged on.
- **Overall accuracy** (both models) — the intuitive top-line and the headline baseline-vs-tuned
  comparison.
- **Per-class precision / recall / F1** — to see *which* distinction the model learned and which
  it didn't (e.g., high `analysis` precision but low recall = it only flags the most obvious
  breakdowns).
- **Weighted-F1** — accounts for any residual class imbalance in the test set.
- **4×4 confusion matrix** — to read the *direction* of errors. Expected confusions, stated as a
  hypothesis to test: `banter → reaction`, `hot_take ↔ analysis`, `hot_take ↔ reaction`.

**Why these and not just accuracy:** the whole point of the project is to find *where the
boundary breaks*. Macro-F1 + per-class metrics + the confusion matrix together pinpoint the
broken boundary and its direction; accuracy hides it.

---

## 6. Definition of Success ("good enough")

Concrete, objective pass/fail criteria for the fine-tuned model on the **held-out test set**:

1. **macro-F1 ≥ 0.65**, and
2. **beats the Groq zero-shot baseline by ≥ 0.10 macro-F1 (absolute)**, and
3. **no single class F1 < 0.45** (no dead class), and
4. **`analysis` F1 ≥ 0.60** — the most decision-relevant boundary, since the realistic product
   use ("surface substantive takes / flag low-effort ones for a human moderator") lives or dies
   on telling real `analysis` from everything else.

**Rationale.** This is a genuinely subjective 4-way task; even careful human annotators
realistically agree only ~0.6–0.7 of the time, which caps how high any model can score. A
macro-F1 around 0.65 that clearly beats a strong zero-shot LLM means the classifier is **useful
as a human-in-the-loop triage tool** — good enough to pre-sort a comment section or surface
candidate "good takes" for a human to confirm — but not good enough to act as an autonomous
moderator. That's the honest deployment bar for a tool like this.

---

## 7. AI Tool Plan

There's no application code to generate here, so AI tools help at three specific points:

1. **Label stress-testing (before annotating).** Feed an LLM the §2 definitions and §3 edge
   cases and ask it to generate boundary comments designed to *break* the taxonomy (banter vs
   reaction, hot_take vs analysis). Have an independent pass try to classify each under the
   rules; any comment that can't be classified cleanly signals a definition to tighten — done
   *before* committing to the full annotation. *(Run as a small multi-agent stress-test; results and
   any definition tweaks recorded in §3 / a stress-test note.)*
2. **Annotation assistance.** Use an LLM to **pre-label** batches, then review and correct every
   pre-assigned label against the §2/§3 rules, with a separate adversarial-verification pass on
   boundary cases. The `pre_labeled` column tracks which examples were machine-pre-labeled. This
   is disclosed in the README's AI-usage section.
3. **Failure-pattern analysis (after evaluation).** Paste the fine-tuned model's wrong
   predictions into an LLM and ask it to surface systematic patterns (a label pair, post length,
   sarcasm, low-information comments). Then **verify each proposed pattern by re-reading the
   actual examples** before writing it up — the LLM proposes, the data confirms.

---

## 8. Stretch Features (planned)

All four, because each reinforces the core rubric rather than competing with it:

- **Deployed interface.** A small Gradio app: paste a comment → predicted label + confidence
  bars. Doubles as the demo-recording surface.
- **Error-pattern analysis.** Go beyond three individual wrong predictions to a *systematic*
  pattern, verified across the error set.
- **Confidence calibration.** Check whether high-confidence predictions are actually right more
  often than low-confidence ones (reliability by confidence bin).
- **Inter-annotator reliability.** A second *independent LLM* labeling pass over 30+ examples,
  reporting Cohen's kappa and analyzing disagreements. (Disclosed as an LLM second pass, not a
  second human.)

---

## 9. Build Plan (milestones)

| # | Milestone | Deliverable | Commit point |
|---|---|---|---|
| 1+2 | Labels + spec | `planning.md` (this doc) | ✅ commit after review |
| 3 | Collect + annotate | `data/takemeter_nba.csv`, `src/collect_data.py` | commit |
| 4 | Baseline | Groq zero-shot results on test set | (part of pipeline) |
| 5 | Fine-tune + eval | `outputs/evaluation_results.json`, `outputs/confusion_matrix.png` | commit |
| 6 | Document + demo | `README.md` (full eval report), stretch analyses, `src/app.py`, demo script | commit |

**Tooling.** Local `uv` venv (Python 3.12), CPU PyTorch + `transformers` + `datasets` +
`scikit-learn` + `groq` + `gradio`. `distilbert-base-uncased` fine-tuned locally on CPU
(~400 training examples × early-stopped epochs ≈ a few minutes on CPU/MPS). The CodePath Colab notebook is the parallel "official"
path and demo backup. Model weights (~270 MB) exceed GitHub's 100 MB limit, so they're
gitignored with reproduction instructions; small outputs (JSON, PNG) are committed.
