# TakeMeter — Demo Video Script & Recording Guide

Target length **3–5 minutes**. You'll screen-record the Gradio app classifying live posts,
narrate one correct and one incorrect prediction, and walk through the evaluation report.

---

## 0. Setup (before recording)

```bash
cd /Users/rona/workspace/codepath/ai201/takemeter-p3
source .venv/bin/activate          # or use .venv/bin/python directly
python src/app.py                  # serves http://127.0.0.1:7860
```

Open `http://127.0.0.1:7860` in your browser. Also open `README.md` (rendered — e.g., on
GitHub or VS Code preview) in a second tab for the evaluation-report walkthrough.

Have these four posts ready to paste (all are real test-set comments with known model output):

| # | Post | Model predicts | True | Use as |
|---|---|---|---|---|
| 1 | *"No there was more spacing league wide in prime Harden's era due to the increased focus on 3 point shooting that started with the 2014 Spurs and 2015 Warriors. Every team started playing that drive and kick to shooter offense..."* | **analysis** (0.99) | analysis | ✅ correct |
| 2 | *"I would rather brunson take the offseason off, he already did FIBA last year"* | **hot_take** (0.99) | hot_take | ✅ correct |
| 3 | *"NOOO not again, I can't watch this team blow another 20-point lead"* | reaction | reaction | ✅ correct |
| 4 | *"At this point, he's worse than a traffic cone."* | **hot_take** (0.99) | banter | ❌ wrong |

---

## 1. Narration script (~4 min)

**[0:00–0:30] Intro.**
> "This is TakeMeter — a fine-tuned DistilBERT classifier that sorts r/nba comments into four
> kinds of take: **analysis**, **hot_take**, **reaction**, and **banter**. I labeled 568 real
> comments, fine-tuned DistilBERT, and compared it to a zero-shot Llama-3.3-70B baseline. Let me
> show it classifying live, then walk through where it works and where it breaks."

**[0:30–1:15] Correct prediction #1 — narrate the reasoning.** Paste post #1, click Classify.
> "This is a structured **analysis**: it makes a causal argument — *the 2014 Spurs and 2015
> Warriors popularized drive-and-kick, which increased league-wide spacing*. There's a stated
> **mechanism**, not just an opinion. The model is 99% confident and correct — and importantly,
> `analysis` is its highest-**precision** class (0.85): when it says analysis, it's almost always
> right, because it has learned to recognize this explicit, reasoned style."

**[1:15–1:50] Correct prediction #2.** Paste post #2.
> "Here's a **hot_take**: 'I'd rather Brunson take the offseason off.' A confident opinion with no
> supporting evidence — it asserts rather than argues. Model says hot_take, 99%, correct."

**[1:50–2:20] Correct prediction #3.** Paste post #3.
> "And a **reaction** — raw in-the-moment emotion, no argument, no joke. Notably this is the one
> class where my fine-tuned model actually **beats** the 70B baseline (F1 0.53 vs 0.39): the big
> LLM badly over-predicts reaction, and fine-tuning fixed that."

**[2:20–3:10] Incorrect prediction — narrate what went wrong.** Paste post #4 ("traffic cone").
> "Now a failure. *'At this point, he's worse than a traffic cone'* is **banter** — it's a joke,
> comedic hyperbole. But the model predicts **hot_take at 98.6% confidence** — confidently wrong.
> Why? It keys on the *surface*: a dismissive, evaluative statement about a player looks like a
> hot take. It can't detect that the **intent is humor** — that 'traffic cone' is a punchline.
> Detecting humor needs world knowledge and pragmatic inference a 66M-parameter model doesn't
> have. This is the model's signature gap: it learned the **lexical surface** of a take, not its
> **communicative intent**."

**[3:10–4:00] Evaluation report walkthrough.** Switch to the rendered README.
> "Here's the honest scorecard. Overall: fine-tuned macro-F1 **0.58** vs the baseline's **0.66** —
> the small model trails the 70B by about 0.08. The confusion matrix shows the dominant error:
> **analysis → hot_take**, 11 of 23 analysis comments. The model has high analysis *precision* but
> low *recall* — it only catches the most obvious breakdowns and dumps subtler reasoning into
> hot_take, because the analysis-vs-hot_take line is a *reasoning* distinction, not a keyword one.
> It's also overconfident — ECE 0.32. The takeaway: fine-tuning a small model on 400 examples
> learns surface patterns well but not the pragmatic reasoning the labels actually require."

---

## 2. Record on macOS

**Option A — built-in screen recorder (simplest):**
1. Press **⌘⇧5** → choose **Record Selected Portion** (or Entire Screen).
2. Click **Options** → set **Microphone** to your mic (so narration is captured).
3. Click **Record**. Do the demo. Stop via the menu-bar ◼ or **⌘⌃Esc**.
4. The `.mov` saves to your Desktop. Note its path.

**Option B — QuickTime:** File → New Screen Recording → set mic → record → File → Save.

Keep it tight (3–4 min) — shorter is easier to fit under the size limit.

---

## 3. Down-convert for GitHub (ffmpeg)

GitHub video facts (looked up):
- **Embed in README** works only via a `user-attachments` URL you get by **dragging the file
  into an Issue/PR/Release comment** — *not* by committing it and linking a repo path (repo-path
  videos don't play). Limit: **10 MB on free plans, 100 MB on paid**.
- Direct repo file push limit is **100 MB** per file regardless.
- Use **MP4 / H.264 video + AAC audio** — the most compatible combo (wrong codec fails even when
  small).

`ffmpeg` is installed. Pick the target for your GitHub plan:

**Free plan — guarantee < 10 MB (two-pass to a computed bitrate):**
```bash
# set DUR to your video length in seconds (e.g. 210 for 3:30)
DUR=210
# total budget 9 MB -> ~ (9*8192)/DUR kbps; reserve 64k for audio
VBIT=$(( (9*8192)/DUR - 64 ))
ffmpeg -y -i ~/Desktop/raw.mov -vf "scale=1280:-2,fps=15" -c:v libx264 -b:v ${VBIT}k \
  -pass 1 -an -f mp4 /dev/null && \
ffmpeg -i ~/Desktop/raw.mov -vf "scale=1280:-2,fps=15" -c:v libx264 -b:v ${VBIT}k \
  -pass 2 -c:a aac -b:a 64k -movflags +faststart demo/takemeter_demo.mp4
```

**Paid plan — high quality under 100 MB (single-pass CRF):**
```bash
ffmpeg -i ~/Desktop/raw.mov -vf "scale=1920:-2" -c:v libx264 -crf 28 -preset veryfast \
  -c:a aac -b:a 128k -movflags +faststart demo/takemeter_demo.mp4
```

Check the size and that it plays:
```bash
ls -lh demo/takemeter_demo.mp4
ffprobe -v error -show_entries format=duration,size:stream=codec_name demo/takemeter_demo.mp4
```
If the free-plan file is still > 10 MB: trim the video, drop `scale` to `854:-2`, or `fps=12`.

---

## 4. Embed in the README

1. On GitHub, open a new **Issue** (you don't have to submit it) or a PR comment.
2. **Drag `demo/takemeter_demo.mp4` into the comment box.** Wait for it to upload — it becomes a
   `https://github.com/user-attachments/assets/...` URL.
3. Copy that URL and paste it into `README.md` under the **Demo** section:
   ```markdown
   ## Demo
   https://github.com/user-attachments/assets/XXXXXXXX
   ```
   GitHub renders a bare `user-attachments` URL on its own line as a video player.
4. Also commit the file at `demo/takemeter_demo.mp4` as a backup (it's < 100 MB).

> Fallback if it won't fit/play: upload to YouTube/Loom (unlisted) and put a clickable thumbnail
> in the README — `[![demo](thumbnail.png)](https://youtu.be/…)`.
