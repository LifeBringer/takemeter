#!/usr/bin/env python3
"""
app.py — TakeMeter deployed interface (stretch feature).
Paste an r/nba comment -> predicted take label + confidence across all four classes.

Usage:
    python src/app.py          # serves at http://127.0.0.1:7860
Requires a trained model in model/ (run train.py first).
"""
import gradio as gr

from predict import predict
from taxonomy import GLOSS, LABELS

EXAMPLES = [
    "Giannis at 28% from three is exactly why teams wall off the paint and dare him to shoot — the spacing just collapses.",
    "Luka is already a top-5 player of all time, no debate.",
    "NOOO not again, I cannot watch this team blow another 20-point lead",
    "Refs really said 'and 1' for breathing on him.",
]


def classify(text):
    if not text or not text.strip():
        return {}, ""
    r = predict(text.strip())[0]
    gloss = f"**{r['label']}** ({r['confidence']*100:.1f}%) — {GLOSS[r['label']]}"
    return r["probs"], gloss


with gr.Blocks(title="TakeMeter") as demo:
    gr.Markdown("# 🏀 TakeMeter\nClassify an r/nba comment as **analysis**, **hot_take**, "
                "**reaction**, or **banter** (fine-tuned DistilBERT).")
    inp = gr.Textbox(label="r/nba comment", lines=3, placeholder="Paste a comment…")
    btn = gr.Button("Classify", variant="primary")
    out_label = gr.Label(num_top_classes=4, label="Confidence by class")
    out_md = gr.Markdown()
    btn.click(classify, inputs=inp, outputs=[out_label, out_md])
    inp.submit(classify, inputs=inp, outputs=[out_label, out_md])
    gr.Examples(EXAMPLES, inputs=inp)


if __name__ == "__main__":
    demo.launch()
