"""Shared inference for the fine-tuned TakeMeter model (used by evaluate.py and app.py)."""
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from taxonomy import ID2LABEL, LABELS

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"

_model = None
_tok = None


def load(model_dir=MODEL_DIR):
    global _model, _tok
    if _model is None:
        _tok = AutoTokenizer.from_pretrained(str(model_dir))
        _model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        _model.eval()
    return _model, _tok


def predict(texts, batch_size=16):
    """Return [{label, confidence, probs:{label:p}}] for each input text."""
    if isinstance(texts, str):
        texts = [texts]
    model, tok = load()
    results = []
    for i in range(0, len(texts), batch_size):
        chunk = list(texts[i:i + batch_size])
        enc = tok(chunk, truncation=True, max_length=128, padding=True, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
        for p in probs:
            idx = int(p.argmax())
            results.append({
                "label": ID2LABEL[idx],
                "confidence": float(p[idx]),
                "probs": {ID2LABEL[j]: float(p[j]) for j in range(len(LABELS))},
            })
    return results
