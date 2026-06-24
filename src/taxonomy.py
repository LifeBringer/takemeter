"""Single source of truth for the TakeMeter label taxonomy.

Imported by the Groq baseline (so the zero-shot prompt matches the annotation rules)
and the Gradio app. The same rules were used to annotate the dataset.
"""

LABELS = ["analysis", "hot_take", "reaction", "banter"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

# One-line gloss per label (used in the app UI).
GLOSS = {
    "analysis": "Structured argument backed by specific, verifiable evidence + a stated mechanism.",
    "hot_take": "Bold, confident claim/ranking/prediction asserted with little or no real support.",
    "reaction": "Immediate emotional response to a moment — a feeling, no constructed joke.",
    "banter": "Primarily humor — a joke, pun, meme, sarcasm, or playful trash-talk.",
}

# Canonical rules — mirror planning.md §2-§3 (sharpened after stress-testing).
RULES = """Classify each r/nba comment by its DOMINANT communicative intent into EXACTLY ONE of four labels:

ANALYSIS — a structured argument backed by SPECIFIC, VERIFIABLE evidence (stats, tactics/scheme, historical comparison, rules/CBA) where REASONING is the point. Requires a stated mechanism/"why" linking evidence to a claim — not just a stat sitting next to a verdict. A real stat used only to prop up a subjective ranking/superlative ("led the league in scoring so he's top-3") is HOT_TAKE, not analysis.

HOT_TAKE — a bold, confident opinion / ranking / prediction asserted with little or no support; any stat is decorative or cherry-picked, not a genuine argument. Asserts rather than argues.

REACTION — an immediate EMOTIONAL response to a specific event/moment (a play, result, trade, injury): a feeling in the moment, little/no argument, and NO constructed joke.

BANTER — primary purpose is HUMOR: a joke, pun, meme, sarcastic one-liner, or playful trash-talk.

BOUNDARY DECISION RULES:
1) banter vs reaction (CONSTRUCTION TEST): if the comment contains ANY deliberately constructed comedic device — metaphor/simile-for-laughs, setup-and-payoff, meme/reference, sustained sarcasm, self-deprecating absurd hypothetical — label BANTER, even if genuine frustration is also present. Reserve REACTION for venting with NO comedic device (raw exclamation, profanity, all-caps despair). Distress emojis alone do NOT make it reaction when the text is a constructed joke.
2) hot_take vs analysis: label ANALYSIS only if removing the opinion framing leaves specific verifiable evidence supporting the claim THROUGH a stated mechanism. A vague/cherry-picked/decorative stat, or a real stat backing only a subjective ranking => HOT_TAKE.
3) reaction vs hot_take: an emotional exclamation tied to the just-happened event => REACTION; a standalone confident prediction/ranking stated as a claim, not bound to a moment => HOT_TAKE."""
