#!/usr/bin/env python3
"""
collect_data.py — Collect a pool of REAL r/nba comments via the Arctic Shift API
(the public Pushshift successor), filter out noise, and emit a length-stratified
candidate set for labeling.

Usage:
    python src/collect_data.py

Outputs:
    data/raw_pool.csv    — all filtered comments (deduped)
    data/candidates.csv  — stratified sample to hand to the labeling workflow

Design notes:
- We sample forward from several ANCHOR dates spread across the 2023-24, 2024-25,
  and 2025-26 seasons so the pool spans many different games/storylines, not one
  news cycle (which would bias the take distribution).
- Arctic Shift caps `limit` at 100/request, so we paginate with `after=<created_utc>`.
- Filtering keeps comments that plausibly express a "take": 5-80 words, not
  deleted/removed/bot/quote/link-only. Very short low-information replies ("this",
  "source?") are dropped on purpose — the taxonomy targets comments that make a
  contribution, and this keeps the four classes cleanly applicable (>90% labelable).
"""
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://arctic-shift.photon-reddit.com/api/comments/search"
UA = "takemeter-research/0.1 (CodePath AI201 project; educational)"
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# Anchor dates (unix UTC) spread across three seasons; we page forward from each.
# Mix of season openers, marquee dates, All-Star, play-in, playoffs, Finals.
ANCHORS = {
    "2023-11-15 early season":   1700000000,
    "2023-12-25 christmas":      1703500000,
    "2024-02-18 all-star":       1708250000,
    "2024-04-16 play-in":        1713240000,
    "2024-05-20 playoffs":       1716190000,
    "2024-06-12 finals":         1718150000,
    "2024-11-12 season opener":  1731400000,
    "2025-01-20 mid season":     1737360000,
    "2025-03-18 stretch run":    1742280000,
    "2025-05-15 playoffs":       1747280000,
    "2025-06-12 finals":         1749700000,
    "2025-11-12 2025-26 open":   1762900000,
}
PER_ANCHOR = 320            # target raw comments collected per anchor
PAGE = 100                  # API max per request
SLEEP = 0.6                 # politeness between requests
FIELDS = "body,created_utc,id,score,author,link_id"

# ---- filtering ----
MIN_WORDS, MAX_WORDS = 5, 80
BOT_AUTHORS = {"AutoModerator", "[deleted]", "RemindMeBot", "sneakpeekbot",
               "B0tRank", "WikiTextBot", "nba-gdt-bot", "MphsBBallBot"}
URL_RE = re.compile(r"https?://\S+")
WS_RE = re.compile(r"\s+")


def fetch(after: int, limit: int = PAGE):
    qs = urllib.parse.urlencode({
        "subreddit": "nba", "after": after, "limit": limit,
        "sort": "asc", "fields": FIELDS,
    })
    req = urllib.request.Request(f"{API}?{qs}", headers={"User-Agent": UA})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8")).get("data") or []
        except Exception as e:
            if attempt == 3:
                print(f"   ! fetch failed after retries: {e}", file=sys.stderr)
                return []
            time.sleep(1.5 * (attempt + 1))
    return []


def clean(body: str) -> str:
    body = body.replace("&amp;", "&").replace("&gt;", ">").replace("&lt;", "<")
    body = URL_RE.sub("", body)
    body = WS_RE.sub(" ", body).strip()
    return body


def keep(c: dict) -> str | None:
    """Return cleaned text if the comment passes filters, else None."""
    body = c.get("body") or ""
    author = c.get("author") or ""
    if author in BOT_AUTHORS or author.lower().endswith("bot"):
        return None
    if body.strip() in ("[deleted]", "[removed]", ""):
        return None
    if body.strip().startswith(">") and body.count(">") >= 1 and len(body) < 120:
        return None  # quote-only reply
    text = clean(body)
    n = len(text.split())
    if n < MIN_WORDS or n > MAX_WORDS:
        return None
    if len(text) < 15:
        return None
    if URL_RE.search(body) and n < 12:
        return None  # mostly-a-link comment
    return text


def collect() -> list[dict]:
    seen_ids, seen_text = set(), set()
    pool = []
    for name, anchor in ANCHORS.items():
        got, after = 0, anchor
        print(f"-> anchor {name} (after={anchor})")
        while got < PER_ANCHOR:
            batch = fetch(after)
            if not batch:
                break
            for c in batch:
                cid = c.get("id")
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)
                text = keep(c)
                if not text:
                    continue
                key = text.lower()
                if key in seen_text:
                    continue
                seen_text.add(key)
                pool.append({
                    "source_id": cid,
                    "text": text,
                    "words": len(text.split()),
                    "created_utc": c.get("created_utc"),
                    "score": c.get("score"),
                    "anchor": name,
                })
                got += 1
            after = batch[-1].get("created_utc", after) + 1
            time.sleep(SLEEP)
        print(f"   kept {got} from this anchor (pool now {len(pool)})")
    return pool


def stratify(pool: list[dict], per_bucket: int = 190) -> list[dict]:
    """Length-stratified sample so analysis-leaning (long) comments are well represented."""
    short = [c for c in pool if c["words"] <= 15]
    mid = [c for c in pool if 16 <= c["words"] <= 35]
    long = [c for c in pool if c["words"] >= 36]
    print(f"buckets — short:{len(short)} mid:{len(mid)} long:{len(long)}")
    # deterministic interleave by score desc then id, take per_bucket from each
    def pick(bucket):
        bucket = sorted(bucket, key=lambda c: (-(c["score"] or 0), c["source_id"]))
        return bucket[:per_bucket]
    cand = pick(long) + pick(mid) + pick(short)
    # de-dup any overlap, keep order
    out, seen = [], set()
    for c in cand:
        if c["source_id"] in seen:
            continue
        seen.add(c["source_id"])
        out.append(c)
    return out


def write_csv(path: Path, rows: list[dict], cols: list[str]):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})


def main():
    DATA.mkdir(exist_ok=True)
    pool = collect()
    if not pool:
        print("No comments collected — check network/API.", file=sys.stderr)
        sys.exit(1)
    write_csv(DATA / "raw_pool.csv", pool,
              ["source_id", "text", "words", "created_utc", "score", "anchor"])
    cand = stratify(pool)
    write_csv(DATA / "candidates.csv", cand,
              ["source_id", "text", "words", "created_utc", "score", "anchor"])
    print(f"\nDONE. raw_pool={len(pool)}  candidates={len(cand)}")
    print(f"  -> {DATA/'raw_pool.csv'}")
    print(f"  -> {DATA/'candidates.csv'}")


if __name__ == "__main__":
    main()
