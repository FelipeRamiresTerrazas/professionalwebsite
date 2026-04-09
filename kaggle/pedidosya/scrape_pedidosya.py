"""
Step 1 — Scrape 2,000 most-recent PedidosYa reviews from Play Store (BR).
Saves: pedidosya_reviews.csv
"""
import sys, pathlib, time
import pandas as pd
from google_play_scraper import reviews, Sort

APP_ID   = "com.pedidosya"
LANG     = "pt"
COUNTRY  = "br"
TARGET   = 2000
BATCH    = 200          # reviews per API call (max allowed)
OUTFILE  = pathlib.Path(__file__).parent / "pedidosya_reviews.csv"

print(f"[scraper] Target: {TARGET} reviews for {APP_ID} | lang={LANG}, country={COUNTRY}")

collected, continuation_token = [], None

while len(collected) < TARGET:
    batch_size = min(BATCH, TARGET - len(collected))
    result, continuation_token = reviews(
        APP_ID,
        lang=LANG,
        country=COUNTRY,
        sort=Sort.NEWEST,
        count=batch_size,
        continuation_token=continuation_token,
    )
    if not result:
        print("[scraper] No more reviews returned — stopping early.")
        break
    collected.extend(result)
    print(f"[scraper] Fetched {len(collected)} / {TARGET} ...", flush=True)
    if continuation_token is None:
        print("[scraper] No continuation token — stopping early.")
        break
    time.sleep(0.5)   # be polite to the API

# Build DataFrame
records = []
for r in collected:
    records.append({
        "reviewId":  r.get("reviewId", ""),
        "score":     r.get("score"),
        "content":   r.get("content", ""),
        "thumbsUp":  r.get("thumbsUpCount", 0),
        "at":        r.get("at"),
        "appVersion": r.get("appVersion", ""),
    })

df = pd.DataFrame(records)
df["at"] = pd.to_datetime(df["at"], errors="coerce")
df.to_csv(OUTFILE, index=False, encoding="utf-8-sig")
print(f"\n[scraper] Done. {len(df)} rows saved to {OUTFILE}")
print(df[["score", "content"]].head(3).to_string())
