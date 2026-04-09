"""
Multi-market VoC scraper — Delivery Hero portfolio (PedidosYa + Foodpanda BD)

Strategy:
  • Scrape 1★ and 2★ reviews *separately* using filter_score_with to ensure
    distinct review pools and avoid duplicate pagination across score bands.
  • No hard total-review cap — keep fetching until TARGET_NEG unique negative
    reviews are collected for each score band.
  • Deduplicates by reviewId across both bands before saving.

Apps:
  PedidosYa BR  →  com.pedidosya | lang=pt, country=br
  PedidosYa BO  →  com.pedidosya | lang=es, country=bo
  PedidosYa AR  →  com.pedidosya | lang=es, country=ar
  Foodpanda BD  →  com.foodpanda.bd | lang=en, country=bd
"""
import json, time, pathlib
from google_play_scraper import reviews, Sort

OUT  = pathlib.Path(__file__).parent / "reviews_delivery_hero_global.json"
BATCH           = 200
TARGET_NEG      = 2000   # target unique negative reviews per market
MAX_CALLS       = 60     # safety: max API calls per score-band per market

TARGETS = [
    {"app": "com.pedidosya",    "lang": "pt", "country": "br", "key": "pedidosya_BR"},
    {"app": "com.pedidosya",    "lang": "es", "country": "bo", "key": "pedidosya_BO"},
    {"app": "com.pedidosya",    "lang": "es", "country": "ar", "key": "pedidosya_AR"},
    {"app": "com.foodpanda.bd", "lang": "en", "country": "bd", "key": "foodpanda_BD"},
]


def scrape_score_band(app_id, lang, country, score, label, target):
    """Fetch reviews for a single star-rating band (1 or 2) using filter_score_with."""
    collected, continuation_token, calls = [], None, 0
    print(f"    [{label}] Band {score}★ ...", end=" ", flush=True)
    while len(collected) < target and calls < MAX_CALLS:
        try:
            result, continuation_token = reviews(
                app_id,
                lang=lang,
                country=country,
                sort=Sort.NEWEST,
                count=BATCH,
                filter_score_with=score,
                continuation_token=continuation_token,
            )
        except Exception as exc:
            print(f"\n      API error: {exc}")
            break
        calls += 1
        if not result:
            break
        collected.extend(result)
        if continuation_token is None:
            break
        time.sleep(0.5)
    print(f"{len(collected)} reviews fetched.")
    return collected


def scrape_market(app_id, lang, country, label, target=TARGET_NEG):
    """Scrape 1★ and 2★ bands separately, merge and deduplicate."""
    print(f"\n[{label}] {app_id} | lang={lang}, country={country} | target={target}")
    per_band = (target + 1) // 2          # split target across two bands

    band1 = scrape_score_band(app_id, lang, country, 1, label, per_band)
    band2 = scrape_score_band(app_id, lang, country, 2, label, per_band)

    seen, rows = set(), []
    for r in band1 + band2:
        rid = r.get("reviewId", "")
        if rid and rid in seen:
            continue
        seen.add(rid)
        rows.append({
            "reviewId":   rid,
            "score":      r.get("score"),
            "content":    r.get("content", ""),
            "thumbsUp":   r.get("thumbsUpCount", 0),
            "at":         str(r.get("at", "")),
            "appVersion": r.get("appVersion", ""),
        })

    print(f"  [{label}] {len(rows)} unique negative reviews (1★+2★).")
    return rows


results = {}
for cfg in TARGETS:
    results[cfg["key"]] = scrape_market(
        cfg["app"], cfg["lang"], cfg["country"], cfg["key"]
    )

OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n[scraper] Saved {sum(len(v) for v in results.values())} total reviews → {OUT}")
for key, rows in results.items():
    scores = [r["score"] for r in rows]
    n1 = scores.count(1)
    n2 = scores.count(2)
    print(f"  {key}: {len(rows)} total  (1★={n1}, 2★={n2})")

