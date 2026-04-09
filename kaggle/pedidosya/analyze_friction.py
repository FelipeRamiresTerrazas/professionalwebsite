"""
Step 2 — Friction Mapping from PedidosYa negative reviews.
Reads:   pedidosya_reviews.csv
Outputs: pedidosya_friction_metrics.json  (used by the React component)
"""
import pathlib, json, re
from collections import Counter
import pandas as pd

BASE  = pathlib.Path(__file__).parent
CSV   = BASE / "pedidosya_reviews.csv"
OUT   = BASE / "pedidosya_friction_metrics.json"

df = pd.read_csv(CSV, encoding="utf-8-sig")
print(f"Total reviews loaded: {len(df)}")

# ── Score distribution ──────────────────────────────────────────────────────
score_counts = df["score"].value_counts().sort_index().to_dict()
print("\nScore distribution:", score_counts)

# ── Isolate negative reviews (1-2 stars) ────────────────────────────────────
neg = df[df["score"].isin([1, 2])].copy()
neg["content"] = neg["content"].fillna("").str.lower()
print(f"\nNegative reviews (1-2 ★): {len(neg)} ({100*len(neg)/len(df):.1f}% of total)")

# ── Keyword patterns (broad, covers PT+ES mixed reviews, no \b on accented) ──
PATTERNS = {
    "busca":    r"(busca|buscar|busco|pesquisa|encontrar|search|procurar)",
    "carrinho": r"(carrinho|carrito|sacola|bag|\bcart\b|pedido n.o adicionado|n.o consigo adicionar)",
    "trava":    r"(trava|travou|travando|bug|bugou|bugando|fecha|fechou|crash|congela|congelou|n.o abre|n.o carrega|nao abre|nao carrega|trava|error|erro|falha)",
    "endereço": r"(endere[cç]|localiza[cç]|gps|local|rua|cep|bairro|direc[cç]|ubicaci)",
    "filtro":   r"(filtro|filtrar|categoria|categor|ordenar|orden|classif|busca por)",
}

freq   = {}
quotes = {}

for kw, pat in PATTERNS.items():
    mask  = neg["content"].str.contains(pat, regex=True, na=False)
    count = int(mask.sum())
    freq[kw] = count
    # grab up to 3 representative authentic quotes (original casing)
    samples = df.loc[neg[mask].index[:3], "content"].tolist()
    quotes[kw] = samples

print("\n── Keyword Frequency (negative reviews) ──")
for kw, cnt in sorted(freq.items(), key=lambda x: -x[1]):
    pct = 100 * cnt / len(neg) if len(neg) else 0
    print(f"  {kw:12s}: {cnt:4d} mentions  ({pct:.1f}% of negative reviews)")

print("\n── Sample quotes ──")
for kw, rows in quotes.items():
    print(f"\n  [{kw}]")
    for q in rows:
        snippet = q[:160].replace("\n", " ")
        print(f"    » {snippet}")

# ── Free-form top friction terms in negative reviews ────────────────────────
STOPWORDS = {
    "que","o","a","de","e","em","um","uma","na","no","para","por","com","se",
    "não","nao","da","do","me","mais","mas","foi","ela","ele","eu","muito",
    "app","aplicativo","pedido","pedidos","pra","já","ja","tem","vez","todo",
    "ate","até","ser","tem","está","esta","sempre","nunca","la","lo","en",
    "es","me","mi","su","le","los","las","por","las","lo","un","una","que",
    "no","ni","con","del","al","el","ya","le","se","si","hay","bien","mal",
    "como","pero","porque","cuando","hacer","hace","tiene","tiempo","veces",
    "cuando","solo","mucho","poco","muy","forma","nada","todo","cada"
}
all_words = " ".join(neg["content"].tolist()).split()
top_words = Counter(
    w.strip(".,!?;:()\"-'«»") for w in all_words
    if len(w) > 3 and w.lower() not in STOPWORDS
).most_common(25)
print("\n── Top 25 words in negative reviews ──")
for w, c in top_words:
    print(f"  {w:20s}: {c}")

# ── Build metrics payload ────────────────────────────────────────────────────
payload = {
    "meta": {
        "total_reviews":    int(len(df)),
        "negative_reviews": int(len(neg)),
        "negative_pct":     round(100 * len(neg) / len(df), 1),
        "score_distribution": {str(k): int(v) for k, v in score_counts.items()},
    },
    "keyword_freq": freq,
    "quotes": quotes,
    "top_words": [{"word": w, "count": c} for w, c in top_words],
}

OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n[analyze] Metrics saved → {OUT}")
