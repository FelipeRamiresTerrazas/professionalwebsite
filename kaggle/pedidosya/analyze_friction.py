"""
Step 2 — Data-driven Friction Mapping from PedidosYa negative reviews.

Pipeline:
  1. Lift Analysis  — find terms that over-index in negative vs positive reviews
  2. TF-IDF + KMeans — auto-discover friction clusters from the corpus
  3. Auto-PATTERNS  — build regex per cluster from top-lift unigrams

Reads:   pedidosya_reviews.csv
Outputs: pedidosya_friction_metrics.json  (used by the React component)
"""
import pathlib, json, re
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

BASE = pathlib.Path(__file__).parent
CSV  = BASE / "pedidosya_reviews.csv"
OUT  = BASE / "pedidosya_friction_metrics.json"

df = pd.read_csv(CSV, encoding="utf-8-sig")
print(f"Total reviews loaded: {len(df)}")

# ── Score distribution ──────────────────────────────────────────────────────
score_counts = df["score"].value_counts().sort_index().to_dict()
print("\nScore distribution:", score_counts)

# ── Split into negative (1-2 ★) and positive (4-5 ★) ───────────────────────
df["content"] = df["content"].fillna("").str.lower()
neg = df[df["score"].isin([1, 2])].copy()
pos = df[df["score"].isin([4, 5])].copy()
print(f"\nNegative reviews (1-2 ★): {len(neg)} ({100*len(neg)/len(df):.1f}%)")
print(f"Positive reviews (4-5 ★): {len(pos)} ({100*len(pos)/len(df):.1f}%)")

# ── STOPWORDS (PT + ES) ─────────────────────────────────────────────────────
STOPWORDS = {
    "que","o","a","de","e","em","um","uma","na","no","para","por","com","se",
    "não","nao","da","do","me","mais","mas","foi","ela","ele","eu","muito",
    "app","aplicativo","pedido","pedidos","pra","já","ja","tem","vez","todo",
    "ate","até","ser","está","esta","sempre","nunca","la","lo","en","es","mi",
    "su","le","los","las","un","una","ni","con","del","al","el","ya","si",
    "hay","bien","mal","como","pero","porque","cuando","hacer","hace","tiene",
    "tiempo","veces","solo","mucho","poco","muy","forma","nada","cada","then",
    "this","that","with","have","from","they","will","been","were","what",
}

def tokenize(text):
    return [
        w.strip(".,!?;:()\"-'«»/\\")
        for w in text.split()
        if len(w) > 3 and w.lower() not in STOPWORDS
    ]

# ── 1. LIFT ANALYSIS ────────────────────────────────────────────────────────
neg_words = Counter(w for row in neg["content"] for w in tokenize(row))
pos_words = Counter(w for row in pos["content"] for w in tokenize(row))
all_vocab  = set(neg_words) | set(pos_words)

lift_scores = {}
for term in all_vocab:
    cnt  = neg_words[term]
    if cnt < 3:
        continue
    neg_rate = cnt / max(len(neg), 1)
    pos_rate = pos_words[term] / max(len(pos), 1)
    lift     = neg_rate / (pos_rate + 1e-5)
    score    = lift * (cnt ** 0.5)          # weight by frequency
    lift_scores[term] = {"lift": round(lift, 2), "count": cnt, "score": round(score, 3)}

top_lift = sorted(lift_scores.items(), key=lambda x: -x[1]["score"])[:40]
print("\n── Top 40 lift terms (negative-overindex) ──")
for term, vals in top_lift:
    print(f"  {term:25s}  lift={vals['lift']:.2f}  cnt={vals['count']:4d}  score={vals['score']:.2f}")

# ── 2. TF-IDF + KMeans CLUSTERING ───────────────────────────────────────────
N_CLUSTERS = 6
corpus = neg["content"].tolist()

tfidf = TfidfVectorizer(
    max_features=400,
    ngram_range=(1, 2),
    min_df=3,
    sublinear_tf=True,
    token_pattern=r"(?u)\b[a-záéíóúãâêôàñüç]{3,}\b",
    stop_words=list(STOPWORDS | {
        "não","nao","mais","app","aplicativo","para","com","que","muito",
        "tem","vez","todo","esta","está","uma","uns","umas","uns","isso",
        "esse","essa","esses","essas","pelo","pela","pelos","pelas",
        "este","esta","estes","estas","aqui","assim","quando","porque",
        "como","ainda","mesmo","também","tambem","agora","fazer",
        "feito","desde","entre","sobre","depois","antes","durante",
        "tudo","nada","algo","alguém","qualquer","todos","todas",
        # ES extras
        "para","todo","todos","todas","esta","este","estos","estas",
        "esse","esses","también","cuando","porque","como","ahora",
        "hacer","hecho","desde","entre","sobre","después","antes",
        "durante","todo","nada","algo","alguien","cualquier",
    }),
)
X = tfidf.fit_transform(corpus)
X_norm = normalize(X)

km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20, max_iter=400)
km.fit(X_norm)
labels = km.labels_

feature_names = np.array(tfidf.get_feature_names_out())
cluster_info  = []

print(f"\n── {N_CLUSTERS} Friction Clusters (KMeans) ──")
used_names = set()
for cid in range(N_CLUSTERS):
    center     = km.cluster_centers_[cid]
    top_idx    = center.argsort()[::-1][:12]
    top_terms  = feature_names[top_idx].tolist()
    size       = int((labels == cid).sum())

    # name: pick highest-lift unigram not already used by another cluster
    candidates = [t for t in top_terms if " " not in t]
    # sort by lift score descending
    candidates.sort(key=lambda t: lift_scores.get(t, {}).get("score", 0), reverse=True)
    best_name  = next((t for t in candidates if t not in used_names), candidates[0] if candidates else top_terms[0])
    used_names.add(best_name)

    # build auto-regex from unigrams in top_terms that are in lift_scores
    unigrams   = [t for t in top_terms if " " not in t and t in lift_scores]
    if not unigrams:
        unigrams = [t for t in top_terms if " " not in t][:6]
    pattern    = r"(?:" + "|".join(re.escape(t) for t in unigrams[:10]) + r")"

    pct = 100 * size / len(neg)
    print(f"\n  Cluster {cid} — '{best_name}'  (n={size}, {pct:.1f}% of neg)")
    print(f"    Terms: {', '.join(top_terms)}")
    print(f"    Regex: {pattern}")

    cluster_info.append({
        "id":       cid,
        "name":     best_name,
        "size":     size,
        "pct_neg":  round(pct, 1),
        "top_terms": top_terms,
        "pattern":  pattern,
    })

# ── 3. MAP AUTO-PATTERNS BACK ONTO neg corpus ────────────────────────────────
print("\n── Keyword Frequency (auto-discovered clusters, negative reviews) ──")
freq   = {}
quotes = {}
for cl in cluster_info:
    name    = cl["name"]
    mask    = neg["content"].str.contains(cl["pattern"], regex=True, na=False)
    count   = int(mask.sum())
    freq[name]   = count
    samples = df.loc[neg[mask].index[:3], "content"].tolist()
    quotes[name] = samples
    pct = 100 * count / len(neg) if len(neg) else 0
    print(f"  {name:20s}: {count:4d} mentions  ({pct:.1f}%)")

# ── Top friction terms (lift-ranked) ────────────────────────────────────────
top_words = [(t, v["count"]) for t, v in top_lift[:25]]
print("\n── Top 25 lift-ranked terms ──")
for w, c in top_words:
    print(f"  {w:25s}: {c}")

# ── Build metrics payload ────────────────────────────────────────────────────
payload = {
    "meta": {
        "total_reviews":    int(len(df)),
        "negative_reviews": int(len(neg)),
        "negative_pct":     round(100 * len(neg) / len(df), 1),
        "score_distribution": {str(k): int(v) for k, v in score_counts.items()},
    },
    "clusters":       cluster_info,
    "keyword_freq":   freq,
    "quotes":         quotes,
    "top_words":      [{"word": w, "count": c} for w, c in top_words],
    "top_lift_terms": [
        {"term": t, **v} for t, v in top_lift[:20]
    ],
}

OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n[analyze] Metrics saved → {OUT}")
