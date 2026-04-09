"""
VoC NLP Analysis - Delivery Hero Global Portfolio
Reads:  kaggle/reviews_delivery_hero_global.json  (PedidosYa BR/BO/AR)
        kaggle/foodpanda/bd_reviews_2025.csv       (Foodpanda BD)
Outputs analysis results to stdout + saves voc_analysis_results.json
"""
import json, re, pathlib
from collections import Counter
import pandas as pd

BASE = pathlib.Path(__file__).parent

# ---- Stopwords ---------------------------------------------------------
STOP_PT = {
    "que","o","a","de","e","em","um","uma","na","no","para","por","com","se",
    "da","do","dos","das","ao","aos","pela","pelo","pelos","pelas",
    "me","te","nos","lhe","lhes","meu","minha","meus","minhas",
    "seu","sua","seus","suas","nosso","nossa","nossos","nossas",
    "este","esta","estes","estas","esse","essa","esses","essas",
    "aquele","aquela","aqueles","aquelas","isto","isso","aquilo",
    "mas","ou","pois","porque","quando","como","onde","quem","qual","quais",
    "mais","menos","muito","pouco","bem","mal","ja","ja","sempre","nunca",
    "tambem","entao","assim","aqui","la","ali","ai","ate","apos",
    "antes","depois","durante","sobre","entre","contra","sem","sob","desde",
    "alem","dentro","fora","perto","longe","junto","logo","agora","hoje",
    "e","sao","foi","foram","ser","estar","esta","estao","era","eram",
    "tem","tinha","tinham","tive","teve","tiveram","ter","haver",
    "ha","havia","vai","vao","ir","vir","veio","vieram","faz","fazem",
    "fez","fizeram","fazer","pode","podem","pude","puderam","poder",
    "deve","devem","devia","dever","quer","querem","queria","querer",
    "nao","app","aplicativo","pedido","pedidos","pra","pq","vc","vcs",
    "vez","cada","todo","toda","todos","todas","nada","tudo",
    "outro","outra","outros","outras","mesmo","mesma","tal","tanto","tanta",
    "sim","otimo","bom","ruim",
}

STOP_ES = {
    "que","el","la","los","las","de","del","en","un","una","unos","unas",
    "al","su","sus","mi","mis","tu","tus","lo","le","les","se","me","te",
    "nos","con","por","para","sin","sobre","entre","desde","hasta",
    "hacia","ante","bajo","durante","segun","tras","contra",
    "este","esta","estos","estas","ese","esa","esos","esas",
    "aquel","aquella","aquellos","aquellas","esto","eso","aquello",
    "y","o","pero","sino","porque","como","cuando","donde","quien","cual",
    "cuales","mas","menos","muy","poco","bien","mal","ya","aun","todavia",
    "tambien","entonces","asi","aqui","ahi","alli","aca","alla","antes",
    "despues","durante","sobre","ademas","incluso","aunque","si","no","ni",
    "tampoco","nunca","siempre","hoy","ahora","luego","pronto",
    "es","son","fue","fueron","ser","estar","esta","estan","era","eran",
    "tiene","tienen","tenia","tenian","tuvo","tuvieron","tener","haber",
    "hay","habia","va","van","ir","venir","vino","vinieron","hace","hacen",
    "hizo","hicieron","hacer","puede","pueden","pudo","pudieron","poder",
    "debe","deben","debia","deber","quiere","quieren","queria","querer",
    "sabe","saben","sabia","saber","dice","dicen","dijo","dijeron","decir",
    "app","aplicacion","pedido","pedidos","veces","cada",
    "todo","toda","todos","todas","nada","algo","alguien","nadie","mismo",
    "misma","otro","otra","otros","otras","tanto","tanta","igual","solo",
    "pues","claro","obvio","bueno","malo","ok",
}

STOP_EN = {
    "the","a","an","of","in","on","at","to","for","with","from","by",
    "about","as","into","through","during","before","after","above",
    "below","between","out","off","over","under","again","then","once",
    "this","that","these","those","it","its","itself","they","them",
    "their","which","who","whom","what","where","when","how","why",
    "i","me","my","myself","we","our","you","your","he","him","his",
    "she","her","they","we","us","ours","its",
    "and","but","or","nor","so","yet","both","either","neither","not",
    "very","just","also","too","even","still","already","now","then",
    "here","there","often","always","never","sometimes","more","most",
    "much","many","few","less","least","same","other","another","each",
    "every","both","all","any","no","only","own","such","than","rather",
    "is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","shall","should","may","might",
    "must","can","could","get","got","make","made","take","took","come",
    "came","go","went","give","gave","say","said","know","knew","think",
    "thought","see","saw","want","use","find","tell","ask","seem","feel",
    "try","leave","call","keep","let","begin","show","hear","play","run",
    "app","order","food","delivery","service","time","good","bad","great",
    "best","worst","really","very","just","like","well","better","worse",
    "ordered","delivered","using","used","one","two","three","day","days",
    "month","months","week","weeks","hour","hours","minute","minutes",
    "good","great","best","well","bad","not","so","too","only",
}

# ---- PATTERNS ----------------------------------------------------------
PYA_PATTERNS = {
    "App crash / nao abre": (
        r"(trava(?:r|u|ndo)?|crash|congela(?:r|u|ndo)?|fecha(?:r|u)?|bugou|bug(?:ado)?|"
        r"n[aã]o\s+abre|nao\s+abre|n[aã]o\s+carrega|"
        r"err[oa](?:r|do|s)?|falha|reinicia|loop\s+infinito|"
        r"se\s+traba|se\s+cuelga|cuelga|traba(?:r)?|falla(?:r|ndo)?|"
        r"no\s+abre|no\s+carga|no\s+funciona|no\s+corre|"
        r"bloquea|bloqueado|pantalla\s+negra|pantalla\s+blanca)"
    ),
    "GPS / endereco": (
        r"(endere[cç](?:o|os)?|localiza[cç](?:ao|oes)?|gps|"
        r"rua|cep|bairro|"
        r"direcci[oo]n|domicilio|ubicaci[oo]n|localizaci[oo]n|"
        r"no\s+reconoce\s+mi)"
    ),
    "Pagamento / cobracas": (
        r"(pagamento|cobr(?:an[cç]a|aram|ado|ar)|cart[aã]o|"
        r"pix|boleto|reembolso|estorno|devolv(?:er|eu|endo)|"
        r"dupla\s+cobran|cobr(?:ou|a)\s+duas|"
        r"pago|cobr(?:aron|ado|ar)|tarjeta|dinero|plata|pesos|"
        r"cr[eé]dito|d[eé]bito|devoluci[oó]n|cargo\s+doble|"
        r"me\s+cobr(?:aron|[oó])\s+dos|cup[oó]n|cupon|descuento\s+no\s+aplic)"
    ),
    "Suporte / atendimento": (
        r"(suporte|atendimento|atendente|reclam(?:a[cç]|ar|ei)|"
        r"n[aã]o\s+resolve|cancelad[ao]|cancelaram|"
        r"soporte|atenci[oó]n|servicio\s+al\s+cliente|reclamo|"
        r"no\s+respond(?:en|i[oó]|e|ieron)|cancel(?:aron|ado|ar)|"
        r"queja|quejas|nadie\s+me\s+ayud|sin\s+respuesta)"
    ),
    "Entrega / pedido": (
        r"(entrega(?:dor|r)?|motoboy|"
        r"pedido\s+errado|item\s+faltando|faltou|"
        r"atraso|atrasad[ao]|demorou|demora|"
        r"entrega(?:ron)?\s+(?:mal|erron|equivoc)|"
        r"pedido\s+equivocad[ao]|falt[oó]|"
        r"tard(?:[oó]|aron|anza)|demor(?:[oó]|aron|ado)|"
        r"no\s+lleg[oó]|nunca\s+lleg[oó]|lleg[oó]\s+(?:mal|frio|equivocado))"
    ),
}

FP_PATTERNS = {
    "Wrong / missing items": (
        r"(wrong|missing|incomplete|incorrect|not\s+received|"
        r"not\s+delivered|different|less\s+quantity|short|"
        r"item(?:s)?\s+(?:missing|wrong|not|different)|"
        r"didn.?t\s+(?:send|include|deliver)|"
        r"half\s+(?:the\s+)?(?:items?|food|order)|substitute)"
    ),
    "App error / not working": (
        r"(app\s+(?:crash|error|bug|broken|not\s+working|doesn.?t\s+work)|"
        r"crash(?:es|ed|ing)?|not\s+(?:loading|working|opening)|"
        r"loading\s+(?:forever|too\s+long|problem)|freeze|frozen|stuck|"
        r"glitch|bug(?:gy)?|keeps?\s+(?:crashing|closing|freezing)|"
        r"can.?t\s+(?:open|order|place|log)|doesn.?t\s+(?:open|work|load))"
    ),
    "Payment / refund": (
        r"(payment\s+(?:fail|failed|error|issue|problem)|"
        r"charged?\s+(?:twice|double|extra|wrong)|refund(?:s|ed)?|"
        r"money\s+(?:not|deducted|taken)|bkash|"
        r"overcharg|double\s+charg|transaction\s+(?:fail|error)|"
        r"wallet|deduct(?:ed)?|didn.?t\s+get\s+(?:my\s+)?(?:money|refund))"
    ),
    "Late delivery / rider": (
        r"(late\s+delivery|too\s+late|very\s+late|"
        r"delay(?:ed)?|took\s+(?:too\s+long|forever|\d+\s+hours?)|"
        r"never\s+arriv|didn.?t\s+arriv|rider\s+(?:late|didn|rude|wrong)|"
        r"waiting\s+(?:for\s+)?\d+|hours?\s+(?:wait|late|delay)|"
        r"cold\s+when\s+arriv|arrived?\s+cold)"
    ),
    "Support / no response": (
        r"(customer\s+(?:service|support|care)|helpline|"
        r"no\s+(?:reply|response|support|help)|ignored|"
        r"complaint(?:s)?|unprofessional|rude\s+(?:staff|rider|service)|"
        r"refund\s+not\s+(?:given|provided|processed)|"
        r"can.?t\s+(?:contact|reach)\s+(?:support|anyone))"
    ),
}

MARKET_STOPWORDS = {
    "pedidosya_BR": [STOP_PT, STOP_EN],
    "pedidosya_BO": [STOP_ES, STOP_EN],
    "pedidosya_AR": [STOP_ES, STOP_EN],
    "foodpanda_BD": [STOP_EN],
}


def analyze_market(reviews_list, patterns, stopword_sets, label):
    """Pattern freq + top terms."""
    texts = [str(r.get("content") or r.get("text") or "").lower() for r in reviews_list]
    total = len(texts)
    stopwords = set()
    for sw in stopword_sets:
        stopwords |= sw

    freq, quotes = {}, {}
    for kw, pat in patterns.items():
        mask = [bool(re.search(pat, t)) for t in texts]
        cnt = sum(mask)
        freq[kw] = cnt
        samples = [reviews_list[i].get("content") or reviews_list[i].get("text", "")
                   for i, hit in enumerate(mask) if hit][:2]
        quotes[kw] = [str(s)[:220] for s in samples]

    all_words = " ".join(texts).split()
    top_words = Counter(
        w.strip(".,!?;:()\"-'/\\|@#") for w in all_words
        if len(w) > 3 and w.lower() not in stopwords
    ).most_common(20)

    top3 = sorted(freq.items(), key=lambda x: -x[1])[:3]

    print(f"\n{'--'*31}")
    print(f"  {label}  |  {total:,} negative reviews")
    print(f"{'--'*31}")
    for kw, cnt in sorted(freq.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / total if total else 0
        print(f"  {kw:38s}: {cnt:5d}  ({pct:.1f}%)")
    print("  TOP 3: " + " | ".join(f"#{i+1} {kw} ({cnt})" for i, (kw, cnt) in enumerate(top3)))
    print(f"  Top terms: {', '.join(w for w, _ in top_words[:12])}")

    return {
        "label": label,
        "total": total,
        "freq": freq,
        "quotes": quotes,
        "top_words": [{"word": w, "count": c} for w, c in top_words],
        "top3": [{"kw": kw, "count": cnt,
                  "pct": round(100 * cnt / total, 1) if total else 0}
                 for kw, cnt in top3],
    }


# ---- Load data ---------------------------------------------------------
with open(BASE / "reviews_delivery_hero_global.json", encoding="utf-8") as f:
    pya_data = json.load(f)

fp_bd = pd.read_csv(BASE / "foodpanda" / "bd_reviews_2025.csv")
fp_bd_neg = fp_bd[fp_bd["overall"] <= 2].copy()
fp_rows = [{"content": str(t)} for t in fp_bd_neg["text"].fillna("")]
print(f"Foodpanda BD negative reviews (overall <= 2): {len(fp_rows):,}")

# ---- Run analysis ------------------------------------------------------
results = {}

results["pedidosya_BR"] = analyze_market(
    pya_data["pedidosya_BR"], PYA_PATTERNS,
    MARKET_STOPWORDS["pedidosya_BR"], "PedidosYa BR (PT)")

results["pedidosya_BO"] = analyze_market(
    pya_data["pedidosya_BO"], PYA_PATTERNS,
    MARKET_STOPWORDS["pedidosya_BO"], "PedidosYa BO (ES)")

results["pedidosya_AR"] = analyze_market(
    pya_data["pedidosya_AR"], PYA_PATTERNS,
    MARKET_STOPWORDS["pedidosya_AR"], "PedidosYa AR (ES)")

results["foodpanda_BD"] = analyze_market(
    fp_rows, FP_PATTERNS,
    MARKET_STOPWORDS["foodpanda_BD"], "Foodpanda BD (EN)")

# ---- Save --------------------------------------------------------------
out_path = BASE / "voc_analysis_results.json"
out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n[analysis] Results saved -> {out_path}")
