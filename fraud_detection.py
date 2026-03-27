"""
=============================================================================
DETECÇÃO DE FRAUDES EM CARTÕES DE CRÉDITO
=============================================================================
Autor: Felipe Ramires Terrazas
Projeto de Portfólio — Data Science

Este script implementa um pipeline completo de Machine Learning para
identificar transações fraudulentas em cartões de crédito, utilizando
o dataset público do Kaggle (284.807 transações, 492 fraudes).

Decisões Técnicas Chave:
- Métrica principal: AUPRC (Area Under Precision-Recall Curve), pois
  Acurácia é enganosa em dados desbalanceados (~99.83% de classe 0).
- Tratamento de desbalanceamento via class_weight='balanced' nos modelos
  E comparação com SMOTE (oversampling sintético).
- Dois modelos comparados: Random Forest e XGBoost.
- Análise de impacto financeiro (Business Impact) para traduzir
  métricas técnicas em valor de negócio.

Requisitos:
  pip install pandas numpy scikit-learn xgboost imbalanced-learn
              matplotlib seaborn joblib
=============================================================================
"""

# ── 0. IMPORTAÇÕES ──────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_curve, auc, average_precision_score,
    roc_auc_score, f1_score
)

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ── CONFIGURAÇÃO ESTÉTICA GLOBAL ────────────────────────────────────────────
# Por quê? Gráficos com estilo "publicação" transmitem profissionalismo.
# O estilo seaborn-v0_8 fornece fundo sutil, grid discreto e tipografia limpa.
plt.style.use("seaborn-v0_8")
sns.set_context("talk", font_scale=0.9)
PALETTE = {"Normal": "#2ecc71", "Fraude": "#e74c3c"}
FIG_DIR = "figures"  # diretório para salvar os gráficos

import os
import sys
# Forçar UTF-8 no stdout para compatibilidade com terminais Windows (cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.makedirs(FIG_DIR, exist_ok=True)


def save_fig(fig, name: str, dpi: int = 200):
    """Salva a figura em alta resolução para uso no portfólio."""
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=dpi,
                bbox_inches="tight", facecolor="white")
    print(f"  [OK] Grafico salvo: {FIG_DIR}/{name}.png")


# =============================================================================
# 1. CARREGAMENTO DOS DADOS
# =============================================================================
print("=" * 70)
print("1. CARREGAMENTO DOS DADOS")
print("=" * 70)

df = pd.read_csv("creditcard.csv")

print(f"  Shape do dataset: {df.shape}")
print(f"  Colunas: {list(df.columns)}")
print(f"  Valores nulos: {df.isnull().sum().sum()}")
print(f"\n  Primeiras linhas:\n{df.head()}\n")


# =============================================================================
# 2. ANÁLISE EXPLORATÓRIA (EDA)
# =============================================================================
print("=" * 70)
print("2. ANÁLISE EXPLORATÓRIA (EDA)")
print("=" * 70)

# ── 2.1 Desequilíbrio da variável alvo ──────────────────────────────────────
# Por quê? Entender a proporção de fraudes é o passo mais crítico.
# Se o modelo simplesmente prever "tudo normal", terá ~99.83% de acurácia,
# mas será completamente inútil. Por isso escolhemos AUPRC como métrica.

class_counts = df["Class"].value_counts()
class_pct = df["Class"].value_counts(normalize=True) * 100

print(f"\n  Distribuição da variável 'Class':")
print(f"    Normal (0): {class_counts[0]:>10,}  ({class_pct[0]:.3f}%)")
print(f"    Fraude (1): {class_counts[1]:>10,}  ({class_pct[1]:.3f}%)")
print(f"    Razão Normal/Fraude: {class_counts[0] / class_counts[1]:.0f}:1")

# Gráfico de barras do desequilíbrio
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(
    ["Normal\n(Class 0)", "Fraude\n(Class 1)"],
    class_counts.values,
    color=[PALETTE["Normal"], PALETTE["Fraude"]],
    edgecolor="white", linewidth=1.5, width=0.5
)
for bar, count, pct in zip(bars, class_counts.values, class_pct.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2000,
            f"{count:,}\n({pct:.2f}%)", ha="center", va="bottom",
            fontweight="bold", fontsize=11)
ax.set_ylabel("Número de Transações")
ax.set_title("Desequilíbrio Extremo entre Classes", fontweight="bold", pad=15)
ax.set_ylim(0, class_counts.max() * 1.15)
sns.despine()
fig.tight_layout()
save_fig(fig, "01_class_imbalance")
plt.close(fig)

# ── 2.2 Distribuição de 'Amount' ────────────────────────────────────────────
# Por quê? O valor da transação é uma feature intuitiva — fraudes podem ter
# padrões distintos (valores muito altos ou padrões incomuns).

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 2.2a — Histograma por classe
for cls, label in [(0, "Normal"), (1, "Fraude")]:
    subset = df[df["Class"] == cls]["Amount"]
    axes[0].hist(subset, bins=80, alpha=0.7, label=label,
                 color=PALETTE[label], edgecolor="white", linewidth=0.3)
axes[0].set_xlabel("Valor da Transação (USD)")
axes[0].set_ylabel("Frequência")
axes[0].set_title("Distribuição de Amount por Classe", fontweight="bold")
axes[0].set_xlim(0, 500)  # zoom na faixa mais densa
axes[0].legend()

# 2.2b — Boxplot comparativo (log-scale)
df_plot = df.copy()
df_plot["Classe"] = df_plot["Class"].map({0: "Normal", 1: "Fraude"})
sns.boxplot(data=df_plot, x="Classe", y="Amount", palette=PALETTE,
            ax=axes[1], showfliers=False)
axes[1].set_ylabel("Valor da Transação (USD)")
axes[1].set_title("Boxplot de Amount (sem outliers)", fontweight="bold")

sns.despine()
fig.tight_layout()
save_fig(fig, "02_amount_distribution")
plt.close(fig)

# ── 2.3 Distribuição de 'Time' ──────────────────────────────────────────────
# Por quê? 'Time' representa segundos desde a primeira transação do dataset.
# Isso revela padrões temporais (ex.: fraudes concentradas à noite).

fig, ax = plt.subplots(figsize=(12, 5))
# Converter segundos para horas para facilitar a interpretação
df["Time_hours"] = df["Time"] / 3600

for cls, label in [(0, "Normal"), (1, "Fraude")]:
    subset = df[df["Class"] == cls]["Time_hours"]
    ax.hist(subset, bins=48, alpha=0.65, label=label,
            color=PALETTE[label], edgecolor="white", linewidth=0.3,
            density=True)  # density=True para comparar forma, não magnitude

ax.set_xlabel("Tempo (horas desde o início da coleta)")
ax.set_ylabel("Densidade")
ax.set_title("Distribuição Temporal das Transações", fontweight="bold")
ax.legend()
sns.despine()
fig.tight_layout()
save_fig(fig, "03_time_distribution")
plt.close(fig)


# ── 2.4 Correlação entre as features PCA (V1–V28) ──────────────────────────
# Por quê? Como V1–V28 já são componentes PCA, esperamos baixa correlação
# entre elas — mas verificar é boa prática.

pca_cols = [f"V{i}" for i in range(1, 29)]
fig, ax = plt.subplots(figsize=(12, 10))
corr_matrix = df[pca_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, cmap="RdBu_r", center=0,
            vmin=-0.3, vmax=0.3, ax=ax, square=True,
            linewidths=0.5, cbar_kws={"shrink": 0.7})
ax.set_title("Matriz de Correlação das Features PCA (V1–V28)",
             fontweight="bold", pad=15)
fig.tight_layout()
save_fig(fig, "04_pca_correlation")
plt.close(fig)


# =============================================================================
# 3. PRÉ-PROCESSAMENTO
# =============================================================================
print("\n" + "=" * 70)
print("3. PRÉ-PROCESSAMENTO")
print("=" * 70)

# ── 3.1 Escalonamento de 'Amount' e 'Time' ─────────────────────────────────
# Por quê? As features V1–V28 já estão escalonadas (saída de PCA), mas
# 'Amount' e 'Time' possuem escalas muito diferentes. Sem escalonamento,
# modelos baseados em distância ou gradiente podem dar peso desproporcional
# a essas variáveis. Mesmo para tree-based models, o escalonamento
# garante coerência no pipeline e facilita futuras comparações.

scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
df["Time_scaled"] = scaler.fit_transform(df[["Time"]])

# Remover colunas originais para evitar duplicidade
df.drop(["Amount", "Time", "Time_hours"], axis=1, inplace=True)

print(f"  [OK] 'Amount' e 'Time' escalonados com StandardScaler.")
print(f"  Shape final: {df.shape}")

# ── 3.2 Separação de Features e Target ──────────────────────────────────────
X = df.drop("Class", axis=1)
y = df["Class"]

# ── 3.3 Split Treino/Teste ──────────────────────────────────────────────────
# Por quê? Usamos stratify=y para preservar a proporção de fraudes em ambos
# os conjuntos. Sem isso, o conjunto de teste poderia ter 0 fraudes.
# random_state fixo garante reprodutibilidade.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Treino: {X_train.shape[0]:,} amostras "
      f"({y_train.sum()} fraudes, {y_train.sum()/len(y_train)*100:.3f}%)")
print(f"  Teste:  {X_test.shape[0]:,} amostras "
      f"({y_test.sum()} fraudes, {y_test.sum()/len(y_test)*100:.3f}%)")


# =============================================================================
# 4. TRATAMENTO DE DESEQUILÍBRIO — DUAS ESTRATÉGIAS
# =============================================================================
print("\n" + "=" * 70)
print("4. TRATAMENTO DE DESEQUILÍBRIO")
print("=" * 70)

# ── Estratégia A: Class Weights ─────────────────────────────────────────────
# Por quê? A forma mais simples e eficiente. O modelo penaliza mais os erros
# na classe minoritária, sem gerar dados sintéticos. Evita overfitting que
# pode ocorrer com oversampling.
print("\n  Estratégia A: Ajuste de pesos de classe (class_weight='balanced')")
print("    → Penaliza mais erros em fraudes, sem criar dados sintéticos.")

# ── Estratégia B: SMOTE ────────────────────────────────────────────────────
# Por quê? SMOTE (Synthetic Minority Oversampling Technique) gera exemplos
# sintéticos interpolando entre vizinhos da classe minoritária. Isso aumenta
# a diversidade do treino, mas deve ser aplicado APENAS no treino (nunca
# no teste) para evitar data leakage.
print("\n  Estratégia B: SMOTE (oversampling sintético)")

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"    Antes do SMOTE: {y_train.value_counts().to_dict()}")
print(f"    Depois do SMOTE: {pd.Series(y_train_smote).value_counts().to_dict()}")
print("    ⚠ SMOTE aplicado APENAS no conjunto de TREINO (sem data leakage).")


# =============================================================================
# 5. MODELAGEM — RANDOM FOREST vs. XGBOOST
# =============================================================================
print("\n" + "=" * 70)
print("5. MODELAGEM")
print("=" * 70)

# Dicionário para armazenar resultados de todos os modelos
results = {}


def evaluate_model(model, X_tr, y_tr, X_te, y_te, model_name: str):
    """
    Treina o modelo, faz predições e calcula as métricas profissionais.

    Por quê esta função?
    - Centralizar a avaliação garante consistência entre modelos.
    - Retorna probabilidades (predict_proba) para curvas PR, não
      apenas predições binárias.

    Métricas escolhidas:
    - AUPRC: Padrão ouro para dados desbalanceados — foca na capacidade
      do modelo de encontrar fraudes sem gerar muitos alarmes falsos.
    - ROC AUC: Complementar, mostra desempenho geral.
    - F1-Score: Média harmônica entre Precision e Recall.
    - Matriz de Confusão: Visualização intuitiva de TP, FP, TN, FN.
    """
    print(f"\n  ── {model_name} ──")

    # Treinar
    model.fit(X_tr, y_tr)

    # Predições
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]  # probabilidade de fraude

    # Métricas
    auprc = average_precision_score(y_te, y_proba)
    roc = roc_auc_score(y_te, y_proba)
    f1 = f1_score(y_te, y_pred)
    cm = confusion_matrix(y_te, y_pred)

    print(f"    AUPRC:   {auprc:.4f}")
    print(f"    ROC AUC: {roc:.4f}")
    print(f"    F1:      {f1:.4f}")
    print(f"\n    Matriz de Confusão:")
    print(f"      TN={cm[0, 0]:>6,}  FP={cm[0, 1]:>4,}")
    print(f"      FN={cm[1, 0]:>6,}  TP={cm[1, 1]:>4,}")
    print(f"\n{classification_report(y_te, y_pred, target_names=['Normal', 'Fraude'])}")

    results[model_name] = {
        "model": model, "y_pred": y_pred, "y_proba": y_proba,
        "auprc": auprc, "roc_auc": roc, "f1": f1, "cm": cm
    }
    return model


# ── 5.1 Random Forest com Class Weights ─────────────────────────────────────
# Por quê Random Forest?
# - Robusto a outliers e ruído.
# - Fornece feature importance nativa.
# - class_weight='balanced' ajusta pesos proporcionalmente ao desbalanceamento.

rf_model = RandomForestClassifier(
    n_estimators=200,           # 200 árvores para estabilidade
    max_depth=20,               # limitar profundidade evita overfitting
    min_samples_leaf=5,         # mínimo de amostras por folha
    class_weight="balanced",    # penaliza mais erros em fraudes
    random_state=42,
    n_jobs=-1                   # paralelismo total
)
evaluate_model(rf_model, X_train, y_train, X_test, y_test,
               "Random Forest (Class Weights)")

# ── 5.2 XGBoost com Class Weights ───────────────────────────────────────────
# Por quê XGBoost?
# - Gradient boosting tende a superar Random Forest em dados tabulares.
# - scale_pos_weight ajusta o peso da classe positiva (fraude).
# - Regularização L1/L2 nativa previne overfitting.

# Calcular scale_pos_weight: proporção de negativos / positivos
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos = neg_count / pos_count

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_pos,  # equivalente a class_weight='balanced'
    reg_alpha=0.1,               # regularização L1
    reg_lambda=1.0,              # regularização L2
    eval_metric="aucpr",         # otimizar AUPRC durante o treino
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1
)
evaluate_model(xgb_model, X_train, y_train, X_test, y_test,
               "XGBoost (Class Weights)")

# ── 5.3 Random Forest com SMOTE ─────────────────────────────────────────────
rf_smote = RandomForestClassifier(
    n_estimators=200, max_depth=20, min_samples_leaf=5,
    random_state=42, n_jobs=-1
    # Sem class_weight='balanced' — SMOTE já equilibra as classes
)
evaluate_model(rf_smote, X_train_smote, y_train_smote, X_test, y_test,
               "Random Forest (SMOTE)")

# ── 5.4 XGBoost com SMOTE ───────────────────────────────────────────────────
xgb_smote = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    reg_alpha=0.1, reg_lambda=1.0,
    eval_metric="aucpr", use_label_encoder=False,
    random_state=42, n_jobs=-1
    # Sem scale_pos_weight — SMOTE já equilibra as classes
)
evaluate_model(xgb_smote, X_train_smote, y_train_smote, X_test, y_test,
               "XGBoost (SMOTE)")


# =============================================================================
# 6. VISUALIZAÇÃO — COMPARAÇÃO DE MODELOS
# =============================================================================
print("\n" + "=" * 70)
print("6. VISUALIZAÇÕES COMPARATIVAS")
print("=" * 70)

# ── 6.1 Curvas Precision-Recall ─────────────────────────────────────────────
# Por quê PR Curve ao invés de ROC Curve?
# Em dados extremamente desbalanceados, a ROC Curve pode ser otimista porque
# a taxa de falsos positivos (FPR) é "diluída" pela enorme classe negativa.
# A Precision-Recall Curve foca exclusivamente na classe minoritária.

fig, ax = plt.subplots(figsize=(10, 7))
colors = ["#2ecc71", "#3498db", "#e67e22", "#9b59b6"]

for (name, res), color in zip(results.items(), colors):
    precision, recall, _ = precision_recall_curve(y_test, res["y_proba"])
    ax.plot(recall, precision, label=f"{name} (AUPRC={res['auprc']:.4f})",
            color=color, linewidth=2)

# Linha de base: proporção de fraudes no dataset (classificador aleatório)
baseline = y_test.sum() / len(y_test)
ax.axhline(y=baseline, color="gray", linestyle="--", alpha=0.7,
           label=f"Baseline (classificador aleatório): {baseline:.4f}")

ax.set_xlabel("Recall (Sensibilidade)", fontsize=12)
ax.set_ylabel("Precision (Precisão)", fontsize=12)
ax.set_title("Curvas Precision-Recall — Comparação de Modelos",
             fontweight="bold", fontsize=14, pad=15)
ax.legend(loc="upper right", fontsize=9, frameon=True, fancybox=True)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])
sns.despine()
fig.tight_layout()
save_fig(fig, "05_precision_recall_curves")
plt.close(fig)

# ── 6.2 Matrizes de Confusão comparativas ───────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, (name, res) in enumerate(results.items()):
    cm = res["cm"]
    # Calcular percentuais
    cm_pct = cm.astype("float") / cm.sum() * 100

    # Anotações com contagem + percentual
    annot = np.array([
        [f"TN\n{cm[0,0]:,}\n({cm_pct[0,0]:.2f}%)",
         f"FP\n{cm[0,1]:,}\n({cm_pct[0,1]:.3f}%)"],
        [f"FN\n{cm[1,0]:,}\n({cm_pct[1,0]:.3f}%)",
         f"TP\n{cm[1,1]:,}\n({cm_pct[1,1]:.3f}%)"]
    ])

    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", ax=axes[idx],
                xticklabels=["Normal", "Fraude"],
                yticklabels=["Normal", "Fraude"],
                cbar=False, linewidths=2, linecolor="white")
    axes[idx].set_ylabel("Real")
    axes[idx].set_xlabel("Predito")
    axes[idx].set_title(name, fontweight="bold", fontsize=11)

fig.suptitle("Matrizes de Confusão — Todos os Modelos",
             fontweight="bold", fontsize=14, y=1.01)
fig.tight_layout()
save_fig(fig, "06_confusion_matrices")
plt.close(fig)

# ── 6.3 Comparação de Métricas (barras agrupadas) ───────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
model_names = list(results.keys())
# Nomes curtos para melhor legibilidade
short_names = ["RF\n(Weights)", "XGB\n(Weights)", "RF\n(SMOTE)", "XGB\n(SMOTE)"]
x = np.arange(len(model_names))
width = 0.25

auprc_vals = [results[m]["auprc"] for m in model_names]
roc_vals = [results[m]["roc_auc"] for m in model_names]
f1_vals = [results[m]["f1"] for m in model_names]

bars1 = ax.bar(x - width, auprc_vals, width, label="AUPRC", color="#e74c3c",
               edgecolor="white")
bars2 = ax.bar(x, roc_vals, width, label="ROC AUC", color="#3498db",
               edgecolor="white")
bars3 = ax.bar(x + width, f1_vals, width, label="F1-Score", color="#2ecc71",
               edgecolor="white")

# Valores sobre as barras
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                f"{height:.3f}", ha="center", va="bottom", fontsize=8,
                fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=10)
ax.set_ylabel("Score")
ax.set_title("Comparação de Métricas entre Modelos", fontweight="bold",
             fontsize=14, pad=15)
ax.set_ylim(0, 1.1)
ax.legend(loc="lower right", fontsize=10)
sns.despine()
fig.tight_layout()
save_fig(fig, "07_metrics_comparison")
plt.close(fig)


# =============================================================================
# 7. FEATURE IMPORTANCE — DIFERENCIAL DE PORTFÓLIO
# =============================================================================
print("\n" + "=" * 70)
print("7. FEATURE IMPORTANCE")
print("=" * 70)

# Por quê? Mostrar quais variáveis o modelo usa para detectar fraude
# demonstra compreensão do problema. Mesmo que V1–V28 sejam anônimas
# (resultado de PCA), a importância revela quais componentes principais
# capturam os padrões de fraude.

# Selecionar o melhor modelo baseado em AUPRC
best_name = max(results, key=lambda k: results[k]["auprc"])
best_model = results[best_name]["model"]
print(f"\n  Melhor modelo (AUPRC): {best_name}")

# Feature Importance do melhor modelo
feature_names = X_train.columns.tolist()

if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
else:
    # Fallback: usar o modelo Random Forest (Class Weights) se o melhor
    # não tiver feature_importances_
    importances = results["Random Forest (Class Weights)"]["model"].feature_importances_
    best_name = "Random Forest (Class Weights)"

feat_imp = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print(f"\n  Top 10 Features mais importantes ({best_name}):")
for i, row in feat_imp.head(10).iterrows():
    print(f"    {row['Feature']:>15s}: {row['Importance']:.4f}")

# Gráfico de Feature Importance (Top 15)
fig, ax = plt.subplots(figsize=(10, 8))
top_n = 15
top_features = feat_imp.head(top_n).sort_values("Importance")

colors_gradient = plt.cm.Reds(np.linspace(0.3, 0.9, top_n))
ax.barh(top_features["Feature"], top_features["Importance"],
        color=colors_gradient, edgecolor="white", linewidth=0.5)

for i, (val, name) in enumerate(zip(top_features["Importance"],
                                     top_features["Feature"])):
    ax.text(val + 0.002, i, f"{val:.4f}", va="center", fontsize=9)

ax.set_xlabel("Importância (Gini / Gain)")
ax.set_title(f"Top {top_n} Features — {best_name}",
             fontweight="bold", fontsize=13, pad=15)
sns.despine()
fig.tight_layout()
save_fig(fig, "08_feature_importance")
plt.close(fig)

# ── XGBoost Feature Importance comparativa ──────────────────────────────────
# Por quê comparar? Diferentes algoritmos podem valorizar features diferentes,
# revelando padrões complementares.

xgb_best_name = [n for n in results if "XGBoost" in n]
if xgb_best_name:
    xgb_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": results[xgb_best_name[0]]["model"].feature_importances_
    }).sort_values("Importance", ascending=False)

    rf_best_name = [n for n in results if "Random Forest" in n][0]
    rf_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": results[rf_best_name]["model"].feature_importances_
    }).sort_values("Importance", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # RF
    top_rf = rf_imp.head(10).sort_values("Importance")
    axes[0].barh(top_rf["Feature"], top_rf["Importance"],
                 color="#2ecc71", edgecolor="white")
    axes[0].set_title(f"Random Forest", fontweight="bold")
    axes[0].set_xlabel("Importância")

    # XGBoost
    top_xgb = xgb_imp.head(10).sort_values("Importance")
    axes[1].barh(top_xgb["Feature"], top_xgb["Importance"],
                 color="#3498db", edgecolor="white")
    axes[1].set_title(f"XGBoost", fontweight="bold")
    axes[1].set_xlabel("Importância")

    fig.suptitle("Comparação de Feature Importance: RF vs. XGBoost",
                 fontweight="bold", fontsize=14, y=1.01)
    sns.despine()
    fig.tight_layout()
    save_fig(fig, "09_feature_importance_comparison")
    plt.close(fig)


# =============================================================================
# 8. ANÁLISE DE IMPACTO FINANCEIRO (BUSINESS IMPACT)
# =============================================================================
print("\n" + "=" * 70)
print("8. ANÁLISE DE IMPACTO FINANCEIRO (BUSINESS IMPACT)")
print("=" * 70)

# Por quê? Traduzir métricas técnicas em valor de negócio é o que separa
# um cientista de dados júnior de um profissional. Stakeholders não entendem
# "AUPRC de 0.85", mas entendem "o modelo economizaria R$ 500K por ano".

# Reconstruir os valores originais de Amount no conjunto de teste
# (precisamos reverter o escalonamento)
# Como o scaler foi ajustado no df inteiro, precisamos do original
df_original = pd.read_csv("creditcard.csv")
_, test_idx = train_test_split(
    df_original.index, test_size=0.2, random_state=42,
    stratify=df_original["Class"]
)
amount_test = df_original.loc[test_idx, "Amount"].values

print(f"\n  Análise para cada modelo:")
print(f"  {'Modelo':<35s} {'Fraudes Bloq.':<15s} {'$ Economizado':<15s} "
      f"{'FP (Incômodo)':<15s} {'$ Fraude Perdido':<15s}")
print(f"  {'-'*95}")

# Custo estimado de um Falso Positivo (incomodar cliente legítimo)
# Suposição conservadora: custo de atendimento + perda temporária de confiança
COST_PER_FP = 10  # USD por falso positivo (ligação, SMS, bloqueio temporário)

business_data = []

for name, res in results.items():
    cm = res["cm"]
    y_pred = res["y_pred"]

    # Verdadeiros Positivos: fraudes corretamente detectadas
    tp_mask = (y_test.values == 1) & (y_pred == 1)
    money_saved = amount_test[tp_mask].sum()

    # Falsos Negativos: fraudes que passaram despercebidas
    fn_mask = (y_test.values == 1) & (y_pred == 0)
    money_lost = amount_test[fn_mask].sum()

    # Falsos Positivos: clientes legítimos incomodados
    fp_count = cm[0, 1]
    fp_cost = fp_count * COST_PER_FP

    # True Positives count
    tp_count = cm[1, 1]

    # Lucro líquido do modelo
    net_benefit = money_saved - fp_cost

    print(f"  {name:<35s} {tp_count:<15,} ${money_saved:<14,.2f} "
          f"{fp_count:<15,} ${money_lost:<14,.2f}")

    business_data.append({
        "Modelo": name,
        "Fraudes Bloqueadas": tp_count,
        "Dinheiro Economizado": money_saved,
        "Dinheiro Perdido (FN)": money_lost,
        "Falsos Positivos": fp_count,
        "Custo FP": fp_cost,
        "Benefício Líquido": net_benefit
    })

business_df = pd.DataFrame(business_data)

# Gráfico de impacto financeiro
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 8a — Dinheiro economizado vs. perdido
short_names_biz = ["RF\n(Weights)", "XGB\n(Weights)", "RF\n(SMOTE)", "XGB\n(SMOTE)"]
x = np.arange(len(business_df))
width = 0.35

axes[0].bar(x - width/2, business_df["Dinheiro Economizado"],
            width, label="$ Economizado (TP)", color="#2ecc71", edgecolor="white")
axes[0].bar(x + width/2, business_df["Dinheiro Perdido (FN)"],
            width, label="$ Perdido (FN)", color="#e74c3c", edgecolor="white")
axes[0].set_xticks(x)
axes[0].set_xticklabels(short_names_biz, fontsize=9)
axes[0].set_ylabel("Valor (USD)")
axes[0].set_title("Impacto Financeiro: Fraudes Bloqueadas vs. Perdidas",
                   fontweight="bold", fontsize=11)
axes[0].legend(fontsize=9)

# 8b — Benefício líquido (economizado - custo de FP)
colors_net = ["#2ecc71" if v > 0 else "#e74c3c"
              for v in business_df["Benefício Líquido"]]
axes[1].bar(x, business_df["Benefício Líquido"], color=colors_net,
            edgecolor="white", width=0.5)
for i, val in enumerate(business_df["Benefício Líquido"]):
    axes[1].text(i, val + 100, f"${val:,.0f}", ha="center",
                 fontsize=9, fontweight="bold")
axes[1].set_xticks(x)
axes[1].set_xticklabels(short_names_biz, fontsize=9)
axes[1].set_ylabel("Benefício Líquido (USD)")
axes[1].set_title(f"Benefício Líquido (Economizado − Custo FP × ${COST_PER_FP})",
                   fontweight="bold", fontsize=11)
axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)

sns.despine()
fig.tight_layout()
save_fig(fig, "10_business_impact")
plt.close(fig)


# =============================================================================
# 9. TABELA RESUMO FINAL
# =============================================================================
print("\n" + "=" * 70)
print("9. TABELA RESUMO FINAL")
print("=" * 70)

summary = pd.DataFrame({
    "Modelo": list(results.keys()),
    "AUPRC": [results[m]["auprc"] for m in results],
    "ROC AUC": [results[m]["roc_auc"] for m in results],
    "F1-Score": [results[m]["f1"] for m in results],
    "TP (Fraudes Det.)": [results[m]["cm"][1, 1] for m in results],
    "FP": [results[m]["cm"][0, 1] for m in results],
    "FN (Fraudes Perdidas)": [results[m]["cm"][1, 0] for m in results],
}).sort_values("AUPRC", ascending=False)

print(f"\n{summary.to_string(index=False)}\n")

# Modelo vencedor
winner = summary.iloc[0]["Modelo"]
winner_auprc = summary.iloc[0]["AUPRC"]
print(f"  [MELHOR MODELO]: {winner}")
print(f"     AUPRC: {winner_auprc:.4f}")
print(f"     Fraudes detectadas: {summary.iloc[0]['TP (Fraudes Det.)']:.0f} "
      f"de {y_test.sum()} no conjunto de teste")


# =============================================================================
# 10. EXPORTAR MODELO FINAL
# =============================================================================
print("\n" + "=" * 70)
print("10. EXPORTAR MODELO FINAL")
print("=" * 70)

import joblib
model_path = "fraud_detection_model.joblib"
best_final_model = results[winner]["model"]
joblib.dump(best_final_model, model_path)
print(f"  [OK] Modelo salvo em: {model_path}")
print(f"  [OK] Todos os graficos salvos em: {FIG_DIR}/")

print("\n" + "=" * 70)
print("PIPELINE CONCLUÍDO COM SUCESSO!")
print("=" * 70)
print(f"""
Resumo do Projeto:
  • Dataset: 284.807 transações (492 fraudes = 0.172%)
  • Métrica principal: AUPRC (padrão ouro para dados desbalanceados)
  • Modelos comparados: Random Forest e XGBoost (com Class Weights e SMOTE)
  • Melhor modelo: {winner} (AUPRC = {winner_auprc:.4f})
  • Gráficos gerados: {len(os.listdir(FIG_DIR))} figuras em '{FIG_DIR}/'

Diferenciais para o Portfólio:
  • Análise de Feature Importance (V1–V28)
  • Análise de Impacto Financeiro (Business Impact)
  • Gráficos com estilo publicação (seaborn-v0_8)
  • Código completamente documentado com justificativas técnicas
""")
