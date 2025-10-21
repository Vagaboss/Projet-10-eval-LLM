"""
compare_ragas.py — Comparaison entre RAG classique et RAG + SQL Tool
"""

import os
import json
import matplotlib.pyplot as plt
import pandas as pd

# --- Chargement des fichiers de résultats ---
RAG_RESULTS_PATH = os.path.join("eval", "results.json")
RAG_SQL_RESULTS_PATH = os.path.join("eval", "results_rag_sql.json")

def load_results(path):
    if not os.path.exists(path):
        print(f"❌ Fichier introuvable : {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

rag = load_results(RAG_RESULTS_PATH)
rag_sql = load_results(RAG_SQL_RESULTS_PATH)

if not rag or not rag_sql:
    print("⚠️ Impossible de charger les fichiers RAG et/ou RAG+SQL.")
    exit(1)

# --- Extraire les métriques principales ---
metrics = ["faithfulness", "answer_similarity", "context_precision", "context_recall"]

rag_scores = [rag["metrics"].get(m, 0) for m in metrics]
rag_sql_scores = [rag_sql["metrics"].get(m, 0) for m in metrics]

# --- Créer un DataFrame pour comparaison ---
df = pd.DataFrame({
    "Métrique": metrics,
    "RAG seul": rag_scores,
    "RAG + SQL": rag_sql_scores,
    "Gain (%)": [round(((b - a) / a * 100), 1) if a != 0 else 0 for a, b in zip(rag_scores, rag_sql_scores)]
})

# --- Affichage console ---
print("\n===== 📊 COMPARATIF DES SCORES RAGAS =====\n")
print(df.to_markdown(index=False))

# --- Graphique comparatif ---
plt.figure(figsize=(8, 5))
x = range(len(metrics))
plt.bar(x, rag_scores, width=0.4, label="RAG seul", align="center", color="#4B9CD3")
plt.bar([i + 0.4 for i in x], rag_sql_scores, width=0.4, label="RAG + SQL Tool", align="center", color="#66CDAA")

plt.xticks([i + 0.2 for i in x], metrics, rotation=20)
plt.ylabel("Score")
plt.title("Comparaison RAG vs RAG + SQL Tool (RAGAS)")
plt.legend()
plt.tight_layout()
plt.show()

# --- Analyse qualitative ---
print("\n===== 🧠 INTERPRÉTATION =====")
for i, m in enumerate(metrics):
    diff = rag_sql_scores[i] - rag_scores[i]
    if diff > 0.05:
        print(f"✅ {m}: amélioration notable (+{diff:.3f})")
    elif diff > 0:
        print(f"➕ {m}: légère amélioration (+{diff:.3f})")
    elif diff < 0:
        print(f"⚠️ {m}: légère baisse ({diff:.3f})")
    else:
        print(f"🔹 {m}: stable")

print("\n✅ Analyse terminée. Le graphique de comparaison s’est affiché avec succès.")
