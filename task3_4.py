#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tâches 3 & 4 : Recherche et évaluation avec différents modèles
Charge les index créés précédemment et calcule MAP, P@1, P@5, P@10.
"""

import os
import re
import pyterrier as pt
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import ir_measures
from ir_measures import MAP, P, Recall
import nltk

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

if not pt.java.started():
    pt.java.init()

# ============================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================
queries_df = pd.DataFrame([
    {"qid": "q1", "query": "regime change Iran"},
    {"qid": "q2", "query": "closing Hormuz strait"},
    {"qid": "q3", "query": "US bases attacked"},
    {"qid": "q4", "query": "supreme leader Khamenei"},
    {"qid": "q5", "query": "Iran nuclear deal"}
])

qrels_raw = pd.read_csv("qrels.csv", sep="\t")
print(f"✅ Qrels chargés : {len(qrels_raw)} jugements")

# Convertir qrels au format ir_measures
qrels_im = qrels_raw.rename(columns={
    "query_id": "query_id",
    "docno": "doc_id",
    "relevance": "relevance"
})[["query_id", "doc_id", "relevance"]]
qrels_im.columns = ["query_id", "doc_id", "relevance"]


# ============================================================
# 2. PRÉTRAITEMENT
# ============================================================
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text, method):
    if not isinstance(text, str) or pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    if method == "lexeme":
        return " ".join(tokens)
    elif method == "stem":
        return " ".join(stemmer.stem(t) for t in tokens)
    elif method == "lemma":
        return " ".join(lemmatizer.lemmatize(t) for t in tokens)
    return ""

# ============================================================
# 3. CHARGEMENT D'INDEX
# ============================================================
def load_index(method):
    base_dir = os.path.abspath("indexes")
    index_path = os.path.join(base_dir, f"index_{method}")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index introuvable : {index_path}")
    print(f"📂 Chargement de l'index {method.upper()} depuis {index_path}")
    return pt.IndexFactory.of(index_path)

# ============================================================
# 4. ÉVALUATION
# ============================================================
preprocess_methods = ["lexeme", "stem", "lemma"]
models = {
    "TF_IDF": "TF_IDF",
    "BM25":   "BM25",
    "PL2":    "PL2",
    "DPH":    "DPH"
}

metrics = [MAP, P@1, P@5, P@10, Recall@30]

all_results = []

for prep in preprocess_methods:
    print(f"\n{'='*50}")
    print(f"Prétraitement : {prep.upper()}")
    print('='*50)

    try:
        index = load_index(prep)
    except Exception as e:
        print(f"❌ Erreur chargement index : {e}")
        continue

    # Prétraiter les requêtes
    queries_proc = queries_df.copy()
    queries_proc["query"] = queries_proc["query"].apply(lambda x: preprocess_text(x, prep))

    for model_name, wmodel in models.items():
        print(f"  Modèle : {model_name}...", end=" ", flush=True)
        try:
            retriever = pt.terrier.Retriever(index, wmodel=wmodel) % 30
            results = retriever.transform(queries_proc)

            # Convertir results au format ir_measures
            run = results.rename(columns={"qid": "query_id", "docno": "doc_id", "score": "score"})
            run = run[["query_id", "doc_id", "score"]]

            # Évaluation avec ir_measures
            scores = ir_measures.calc_aggregate(metrics, qrels_im, run)

            row = {"preprocessing": prep, "model": model_name}
            for m, v in scores.items():
                row[str(m)] = round(v, 4)
            all_results.append(row)
            print("✅")

        except Exception as e:
            print(f"❌ Erreur : {e}")

# ============================================================
# 5. SYNTHÈSE FINALE
# ============================================================
if all_results:
    summary = pd.DataFrame(all_results)
    print("\n" + "="*70)
    print("RÉSULTATS FINAUX")
    print("="*70)
    print(summary.to_string(index=False))
    summary.to_csv("evaluation_results.csv", index=False)
    print("\n✅ Résultats sauvegardés dans evaluation_results.csv")
else:
    print("\n❌ Aucun résultat généré. Vérifiez vos index.")