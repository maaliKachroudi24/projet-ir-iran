#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tâches 3 & 4 : Recherche et évaluation avec différents modèles
"""

import os
import re
import glob
import pyterrier as pt
import pandas as pd
import ir_measures
from ir_measures import MAP, P, Recall, RPrec
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

if not pt.java.started():
    pt.java.init()

# ============================================================
# 1. CONFIGURATION
# ============================================================
TOPICS = [
    {"qid": "MB01", "query": "regime change Iran"},
    {"qid": "MB02", "query": "closing Hormuz strait"},
    {"qid": "MB03", "query": "US bases attacked"},
    {"qid": "MB04", "query": "supreme leader Khamenei"},
    {"qid": "MB05", "query": "Iran nuclear deal"},
    {"qid": "MB06", "query": "Middle East conflict escalation"},
]

queries_df = pd.DataFrame(TOPICS)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# 2. CHARGEMENT DES QRELS
# ============================================================
def load_qrels(path="collection/qrels.txt"):
    """
    Lit qrels.txt (format TREC : qid 0 docno relevance)
    et retourne un DataFrame compatible ir_measures.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                qid, _, docno, rel = parts
                rows.append({"query_id": qid, "doc_id": docno, "relevance": int(rel)})
    df = pd.DataFrame(rows)
    print(f"✅ Qrels chargés : {len(df)} jugements")
    return df

# ============================================================
# 3. PRÉTRAITEMENT
# ============================================================
stemmer    = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text, method):
    if not isinstance(text, str):
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
# 4. CHARGEMENT DE L'INDEX  ← CORRECTION PRINCIPALE
# ============================================================
def find_index_path(method, base_dir="indexes"):
    """
    IterDictIndexer crée parfois un sous-dossier supplémentaire.
    Cette fonction cherche data.properties de manière récursive
    et retourne le dossier qui le contient.
    """
    base = os.path.abspath(base_dir)
    top  = os.path.join(base, f"index_{method}")

    # Recherche récursive de data.properties
    candidates = glob.glob(os.path.join(top, "**", "data.properties"), recursive=True)
    if candidates:
        return os.path.dirname(candidates[0])

    # Vérification directe
    if os.path.exists(os.path.join(top, "data.properties")):
        return top

    raise FileNotFoundError(
        f"Aucun data.properties trouvé sous {top}\n"
        f"Vérifiez que task_1_2.py a bien créé les index."
    )


def load_index(method):
    idx_path = find_index_path(method)
    print(f"  📂 Index {method.upper()} → {idx_path}")
    return pt.IndexFactory.of(idx_path)

# ============================================================
# 5. TÂCHE 3 : RETRIEVAL
# ============================================================
MODELS = {
    "TF_IDF": "TF_IDF",
    "BM25":   "BM25",
    "PL2":    "PL2",
    "DPH":    "DPH",
}

def run_all_experiments():
    all_runs = {}

    for method in ["lexeme", "stem", "lemma"]:
        print(f"\n{'='*55}")
        print(f"  Prétraitement : {method.upper()}")
        print(f"{'='*55}")

        try:
            index = load_index(method)
        except FileNotFoundError as e:
            print(f"  ❌ {e}")
            continue

        # Requêtes prétraitées avec la même méthode que l'index
        q_proc = queries_df.copy()
        q_proc["query"] = q_proc["query"].apply(lambda x: preprocess_text(x, method))

        for model_name, wmodel in MODELS.items():
            key = f"{method}_{model_name}"
            print(f"  🔍 Modèle {model_name}...", end=" ", flush=True)
            try:
                retriever = pt.terrier.Retriever(index, wmodel=wmodel) % 30
                results   = retriever.transform(q_proc)
                all_runs[key] = results

                out = os.path.join(RESULTS_DIR, f"run_{key}.csv")
                results.to_csv(out, index=False)
                print(f"✅  ({len(results)} lignes)")
            except Exception as e:
                print(f"❌  Erreur : {e}")

    return all_runs

# ============================================================
# 6. TÂCHE 4 : ÉVALUATION
# ============================================================
METRICS = [MAP, P@1, P@5, P@10, Recall@30, RPrec]

def evaluate_all(all_runs, qrels_df):
    summary_rows = []

    print(f"\n{'='*55}")
    print("  ÉVALUATION")
    print(f"{'='*55}")

    for key, run_pt in all_runs.items():
        method, model = key.split("_", 1)

        # Convertir au format ir_measures : query_id, doc_id, score
        run_im = (run_pt
                  .rename(columns={"qid": "query_id", "docno": "doc_id", "score": "score"})
                  [["query_id", "doc_id", "score"]])

        try:
            scores = ir_measures.calc_aggregate(METRICS, qrels_df, run_im)
            row = {"experiment": key, "method": method, "model": model}
            for m, v in scores.items():
                row[str(m)] = round(v, 4)
            summary_rows.append(row)

            # Affichage résumé ligne par ligne
            map_val = scores.get(MAP, "-")
            p5_val  = scores.get(P@5, "-")
            print(f"  {key:<25}  MAP={map_val:.4f}  P@5={p5_val:.4f}")

        except Exception as e:
            print(f"  ❌ Erreur évaluation {key}: {e}")

    summary_df = pd.DataFrame(summary_rows)
    out = os.path.join(RESULTS_DIR, "evaluation_summary.csv")
    summary_df.to_csv(out, index=False)
    print(f"\n✅ Résumé sauvegardé → {out}")
    return summary_df


def print_summary(summary_df):
    if summary_df.empty:
        print("\n❌ Aucun résultat à afficher.")
        return

    print("\n" + "="*70)
    print("TABLEAU RÉCAPITULATIF")
    print("="*70)
    print(summary_df.to_string(index=False))
    print("="*70)

    # Trouve la colonne MAP (peut s'appeler "AP" ou "MAP" selon ir_measures)
    map_col = next((c for c in summary_df.columns if c in ("AP", "nDCG", "MAP")), None)
    if map_col:
        best = summary_df.loc[summary_df[map_col].idxmax()]
        print(f"\n🏆 Meilleure stratégie ({map_col}) : {best['experiment']}  = {best[map_col]:.4f}")


def plot_recall_precision(all_runs, qrels_df):
    """Courbes Rappel-Précision interpolées (11 niveaux standard)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        recall_levels = [i / 10 for i in range(11)]
        # Compatibilité selon la version de ir_measures
        try:
            rp_metrics = [ir_measures.IPrec @ r for r in recall_levels]
        except Exception:
            try:
                rp_metrics = [ir_measures.iprec_at_recall[r] for r in recall_levels]
            except Exception:
                print("  ⚠️  ir_measures ne supporte pas les métriques iprec — courbes ignorées")
                return

        fig, ax = plt.subplots(figsize=(10, 7))
        plotted = False

        for key, run_pt in all_runs.items():
            run_im = (run_pt
                      .rename(columns={"qid": "query_id", "docno": "doc_id", "score": "score"})
                      [["query_id", "doc_id", "score"]])
            try:
                scores    = ir_measures.calc_aggregate(rp_metrics, qrels_df, run_im)
                prec_vals = [scores.get(m, 0.0) for m in rp_metrics]
                ax.plot(recall_levels, prec_vals, marker='o', label=key)
                plotted = True
            except Exception as e:
                print(f"  ⚠️  Courbe R-P impossible pour {key}: {e}")

        if plotted:
            ax.set_xlabel("Rappel")
            ax.set_ylabel("Précision")
            ax.set_title("Courbes Rappel-Précision — Toutes stratégies")
            ax.legend(loc="upper right", fontsize=7)
            ax.grid(True)
            fig.tight_layout()

            plot_path = os.path.join(RESULTS_DIR, "recall_precision_curves.png")
            fig.savefig(plot_path, dpi=150)
            print(f"\n📈 Courbes R-P → {plot_path}")
        else:
            print("\n⚠️  Aucune courbe générée.")

    except ImportError:
        print("  ⚠️  matplotlib non installé — pip install matplotlib")

# ============================================================
# 7. MAIN
# ============================================================
def main():
    print("\n" + "="*55)
    print("SRI PROJECT — TÂCHES 3 & 4")
    print("="*55)

    qrels_df = load_qrels("collection/qrels.txt")

    # ── Tâche 3 : Retrieval ──
    all_runs = run_all_experiments()

    if not all_runs:
        print("\n❌ Aucun index chargé. Vérifiez que task_1_2.py a créé le dossier indexes/")
        return

    # ── Tâche 4 : Évaluation ──
    summary_df = evaluate_all(all_runs, qrels_df)
    plot_recall_precision(all_runs, qrels_df)
    print_summary(summary_df)

    print("\n🎉 DONE — Résultats dans le dossier /results/")


if __name__ == "__main__":
    main()