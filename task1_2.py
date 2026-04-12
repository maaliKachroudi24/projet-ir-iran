
import os
import re
import time
import requests
import pandas as pd
import pyterrier as pt
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ============================================================
# 1. REQUÊTES
# ============================================================
QUERIES = [
    {"id": "q1", "text": "regime change Iran"},
    {"id": "q2", "text": "closing Hormuz strait"},
    {"id": "q3", "text": "US bases attacked"},
    {"id": "q4", "text": "supreme leader Khamenei"},
    {"id": "q5", "text": "Iran nuclear deal"}
]

# ============================================================
# 2. COLLECTION MASTODON
# ============================================================
class MastodonCollector:
    def __init__(self, instance="https://mastodon.social"):
        self.api = f"{instance}/api/v1"

    def search_hashtag(self, hashtag, limit=50):
        url = f"{self.api}/timelines/tag/{hashtag}"
        params = {"limit": min(limit, 40), "local": False}
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            posts = r.json()
            results = []
            for p in posts:
                results.append({
                    "id": p["id"],
                    "text": self._clean_html(p["content"]),
                    "author": p["account"]["username"],
                    "url": p["url"],
                    "date": p["created_at"]
                })
            return results
        except Exception as e:
            print(f"  ⚠️ Erreur hashtag #{hashtag}: {e}")
            return []

    def _clean_html(self, html):
        clean = re.sub(r'<[^>]+>', '', html)
        clean = re.sub(r'&[a-z]+;', ' ', clean)
        return clean.strip()

    def collect_for_query(self, query_text, target=100):
        words = [w for w in query_text.lower().split() if len(w) > 2 and w not in ("the","and","for","with")]
        hashtags = list(set(words))[:3]
        if "iran" not in hashtags:
            hashtags.append("iran")
        print(f"  🔍 Hashtags: {hashtags}")

        all_posts = []
        for tag in hashtags:
            posts = self.search_hashtag(tag, limit=target)
            for p in posts:
                p["source_tag"] = tag
            all_posts.extend(posts)
            time.sleep(0.5)

        seen = set()
        unique = []
        for p in all_posts:
            if p["id"] not in seen:
                seen.add(p["id"])
                unique.append(p)
        return unique[:target]

def build_collection():
    collector = MastodonCollector()
    corpus_records = []
    qrels_records = []

    for q in QUERIES:
        qid = q["id"]
        qtext = q["text"]
        print(f"\n📥 Requête {qid} : '{qtext}'")
        posts = collector.collect_for_query(qtext, target=100)
        print(f"   → {len(posts)} posts uniques")

        for idx, post in enumerate(posts):
            doc_id = f"mastodon_{post['id']}"
            if not any(d["docno"] == doc_id for d in corpus_records):
                corpus_records.append({
                    "docno": doc_id,
                    "text": post["text"],
                    "author": post["author"],
                    "url": post["url"],
                    "date": post["date"]
                })
            rel = 1 if idx < 30 else 0
            qrels_records.append({
                "query_id": qid,
                "docno": doc_id,
                "relevance": rel
            })

    corpus_df = pd.DataFrame(corpus_records)
    qrels_df = pd.DataFrame(qrels_records)
    return corpus_df, qrels_df

# ============================================================
# 3. INDEXATION (avec chemins absolus)
# ============================================================
def preprocess(text, method):
    if not isinstance(text, str) or pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    if method == "lexeme":
        return " ".join(tokens)
    elif method == "stem":
        stemmer = PorterStemmer()
        return " ".join(stemmer.stem(t) for t in tokens)
    elif method == "lemma":
        lemmatizer = WordNetLemmatizer()
        return " ".join(lemmatizer.lemmatize(t) for t in tokens)
    else:
        raise ValueError("method doit être 'lexeme', 'stem' ou 'lemma'")

def build_index(corpus_df, method, base_dir="indexes"):
    """Crée un index PyTerrier avec un chemin absolu."""
    # Créer le répertoire avec chemin absolu
    abs_base = os.path.abspath(base_dir)
    index_dir = os.path.join(abs_base, f"index_{method}")
    os.makedirs(index_dir, exist_ok=True)
    
    print(f"\n🔨 Indexation avec {method.upper()} dans {index_dir}")
    df = corpus_df.copy()
    df["processed"] = df["text"].apply(lambda x: preprocess(x, method))
    index_data = df[["docno", "processed"]].rename(columns={"processed": "text"})
    
    # Augmenter la taille max du champ docno (IDs Mastodon longs)
    indexer = pt.IterDictIndexer(index_dir, overwrite=True, meta={"docno": 50})
    index_ref = indexer.index(index_data.to_dict(orient="records"))
    print(f"   ✅ Index sauvegardé")
    return index_ref

# ============================================================
# 4. MAIN
# ============================================================
def main():
    if not pt.started():
        pt.init()

    print("=" * 60)
    print("TÂCHE 1 : Construction de la collection de test (Mastodon)")
    print("=" * 60)

    corpus_df, qrels_df = build_collection()

    # Sauvegarder les fichiers
    corpus_df.to_csv("corpus.csv", index=False, encoding="utf-8")
    qrels_df.to_csv("qrels.csv", index=False, sep="\t")
    print(f"\n✅ Corpus : {len(corpus_df)} documents uniques → corpus.csv")
    print(f"✅ Qrels   : {len(qrels_df)} jugements → qrels.csv")

    print("\n" + "=" * 60)
    print("TÂCHE 2 : Indexation (lexèmes, stems, lemmes)")
    print("=" * 60)

    # Créer les trois index
    for method in ["lexeme", "stem", "lemma"]:
        build_index(corpus_df, method)

    print("\n🎉 Tâches 1 et 2 terminées avec succès !")
    print("   Fichiers générés : corpus.csv, qrels.csv, indexes/")

if __name__ == "__main__":
    main()