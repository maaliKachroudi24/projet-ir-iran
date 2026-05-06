import os
import re
import time
import json
import random
import pyterrier as pt
from playwright.sync_api import sync_playwright
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ============================================================
# 1. TOPICS
# ============================================================
TOPICS = [
  {"num": "MB01", "title": "regime change Iran"},
  {"num": "MB02", "title": "closing Hormuz strait"},
  {"num": "MB03", "title": "US bases attacked"},
  {"num": "MB04", "title": "supreme leader Khamenei"},
  {"num": "MB05", "title": "Iran nuclear deal"},
  {"num": "MB06", "title": "Middle East conflict escalation"}
]

# ============================================================
# 2. SCRAPING (OPTIONAL - NOT USED NOW)
# ============================================================
def scrape_tweets_for_query(page, query_text, target=100):

  url = f"https://x.com/search?q={query_text.replace(' ', '%20')}&src=typed_query"
  page.goto(url, wait_until="domcontentloaded")

  time.sleep(5)

  collected = []
  seen_ids = set()
  scrolls = 0

  while len(collected) < target and scrolls < 60:

    tweets = page.query_selector_all("article[data-testid='tweet']")

    for tweet in tweets:
      try:
        text_el = tweet.query_selector("div[data-testid='tweetText']")
        if not text_el:
          continue

        text = text_el.inner_text().strip()

        link_el = tweet.query_selector("a[href*='/status/']")
        if not link_el:
          continue

        href = link_el.get_attribute("href")
        match = re.search(r"/status/(\d+)", href)
        if not match:
          continue

        doc_id = match.group(1)

        if doc_id in seen_ids:
          continue

        seen_ids.add(doc_id)

        author_el = tweet.query_selector("div[data-testid='User-Name']")
        author = author_el.inner_text().split("\n")[0] if author_el else "unknown"

        time_el = tweet.query_selector("time")
        created_at = time_el.get_attribute("datetime") if time_el else ""

        collected.append({
          "id": doc_id,
          "text": text,
          "author": author,
          "date": created_at
        })

        if len(collected) >= target:
          break

      except:
        continue

    page.evaluate(f"window.scrollBy(0, {random.randint(1000,2000)})")
    time.sleep(random.uniform(2,4))
    scrolls += 1

  return collected


# ============================================================
# 3. PREPROCESSING
# ============================================================
def preprocess(text, method):

  if not isinstance(text, str):
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
    raise ValueError("method must be lexeme, stem, lemma")


# ============================================================
# 4. LOAD EXISTING DATA
# ============================================================
def load_corpus(path="collection/corpus_tweets.json"):

  if not os.path.exists(path):
    raise FileNotFoundError("❌ corpus file not found. Run scraping first.")

  with open(path, "r", encoding="utf-8") as f:
    corpus = json.load(f)

  print(f"📂 Loaded corpus: {len(corpus)} documents")
  return corpus


# ============================================================
# 5. INDEXATION
# ============================================================
def build_index(corpus, method, base_dir="indexes"):

  abs_base = os.path.abspath(base_dir)
  index_dir = os.path.join(abs_base, f"index_{method}")
  os.makedirs(index_dir, exist_ok=True)

  print(f"\n🔨 Building index: {method}")

  import pandas as pd
  df = pd.DataFrame(corpus)

  df["processed"] = df["text"].apply(lambda x: preprocess(x, method))

  index_data = df[["id", "processed"]].rename(
    columns={"id": "docno", "processed": "text"}
  )

  indexer = pt.IterDictIndexer(index_dir, overwrite=True, meta={"docno": 50})
  index_ref = indexer.index(index_data.to_dict(orient="records"))

  print(f"✅ Index saved in: {index_dir}")

  return index_ref


# ============================================================
# 6. MAIN (INDEX ONLY)
# ============================================================
def main():

  if not pt.started():
    pt.init()

  print("\n==============================")
  print("SRI PROJECT - INDEXING ONLY")
  print("==============================")

  # ✅ Load existing data
  corpus = load_corpus()

  # Build indexes
  indexes = {}
  for method in ["lexeme", "stem", "lemma"]:
    indexes[method] = build_index(corpus, method)

  print("\n🎉 DONE: All indexes created successfully")


# ============================================================
if __name__ == "__main__":
  main()