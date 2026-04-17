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
# 1. TOPICS (6 REQUÊTES)
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
# 2. SCRAPING TWITTER (Playwright)
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


def collect_all_queries():

    all_data = {}

    with sync_playwright() as p:
        
        browser = p.chromium.connect_over_cdp("http://localhost:9222")

        context = browser.contexts[0]
        page = context.new_page()

        for topic in TOPICS:
            print(f"\n📥 {topic['num']} - {topic['title']}")

            posts = scrape_tweets_for_query(page, topic["title"], 100)
            print(f"   → {len(posts)} tweets")

            all_data[topic["num"]] = posts
            time.sleep(5)

        browser.close()

    return all_data

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
# 4. BUILD DATASET (600 tweets)
# ============================================================
def build_dataset():

    all_data = collect_all_queries()

    corpus = []
    qrels = []

    for topic in TOPICS:
        qid = topic["num"]
        posts = all_data[qid]

        for i, post in enumerate(posts):

            corpus.append({
                "id": post["id"],
                "timestamp": post["date"],
                "user": post["author"],
                "text": post["text"],
                "lang": "en",
                "retweets": 0,
                "likes": 0
            })

            relevance = 1 if i < 30 else 0
            qrels.append(f"{qid} 0 {post['id']} {relevance}")

    return corpus, qrels

# ============================================================
# 5. SAVE FILES
# ============================================================
def save_files(corpus, qrels):

    os.makedirs("collection", exist_ok=True)

    with open("collection/corpus_tweets.json", "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)

    with open("collection/topics.json", "w", encoding="utf-8") as f:
        json.dump(TOPICS, f, indent=2, ensure_ascii=False)

    with open("collection/qrels.txt", "w", encoding="utf-8") as f:
        for line in qrels:
            f.write(line + "\n")

    print("\n📁 Files generated in /collection")

# ============================================================
# 6. INDEXATION
# ============================================================
def build_index(corpus, method, base_dir="indexes"):

    abs_base = os.path.abspath(base_dir)
    index_dir = os.path.join(abs_base, f"index_{method}")
    os.makedirs(index_dir, exist_ok=True)

    print(f"\n🔨 Index {method}")

    import pandas as pd
    df = pd.DataFrame(corpus)

    df["processed"] = df["text"].apply(lambda x: preprocess(x, method))

    index_data = df[["id", "processed"]].rename(
        columns={"id": "docno", "processed": "text"}
    )

    indexer = pt.IterDictIndexer(index_dir, overwrite=True, meta={"docno": 50})
    indexer.index(index_data.to_dict(orient="records"))

# ============================================================
# 7. MAIN
# ============================================================
def main():

    if not pt.started():
        pt.init()

    print("\n==============================")
    print("SRI PROJECT - TWITTER SCRAPING")
    print("==============================")

    corpus, qrels = build_dataset()

    save_files(corpus, qrels)

    for method in ["lexeme", "stem", "lemma"]:
        build_index(corpus, method)

    print("\n🎉 DONE: 600 tweets collected and indexed")

# ============================================================
if __name__ == "__main__":
    main()