import os
import re
import time
import json
import random
import threading
from queue import Queue

import pyterrier as pt
from playwright.sync_api import sync_playwright

# ============================================================
# CONFIG
# ============================================================

TARGET_TWEETS = 100
MAX_RETRIES = 5
MAX_TOTAL_ATTEMPTS = 8

# ============================================================
# TOPICS
# ============================================================

TOPICS = [
    {"num": f"MB{i+1:02d}", "title": t}
    for i, t in enumerate([
        "regime change Iran","closing Hormuz strait","US bases attacked",
        "supreme leader Khamenei","Iran nuclear deal","Middle East conflict escalation",
        "Iran Israel conflict","US Iran tensions","oil prices Middle East war",
        "Gulf security crisis","nuclear sanctions Iran","Strait of Hormuz blockade",
        "Hezbollah Israel war","Yemen Houthis attacks","US military bases Middle East",
        "diplomatic talks Iran US","missile attacks Gulf region","OPEC oil disruption",
        "Israel Iran escalation news","Iran proxy wars","Middle East crisis",
        "Tehran protests regime","sanctions Iran economy","oil shipping threats Gulf",
        "US foreign policy Iran"
    ])
]

# ============================================================
# GLOBALS
# ============================================================

corpus_global = []
qrels_global = []
attempts = {}
lock = threading.Lock()

# ============================================================
# QUERY EXPANSION
# ============================================================

def expand_query(q):
    return [
        q,
        q + " news",
        q + " latest",
        q + " update",
        q + " war",
        q + " conflict"
    ]

# ============================================================
# SCRAPE SAFE VERSION (ANTI TIMEOUT + ANTI CRASH)
# ============================================================

def scrape(page, query, target):

    url = f"https://x.com/search?q={query.replace(' ', '%20')}&src=typed_query"

    # 🔥 SAFE NAVIGATION
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
    except Exception as e:
        print(f"⚠️ goto failed: {query}")
        return []

    time.sleep(5)

    collected, seen = [], set()

    for _ in range(120):

        try:
            tweets = page.query_selector_all("article[data-testid='tweet']")
        except:
            continue

        for t in tweets:

            if len(collected) >= target:
                return collected

            try:
                text_el = t.query_selector("div[data-testid='tweetText']")
                link_el = t.query_selector("a[href*='/status/']")

                if not text_el or not link_el:
                    continue

                text = text_el.inner_text().strip()
                href = link_el.get_attribute("href")

                match = re.search(r"/status/(\d+)", href)
                if not match:
                    continue

                doc_id = match.group(1)

                if doc_id in seen:
                    continue

                seen.add(doc_id)

                author_el = t.query_selector("div[data-testid='User-Name']")
                author = author_el.inner_text().split("\n")[0] if author_el else "unknown"

                time_el = t.query_selector("time")
                timestamp = time_el.get_attribute("datetime") if time_el else ""

                collected.append({
                    "id": doc_id,
                    "timestamp": timestamp,
                    "user": author,
                    "text": text,
                    "lang": "en",
                    "retweets": 0,
                    "likes": 0
                })

                print(f"📊 {len(collected)}/100", end="\r")

            except:
                continue

        # 🔥 SAFE SCROLL
        try:
            page.evaluate("window.scrollBy(0, 2500)")
        except:
            pass

        time.sleep(random.uniform(2, 3))

    return collected

# ============================================================
# RETRY SYSTEM
# ============================================================

def collect_with_retry(page, query):

    for attempt in range(MAX_RETRIES):

        for q in expand_query(query):

            print(f"\n🔄 Attempt {attempt+1} → {q}")

            data = scrape(page, q, TARGET_TWEETS)

            if len(data) == TARGET_TWEETS:
                return data

        try:
            page.reload(timeout=60000)
        except:
            pass

        time.sleep(random.uniform(5, 10))

    return []

# ============================================================
# SAVE SAFE
# ============================================================

def save():

    os.makedirs("collection", exist_ok=True)

    with open("collection/corpus_tweets.json", "w", encoding="utf-8") as f:
        json.dump(corpus_global, f, indent=2, ensure_ascii=False)

    with open("collection/qrels.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(qrels_global))

    with open("collection/topics.json", "w", encoding="utf-8") as f:
        json.dump(TOPICS, f, indent=2, ensure_ascii=False)

    print(f"\n📁 SAVED → corpus={len(corpus_global)}")

# ============================================================
# WORKER (ANTI CRASH FINAL VERSION)
# ============================================================

def worker(queue):

    with sync_playwright() as p:

        browser = p.chromium.connect_over_cdp("http://localhost:9222")
        page = browser.contexts[0].new_page()

        while not queue.empty():

            try:
                topic = queue.get()
                qid = topic["num"]
                title = topic["title"]

                attempts[qid] = attempts.get(qid, 0) + 1

                print(f"\n🧵 {qid} attempt {attempts[qid]}")

                data = collect_with_retry(page, title)

                if len(data) == 100:

                    local_qrels = []

                    for i, d in enumerate(data):

                        corpus_global.append(d)

                        rel = 1 if i < 30 else 0
                        local_qrels.append(f"{qid} 0 {d['id']} {rel}")

                    with lock:
                        qrels_global.extend(local_qrels)
                        save()

                    print(f"✅ {qid} DONE")

                else:

                    if attempts[qid] < MAX_TOTAL_ATTEMPTS:
                        print(f"🔁 REQUEUE {qid}")
                        queue.put(topic)
                    else:
                        print(f"❌ FINAL FAIL {qid}")

                queue.task_done()

            except Exception as e:
                print(f"🔥 THREAD ERROR: {e}")
                queue.task_done()
                continue

# ============================================================
# RUN
# ============================================================

def run():

    queue = Queue()

    for t in TOPICS:
        queue.put(t)

    threads = []

    for _ in range(5):
        t = threading.Thread(target=worker, args=(queue,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

# ============================================================
# MAIN
# ============================================================

def main():

    if not pt.java.started():
        pt.java.init()

    print("\n🚀 START SCRAPING")

    run()

    print("\n🎉 DONE - STABLE VERSION COMPLETED")

if __name__ == "__main__":
    main()