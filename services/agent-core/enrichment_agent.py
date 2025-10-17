import os
import re
import json
import time
import boto3
import requests
import threading
import pandas as pd
from html import unescape
from datetime import datetime
from dotenv import load_dotenv
from collections import Counter
from urllib.parse import urlparse
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from concurrent.futures import ThreadPoolExecutor, as_completed

CONFIDENCE_RECORDS = []

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.environ.get("GOOGLE_SEARCH_ENGINE_ID")
BING_API_KEY = os.environ.get("BING_API_KEY")
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")

bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")

# Initialize AgentCore App
app = BedrockAgentCoreApp()

MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "6"))
LLM_MAX_CONCURRENCY = int(os.environ.get("LLM_MAX_CONCURRENCY", "3"))
SEARCH_MAX_CONCURRENCY = int(os.environ.get("SEARCH_MAX_CONCURRENCY", "6"))
FETCH_MAX_CONCURRENCY = int(os.environ.get("FETCH_MAX_CONCURRENCY", "6"))
LLM_TIMEOUT_SEC = float(os.environ.get("LLM_TIMEOUT_SEC", "25"))
HTTP_TIMEOUT_SEC = float(os.environ.get("HTTP_TIMEOUT_SEC", "20"))
FETCH_TIMEOUT_SEC = float(os.environ.get("FETCH_TIMEOUT_SEC", "12"))
LLM_COOLDOWN_SEC = float(os.environ.get("LLM_COOLDOWN_SEC", "0.35"))
SEARCH_COOLDOWN_SEC = float(os.environ.get("SEARCH_COOLDOWN_SEC", "0.2"))
MAX_SNIPPETS_TOTAL = int(os.environ.get("MAX_SNIPPETS_TOTAL", "80"))
PER_QUERY_RESULTS = int(os.environ.get("PER_QUERY_RESULTS", "10"))
MAX_QUERIES = int(os.environ.get("MAX_QUERIES", "12"))
PROMPT_CONTEXT_CHARS = int(os.environ.get("PROMPT_CONTEXT_CHARS", "1800"))
PROMPT_SNIPPETS_CHARS = int(os.environ.get("PROMPT_SNIPPETS_CHARS", "5000"))
PASS_BATCH_SIZE = int(os.environ.get("PASS_BATCH_SIZE", "32"))
MIN_CONFIDENCE = float(os.environ.get("MIN_CONFIDENCE", "0.7"))
INCLUDE_DOMAIN_QUERIES = os.environ.get("INCLUDE_DOMAIN_QUERIES", "false").lower() == "true"
ENABLE_FETCH_HTML = os.environ.get("ENABLE_FETCH_HTML", "true").lower() == "true"
MAX_FETCH_PAGES = int(os.environ.get("MAX_FETCH_PAGES", "20"))
MAX_FETCH_BYTES = int(os.environ.get("MAX_FETCH_BYTES", "350000"))

LOGS = []
LOG_LOCK = threading.Lock()
LLM_SEM = threading.Semaphore(LLM_MAX_CONCURRENCY)
SEARCH_SEM = threading.Semaphore(SEARCH_MAX_CONCURRENCY)
FETCH_SEM = threading.Semaphore(FETCH_MAX_CONCURRENCY)

SCHEMA_CACHE = {"forward": {}, "reverse": {}, "labels": {}}

def jlog(event_type, **data):
    payload = {"ts": datetime.utcnow().isoformat() + "Z", "event": event_type, **data}
    with LOG_LOCK:
        LOGS.append(payload)
        print(json.dumps(payload, ensure_ascii=False))

def write_logs_json(path):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(LOGS, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(json.dumps({"ts": datetime.utcnow().isoformat() + "Z", "event": "log_write_error", "error": str(e)}))

def _retry(fn, retries=3, backoff=0.8, base_delay=0.6, name="op"):
    err = None
    for i in range(1, retries + 1):
        try:
            out = fn()
            if out is not None:
                return out
        except Exception as e:
            err = str(e)
        delay = base_delay * (backoff ** (i - 1))
        jlog("retry_attempt", op=name, attempt=i, delay_sec=round(delay, 3), error=err or "")
        time.sleep(delay)
    jlog("retry_failed", op=name, error=err or "unknown")
    return None

def http_get_json(url, params=None, headers=None):
    def _call():
        with SEARCH_SEM:
            r = requests.get(url, params=params, headers=headers, timeout=HTTP_TIMEOUT_SEC)
            time.sleep(SEARCH_COOLDOWN_SEC)
            return r.json()
    return _retry(_call, retries=3, name="http_get_json")

def fetch_url_text(url):
    def _call():
        with FETCH_SEM:
            try:
                r = requests.get(
                    url,
                    timeout=FETCH_TIMEOUT_SEC,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.5"
                    },
                    allow_redirects=True
                )
                r.raise_for_status()  # Raise exception for bad status codes

                content_type = (r.headers.get("Content-Type") or "").lower()
                if "text/html" not in content_type and "application/xhtml" not in content_type:
                    return None

                b = r.content[:MAX_FETCH_BYTES]
                # Try multiple encodings
                for encoding in [r.apparent_encoding, 'utf-8', 'latin-1']:
                    try:
                        t = b.decode(encoding or 'utf-8', errors="ignore")
                        break
                    except:
                        continue

                t = re.sub(r"(?is)<(script|style|noscript|template|svg|meta|link|iframe)[^>]*>.*?</\1>", " ", t)
                t = re.sub(r"(?is)<[^>]+>", " ", t)
                t = unescape(t)
                t = re.sub(r"\s+", " ", t).strip()
                time.sleep(SEARCH_COOLDOWN_SEC)
                return t if t else None
            except requests.exceptions.Timeout:
                raise Exception(f"Timeout fetching {url}")
            except requests.exceptions.ConnectionError as e:
                raise Exception(f"Connection error: {str(e)[:100]}")
            except requests.exceptions.HTTPError as e:
                raise Exception(f"HTTP {e.response.status_code}: {url}")
            except Exception as e:
                raise Exception(f"Fetch error: {str(e)[:100]}")
    return _retry(_call, retries=2, base_delay=0.5, name="fetch_url_text")

def google_search_single(query, num_results=90, hl="en", lr="lang_en"):
    if not GOOGLE_API_KEY or not GOOGLE_SEARCH_ENGINE_ID:
        return []
    jlog("search_start", provider="google_cse", query=query)
    j = http_get_json("https://www.googleapis.com/customsearch/v1", {"key": GOOGLE_API_KEY, "cx": GOOGLE_SEARCH_ENGINE_ID, "q": query, "num": num_results, "hl": hl, "lr": lr})
    items = (j or {}).get("items", []) if isinstance(j, dict) else []
    out = []
    for it in items:
        rec = {"provider": "google_cse", "title": it.get("title", ""), "link": it.get("link", ""), "displayLink": it.get("displayLink", ""), "snippet": it.get("snippet", "")}
        out.append(rec)
        jlog("search_hit", provider="google_cse", query=query, title=rec["title"], domain=rec["displayLink"], url=rec["link"])
    jlog("search_done", provider="google_cse", query=query, hits=len(out))
    return out

def bing_search_single(query, num_results=10, mkt="en-US"):
    if not BING_API_KEY:
        return []
    jlog("search_start", provider="bing", query=query)
    headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
    params = {"q": query, "count": num_results, "mkt": mkt, "textDecorations": False, "textFormat": "Raw"}
    j = http_get_json("https://api.bing.microsoft.com/v7.0/search", params=params, headers=headers)
    web_pages = (j or {}).get("webPages", {}).get("value", []) if isinstance(j, dict) else []
    out = []
    for it in web_pages:
        rec = {"provider": "bing", "title": it.get("name", ""), "link": it.get("url", ""), "displayLink": urlparse(it.get("url", "")).hostname or "", "snippet": it.get("snippet", "")}
        out.append(rec)
        jlog("search_hit", provider="bing", query=query, title=rec["title"], domain=rec["displayLink"], url=rec["link"])
    jlog("search_done", provider="bing", query=query, hits=len(out))
    return out

def serpapi_google(query, num_results=10):
    if not SERPAPI_KEY:
        return []
    jlog("search_start", provider="serpapi_google", query=query)
    params = {"engine": "google", "q": query, "api_key": SERPAPI_KEY, "num": num_results}
    j = http_get_json("https://serpapi.com/search.json", params=params)
    items = (j or {}).get("organic_results", []) if isinstance(j, dict) else []
    out = []
    for it in items:
        rec = {"provider": "serpapi_google", "title": it.get("title", ""), "link": it.get("link", ""), "displayLink": urlparse(it.get("link", "")).hostname or "", "snippet": it.get("snippet", "")}
        out.append(rec)
        jlog("search_hit", provider="serpapi_google", query=query, title=rec["title"], domain=rec["displayLink"], url=rec["link"])
    jlog("search_done", provider="serpapi_google", query=query, hits=len(out))
    return out

def search_all_providers(query, per_query=10):
    results = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = []
        futs.append(ex.submit(google_search_single, query, per_query))
        futs.append(ex.submit(bing_search_single, query, per_query))
        futs.append(ex.submit(serpapi_google, query, per_query))
        for fut in as_completed(futs):
            hits = fut.result() or []
            results.extend(hits)
    return results

def google_search_multi(queries, per_query=10, max_total=80):
    seen = set()
    results = []
    with ThreadPoolExecutor(max_workers=min(len(queries), SEARCH_MAX_CONCURRENCY)) as ex:
        futs = {ex.submit(search_all_providers, q, per_query): q for q in queries}
        for fut in as_completed(futs):
            hits = fut.result() or []
            for h in hits:
                key = h.get("link")
                if key and key not in seen and len(results) < max_total:
                    seen.add(key)
                    results.append(h)
    jlog("search_aggregate_done", total=len(results), unique_domains=len({urlparse(r.get('link','')).hostname for r in results if r.get('link')}))
    return results

def ask_claude(prompt: str, max_tokens=1200, temp=0.0):
    jlog("llm_send", tokens=max_tokens, prompt_chars=len(prompt))
    def _task():
        with LLM_SEM:
            resp = bedrock.converse(
                modelId="anthropic.claude-3-haiku-20240307-v1:0",
                system=[{"text": "You are a stateless data fulfillment API. Respond deterministically and follow instructions exactly."}],
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": temp}
            )
            time.sleep(LLM_COOLDOWN_SEC)
            return resp["output"]["message"]["content"][0]["text"].strip()
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_task)
        try:
            out = fut.result(timeout=LLM_TIMEOUT_SEC)
        except Exception as e:
            jlog("llm_timeout", error=str(e))
            return ""
    if out is None:
        jlog("llm_error", message="no_output")
        return ""
    jlog("llm_recv", preview=out[:200].replace("\n", " "), out_chars=len(out))
    return out

def domain_authoritative(domain: str) -> bool:
    d = (domain or "").lower()
    return any([
        # Government & Educational
        d.endswith(".gov"),
        d.endswith(".edu"),
        d.endswith(".gov.uk"),
        d.endswith(".gov.au"),
        d.endswith(".gov.ca"),
        d.endswith(".org"),  # Non-profit organizations
        d.endswith(".ac.uk"),  # UK academic

        # Official documentation & support
        d.startswith("docs."),
        d.startswith("developer."),
        d.startswith("api."),
        "support." in d,
        "help." in d,
        "official" in d,
        "press" in d,
        "about." in d,

        # Major Entertainment & Media - Movies/TV
        "imdb.com" in d,
        "wikipedia.org" in d,
        "themoviedb.org" in d,
        "boxofficemojo.com" in d,
        "rottentomatoes.com" in d,
        "filmratings.com" in d,
        "bbfc.co.uk" in d,  # British Board of Film Classification
        "mpaa.org" in d,  # Motion Picture Association
        "oscars.org" in d,
        "variety.com" in d,
        "hollywoodreporter.com" in d,
        "metacritic.com" in d,
        "allmovie.com" in d,
        "letterboxd.com" in d,
        "mubi.com" in d,
        "criterion.com" in d,
        "bfi.org.uk" in d,  # British Film Institute
        "afi.com" in d,  # American Film Institute
        "imdb.com" in d,
        "tmdb.org" in d,

        # Streaming Services (Official Sources)
        "netflix.com" in d,
        "primevideo.com" in d,
        "amazon.com" in d,
        "hulu.com" in d,
        "disneyplus.com" in d,
        "hbomax.com" in d,
        "max.com" in d,
        "apple.com" in d,
        "paramountplus.com" in d,
        "peacocktv.com" in d,
        "crunchyroll.com" in d,
        "funimation.com" in d,

        # Studios & Production Companies
        "warnerbros.com" in d,
        "universalpictures.com" in d,
        "paramountpictures.com" in d,
        "sonypictures.com" in d,
        "foxmovies.com" in d,
        "mgm.com" in d,
        "lionsgate.com" in d,
        "a24films.com" in d,
        "focus-features.com" in d,
        "searchlightpictures.com" in d,

        # News & Entertainment Media
        "deadline.com" in d,
        "thewrap.com" in d,
        "indiewire.com" in d,
        "screendaily.com" in d,
        "empireonline.com" in d,
        "entertainment.com" in d,
        "ew.com" in d,  # Entertainment Weekly

        # Book & Literature
        "goodreads.com" in d,
        "amazon.com" in d,
        "barnesandnoble.com" in d,
        "publishersweekly.com" in d,
        "bookpage.com" in d,
        "kirkusreviews.com" in d,
        "librarything.com" in d,
        "worldcat.org" in d,
        "loc.gov" in d,  # Library of Congress
        "isbn.org" in d,
        "publishersmarketplace.com" in d,

        # Music & Audio
        "spotify.com" in d,
        "music.apple.com" in d,
        "youtube.com" in d,
        "allmusic.com" in d,
        "discogs.com" in d,
        "musicbrainz.org" in d,
        "last.fm" in d,
        "billboard.com" in d,
        "rollingstone.com" in d,
        "pitchfork.com" in d,
        "grammy.com" in d,

        # Gaming
        "ign.com" in d,
        "gamespot.com" in d,
        "polygon.com" in d,
        "kotaku.com" in d,
        "eurogamer.net" in d,
        "pcgamer.com" in d,
        "nintendo.com" in d,
        "playstation.com" in d,
        "xbox.com" in d,
        "steam.com" in d,
        "epicgames.com" in d,
        "metacritic.com" in d,

        # Technology & Software
        "github.com" in d,
        "stackoverflow.com" in d,
        "microsoft.com" in d,
        "apple.com" in d,
        "google.com" in d,
        "mozilla.org" in d,
        "w3.org" in d,
        "ietf.org" in d,
        "ieee.org" in d,

        # Academic & Research
        "arxiv.org" in d,
        "scholar.google.com" in d,
        "researchgate.net" in d,
        "academia.edu" in d,
        "jstor.org" in d,
        "sciencedirect.com" in d,
        "springer.com" in d,
        "nature.com" in d,
        "science.org" in d,
        "pubmed.gov" in d,
        "ncbi.nlm.nih.gov" in d,

        # News & Information
        "reuters.com" in d,
        "apnews.com" in d,
        "bbc.com" in d,
        "bbc.co.uk" in d,
        "cnn.com" in d,
        "nytimes.com" in d,
        "theguardian.com" in d,
        "washingtonpost.com" in d,
        "wsj.com" in d,
        "bloomberg.com" in d,
        "forbes.com" in d,
        "economist.com" in d,

        # Sports
        "espn.com" in d,
        "nfl.com" in d,
        "nba.com" in d,
        "mlb.com" in d,
        "nhl.com" in d,
        "fifa.com" in d,
        "uefa.com" in d,
        "olympic.org" in d,
        "sports-reference.com" in d,

        # Regional Entertainment (International)
        # Bollywood/Indian
        "bollywoodhungama.com" in d,
        "filmfare.com" in d,
        "pinkvilla.com" in d,
        "koimoi.com" in d,
        "indiatoday.in" in d,
        "timesofindia.com" in d,

        # Pakistani
        "dawn.com" in d,
        "geo.tv" in d,
        "arydigital.tv" in d,
        "hum.tv" in d,

        # Korean
        "koreaboo.com" in d,
        "soompi.com" in d,
        "allkpop.com" in d,
        "hancinema.net" in d,

        # Japanese
        "myanimelist.net" in d,
        "anidb.net" in d,
        "animenewsnetwork.com" in d,
        "crunchyroll.com" in d,

        # Chinese
        "douban.com" in d,
        "mtime.com" in d,

        # Latin American
        "univision.com" in d,
        "telemundo.com" in d,

        # Other trusted patterns
        "wiki" in d,  # Catches all wiki sites
        ".mil" in d,  # Military domains
    ])

def detect_regex_from_examples(series: pd.Series) -> str:
    vals = [str(v) for v in series.dropna().astype(str).tolist()][:200]
    if not vals:
        return r".+"
    if all(re.fullmatch(r"\d+", v or "") for v in vals):
        lengths = {len(v) for v in vals}
        if len(lengths) == 1:
            L = list(lengths)[0]
            return rf"\d{{{L}}}"
        return r"\d+"
    if all(re.fullmatch(r"[A-Z0-9\-]+", v) for v in vals):
        lens = {len(v) for v in vals}
        if len(lens) == 1:
            L = list(lens)[0]
            return rf"[A-Z0-9\-]{{{L}}}"
        return r"[A-Z0-9\-]+"
    if all(re.fullmatch(r"[A-Za-z0-9_\-\.]+@[A-Za-z0-9\.\-]+\.[A-Za-z]{2,}", v) for v in vals):
        return r"[A-Za-z0-9_\-\.]+@[A-Za-z0-9\.\-]+\.[A-Za-z]{2,}"
    if all(re.fullmatch(r"https?://\S+", v) for v in vals):
        return r"https?://\S+"
    if all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", v) for v in vals):
        return r"\d{4}-\d{2}-\d{2}"
    return r".+"

def flexible_regex_match(value, expected_regex):
    if not value or not str(value).strip():
        return False
    v = str(value).strip()
    if not expected_regex or expected_regex in [".+", r".+", ""]:
        return True
    if re.fullmatch(expected_regex, v):
        return True
    if r"\d" in expected_regex:
        nums = re.findall(r'\d+', v)
        if nums:
            return True
    if "[A-Z0-9" in expected_regex or "[A-Za-z0-9" in expected_regex:
        if re.search(r'[A-Za-z0-9]', v):
            return True
    if len(v) >= 3 and not v.isspace():
        return True
    return False

def sanity_check_column_text(value: str, expected_regex: str) -> str:
    if value is None:
        return "none"
    v = str(value).strip()
    if v == "":
        return "empty"
    if expected_regex and expected_regex not in [".+", r".+", ""]:
        if not flexible_regex_match(v, expected_regex):
            return "regex_mismatch_warning"
    if len(v) > 512:
        return "too_long"
    return "ok"

def compute_confidence(model_conf: float, source_hint: str, recall_hits: int, recall_used: int):
    base = 0.6
    if domain_authoritative(source_hint):
        base = 0.9
    try:
        observed = float(model_conf)
    except:
        observed = 0.0
    recall_factor = 0.0
    if recall_hits > 0:
        recall_factor = min(0.1, (recall_used / max(1, recall_hits)) * 0.1)
    final_conf = max(0.0, min(1.0, 0.4 * observed + 0.5 * base + recall_factor))
    return final_conf

def clean_dynamic(value, column_series: pd.Series):
    try:
        if pd.api.types.is_numeric_dtype(column_series):
            if all(str(x).strip().isdigit() or (isinstance(x, (int, float)) and float(x).is_integer()) for x in column_series.dropna()):
                return int(float(str(value).strip()))
            return float(str(value).strip().replace(",", ""))
        if pd.api.types.is_string_dtype(column_series) or column_series.dtype == object:
            return str(value).strip().strip("'").strip('"')
        return value
    except:
        return value

STOP = set("""
a an and the or of to in for with on by as at from this that these those is are was were be been being it its his her their them they we you i he she who which what when where why how into over under about after before during between while through across per each other than up out off so such not no yes true false film movie feature motion picture tv series season episode cast crew review rating score synopsis plot story storyline overview release date year runtime certificate certification genre distributor studio production company netflix amazon prime hbo hulu disney apple fox warner universal paramount sony columbia pictures
""".split())

def normalize_text(s):
    s = re.sub(r"[\u2018\u2019\u201C\u201D]", '"', str(s))
    s = re.sub(r"[^\w\s\-:,.()'/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def keyphrases_from_synopsis(s, top_k=6):
    s = normalize_text(s).lower()
    tokens = [t for t in re.split(r"[^\w']+", s) if t and t not in STOP and not t.isdigit()]
    counts = Counter(tokens)
    ranked = [w for w, c in counts.most_common(50) if len(w) >= 5]
    phrases = []
    for n in [4,3,2]:
        for i in range(len(tokens)-n+1):
            seg = tokens[i:i+n]
            if all(t not in STOP for t in seg):
                phrases.append(" ".join(seg))
    uniq = []
    for p in phrases:
        if p not in uniq and len(uniq) < top_k:
            uniq.append(p)
    keep = [p for p in uniq] + ranked[:max(0, top_k - len(uniq))]
    return [p for p in keep if p]

def truncate_text(s, max_chars):
    s = s or ""
    if len(s) <= max_chars:
        return s
    head = max_chars // 2
    tail = max_chars - head
    return s[:head] + "\n...\n" + s[-tail:]

def profile_column(series: pd.Series, k=5):
    non_null = series.dropna()
    dtype = str(series.dtype)
    regex = detect_regex_from_examples(series)
    samples = list(map(lambda x: str(x)[:120], non_null.astype(str).head(k)))
    lengths = list(non_null.astype(str).map(len))
    lens = {"min": int(min(lengths)) if lengths else 0, "max": int(max(lengths)) if lengths else 0, "median": int(pd.Series(lengths).median()) if lengths else 0}
    return {"dtype": dtype, "regex": regex, "len_stats": lens, "samples": samples}

def dataframe_signature(df: pd.DataFrame, k=5, max_cols=30):
    cols = list(df.columns)[:max_cols]
    sig = {}
    for c in cols:
        sig[c] = profile_column(df[c], k=k)
    return sig

def title_variants(t):
    v = [t]
    t2 = t.replace(" Part 2", " Part Two").replace(" part 2", " Part Two")
    v.append(t2)
    t3 = t2.replace("Part Two", "Part II")
    v.append(t3)
    v.append(t.replace(":", " : "))
    v.append(t.replace(" - ", ": "))
    return list(dict.fromkeys([x.strip() for x in v if x.strip()]))

def schema_agent_prompt(sig):
    return (
        "You are a data schema analyst. Infer human-meaningful column names from observed values. "
        "Return strict JSON with key 'columns' as a list of objects with keys: original, proposed, human_label. "
        "Rules: proposed must be concise snake_case without special characters; human_label is a short natural phrase users would say. "
        "If original is already good, keep proposed equal to a normalized variant of original."
        "\nSchema:\n" + json.dumps(sig, ensure_ascii=False)
    )

def propose_column_names(df: pd.DataFrame):
    sig = dataframe_signature(df, k=6, max_cols=min(60, len(df.columns)))
    out = ask_claude(schema_agent_prompt(sig), max_tokens=1200, temp=0.0) or "{}"
    try:
        j = json.loads(out)
        cols = j.get("columns", [])
    except:
        cols = []
    forward = {}
    reverse = {}
    labels = {}
    seen = set()
    for c in df.columns:
        forward[c] = re.sub(r"[^a-z0-9_]+", "_", c.lower().replace(" ", "_")).strip("_") or c
    for it in cols:
        o = str(it.get("original", "")).strip()
        p = str(it.get("proposed", "")).strip()
        h = str(it.get("human_label", "")).strip()
        if not o or o not in df.columns:
            continue
        p = re.sub(r"[^a-z0-9_]+", "_", p.lower()).strip("_") or forward[o]
        if p in seen and p != forward[o]:
            p = f"{p}_{len(seen)}"
        seen.add(p)
        forward[o] = p
        reverse[p] = o
        labels[p] = h or o.replace("_", " ")
    for p, o in list(reverse.items()):
        labels.setdefault(p, o.replace("_", " "))
    return {"forward": forward, "reverse": reverse or {v: k for k, v in forward.items()}, "labels": labels}

def apply_temporary_schema(df: pd.DataFrame):
    m = propose_column_names(df)
    SCHEMA_CACHE["forward"] = m["forward"]
    SCHEMA_CACHE["reverse"] = m["reverse"]
    SCHEMA_CACHE["labels"] = m["labels"]
    renamed = df.rename(columns=SCHEMA_CACHE["forward"]) if SCHEMA_CACHE["forward"] else df.copy()
    jlog("schema_agent_applied", forward=SCHEMA_CACHE["forward"], reverse=SCHEMA_CACHE["reverse"]) 
    return renamed

def restore_original_schema(df: pd.DataFrame):
    rev = SCHEMA_CACHE.get("reverse", {})
    return df.rename(columns=rev) if rev else df

def human_label_for(field):
    lbls = SCHEMA_CACHE.get("labels", {})
    return lbls.get(field, field.replace("_", " "))

def row_context(row: pd.Series):
    ctx = {}
    for col in row.index:
        v = row.get(col, None)
        if pd.isna(v):
            continue
        s = str(v).strip()
        if not s:
            continue
        ctx[col] = s
    return ctx

def disambiguation_from_row(ctx: dict):
    year = ""
    rating = ""
    synopsis = ""
    title_hint = ""
    for k, v in ctx.items():
        lk = k.lower()
        if "year" in lk or "release" in lk:
            m = re.search(r"\d{4}", v)
            if m:
                year = m.group(0)
        if "rating" in lk or "score" in lk:
            rating = v
        if any(x in lk for x in ["synopsis", "plot", "overview", "story", "description"]):
            synopsis = v
        if any(x in lk for x in ["name", "title", "movie"]):
            title_hint = v
    variants = title_variants(title_hint) if title_hint else []
    phrases = keyphrases_from_synopsis(synopsis, top_k=6) if synopsis else []
    return {"year": year, "rating": rating, "title_hint": title_hint, "title_variants": variants, "synopsis_keyphrases": phrases}

def build_context_pack(row: pd.Series, df: pd.DataFrame, target_field: str, max_ctx=PROMPT_CONTEXT_CHARS):
    ctx_dict = row_context(row.drop(labels=[target_field]) if target_field in row.index else row)
    disamb = disambiguation_from_row(ctx_dict)
    sig = dataframe_signature(df)
    target_profile = sig.get(target_field, {"regex": ".+", "dtype": "object", "len_stats": {}, "samples": []})
    human_label = human_label_for(target_field)

    is_title_col = bool(re.search(r"title", target_field, re.I))
    has_episode_context = any(
        re.search(r"(episode|season|ep\\b|s\\d+e\\d+)", c, re.I) or
        (df[c].astype(str).str.contains(r"\\bS\\d+E\\d+\\b", case=False, na=False).any() if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object else False)
        for c in df.columns
    )

    if is_title_col and has_episode_context:
        human_label = "TV series title"
        natural_goal = "Find the TV SERIES (parent show) title for the entity identified by the row facts. If both a series and an episode title appear, return the SERIES title only."
    else:
        natural_goal = f"Find {human_label} for the entity identified by the row facts. Never use the raw column label \"{target_field}\" in queries; use natural phrasing."
    pack = {
        "row_facts": ctx_dict,
        "target_field": target_field,
        "semantic_target": {"canonical": target_field, "human_label": human_label_for(target_field), "natural_goal": natural_goal},
        "target_profile": target_profile,
        "disambiguation": disamb,
        "schema_overview": {k: {"regex": v["regex"], "dtype": v["dtype"]} for k, v in sig.items()}
    }
    text = json.dumps(pack, ensure_ascii=False)
    return truncate_text(text, max_ctx), pack

def build_understanding_prompt(field_name, context_text):
#     return f"""
# You are a schema-aware, retrieval-first inference engine. Treat column names as machine labels only. Infer the human meaning.

# ContextJSON:
# {context_text}

# Rules:
# - Use ContextJSON.semantic_target.human_label and natural_goal to define what to search.
# - Never include the raw column label in queries.
# - Prefer official or authoritative sources.

# Output only JSON with keys:
# - pattern_summary
# - best_query
# - alt_queries
# - disambiguation_factors
# """
    return f"""
    You are a schema-aware, retrieval-first inference engine. Treat raw column labels as machine labels only. Infer human meaning from context.

    ContextJSON:
    {context_text}

    Rules:
    - Use ContextJSON.semantic_target.human_label and natural_goal to define what to search.
    - If the context suggests TV data (season/episode present), ALWAYS prefer the SERIES (parent show) title over any episode/chapter title.
    - Never include the raw column label in queries.
    - Prefer official or authoritative sources.

    Output only JSON with keys:
    - pattern_summary
    - best_query
    - alt_queries
    - disambiguation_factors
    """

def build_query_prompt(field_name, context_text):
    return f"""
Craft one elite web search query to satisfy ContextJSON.semantic_target.natural_goal with ≥0.95 confidence.

ContextJSON:
{context_text}

Return only the query string.
"""

def build_extract_prompt(field_name, context_text, snippets_text):
    return f"""
You are an evidence-bound extractor. Return ONLY valid JSON, nothing else.

ContextJSON:
{context_text}

Snippets:
{truncate_text(snippets_text, PROMPT_SNIPPETS_CHARS)}

Task:
1) Use ContextJSON.semantic_target.human_label to determine what value to extract, not the raw column label.
2) Ensure the value matches ContextJSON.target_profile.regex if applicable.
3) Return ONLY a JSON object with these keys: "value", "evidence_quote", "source_hint", "confidence" (0..1).
4) If insufficient data, return {{"value": "UNKNOWN", "evidence_quote": "", "source_hint": "", "confidence": 0}}
5) Do NOT include any explanatory text before or after the JSON.
"""

def build_verifier_prompt(field_name, candidate, context_text, snippets_text):
    return f"""
You are a strict verifier. Your ONLY job is to output YES or NO.

Candidate value to verify: {candidate}

ContextJSON:
{context_text}

Snippets:
{truncate_text(snippets_text, PROMPT_SNIPPETS_CHARS)}

Rules:
1. Output ONLY the word "YES" or "NO" - nothing else
2. Return YES if the candidate appears explicitly in Snippets or ContextJSON.row_facts AND matches ContextJSON.target_profile.regex
3. Return NO otherwise
4. Do NOT repeat the candidate value
5. Do NOT add explanations

Output (YES or NO):
"""

def score_query(q):
    ops = sum(op in q for op in ['"', 'intitle:', 'inurl:', 'OR', 'AND'])
    length = len(q)
    quoted = q.count('"') // 2
    diversity = len(set(re.split(r"\W+", q.lower()))) / max(1, len(q.split()))
    spec = min(1.0, (quoted + ops) / 6.0)
    len_score = min(1.0, length / 140.0)
    return round(0.45*spec + 0.35*len_score + 0.20*diversity, 3)

def make_open_web_variants(seed_query, context_pack, max_q=MAX_QUERIES):
    dis = context_pack.get("disambiguation", {})
    year = dis.get("year", "")
    rating = dis.get("rating", "")
    variants = dis.get("title_variants", []) or []
    phrases = dis.get("synopsis_keyphrases", []) or []
    base = seed_query.strip()
    extras = []
    for tv in variants[:3]:
        extras.append(f'{base} "{tv}"')
    if phrases:
        extras += [f'{base} "{phrases[0]}"', f'{base} "{phrases[0]}" "{year}"' if year else base]
    if year:
        extras += [f'{base} "{year}"', f'intitle:"{year}" {base}', f'inurl:"{year}" {base}']
    if rating:
        extras += [f'{base} rating "{rating}"', f'{base} rated "{rating}"']
    synonyms = ["movie", "film", "\"motion picture\"", "\"feature film\""]
    extras += [f'{base} {s}' for s in synonyms]
    if INCLUDE_DOMAIN_QUERIES:
        extras += [f'{base} site:imdb.com', f'{base} site:wikipedia.org', f'{base} site:themoviedb.org', f'{base} site:rottentomatoes.com', f'{base} site:boxofficemojo.com']
    uniq = []
    for e in [base] + [x for x in extras if x]:
        e = re.sub(r"\s+", " ", e).strip()
        if e and e not in uniq:
            uniq.append(e)
    scored = [{"query": q, "score": score_query(q)} for q in uniq]
    scored.sort(key=lambda x: x["score"], reverse=True)
    jlog("query_candidates_scored", top=[scored[i] for i in range(min(max_q, len(scored)))])
    return [x["query"] for x in scored[:max_q]]

def assemble_snippets(snips, fetched_map=None):
    rows = []
    for s in snips:
        extra = ""
        if fetched_map and s.get("link") in fetched_map and fetched_map[s.get("link")]:
            extra = fetched_map[s.get("link")][:400]
        rows.append(f"Title: {s.get('title','')}\nDomain: {s.get('displayLink','')}\nURL: {s.get('link','')}\nSnippet: {s.get('snippet','')}\nPage: {extra}")
    return "\n\n".join(rows)

def intelligent_evidence_search(candidate, snippets_text, evidence_quote):
    if not candidate or not snippets_text:
        return 0
    candidate_normalized = candidate.lower().strip()

    # PASS 1: Exact substring match (handles short values like "V, L, S", "TV-14")
    if evidence_quote and candidate_normalized in evidence_quote.lower():
        snippet_blocks = snippets_text.split("\n\n")
        count = sum(1 for block in snippet_blocks if candidate_normalized in block.lower())
        return max(1, count)

    # PASS 2: Direct substring search in snippets (for short/formatted values)
    snippet_blocks = snippets_text.split("\n\n")
    exact_match_count = sum(1 for block in snippet_blocks if candidate_normalized in block.lower())
    if exact_match_count > 0:
        return exact_match_count

    # PASS 3: Token-based matching (for longer values)
    # Use shorter tokens (2+ chars instead of 4+) to handle abbreviations
    candidate_tokens = set(re.findall(r'\w{2,}', candidate_normalized))
    if not candidate_tokens:
        # If no tokens at all, try single characters (for codes like "V, L, S")
        candidate_chars = set(re.findall(r'[a-z0-9]', candidate_normalized))
        if not candidate_chars:
            return 0
        # Check if individual chars appear in blocks
        evidence_count = 0
        for block in snippet_blocks:
            block_normalized = block.lower()
            block_chars = set(re.findall(r'[a-z0-9]', block_normalized))
            # If at least 50% of candidate chars appear, count it
            overlap = len(candidate_chars & block_chars)
            if overlap / len(candidate_chars) >= 0.5:
                evidence_count += 1
        return evidence_count

    # Token-based search with lower threshold (50% instead of 70%)
    evidence_count = 0
    for block in snippet_blocks:
        block_normalized = block.lower()
        block_tokens = set(re.findall(r'\w{2,}', block_normalized))
        overlap = len(candidate_tokens & block_tokens)
        if overlap / len(candidate_tokens) >= 0.5:  # Reduced from 0.7 to 0.5
            evidence_count += 1
    return evidence_count

def enrich_field(row, field_name, df, min_confidence=MIN_CONFIDENCE, max_sites=MAX_SNIPPETS_TOTAL):
    global CONFIDENCE_RECORDS
    series_id = row.get("Series ID") or row.get("series_id") or row.name  # adjust column name

    jlog("enrich_start", field=field_name)
    context_text, pack = build_context_pack(row, df, field_name)
    natural_goal = pack.get("semantic_target", {}).get("natural_goal", "")
    target_regex = pack.get("target_profile", {}).get("regex") or ".+"
    jlog("semantic_objective", field=field_name, canonical=pack.get("semantic_target", {}).get("canonical"), human_label=pack.get("semantic_target", {}).get("human_label"), natural_goal=natural_goal)
    sanity = sanity_check_column_text(row.get(field_name), target_regex)
    if sanity != "ok":
        jlog("column_text_anomaly", field=field_name, state=sanity)
    understanding_out = ask_claude(build_understanding_prompt(field_name, context_text), max_tokens=900, temp=0.0) or "{}"
    try:
        understanding = json.loads(understanding_out)
    except:
        understanding = {}
    pattern_summary = understanding.get("pattern_summary", "")
    best_query_hint = understanding.get("best_query", "")
    alt_queries = understanding.get("alt_queries", []) or []
    disamb = understanding.get("disambiguation_factors", []) or []
    jlog("understanding", field=field_name, pattern_summary=pattern_summary, best_query=best_query_hint, alt_queries=alt_queries, disambiguation=disamb)
    if not best_query_hint:
        best_query_hint = ask_claude(build_query_prompt(field_name, context_text), max_tokens=300, temp=0.0)
    queries = make_open_web_variants(best_query_hint or "", pack)
    for q in alt_queries:
        if q and q not in queries and len(queries) < MAX_QUERIES:
            queries.append(q)
    jlog("queries_finalized", field=field_name, total=len(queries), queries=queries)
    all_snips = google_search_multi(queries, per_query=PER_QUERY_RESULTS, max_total=max_sites)
    recall_hits = len(all_snips)
    fetched_map = {}
    if ENABLE_FETCH_HTML and all_snips:
        subset = all_snips[:min(MAX_FETCH_PAGES, len(all_snips))]
        with ThreadPoolExecutor(max_workers=FETCH_MAX_CONCURRENCY) as ex:
            futs = {ex.submit(fetch_url_text, s.get("link")): s.get("link") for s in subset if s.get("link")}
            for fut in as_completed(futs):
                url = futs[fut]
                text = fut.result()
                fetched_map[url] = text
                jlog("fetch_page_done", url=url, has_text=bool(text))
    snippets_text = assemble_snippets(all_snips, fetched_map=fetched_map if ENABLE_FETCH_HTML else None)
    if all_snips:
        extract_out = ask_claude(build_extract_prompt(field_name, context_text, snippets_text), max_tokens=900, temp=0.0) or "{}"
        try:
            data = json.loads(extract_out)
        except:
            # Try to extract JSON from response (LLM sometimes adds extra text)
            json_match = re.search(r'\{.*\}', extract_out, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    jlog("json_extraction_recovered", field=field_name)
                except:
                    jlog("json_parse_failed", field=field_name, raw_output=extract_out[:500])
                    data = {}
            else:
                jlog("json_parse_failed", field=field_name, raw_output=extract_out[:500])
                data = {}
        candidate = str(data.get("value", "")).strip()
        evidence_quote = str(data.get("evidence_quote", "")).strip()
        source_hint = str(data.get("source_hint", "")).strip()

        # Log if we got an empty candidate
        if not candidate or candidate.upper() == "UNKNOWN":
            jlog("empty_candidate_detected", field=field_name, data_keys=list(data.keys()), raw_output_preview=extract_out[:300])
        if not source_hint and all_snips:
            try:
                parsed = urlparse(all_snips[0].get("link", ""))
                source_hint = parsed.hostname or all_snips[0].get("displayLink", "")
            except:
                source_hint = all_snips[0].get("displayLink", "")
        try:
            model_conf = float(data.get("confidence", 0))
        except:
            model_conf = 0.0
        recall_used = intelligent_evidence_search(candidate, snippets_text, evidence_quote)
        computed_conf = compute_confidence(model_conf, source_hint, recall_hits, recall_used)
        jlog("candidate_proposed", field=field_name, value=candidate, model_conf=round(model_conf, 3), computed_conf=round(computed_conf, 3), source_hint=source_hint, recall_hits=recall_hits, recall_used=recall_used)
        verifier_out = ask_claude(build_verifier_prompt(field_name, candidate, context_text, snippets_text), max_tokens=120, temp=0.0) or "NO"

        # Robust verdict parsing - handle malformed responses
        verdict_raw = verifier_out.strip().upper()
        if "YES" in verdict_raw:
            verdict = "YES"
        elif "NO" in verdict_raw:
            verdict = "NO"
        else:
            # If neither YES nor NO found, default to NO and log warning
            verdict = "NO"
            jlog("verifier_malformed", field=field_name, raw_response=verifier_out[:200])

        jlog("verifier_result", field=field_name, verdict=verdict, raw=verifier_out[:100] if verdict_raw != verdict else None)

        # Acceptance logic with flexible recall requirements
        passes_confidence = computed_conf >= min_confidence
        passes_regex = flexible_regex_match(candidate, target_regex)
        passes_verdict = verdict == "YES"

        # Accept if ALL conditions meet, with flexible recall:
        # - For high confidence (>=0.85) or authoritative sources, allow recall_used=0
        # - Otherwise require recall_used > 0
        is_authoritative = domain_authoritative(source_hint)
        allow_zero_recall = (computed_conf >= 0.85) or is_authoritative or (model_conf >= 0.9)

        if passes_verdict and passes_confidence and passes_regex and (recall_used > 0 or allow_zero_recall):
            jlog("accept", field=field_name, value=candidate, confidence=round(computed_conf, 3), recall_used=recall_used, zero_recall_accepted=recall_used == 0)
            CONFIDENCE_RECORDS.append({
                "series_id": series_id,
                "field_name": field_name,
                # mark as "accept-true" if zero_recall_accepted is True, else just "accept"
                "status": "accept-true" if zero_recall_flag else "accept",
                "value": candidate,
                "confidence": round(computed_conf, 3)
            })
            return clean_dynamic(candidate, df[field_name])

        # Log detailed rejection reason
        rejection_reasons = []
        if not passes_verdict:
            rejection_reasons.append("verifier_rejected")
        if not passes_confidence:
            rejection_reasons.append(f"low_confidence({round(computed_conf, 3)}<{min_confidence})")
        if not passes_regex:
            rejection_reasons.append(f"regex_mismatch")
        if recall_used == 0 and not allow_zero_recall:
            rejection_reasons.append("zero_recall_not_allowed")

        jlog("reject", field=field_name, reason=",".join(rejection_reasons), regex=target_regex, confidence=round(computed_conf, 3), recall_used=recall_used, model_conf=round(model_conf, 3))
        CONFIDENCE_RECORDS.append({
        "series_id": series_id,
        "field_name": field_name,
        "status": "reject",
        "value": candidate,
        "confidence": round(computed_conf, 3)
    })
    fallback_prompt = f"""
You are an extractor operating without web evidence. Be VERY conservative.

ContextJSON:
{context_text}

ONLY output a value if it appears EXPLICITLY in ContextJSON.row_facts. Do NOT infer or guess.
If the exact value for "{pack.get('semantic_target',{}).get('human_label','')}" is NOT in the row_facts, output UNKNOWN.
The value must match ContextJSON.target_profile.regex if applicable.
Output only the value or UNKNOWN.
"""
    val = (ask_claude(fallback_prompt, max_tokens=160, temp=0.0) or "UNKNOWN").strip()
    if val.upper() == "UNKNOWN":
        jlog("fallback_unknown", field=field_name)
        return None
    if not flexible_regex_match(val, target_regex):
        jlog("fallback_reject", field=field_name, value=val, reason="regex_mismatch", regex=target_regex)
        return None

    # Additional validation: check if value seems reasonable
    if len(val) < 2 or val.lower() in ['na', 'n/a', 'none', 'null', 'unknown']:
        jlog("fallback_reject", field=field_name, value=val, reason="invalid_fallback_value")
        return None

    jlog("fallback_accept", field=field_name, value=val)
    CONFIDENCE_RECORDS.append({
            "series_id": series_id,
            "field_name": field_name,
            "status": "fallback_accept",
            "value": val,
            "confidence": 0.50  # in fallback, value itself is confidence
        })
    return clean_dynamic(val, df[field_name])

def report_missing(df):
    report = []
    missing = df[df.isnull().any(axis=1)]
    for idx, row in missing.iterrows():
        for col in row.index[row.isnull()]:
            report.append({"row_index": idx, "missing_column": col, "context": {c: str(v) for c, v in row.items() if pd.notna(v)}})
    return report

def process_item(df, item, min_confidence):
    row_idx, col = item["row_index"], item["missing_column"]
    jlog("row_process_start", row=row_idx, field=col)
    try:
        value = enrich_field(df.loc[row_idx], col, df=df, min_confidence=min_confidence)
        if value is not None and str(value).strip() != "":
            jlog("row_update", row=row_idx, field=col, value=str(value))
            return (row_idx, col, value, None)
        else:
            jlog("row_no_update", row=row_idx, field=col)
            return (row_idx, col, None, None)
    except Exception as e:
        jlog("row_error", row=row_idx, field=col, error=str(e))
        return (row_idx, col, None, str(e))

@app.entrypoint
def invoke(payload: dict) -> dict:
    output_path = "filledmissingdata.csv"
    min_confidence = MIN_CONFIDENCE
    max_passes = 3
    id_column = None
    log_path = "filledmissingdatalogs.json"

    csv_path = payload.get("csv_s3_path")
    output_s3_path = payload.get("output_s3_path", None)
    logs_s3_path = payload.get("logs_s3_path", "s3://enrichment-agent-s3/logs/logs.json")

    if not csv_path:
        raise ValueError("csv_s3_path is missing in payload")

    s3 = boto3.client("s3")

    # ---------- Read input CSV ----------
    if csv_path.startswith("s3://"):
        bucket, key = csv_path.replace("s3://", "").split("/", 1)
        tmp_path = "/tmp/input.csv"
        s3.download_file(bucket, key, tmp_path)
        df = pd.read_csv(tmp_path)
    else:
        df = pd.read_csv(csv_path)

    # ---------- Read fixed knowledge base from S3 ----------
    kb_s3_path = "s3://enrichment-agent-s3/knowledge_base/media_series.csv"
    kb_bucket, kb_key = kb_s3_path.replace("s3://", "").split("/", 1)
    kb_tmp_path = "/tmp/knowledge_base.csv"
    s3.download_file(kb_bucket, kb_key, kb_tmp_path)
    knowledge_base_df = pd.read_csv(kb_tmp_path)

    # ---------- Union input CSV with knowledge base ----------
    df = pd.concat([df, knowledge_base_df], ignore_index=True)
    df = df.drop_duplicates(subset=["series_id"], keep="first")

    jlog(
        "pipeline_start",
        rows=len(df),
        source=csv_path,
        min_confidence=min_confidence,
        max_workers=MAX_WORKERS,
        open_web=True,
        providers={
            "google_cse": bool(GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID),
            "bing": bool(BING_API_KEY),
            "serpapi_google": bool(SERPAPI_KEY)
        },
        fetch_html=ENABLE_FETCH_HTML
    )

    df_sem = apply_temporary_schema(df)

    # ---------- Enrichment Passes ----------
    for i in range(max_passes):
        jlog("pass_start", pass_index=i + 1)
        rep = report_missing(df_sem)
        jlog("missing_report", count=len(rep))
        if not rep:
            jlog("no_missing_left")
            break

        for start in range(0, len(rep), PASS_BATCH_SIZE):
            batch = rep[start:start + PASS_BATCH_SIZE]
            jlog("batch_start", pass_index=i + 1, batch_start=start, batch_size=len(batch))
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futs = [ex.submit(process_item, df_sem, item, min_confidence) for item in batch]
                for fut in as_completed(futs):
                    row_idx, col, value, err = fut.result()
                    if value is not None and err is None:
                        df_sem.at[row_idx, col] = value
            jlog("batch_end", pass_index=i + 1, batch_start=start, batch_size=len(batch))
        jlog("pass_end", pass_index=i + 1)

    global CONFIDENCE_RECORDS

    # Convert shared list to DataFrame
    conf_df = pd.DataFrame(CONFIDENCE_RECORDS)
    conf_df = conf_df[conf_df["status"].str.lower() != "reject"]


    # Print it
    print("=== Enrichment Confidence Table ===")
    print(conf_df)

    df_final = restore_original_schema(df_sem)

    # ---------- Filter out rows originally in knowledge base ----------
    df_final["series_id"] = df_final["series_id"].astype(str)
    knowledge_base_df["series_id"] = knowledge_base_df["series_id"].astype(str)
    df_final = df_final[~df_final["series_id"].isin(knowledge_base_df["series_id"])].reset_index(drop=True)

    # ---------- Read original CSV for comparison ----------
    if csv_path.startswith("s3://"):
        bucket, key = csv_path.replace("s3://", "").split("/", 1)
        tmp_path = "/tmp/input.csv"
        s3.download_file(bucket, key, tmp_path)
        original_df = pd.read_csv(tmp_path)
    else:
        original_df = pd.read_csv(csv_path)

    original_df["series_id"] = original_df["series_id"].astype(str)
    original_df = original_df[original_df["series_id"].isin(df_final["series_id"])].reset_index(drop=True)
    
    # ---------- Identify common columns ----------
    common_cols = [col for col in df_final.columns if col in original_df.columns]

    # ---------- Missing value flags ----------
    for col in common_cols:
        missing_col = f"{col}_missing_value"
        df_final[missing_col] = (
            (original_df[col].isna() | (original_df[col].astype(str).str.strip() == ""))
            & (~df_final[col].isna()) & (df_final[col].astype(str).str.strip() != "")
        ).astype(int)

    # Add confidence column ONLY for fields that were missing/enriched
    for col in common_cols:
        conf_col = f"{col}_conf"
        missing_col = f"{col}_missing_value"

        df_final[conf_col] = 0.0  # initialize

        for idx, row in df_final.iterrows():
            # If this field was missing and now filled
            if missing_col in df_final.columns and df_final.at[idx, missing_col] == 1:
                sid = df_final.at[idx, "series_id"]
                
                # Find matching record in conf_df
                match = conf_df[
                    (conf_df["series_id"] == sid) &
                    (conf_df["field_name"] == col)
                ]

                if not match.empty:
                    df_final.at[idx, conf_col] = match["confidence"].values[-1]  # latest/confident one
                else:
                    df_final.at[idx, conf_col] = 0.5  # fallback confidence
            else:
                df_final.at[idx, conf_col] = 0.0
    # Add manual review column based on conf_df status
    for col in common_cols:
        review_col = f"{col}_manual_review"
        df_final[review_col] = 0  # initialize with 0

        for idx, row in df_final.iterrows():
            sid = df_final.at[idx, "series_id"]

            # Find matching record in conf_df
            match = conf_df[
                (conf_df["series_id"] == sid) &
                (conf_df["field_name"] == col)
            ]

            if not match.empty:
                latest_status = match["status"].values[-1]

                # If status is "accept-true" or "fallback_accept", mark for review
                if latest_status in ["accept-true", "fallback_accept"]:
                    df_final.at[idx, review_col] = 1
                else:
                    df_final.at[idx, review_col] = 0
            else:
                df_final.at[idx, review_col] = 0

    # ---------- Save final CSV ----------
    df_final.to_csv(output_path, index=False)
    jlog("pipeline_saved", path=output_path)

    # ---------- Save logs ----------
    write_logs_json(log_path)
    jlog("pipeline_logs_written", path=log_path)
    jlog("pipeline_end", rows=len(df_final))

    # ---------- Upload CSV to S3 ----------
    if output_s3_path and output_s3_path.startswith("s3://"):
        bucket, key = output_s3_path.replace("s3://", "").split("/", 1)
        s3.upload_file(output_path, bucket, key)
        jlog("pipeline_uploaded_s3", path=output_s3_path)

    # ---------- Upload logs to S3 ----------
    if logs_s3_path and logs_s3_path.startswith("s3://"):
        bucket, key = logs_s3_path.replace("s3://", "").split("/", 1)
        s3.upload_file(log_path, bucket, key)
        jlog("pipeline_logs_uploaded_s3", path=logs_s3_path)

    return df_final

# ---------- MAIN RUN ----------
if __name__ == "__main__":
    app.run()

