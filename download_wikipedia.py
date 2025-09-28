# wikipedia_abstracts_10k.py
# Parallel fetch of random Wikipedia abstracts (random languages),
# truncated to ≤256 LLM tokens at punctuation boundaries, saved to one JSON.

import asyncio, aiohttp, random, json, re, time
import tiktoken

# ---- Config ----
N_SAMPLES      = 10_000
CONCURRENCY    = 32          # increase/decrease if you get many 429s
MAX_RETRIES    = 6
TIMEOUT_SECS   = 12
OUTFILE        = "wikipedia_abstracts_10k.json"
SEED           = 42

# Broad set of language codes (deduped, includes large projects)
WIKI_LANGS = list(dict.fromkeys("""
en es de fr it pt ru ja zh ar nl pl sv uk vi cs fi tr ko fa hu he id ro no da el bg sr sk ca eo ms et hr lt sl th bn hi ur ta te ml mr gu kn pa az kk be lv eu gl cy is ga af mk hy ka bs sq mn ne si km la yo sw am zu xh my lo tg uz tt ky tl ceb war
""".split()))

# Sentence/punctuation boundary (supports many scripts): split AFTER these marks,
# allowing trailing quotes/brackets.
_SENT_END = r"[\.!\?;:…।！？。؛]"
SENT_SPLIT_RE = re.compile(rf'(?<= {_SENT_END} )["»”’\)\]\}}]*\s+'.replace(" ", ""), re.UNICODE)
WS_RE = re.compile(r"\s+", re.UNICODE)

ENC = tiktoken.get_encoding("cl100k_base")


def normalize(text: str) -> str:
    return WS_RE.sub(" ", (text or "").strip())


def truncate_to_256_tokens_at_punct(text: str, max_tokens: int = 256) -> tuple[str, int]:
    """
    Build output by concatenating sentences until adding the next one would exceed max_tokens.
    If the very first sentence alone exceeds the limit, we return that whole first sentence
    (never a mid-sentence cut).
    Returns (text, token_count).
    """
    text = normalize(text)
    if not text:
        return "", 0

    sentences = SENT_SPLIT_RE.split(text)
    if not sentences:
        # No clear punctuation; keep the whole thing if it's short, else cut at last space before limit.
        toks = ENC.encode(text)
        if len(toks) <= max_tokens:
            return text, len(toks)
        # fallback: walk back to a whitespace boundary
        # (still avoids mid-word cut; only used when no punctuation found)
        decoded = ENC.decode(toks[:max_tokens])
        # trim to last space to avoid "sudden cut"
        last_space = decoded.rfind(" ")
        safe = decoded if last_space < 0 else decoded[:last_space]
        return safe.strip(), len(ENC.encode(safe))

    out, total = [], 0
    for i, s in enumerate(sentences):
        s = normalize(s)
        if not s:
            continue
        n = len(ENC.encode(s))
        if i == 0 and n > max_tokens:
            return s, n  # first sentence too long: keep it whole per rule
        if total + n > max_tokens and out:
            break
        out.append(s)
        total += n
        if total >= max_tokens:
            break
    res = normalize(" ".join(out))
    return res, len(ENC.encode(res))


async def fetch_one(session: aiohttp.ClientSession, lang: str) -> dict | None:
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/random/summary"
    backoff = 0.8
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with session.get(url, timeout=TIMEOUT_SECS) as r:
                if r.status in (429, 503, 502):
                    # backoff with jitter
                    await asyncio.sleep(backoff + random.random() * 0.6)
                    backoff *= 1.7
                    continue
                r.raise_for_status()
                data = await r.json()
                extract = data.get("extract") or ""
                if not extract.strip():
                    raise ValueError("empty extract")
                text, tok_count = truncate_to_256_tokens_at_punct(extract, 256)
                if not text:
                    raise ValueError("empty after truncate")
                return {
                    "lang": lang,
                    "title": data.get("title", ""),
                    "url": (data.get("content_urls", {}) or {}).get("desktop", {}).get("page", ""),
                    "tokens": tok_count,
                    "text": text,
                }
        except Exception:
            # brief backoff before retry
            await asyncio.sleep(backoff + random.random() * 0.5)
            backoff *= 1.6
    return None


async def worker(name: int, session: aiohttp.ClientSession, rng: random.Random,
                 out_list: list, seen: set, lock: asyncio.Lock, target: int):
    while True:
        async with lock:
            if len(out_list) >= target:
                return
        lang = rng.choice(WIKI_LANGS)
        item = await fetch_one(session, lang)
        if not item:
            continue
        key = (item["lang"], item["title"])
        async with lock:
            if key in seen:
                continue
            seen.add(key)
            out_list.append(item)
            n = len(out_list)
            if n % 100 == 0:
                print(f"[progress] {n}/{target}")


async def main():
    rng = random.Random(SEED)
    results: list[dict] = []
    seen: set[tuple[str, str]] = set()
    lock = asyncio.Lock()

    conn = aiohttp.TCPConnector(limit=CONCURRENCY * 2)
    headers = {
        "Accept": "application/json",
        "User-Agent": "abstracts-collector/1.0 (+research; contact@example.com)"
    }

    async with aiohttp.ClientSession(connector=conn, headers=headers) as session:
        tasks = [
            asyncio.create_task(worker(i, session, rng, results, seen, lock, N_SAMPLES))
            for i in range(CONCURRENCY)
        ]
        t0 = time.time()
        await asyncio.gather(*tasks)
        dt = time.time() - t0
        print(f"Collected {len(results)} samples in {dt:.1f}s")

    # Save one JSON (list of dicts)
    with open(OUTFILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Wrote: {OUTFILE}")


if __name__ == "__main__":
    asyncio.run(main())
