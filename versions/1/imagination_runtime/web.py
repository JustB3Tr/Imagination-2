from __future__ import annotations

import datetime
import html
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
from typing import Callable, Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

# User asks for "recent events" / "things to do" — raw DDGS often returns Wikipedia for place names.
_EVENT_INTENT = re.compile(
    r"\b(events?|happening|concerts?|festivals?|things to do|what(?:'s|s| is) on)\b",
    re.I,
)
_TIME_SENSITIVE = re.compile(
    r"\b(recent|latest|current|today|now|this week|this month|upcoming|tonight|weekend)\b",
    re.I,
)
_ENCYCLOPEDIA_HOST = (
    "wikipedia.org",
    "wiktionary.org",
    "britannica.com",
    "fandom.com",
    "wikivoyage.org",
)


def is_events_style_query(query: str) -> bool:
    q = (query or "").lower()
    if _EVENT_INTENT.search(q):
        return True
    return bool(_TIME_SENSITIVE.search(q) and re.search(r"\b(event|events|concert|festival|fair|parade)\b", q))


def refine_search_query(query: str) -> str:
    """Widen/narrow DDGS query so local listings rank better than encyclopedia pages."""
    q = (query or "").strip()
    if not q or not is_events_style_query(q):
        return q
    low = q.lower()
    now = datetime.datetime.now()
    bits = [q]
    if "calendar" not in low and "eventbrite" not in low:
        bits.append(f"upcoming events calendar {now.strftime('%B %Y')}")
    elif str(now.year) not in q:
        bits.append(str(now.year))
    return " ".join(bits)


def _events_listing_tier(r: Dict) -> int:
    """
    Lower is better for event-style queries. Encyclopedias and generic wikis sink to the bottom.
    """
    url = (r.get("url") or "").lower()
    dom = (r.get("domain") or "").lower()
    title = (r.get("title") or "").lower()
    if any(h in url for h in _ENCYCLOPEDIA_HOST):
        return 40
    if any(x in dom for x in ("eventbrite.", "ticketmaster.", "meetup.", "stubhub.")):
        return 0
    if dom.endswith(".gov") or "visitarizona" in dom or dom.startswith("visit") or "tourism" in dom:
        return 5
    if "chamber" in dom or "downtow" in dom or "mainstreet" in dom:
        return 6
    if "event" in dom or "calendar" in dom:
        return 7
    if any(w in title for w in ("event", "calendar", "festival", "concert", "tickets", "register")):
        return 8
    if r.get("trusted"):
        return 15
    return 20


def get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def trusted(url: str, trusted_domains: List[str]) -> bool:
    d = get_domain(url)
    return any(d.endswith(x) for x in trusted_domains)


def web_search(
    query: str,
    *,
    max_results: int,
    max_sources: int,
    trusted_domains: List[str],
    cache: Dict[str, List[Dict]],
) -> List[Dict]:
    original = (query or "").strip()
    search_q = refine_search_query(original)
    key = search_q.lower()
    if key in cache:
        return cache[key]

    events_mode = is_events_style_query(original)
    if events_mode:
        # Need a wide DDGS pool so reranking can fill max_sources; cap avoids runaway API use.
        fetch_cap = min(120, max(max_results * 2, max_sources * 8, 32))
    else:
        fetch_cap = max_results

    results: List[Dict] = []
    seen = set()
    with DDGS() as ddgs:
        for r in ddgs.text(search_q, max_results=fetch_cap):
            url = (r.get("href") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            title = (r.get("title") or "").strip()
            snippet = (r.get("body") or "").strip()
            results.append(
                {
                    "url": url,
                    "title": title or get_domain(url) or "Untitled",
                    "snippet": snippet,
                    "trusted": trusted(url, trusted_domains),
                    "domain": get_domain(url),
                }
            )

    if events_mode:
        results.sort(
            key=lambda x: (
                _events_listing_tier(x),
                not x["trusted"],
                x["domain"],
                x["title"],
            )
        )
    else:
        results.sort(key=lambda x: (not x["trusted"], x["domain"], x["title"]))
    trimmed = results[:max_sources]
    cache[key] = trimmed
    return trimmed


def fetch_page(
    url: str,
    *,
    timeout_s: int,
    max_chars: int,
    cache: Dict[str, str],
) -> str:
    if url in cache:
        return cache[url]

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=timeout_s)
        r.raise_for_status()

        ctype = (r.headers.get("content-type") or "").lower()
        if "text/html" not in ctype and "application/xhtml" not in ctype:
            cache[url] = ""
            return ""

        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(
            ["script", "style", "nav", "footer", "header", "aside", "form", "noscript", "svg"]
        ):
            tag.decompose()

        chunks: List[str] = []
        for el in soup.find_all(["h1", "h2", "h3", "p", "li"]):
            txt = el.get_text(" ", strip=True)
            if txt:
                chunks.append(txt)

        text = " ".join(chunks)
        text = re.sub(r"\s+", " ", text).strip()[:max_chars]
        cache[url] = text
        return text
    except Exception:
        cache[url] = ""
        return ""


def fetch_parallel(
    urls: List[str],
    *,
    timeout_s: int,
    max_chars: int,
    cache: Dict[str, str],
    on_fetched: Optional[Callable[[str, int], None]] = None,
) -> List[str]:
    """
    Fetch URLs in parallel. If on_fetched is set, call it with (domain, index) after each URL finishes
    (completion order may differ from input order).
    """
    if not urls:
        return []

    if on_fetched is None:
        with ThreadPoolExecutor(max_workers=min(4, max(1, len(urls)))) as exe:
            return list(
                exe.map(
                    lambda u: fetch_page(u, timeout_s=timeout_s, max_chars=max_chars, cache=cache),
                    urls,
                )
            )

    texts: List[str] = [""] * len(urls)
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(urls)))) as exe:
        future_to_idx = {
            exe.submit(fetch_page, u, timeout_s=timeout_s, max_chars=max_chars, cache=cache): i
            for i, u in enumerate(urls)
        }
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                texts[idx] = fut.result()
            except Exception:
                texts[idx] = ""
            domain = get_domain(urls[idx])
            on_fetched(domain, idx)
    return texts


def fetch_urls_sequential(
    urls: List[str],
    *,
    timeout_s: int,
    max_chars: int,
    cache: Dict[str, str],
    before_each: Optional[Callable[[str, int], None]] = None,
) -> List[str]:
    """
    Fetch URLs one-by-one; optional before_each(domain, index) runs before each fetch
    so the UI can show 'Reading example.com…' while the request is in flight.
    """
    out: List[str] = []
    for i, url in enumerate(urls):
        if before_each is not None:
            before_each(get_domain(url), i)
        out.append(fetch_page(url, timeout_s=timeout_s, max_chars=max_chars, cache=cache))
    return out


def sources_to_cards(sources: List[Dict]) -> str:
    if not sources:
        return "<div class='sources-empty'>Run a search or ask a current-events question to populate source cards.</div>"

    cards: List[str] = []
    for s in sources:
        trust_badge = "TRUSTED" if s.get("trusted") else "OTHER"
        title = html.escape(s.get("title") or "")
        url = html.escape(s.get("url") or "")
        domain = html.escape(s.get("domain") or "")
        snippet = html.escape((s.get("snippet") or "")[:180])
        idx = html.escape(str(s.get("idx") or ""))

        cards.append(
            f"""
        <a class="source-card" href="{url}" target="_blank" rel="noopener">
          <div class="source-row">
            <div class="source-badge">[{idx}]</div>
            <div class="trust-pill {'trust-on' if s.get('trusted') else 'trust-off'}">{trust_badge}</div>
          </div>
          <div class="source-title">{title}</div>
          <div class="source-domain">{domain}</div>
          <div class="source-snippet">{snippet}</div>
        </a>
        """
        )

    return "<div class='sources-grid'>" + "\n".join(cards) + "</div>"
