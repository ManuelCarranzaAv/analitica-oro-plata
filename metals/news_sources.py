"""Noticias de medios financieros serios y organismos oficiales del mercado de metales.

Google News sólo se usa como buscador: cada búsqueda se limita con ``site:`` a TRUSTED_SOURCES y cada
resultado se acepta sólo si su medio de origen (``<source url>``) es uno de ellos.
No se descargan cuerpos de artículos ni se accede a contenido de suscripción.
"""
import calendar
import hashlib
import html
import re
from urllib.parse import quote_plus, urlparse

import feedparser
import pandas as pd
import requests

from .config import DATA
from .news_filter import LOW_VALUE, TRUSTED_SOURCES, impact_score, matched_metals, trusted_domain
from .storage import now


def trusted_search(query):
    sites = " OR ".join(f"site:{domain}" for domain in TRUSTED_SOURCES)
    return "https://news.google.com/rss/search?q=" + quote_plus(f"{query} ({sites}) when:7d") + "&hl=en-US&gl=US&ceid=US:en"


TARGETED_FEEDS = {
    "Medios confiables — precio del oro": trusted_search('"gold price" OR "gold prices" OR bullion'),
    "Medios confiables — precio de la plata": trusted_search('"silver price" OR "silver prices" OR "silver market"'),
    "Medios confiables — factores del oro y la plata":
        trusted_search('(gold OR silver) (Fed OR inflation OR dollar OR yields OR "central bank" OR ETF OR demand)'),
}


def clean(value):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", value or ""))).strip()


def brief_metadata(title, summary, max_words=25):
    """Conservar un fragmento literal breve que aún mencione el metal.

    Título + extracto nunca exceden max_words. Los puntos suspensivos señalan
    cortes. Se prefieren ventanas del resumen con una mención temática real.
    """
    title_words, summary_words = clean(title).split(), clean(summary).split()
    # Reservar al menos diez palabras al extracto cuando el título es largo.
    title_budget = min(len(title_words), 15) if summary_words else max_words
    short_title = " ".join(title_words[:title_budget])
    if len(title_words) > title_budget:
        short_title += "…"
    remaining = max_words - len(short_title.split())
    target = matched_metals(title, summary)
    best_start, best_score = 0, -1
    for start in range(max(1, len(summary_words) - remaining + 1)):
        excerpt = " ".join(summary_words[start:start + remaining])
        score = len(matched_metals(short_title, excerpt) & target)
        if score > best_score:
            best_start, best_score = start, score
        if score == len(target):
            break
    excerpt = " ".join(summary_words[best_start:best_start + remaining])
    if excerpt and best_start:
        excerpt = "…" + excerpt
    if excerpt and best_start + remaining < len(summary_words):
        excerpt += "…"
    return short_title, excerpt


def parse_feed(body, source, fetched_at, window_days=31, trusted_only=False):
    """Validar fecha, URL, tema y medio del RSS; devolver sólo metadatos públicos.

    Con trusted_only, el medio de origen de cada entrada debe estar en TRUSTED_SOURCES, se descartan los
    titulares de bajo valor y la fuente guardada es 'Medio · dominio' (por ejemplo 'Reuters · reuters.com').
    """
    feed = feedparser.parse(body)
    if not feed.entries:
        raise ValueError("RSS sin entradas legibles; no se sustituye por noticias simuladas.")
    asof = pd.to_datetime(fetched_at, utc=True)
    minimum = asof - pd.Timedelta(days=window_days)
    rows = []
    for entry in feed.entries:
        url = entry.get("link", "")
        parsed_url = urlparse(url)
        if parsed_url.scheme != "https" or not parsed_url.hostname:
            continue
        published_raw = str(entry.get("published", ""))
        date = entry.get("published_parsed") or entry.get("updated_parsed")
        try:
            if published_raw.isdigit():
                published = pd.Timestamp(int(published_raw), unit="s", tz="UTC")
            elif date:
                published = pd.Timestamp(calendar.timegm(date), unit="s", tz="UTC")
            else:
                continue
        except (ValueError, OverflowError):
            continue
        if not minimum <= published <= asof:
            continue
        title, summary = clean(entry.get("title", "")), clean(entry.get("summary", ""))
        if not title or not matched_metals(title, summary):
            continue
        entry_source = source
        if trusted_only:
            origin = urlparse(str((entry.get("source") or {}).get("href", ""))).hostname or ""
            domain = trusted_domain(origin)
            if not domain or LOW_VALUE.search(f"{title} {summary}"):
                continue
            entry_source = f"{TRUSTED_SOURCES[domain]} · {domain}"
        title, summary = brief_metadata(title, summary)
        if not matched_metals(title, summary):
            continue
        rows.append({"id": hashlib.sha256(url.encode()).hexdigest(), "title": title,
                     "summary": summary, "url": url, "source": entry_source,
                     "published_at": published.isoformat(), "fetched_at": asof.isoformat()})
    return rows


def select_balanced(rows, per_metal=5):
    """Unión de N titulares por metal, sin duplicar artículos: primero los que hablan de factores de precio."""
    if not 1 <= per_metal <= 20:
        raise ValueError("El objetivo por metal debe estar entre 1 y 20.")
    selected, counts = {}, {}
    ordered = sorted(rows, key=lambda r: (impact_score(r["title"], r["summary"]) > 0, pd.Timestamp(r["published_at"]), r["id"]),
                     reverse=True)
    for metal in ("gold", "silver"):
        titles, ids = set(), set()
        for row in ordered:
            normalized = re.sub(r"\W+", " ", row["title"].lower()).strip()
            if metal not in matched_metals(row["title"], row["summary"]) or row["id"] in ids or normalized in titles:
                continue
            selected[row["id"]] = row
            ids.add(row["id"])
            titles.add(normalized)
            if len(ids) == per_metal:
                break
        counts[metal] = len(ids)
    return sorted(selected.values(), key=lambda r: pd.Timestamp(r["published_at"]), reverse=True), counts


def fetch_targeted_news(per_metal=5, window_days=31):
    """Obtener noticias reales; el llamador guarda rows con storage.upsert.

    Retorna (rows, diagnostics), sin escribir la base de datos ni llamar a Gemini.
    diagnostics contiene coverage, deficits, sources y el instante de captura.
    Una fuente caída no cancela las otras y nunca se rellena una cuota con ficción.
    """
    if not 1 <= per_metal <= 20 or not 1 <= window_days <= 90:
        raise ValueError("Objetivo de 1 a 20 y ventana de 1 a 90 días.")
    fetched_at = now()
    candidates, sources = [], []
    with requests.Session() as client:
        client.headers["User-Agent"] = "MetalsAcademicMVP/0.2 (public RSS metadata research)"
        for source, url in TARGETED_FEEDS.items():
            diagnostic = {"source": source, "url": url, "sha256": "", "rows": 0}
            try:
                response = client.get(url, timeout=(10, 30))
                response.raise_for_status()
                body = response.content
                digest = hashlib.sha256(body).hexdigest()
                raw = DATA / "raw" / f"{digest[:12]}_targeted_news.xml"
                raw.parent.mkdir(parents=True, exist_ok=True)
                if not raw.exists():
                    raw.write_bytes(body)
                rows = parse_feed(body, source, fetched_at, window_days, trusted_only=True)
                candidates.extend(rows)
                diagnostic.update(status="ok" if rows else "empty", rows=len(rows),
                                  sha256=digest, detail="Sólo metadatos RSS, filtro temático y temporal aplicado.")
            except (requests.RequestException, ValueError, OSError) as exc:
                diagnostic.update(status="error", detail=f"{type(exc).__name__}: {str(exc)[:250]}")
            sources.append(diagnostic)
    selected, coverage = select_balanced(candidates, per_metal)
    return selected, {"fetched_at": fetched_at, "window_days": window_days,
                      "per_metal": per_metal, "coverage": coverage,
                      "deficits": {metal: per_metal - count for metal, count in coverage.items()},
                      "sources": sources}
