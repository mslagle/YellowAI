"""
Internet Search & Webpage Reader for YellowAI (Ollama Agent).
Provides free, API-key-free web search via DuckDuckGo (ddgs), clean HTML text extraction,
and persistent disk caching so discovered Pokémon Yellow Recharged walkthroughs,
type matchups, and puzzle solutions persist across runs.
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup

import config


class WebSearchClient:
    """
    Executes internet searches and webpage fetches with persistent local caching.
    Designed to expose native Ollama function-calling tools to the local VLM.
    """

    def __init__(self, cache_path: Optional[Path] = None, timeout: float = 12.0):
        self.cache_path = Path(cache_path) if cache_path else config.WEB_KNOWLEDGE_CACHE_FILE
        self.timeout = timeout
        self.cache: Dict[str, Dict[str, Any]] = {"searches": {}, "pages": {}, "facts": []}
        self._load_cache()

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.cache["searches"] = data.get("searches", {})
                self.cache["pages"] = data.get("pages", {})
                self.cache["facts"] = data.get("facts", [])
        except Exception as e:
            print(f"[WebSearch] Warning: Could not load cache ({e})")

    def _save_cache(self) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[WebSearch] Warning: Could not save cache ({e})")

    def add_verified_fact(self, topic: str, summary: str, source_url: str = "") -> None:
        """Persists a distilled internet-researched fact for future runs."""
        clean_summary = " ".join((summary or "").strip().split())
        if not clean_summary:
            return
        entry = {
            "topic": topic.strip() or "Pokemon Recharged Yellow",
            "summary": clean_summary[:600],
            "source_url": source_url.strip(),
            "timestamp": datetime.now().isoformat(),
        }
        # Avoid duplicate summaries
        for existing in self.cache["facts"]:
            if existing.get("summary", "").lower() == entry["summary"].lower():
                return
        self.cache["facts"].append(entry)
        if len(self.cache["facts"]) > 60:
            self.cache["facts"] = self.cache["facts"][-60:]
        self._save_cache()

    def get_relevant_cached_facts(self, query_context: str, limit: int = 5) -> List[str]:
        """Returns cached internet facts relevant to the current area, opponent, or quest."""
        facts = self.cache.get("facts", [])
        if not facts:
            return []
        q_words = {w for w in re.findall(r"[a-z0-9]+", (query_context or "").lower()) if len(w) > 2}
        scored: List[tuple] = []
        for idx, item in enumerate(facts):
            text = f"{item.get('topic', '')} {item.get('summary', '')}".lower()
            words = set(re.findall(r"[a-z0-9]+", text))
            overlap = len(q_words & words)
            scored.append((overlap, idx, item))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        selected = [item.get("summary", "") for ov, _, item in scored[:limit] if ov > 0 or not q_words]
        if not selected:
            selected = [item.get("summary", "") for item in facts[-min(3, limit):]]
        return [s for s in selected if s]

    def search(self, query: str, max_results: int = 5, use_cache: bool = True) -> Dict[str, Any]:
        """Searches the internet using DuckDuckGo and caches results locally."""
        clean_query = " ".join((query or "").strip().split())
        if not clean_query:
            return {"query": "", "results": [], "error": "Empty query"}

        cache_key = clean_query.lower()
        if use_cache and cache_key in self.cache["searches"]:
            cached = self.cache["searches"][cache_key]
            # Refresh cache only if older than 7 days
            if time.time() - cached.get("ts", 0) < 7 * 86400:
                return {
                    "query": clean_query,
                    "cached": True,
                    "results": cached.get("results", [])[:max_results],
                }

        results: List[Dict[str, str]] = []
        error_msg: Optional[str] = None
        try:
            from ddgs import DDGS
            raw = list(DDGS().text(clean_query, max_results=max_results))
            for r in raw:
                results.append({
                    "title": str(r.get("title", "")).strip(),
                    "url": str(r.get("href", r.get("link", ""))).strip(),
                    "snippet": str(r.get("body", r.get("snippet", ""))).strip(),
                })
        except Exception as e:
            error_msg = str(e)

        if results:
            self.cache["searches"][cache_key] = {
                "ts": time.time(),
                "timestamp": datetime.now().isoformat(),
                "results": results,
            }
            self._save_cache()

        return {
            "query": clean_query,
            "cached": False,
            "results": results,
            **({"error": error_msg} if error_msg else {}),
        }

    def fetch_webpage(self, url: str, max_chars: int = 3500) -> Dict[str, Any]:
        """Fetches and extracts readable text from a webpage URL."""
        clean_url = (url or "").strip()
        if not clean_url.startswith(("http://", "https://")):
            return {"url": clean_url, "text": "", "error": "Invalid URL scheme"}

        if clean_url in self.cache["pages"]:
            cached = self.cache["pages"][clean_url]
            if time.time() - cached.get("ts", 0) < 7 * 86400:
                return {
                    "url": clean_url,
                    "cached": True,
                    "title": cached.get("title", ""),
                    "text": cached.get("text", "")[:max_chars],
                }

        try:
            resp = requests.get(
                clean_url,
                timeout=self.timeout,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) YellowAI/1.0"},
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside"]):
                tag.decompose()
            title = soup.title.string.strip() if soup.title and soup.title.string else ""
            raw_text = soup.get_text(separator="\n")
            lines = [" ".join(line.split()) for line in raw_text.splitlines()]
            cleaned = "\n".join(line for line in lines if len(line) > 2)
            truncated = cleaned[:max_chars]

            self.cache["pages"][clean_url] = {
                "ts": time.time(),
                "title": title,
                "text": cleaned[:8000],
            }
            self._save_cache()
            return {"url": clean_url, "cached": False, "title": title, "text": truncated}
        except Exception as e:
            return {"url": clean_url, "text": "", "error": str(e)}

    @staticmethod
    def get_ollama_tool_schemas() -> List[Dict[str, Any]]:
        """Returns Ollama /api/chat compatible tool schemas for internet research."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": (
                        "Search the internet for Pokemon Recharged Yellow walkthroughs, "
                        "Pokemon Yellow maps, gym leader teams, item locations, HM progression, "
                        "type effectiveness, or solutions when lost or planning how to beat the game."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query, e.g. 'Pokemon Recharged Yellow Pewter Gym Brock strategy' or 'Pokemon Yellow Mt Moon map walkthrough'",
                            }
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch_webpage",
                    "description": "Read the text content of a URL returned by web_search to inspect detailed walkthrough steps or Pokémon data.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "Full HTTP/HTTPS URL to fetch.",
                            }
                        },
                        "required": ["url"],
                    },
                },
            },
        ]

    def execute_tool_call(self, name: str, arguments: Dict[str, Any]) -> str:
        """Executes a tool call from Ollama and returns serialized JSON string."""
        if name == "web_search":
            q = str(arguments.get("query", "")).strip()
            res = self.search(q, max_results=5)
            return json.dumps(res, ensure_ascii=False)
        if name == "fetch_webpage":
            u = str(arguments.get("url", "")).strip()
            res = self.fetch_webpage(u, max_chars=3500)
            return json.dumps(res, ensure_ascii=False)
        return json.dumps({"error": f"Unknown tool: {name}"})
