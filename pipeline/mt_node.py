import asyncio
import os
from collections import OrderedDict

import deepl
import requests
from utils.logger import logger


class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


class MTNode:
    GOOGLE_TRANSLATE_URL = "https://translation.googleapis.com/language/translate/v2"

    def __init__(self, source_lang: str, target_lang: str):
        self.source_lang = source_lang
        self.target_lang = target_lang

        # --- Primary: Google Cloud Translate (simple API key auth) ---
        self.google_api_key = os.getenv("GOOGLE_TRANSLATE_API_KEY")
        if not self.google_api_key:
            logger.warning(
                "[MTNode] GOOGLE_TRANSLATE_API_KEY not set; Google Translate (primary) disabled."
            )

        # --- Fallback: DeepL ---
        self.deepl_auth_key = os.getenv("DEEPL_API_KEY")
        self.translator = None
        if self.deepl_auth_key and self.deepl_auth_key != "your_deepl_api_key":
            try:
                self.translator = deepl.Translator(self.deepl_auth_key)
            except Exception as e:
                logger.warning(f"[MTNode] Failed to init DeepL client: {e}")

        if not self.google_api_key and not self.translator:
            logger.warning(
                "[MTNode] No translation provider configured (Google or DeepL). "
                "Translation will pass through original text."
            )

        self.cache = LRUCache(512)

    def _map_deepl_target(self, lang: str):
        # DeepL specific mapping e.g. en -> EN-US
        lang = lang.upper()
        if lang == "EN":
            return "EN-US"
        if lang == "PT":
            return "PT-PT"
        return lang

    def _map_google_target(self, lang: str):
        return lang.split("-")[0].lower()

    def _google_translate_rest(self, text: str):
        """Call Google Cloud Translate v2 REST API using a simple API key."""
        target = self._map_google_target(self.target_lang)
        source = self._map_google_target(self.source_lang)

        params = {
            "key": self.google_api_key,
            "q": text,
            "target": target,
            "source": source,
            "format": "text",
        }
        resp = requests.post(self.GOOGLE_TRANSLATE_URL, data=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data["data"]["translations"][0]["translatedText"]

    def _do_translate(self, text: str):
        cache_key = (text, self.source_lang, self.target_lang)
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        res_text = None

        # Try Google Translate first (primary, for now)
        if self.google_api_key:
            try:
                res_text = self._google_translate_rest(text)
                logger.info(
                    f"[MTNode] Google Translate (API key) used: {self.source_lang}->{self.target_lang}"
                )
            except Exception as e:
                logger.warning(f"[MTNode] Google Translate failed, falling back: {e}")

        # Fallback to DeepL
        if not res_text and self.translator:
            try:
                target = self._map_deepl_target(self.target_lang)
                result = self.translator.translate_text(text, target_lang=target)
                res_text = result.text
                logger.info(
                    f"[MTNode] DeepL used for translation: {self.source_lang}->{self.target_lang}"
                )
            except Exception as e:
                logger.error(f"[MTNode] DeepL failed: {e}")

        if not res_text:
            logger.error(
                f"[MTNode] All MT providers failed or unavailable for text: {text}"
            )
            return text  # fallback to original text

        self.cache.put(cache_key, res_text)
        return res_text

    async def translate(self, text: str) -> str:
        if not text.strip():
            return text
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._do_translate, text)
