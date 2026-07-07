import asyncio
import os
from collections import OrderedDict

import deepl
from utils.logger import logger

try:
    from google.cloud import translate_v2 as translate
except ImportError:
    translate = None


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
    def __init__(self, source_lang: str, target_lang: str):
        self.source_lang = source_lang
        self.target_lang = target_lang

        # --- DeepL setup ---
        self.deepl_auth_key = os.getenv("DEEPL_API_KEY")
        self.translator = None
        if self.deepl_auth_key and self.deepl_auth_key != "your_deepl_api_key":
            try:
                self.translator = deepl.Translator(self.deepl_auth_key)
            except Exception as e:
                logger.warning(f"[MTNode] Failed to init DeepL client: {e}")

        # --- Google Translate setup (uses service account credentials, not api_key) ---
        self.google_client = None
        google_creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if translate and google_creds_path and os.path.exists(google_creds_path):
            try:
                self.google_client = translate.Client()
            except Exception as e:
                logger.warning(f"[MTNode] Failed to init Google Translate client: {e}")

        if not self.translator and not self.google_client:
            logger.warning(
                "[MTNode] No translation provider configured (DeepL or Google). "
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

    def _do_translate(self, text: str):
        cache_key = (text, self.source_lang, self.target_lang)
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        res_text = None

        # Try DeepL
        if self.translator:
            try:
                target = self._map_deepl_target(self.target_lang)
                result = self.translator.translate_text(text, target_lang=target)
                res_text = result.text
                logger.info(
                    f"[MTNode] DeepL used for translation: {self.source_lang}->{self.target_lang}"
                )
            except Exception as e:
                logger.warning(f"[MTNode] DeepL failed, falling back: {e}")

        # Fallback to Google
        if not res_text and self.google_client:
            try:
                target = self._map_google_target(self.target_lang)
                source = self._map_google_target(self.source_lang)
                result = self.google_client.translate(
                    text, target_language=target, source_language=source
                )
                res_text = result["translatedText"]
                logger.info(
                    f"[MTNode] Google Translate used: {self.source_lang}->{self.target_lang}"
                )
            except Exception as e:
                logger.error(f"[MTNode] Google Translate failed: {e}")

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
