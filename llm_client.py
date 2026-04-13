from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx
import streamlit as st


DEFAULT_TIMEOUT = 90.0


class LLMClient:

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = (
            base_url
            or st.secrets.get("LLM_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self.model = model or st.secrets.get("LLM_MODEL", "gpt-4o-mini")
        self.api_key = api_key or st.secrets.get("LLM_API_KEY", "")
        self._client: Optional[httpx.AsyncClient] = None
        self._timeout = timeout

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 400,
        temperature: float = 0.5,
    ) -> str:
        client = self._ensure_client()
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        url = f"{self.base_url}/chat/completions"
        resp = await client.post(url, json=payload, headers=self._headers())
        if resp.status_code != 200:
            raise RuntimeError(f"LLM API Error {resp.status_code}: {resp.text}")
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"LLM response malformed: {e} | body={data!r}")

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def __aenter__(self) -> "LLMClient":
        self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()
