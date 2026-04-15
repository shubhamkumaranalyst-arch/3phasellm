import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from ollama import AsyncClient


class ModelManager:
    """Centralized model registry + timeout/fallback logic with warm-up support."""

    def __init__(
        self,
        client: AsyncClient,
        models_path: Path,
        request_timeout_seconds: float = 60.0,
    ) -> None:
        self.client = client
        self.models_path = models_path
        self.timeout = request_timeout_seconds
        self.config = self._load_config()
        self._warmed_models: set[str] = set()

    def _load_config(self) -> Dict[str, Any]:
        with self.models_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def resolve(self, route_key: str) -> str:
        routing = self.config.get("routing", {})
        registry = self.config.get("registry", {})
        fallback = self.config.get("fallbacks", {}).get(route_key) or self.config.get("fallbacks", {}).get("default")
        alias = routing.get(route_key) or routing.get("default")
        model = registry.get(alias, alias)
        if not model and fallback:
            return registry.get(fallback, fallback)
        return model

    def fallback_for(self, route_key: str) -> Optional[str]:
        registry = self.config.get("registry", {})
        fallback_alias = self.config.get("fallbacks", {}).get(route_key) or self.config.get("fallbacks", {}).get("default")
        if not fallback_alias:
            return None
        return registry.get(fallback_alias, fallback_alias)

    async def warm_models(self, route_keys: List[str]) -> None:
        async def warm_one(route_key: str) -> None:
            model = self.resolve(route_key)
            if not model or model in self._warmed_models:
                return
            try:
                await asyncio.wait_for(
                    self.client.generate(
                        model=model,
                        prompt="warmup",
                        keep_alive="30m",
                    ),
                    timeout=min(self.timeout, 15.0),
                )
                self._warmed_models.add(model)
            except Exception:
                # Non-fatal warm-up failures are tolerated.
                return

        await asyncio.gather(*(warm_one(k) for k in route_keys))

    async def chat(self, route_key: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        primary = self.resolve(route_key)
        fallback = self.fallback_for(route_key)
        try:
            return await asyncio.wait_for(
                self.client.chat(model=primary, messages=messages, keep_alive="30m"),
                timeout=self.timeout,
            )
        except Exception:
            if fallback and fallback != primary:
                return await asyncio.wait_for(
                    self.client.chat(model=fallback, messages=messages, keep_alive="30m"),
                    timeout=self.timeout,
                )
            raise

    async def generate(self, route_key: str, prompt: str) -> Dict[str, Any]:
        primary = self.resolve(route_key)
        fallback = self.fallback_for(route_key)
        try:
            return await asyncio.wait_for(
                self.client.generate(model=primary, prompt=prompt, keep_alive="30m"),
                timeout=self.timeout,
            )
        except Exception:
            if fallback and fallback != primary:
                return await asyncio.wait_for(
                    self.client.generate(model=fallback, prompt=prompt, keep_alive="30m"),
                    timeout=self.timeout,
                )
            raise
