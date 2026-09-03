from __future__ import annotations

import asyncio
import base64
import json
import re
import time
from typing import Any

from openai import AsyncOpenAI

from config.settings import ActorConfig
from logger_config import logger
from modules.base_agent import BaseAgent

ROUTER_CATEGORIES: tuple[str, ...] = ("vehicle", "seating", "decal")

ROUTER_SYSTEM_PROMPT = (
    "You classify a product photo for a procedural 3D modeling pipeline. "
    "Always respond with valid JSON only."
)

ROUTER_USER_PROMPT = """Look at the reference image and answer with JSON:
{"object": "<2-6 word name of the object>", "categories": [<zero or more of "vehicle", "seating", "decal">]}

Category definitions:
- "vehicle": car, truck, bus, bicycle, motorcycle, scooter, airplane, jet, helicopter, drone, boat, train, cart, or any other vehicle.
- "seating": chair, sofa, couch, loveseat, armchair, bench, stool, chaise lounge, or other seating / upholstered furniture.
- "decal": a ceramic, glass, metal, or plastic vessel, plate, vase, mug, bowl, or similar body whose surface carries painted, printed, glazed, engraved, or floral ornament.

Use an empty list when none apply. Output the JSON object only."""

_OUTER_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse(text: str) -> tuple[str | None, set[str] | None]:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    m = _OUTER_BRACE_RE.search(text)
    if not m:
        return None, None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None, None
    name = obj.get("object")
    cats = obj.get("categories")
    if not isinstance(cats, list):
        return (str(name) if name else None), None
    keep = {str(c).strip().lower() for c in cats}
    keep &= set(ROUTER_CATEGORIES)
    return (str(name) if name else None), keep


class RouterAgent(BaseAgent):
    """One cheap VLM call per prompt: which coder handbooks does this object need?

    Returns a set of handbook keys (subset of ROUTER_CATEGORIES) plus a short object
    name. Any failure returns (None, None), which callers treat as "use the full,
    unmodified coder system prompt" so the router can never make things worse than
    the baseline pipeline.
    """

    actor = "router"

    def __init__(self, client: AsyncOpenAI, settings: ActorConfig, *, max_retries: int = 2) -> None:
        super().__init__(client, settings)
        self.max_retries = max_retries
        self._sem = asyncio.Semaphore(max(1, settings.workers))

    async def classify(
        self,
        *,
        task_id: str,
        image_bytes: bytes,
        image_mime: str,
        seed: int | None = None,
    ) -> tuple[str | None, set[str] | None]:
        t0 = time.time()
        ref_b64 = base64.b64encode(image_bytes).decode()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{ref_b64}"}},
                    {"type": "text", "text": ROUTER_USER_PROMPT},
                ],
            },
        ]
        extra_body: dict[str, Any] = {}
        if self.backend == "vllm" and self.enable_thinking is not None:
            extra_body["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        last_err: str | None = None
        for attempt in range(self.max_retries + 1):
            try:
                async with self._sem:
                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature if attempt == 0 else 0.3,
                        seed=(seed if seed is not None else self.seed) + attempt,
                        extra_body=extra_body,
                    )
                text = resp.choices[0].message.content or ""
                name, cats = _parse(text)
                if cats is None:
                    raise ValueError(f"unparseable router output: {text[:120]!r}")
                logger.info(
                    f"[Router] Task {task_id} | object={name!r} | categories={sorted(cats)} | "
                    f"Elapsed: {time.time() - t0:.1f}s"
                )
                return name, cats
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                logger.warning(f"[Router] Task {task_id} attempt {attempt + 1} failed: {last_err}")
        logger.warning(f"[Router] Task {task_id} giving up ({last_err}); using full coder prompt")
        return None, None
