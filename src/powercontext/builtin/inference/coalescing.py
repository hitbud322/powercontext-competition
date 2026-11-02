# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Embedding request coalescing and bounded vector cache.

Implements a wrapper that shares concurrent requests to the same text,
caches successful vectors with LRU eviction, and preserves per-call
usage accounting and cancellation isolation.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage

if TYPE_CHECKING:
    from powercontext.builtin.artifacts.memory.models import EmbeddingProfile
    from powercontext.builtin.inference.protocols import EmbeddingModel


class _BatchInfo:
    """Metadata for one underlying model call batch."""

    __slots__ = ("texts", "usage", "exception", "done")

    def __init__(self, texts: list[str]) -> None:
        self.texts = tuple(texts)
        self.usage: InferenceUsage | None = None
        self.exception: BaseException | None = None
        self.done: bool = False


class CoalescingEmbeddingModel:
    """Wrap an EmbeddingModel with cross-call request coalescing and LRU caching.

    Same-text requests arriving concurrently within the same event loop share a
    single underlying model call. Successful vectors are cached up to ``cache_size``
    entries with LRU eviction. ``cache_size=0`` disables cross-call caching but
    still shares in-flight requests.
    """

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")  # noqa: TRY003
        if coalesce_ms < 0:
            raise ValueError("coalesce_ms must be >= 0")  # noqa: TRY003
        if cache_size < 0:
            raise ValueError("cache_size must be >= 0")  # noqa: TRY003

        self._model = model
        self.profile: EmbeddingProfile = model.profile
        self._max_batch_size = max_batch_size
        self._coalesce_ms = coalesce_ms
        self._cache_size = cache_size

        # LRU cache: text -> EmbeddingVector
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()

        # Shared lock
        self._lock = asyncio.Lock()

        # --- Coalescing state (all protected by _lock) ---
        # Pending texts: text -> list of per-call waiter futures
        self._pending: dict[str, list[asyncio.Future[Any]]] = {}
        # Inflight (batch running): text -> _BatchInfo (only while batch NOT done)
        self._inflight: dict[str, _BatchInfo] = {}
        # Texts currently executing: text -> _BatchInfo (for usage tracking during wait)
        self._executing: dict[str, _BatchInfo] = {}
        # Queue of texts waiting for dispatch
        self._queue: list[str] = []
        # Coalesce window timer
        self._timer: asyncio.Handle | None = None

    # --- Cache ---

    def _cache_get(self, text: str) -> EmbeddingVector | None:
        if self._cache_size <= 0:
            return None
        if text in self._cache:
            self._cache.move_to_end(text)
            return self._cache[text]
        return None

    def _cache_put(self, text: str, vector: EmbeddingVector) -> None:
        if self._cache_size <= 0:
            return
        if text in self._cache:
            self._cache.move_to_end(text)
            self._cache[text] = vector
        else:
            self._cache[text] = vector
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

    # --- Public API ---

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))

        # Plan: for each text, track what we need to wait on and which batch_info it belongs to
        # We'll use a simple mapping approach:
        # - For each input text: (index, text, waiter_or_vector_or_exception, batch_info_or_None)

        plan: list[tuple[int, str, Any, _BatchInfo | None]] = []
        futures_to_wait: set[asyncio.Future[Any]] = set()

        async with self._lock:
            for idx, text in enumerate(texts):
                # 1. Cache hit - no wait needed, no batch tracking
                cached = self._cache_get(text)
                if cached is not None:
                    plan.append((idx, text, cached, None))
                    continue

                # 2. Check inflight (batch still running)
                bi = self._inflight.get(text)
                if bi is not None:
                    # Wait for this batch
                    waiter = asyncio.get_running_loop().create_future()
                    self._pending.setdefault(text, []).append(waiter)
                    futures_to_wait.add(waiter)
                    plan.append((idx, text, waiter, bi))
                    continue

                # 3. Check executing (batch already running, moved from inflight but not done yet)
                bi = self._executing.get(text)
                if bi is not None:
                    waiter = asyncio.get_running_loop().create_future()
                    self._pending.setdefault(text, []).append(waiter)
                    futures_to_wait.add(waiter)
                    plan.append((idx, text, waiter, bi))
                    continue

                # 4. New text - add to queue
                waiter = asyncio.get_running_loop().create_future()
                if text not in self._pending:
                    self._pending[text] = [waiter]
                    self._queue.append(text)
                    # Start coalesce timer if not running
                    if self._timer is None and self._coalesce_ms > 0:
                        loop = asyncio.get_running_loop()
                        self._timer = loop.call_later(self._coalesce_ms / 1000.0, self._dispatch)
                else:
                    self._pending[text].append(waiter)

                futures_to_wait.add(waiter)
                plan.append((idx, text, waiter, None))  # batch_info set after dispatch

            # Flush queue so all texts have batch_info (important for usage tracking)
            while self._queue:
                self._dispatch()

            # After dispatch, fill in batch_info for new texts in plan
            for i, (idx, text, val, bi) in enumerate(plan):
                if bi is None:
                    # Look up batch_info now
                    new_bi = self._inflight.get(text) or self._executing.get(text)
                    if new_bi is not None:
                        plan[i] = (idx, text, val, new_bi)

        # Wait for all needed futures (shielded to isolate cancellation)
        if futures_to_wait:
            try:
                await asyncio.gather(*(asyncio.shield(f) for f in futures_to_wait))
            except asyncio.CancelledError:
                raise

        # Collect results
        vectors: list[EmbeddingVector] = [None] * len(texts)  # type: ignore[list-item]
        total_usage = InferenceUsage(requests=0)
        batches_seen: set[_BatchInfo] = set()

        for idx, text, val, bi in plan:
            # Get the vector (or exception)
            if isinstance(val, asyncio.Future):
                fut_result = val.result()
                if isinstance(fut_result, BaseException):
                    raise fut_result
                vectors[idx] = fut_result
            elif isinstance(val, BaseException):
                raise val
            elif isinstance(val, tuple):
                # EmbeddingVector is tuple[float, ...]
                vectors[idx] = val
            else:
                # Shouldn't happen at this point
                raise RuntimeError(f"Unexpected value for text {text!r}: {val!r}")  # noqa: TRY003

            # Track usage
            if bi is not None and bi not in batches_seen:
                batches_seen.add(bi)
                if bi.exception is not None:
                    raise bi.exception
                if bi.usage is not None:
                    total_usage.requests += bi.usage.requests
                    if bi.usage.input_tokens is not None:
                        total_usage.input_tokens = (total_usage.input_tokens or 0) + bi.usage.input_tokens
                    if bi.usage.output_tokens is not None:
                        total_usage.output_tokens = (total_usage.output_tokens or 0) + bi.usage.output_tokens

        # Final check
        result = tuple(v for v in vectors if v is not None)
        if len(result) != len(texts):
            raise RuntimeError(f"Expected {len(texts)} vectors, got {len(result)}")  # noqa: TRY003

        return EmbeddingResult(vectors=result, usage=total_usage)

    # --- Dispatch & Execution ---

    def _dispatch(self) -> None:
        """Dispatch next batch (caller must hold _lock)."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

        if not self._queue:
            return

        batch_texts = self._queue[: self._max_batch_size]
        self._queue = self._queue[self._max_batch_size :]

        bi = _BatchInfo(batch_texts)
        for text in batch_texts:
            self._inflight[text] = bi

        asyncio.create_task(self._execute(bi))

    async def _execute(self, batch_info: _BatchInfo) -> None:
        texts = batch_info.texts

        # Move from inflight to executing (marking batch as running)
        async with self._lock:
            for text in texts:
                # Move inflight -> executing
                cur = self._inflight.pop(text, None)
                if cur is not None:
                    self._executing[text] = cur

        # Collect waiting futures
        waiting_map: dict[str, list[asyncio.Future[Any]]] = {}
        async with self._lock:
            for text in texts:
                waiting_map[text] = self._pending.pop(text, [])

        try:
            result = await self._model.embed(texts)

            if len(result.vectors) != len(texts):
                raise ValueError(
                    f"Model returned {len(result.vectors)} vectors for {len(texts)} texts"
                )

            vectors_by_text: dict[str, EmbeddingVector] = {}
            for text, vec in zip(texts, result.vectors, strict=True):
                vectors_by_text[text] = vec

            batch_info.usage = result.usage

            # Update cache
            async with self._lock:
                for text, vec in vectors_by_text.items():
                    self._cache_put(text, vec)

            # Resolve waiting futures
            for text, vec in vectors_by_text.items():
                for fut in waiting_map.get(text, []):
                    if not fut.done():
                        fut.set_result(vec)

            batch_info.done = True

        except BaseException as exc:
            batch_info.exception = exc
            # Resolve waiting futures with exception
            for text in texts:
                for fut in waiting_map.get(text, []):
                    if not fut.done():
                        fut.set_result(exc)
            batch_info.done = True

        # Clean up executing
        async with self._lock:
            for text in texts:
                self._executing.pop(text, None)
