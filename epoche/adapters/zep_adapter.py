"""
Zep adapter (experimental).

Sends the entry as a user message to a fresh Zep thread, waits for graph
ingestion, then reads back the extracted facts/entities as statements.
Requires `pip install zep-cloud` and ZEP_API_KEY.

Status: code path implemented, not yet validated against a live Zep account.
Contributions welcome - see README "Adding an adapter".
"""

import asyncio
import os
import uuid


class ZepAdapter:
    name = "zep"

    def __init__(self):
        try:
            from zep_cloud.client import AsyncZep  # type: ignore
        except ImportError as e:
            raise SystemExit("Zep adapter requires: pip install zep-cloud") from e
        self._client = AsyncZep(api_key=os.environ["ZEP_API_KEY"])

    async def extract(self, entry: str):
        user_id = f"epoche-{uuid.uuid4().hex[:8]}"
        thread_id = uuid.uuid4().hex
        await self._client.user.add(user_id=user_id)
        await self._client.thread.create(thread_id=thread_id, user_id=user_id)
        await self._client.thread.add_messages(
            thread_id=thread_id,
            messages=[{"role": "user", "name": "user", "content": entry}],
        )
        await asyncio.sleep(20)  # allow graph ingestion
        edges = await self._client.graph.edge.get_by_user_id(user_id=user_id)
        return [{"statement": e.fact, "kind": None} for e in (edges or []) if e.fact]
