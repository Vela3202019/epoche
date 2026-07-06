"""
Mem0 adapter (experimental).

Feeds the entry to Mem0 as a user message and reads back what Mem0 stored.
Requires `pip install mem0ai` and Mem0/OpenAI credentials.

Status: code path implemented, not yet validated against a live Mem0 account.
Contributions welcome - see README "Adding an adapter".
"""


class Mem0Adapter:
    name = "mem0"

    def __init__(self):
        try:
            from mem0 import AsyncMemory  # type: ignore
        except ImportError as e:
            raise SystemExit("Mem0 adapter requires: pip install mem0ai") from e
        self._memory_cls = AsyncMemory

    async def extract(self, entry: str):
        memory = await self._memory_cls.from_config({})
        result = await memory.add(
            messages=[{"role": "user", "content": entry}],
            user_id="membounds-eval",
            infer=True,
        )
        return [
            {"statement": m.get("memory", ""), "kind": None}
            for m in result.get("results", [])
        ]
