"""
Adapter interface for systems under test.

A memory system adapter takes one first-person entry and returns the memories
the system would store, as a flat list of neutral statements:

    [{"statement": "Had coffee with Mara ...", "kind": "event"}, ...]

kind is one of: "event", "emotion", "self_knowledge" - or omitted if the
system doesn't distinguish. Epoché scores WHAT was stored, not the
system's internal schema.
"""

from typing import Any, Dict, List, Protocol


class MemoryAdapter(Protocol):
    name: str

    async def extract(self, entry: str) -> List[Dict[str, Any]]:
        """Return the memories the system would store for this entry."""
        ...
