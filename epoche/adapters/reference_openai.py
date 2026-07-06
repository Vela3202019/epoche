"""
Reference adapter: a boundary-aware extraction prompt over the OpenAI API.

This is the baseline row of the Epoché leaderboard - a single careful
prompt, no pipeline. It exists so that dedicated memory systems can be
compared against "just a good prompt".
"""

import json
import os

from openai import AsyncOpenAI

EXTRACTION_PROMPT = """You extract personal memories from a first-person journal entry so they can be stored in a long-term memory system.

Extract three kinds of memories:
- event: something that actually happened in the writer's life (including their own past)
- emotion: a feeling the writer expressed or unmistakably implied about themselves
- self_knowledge: a pattern, belief, preference, or goal the writer explicitly claims about themselves

EXCLUSIONS - a memory system that fabricates is worse than one that misses:
1. Not the writer's: stories other people told, and content from books/podcasts/movies. Hearing the story can be an event; the story itself is not.
2. Not yet real: future plans, TODOs, intentions, and hypotheticals are not events. A stated want may be self_knowledge. Real actions already taken toward a future event ARE events.
3. Not stated: do not infer how the writer "must have felt" from events alone. Watch for sarcasm - sarcastic praise is negative emotion.
4. Not claimed: never generalize one event into a pattern or belief yourself. Advice from others is not the writer's belief unless they explicitly adopt it.
5. Dreams stay dreams: having a dream is an event; the dream's content is not.
6. Negation is not occurrence: things that almost happened or didn't happen are not events.
7. Fiction stays fiction: scenes the writer wrote or imagined are not events; the writing session is.

Each memory must be atomic (one event per memory) and first-person.

Return JSON only:
{"memories": [{"statement": "...", "kind": "event|emotion|self_knowledge"}]}"""


class ReferenceOpenAIAdapter:
    name = "reference-prompt"

    def __init__(self, model: str = "gpt-5.2"):
        self.model = model
        self.name = f"reference-prompt ({model})"
        self._client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    async def extract(self, entry: str):
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": entry},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        return json.loads(response.choices[0].message.content).get("memories", [])
