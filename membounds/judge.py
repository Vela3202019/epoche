"""
Judge: aligns extracted statements to gold and classifies boundary violations.

The judge is intentionally a different model family (DeepSeek by default)
from the reference adapter's extractor (OpenAI) to avoid self-preference bias.
Override with EVAL_JUDGE_MODEL / EVAL_JUDGE_BASE_URL / DEEPSEEK_API_KEY.
"""

import json
import os

from openai import AsyncOpenAI

JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "deepseek-chat")
JUDGE_BASE_URL = os.getenv("EVAL_JUDGE_BASE_URL", "https://api.deepseek.com")

JUDGE_PROMPT = """You are grading a memory-extraction system against a gold reference for one journal entry.

You receive:
- ENTRY: the original first-person journal entry
- SHOULD_EXTRACT: memories a careful human annotator extracted. Items marked "optional": true never count against the system when missed. Emotion items may list accept_synonyms.
- EXTRACTED: statements the system stored
- TRAPS: boundaries that must not be crossed, each with a violation type

Grade:

1. MATCHING - For each SHOULD_EXTRACT item, decide whether some EXTRACTED statement expresses the same memory (paraphrase and partial wording are fine; kind labels may differ - judge the content). One extracted statement can match at most one gold item.

2. UNMATCHED EXTRACTED - classify every extracted statement that matched nothing:
   - "grounded": genuinely supported by the entry; the annotator just didn't list it. Harmless.
   - "violation": crosses a listed trap OR is otherwise unsupported by the entry.
     If it crosses a trap, use that trap's violation type; otherwise use "unsupported".

Return JSON only:
{
  "matches": [{"gold_index": 0, "extracted_index": 1}],
  "unmatched_extracted": [
    {"extracted_index": 2, "classification": "violation", "violation": "future", "reason": "..."},
    {"extracted_index": 3, "classification": "grounded", "violation": null, "reason": "..."}
  ],
  "missed_gold": [0]
}"""

_client = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise SystemExit("DEEPSEEK_API_KEY is required for the judge")
        _client = AsyncOpenAI(api_key=api_key, base_url=JUDGE_BASE_URL)
    return _client


def build_payload(case: dict, extracted: list) -> str:
    return json.dumps({
        "ENTRY": case["diary"],
        "SHOULD_EXTRACT": case["should_extract"],
        "EXTRACTED": extracted,
        "TRAPS": case.get("traps", []),
    }, ensure_ascii=False, indent=2)


async def judge_case(case: dict, extracted: list) -> dict:
    response = await get_client().chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": build_payload(case, extracted)},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    return json.loads(response.choices[0].message.content)
