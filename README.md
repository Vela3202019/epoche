# Epoché

**A boundary-fidelity benchmark for AI memory: does the system remember only what was actually given?**

*Epoché (ἐποχή): in Husserl's phenomenology, the discipline of suspending judgment — asserting
nothing beyond what is given in experience. A trustworthy memory system practices epoché at
write time: it does not store the friend's story as yours, the plan as the event, the dream
as the day, or the feeling you never expressed.*

Every major memory benchmark (LoCoMo, LongMemEval, BEAM) measures *retrieval*: feed a long
conversation in, ask questions later, score the answers. What got **written** to memory is
invisible — and write-time extraction is where hallucinations are born and then propagate
([HaluMem, 2025](https://arxiv.org/abs/2511.03506)).

Epoché measures the write path, and specifically **where** it breaks: not *how much* a
system fabricates, but *which boundaries* it crosses.

## The boundaries

A personal memory system that fabricates memories is worse than one that misses them.
Epoché encodes the boundaries that separate a person's actual life from everything
adjacent to it:

| boundary | example failure |
|---|---|
| **secondhand** | a friend's airport story, a podcast anecdote, a therapist's claim → stored as the user's own |
| **future / hypothetical** | "tomorrow I fly to Seattle", TODOs, counterfactual regrets → stored as events |
| **dream** | dream content → stored as a real event |
| **negation** | "I almost texted my ex" → stored as "texted ex" |
| **fiction** | a scene the user *wrote* → stored as the user's life |
| **invented emotion** | "fixed the bug" → stored as "felt relieved" (never stated); sarcasm read literally |
| **invented belief** | one event → generalized into a pattern the user never claimed |
| **stale update** | "I used to think X, now Y" → X stored as current |
| **atomicity** | three events → one mushed memory |

Plus **regression guards** — cases where the *correct* behavior is to extract (own-past
recollections, real actions toward future events, adopted advice, shared "we" experiences)
— so a system can't win by refusing to remember. Suppression is not fidelity.

31 gold-labeled first-person entries across 14 categories, including code-switched
Chinese/English and Spanish/English cases.

## How scoring works

1. An **adapter** feeds each entry to the system under test and returns the memories it
   stored, as neutral statements (see `epoche/adapters/base.py` — ~10 lines to implement).
2. **Deterministic bookkeeping + a cross-family LLM judge** (DeepSeek by default, never the
   same family as the reference extractor) aligns statements to gold, then classifies every
   unmatched statement as *grounded* (harmless) or a *violation* tagged with its boundary type.
3. Reports: recall, precision (1 − violation rate), and a **violations-by-boundary table** —
   per category, with mean±std across `--trials N` repeat runs and a reproducible manifest
   (dataset hash, models, git rev).

```bash
pip install -e .
# .env: OPENAI_API_KEY (reference adapter), DEEPSEEK_API_KEY (judge)

python -m epoche.runner --adapter reference --trials 3
python -m epoche.runner --adapter reference --case dream-content
```

## Results

2 trials × 31 cases, judge `deepseek-chat` (full reports in `results/`):

| system | recall | precision | boundary violations / trial |
|---|---|---|---|
| reference-prompt (gpt-5.2) | 0.965 | 0.924 | 7.5 |

Even a boundary-aware prompt leaks: the reference adapter's violations concentrate in
**future/hypothetical** (6), **stale updates** (4 — superseded beliefs stored as current),
**invented beliefs** (3), and **sarcasm read literally**. Baselines, dreams, negation,
fiction, and all regression guards hold. That spread is the point of the benchmark:
the remaining failures are exactly the boundaries nobody measures.

*Adapters for Mem0 and Zep are implemented but not yet validated against live accounts
(`pip install -e ".[mem0]"` / `".[zep]"`). PRs welcome — the most useful contribution right
now is a validated run of a real memory product.*

## Design principles

- **Cross-family judging.** The judge never shares a model family with the system it grades.
- **Violation-first metrics.** Precision penalizes only boundary violations; extra memories
  that are genuinely grounded in the entry cost nothing.
- **Guards against over-correction.** Regression-guard cases make "extract less" a losing
  strategy, so precision can't be bought with amnesia.
- **Reproducible runs.** Every report carries a manifest; two runs are comparable only if
  dataset hashes match.
- **Small enough to read.** 31 cases you can audit in an hour beats 15,000 you can't.
  (For bulk-scale hallucination measurement, use [HaluMem](https://arxiv.org/abs/2511.03506);
  Epoché is the structured complement that tells you *which* boundary broke.)

## Adding an adapter

Implement one async method:

```python
class MySystemAdapter:
    name = "my-system"

    async def extract(self, entry: str) -> list[dict]:
        # feed entry to your system, return what it stored
        return [{"statement": "...", "kind": "event"}]
```

Register it in `epoche/adapters/__init__.py`, then
`python -m epoche.runner --adapter my-system --trials 3`.

## Limitations

- Gold labels were authored by one person, not independent annotators; treat absolute
  numbers as signal and cross-system deltas as the reliable measurement.
- Entries are synthetic (no real diary data) and short; production inputs are messier.
- The grounded/violation split is an LLM judgment — spot-check `verdict` in the JSON
  output when a number surprises you. A second-judge agreement script is planned
  (the memora prototype reported 0.93/0.88 inter-judge agreement).

## Origins

Epoché grew out of the eval suite for [memora](https://github.com/Bella3202019/memora),
a personal memory graph, where a version of this method cut extraction hallucinations 78%
(mean F1 0.838 → 0.926) in one measured prompt iteration.

MIT license.
