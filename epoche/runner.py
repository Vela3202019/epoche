"""
Epoché runner.

    python -m epoche.runner --adapter reference --trials 3
    python -m epoche.runner --adapter reference --case dream-content

Writes results/<adapter>-<stamp>.json and .md with a per-category boundary report.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from epoche.adapters import load_adapter
from epoche.judge import JUDGE_MODEL, judge_case

load_dotenv()
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"


def score_case(case: dict, extracted: list, verdict: dict) -> dict:
    required = [g for g in case["should_extract"] if not g.get("optional")]
    matched_gold = {m["gold_index"] for m in verdict.get("matches", [])}
    required_indices = {
        i for i, g in enumerate(case["should_extract"]) if not g.get("optional")
    }
    matched_required = len(matched_gold & required_indices)
    violations = [
        u for u in verdict.get("unmatched_extracted", [])
        if u.get("classification") == "violation"
    ]
    n_extracted = len(extracted)
    recall = matched_required / len(required) if required else 1.0
    precision = (n_extracted - len(violations)) / n_extracted if n_extracted else 1.0
    return {
        "recall": round(recall, 3),
        "precision": round(precision, 3),
        "n_extracted": n_extracted,
        "n_required": len(required),
        "violations": [
            {"violation": v.get("violation") or "unsupported", "reason": v.get("reason", "")}
            for v in violations
        ],
    }


async def run_case(case: dict, adapter) -> dict:
    start = time.monotonic()
    extracted = await adapter.extract(case["diary"])
    latency_ms = int((time.monotonic() - start) * 1000)
    verdict = await judge_case(case, extracted)
    return {
        "id": case["id"],
        "category": case["category"],
        "latency_ms": latency_ms,
        "scores": score_case(case, extracted, verdict),
        "extracted": extracted,
        "verdict": verdict,
    }


async def run_trial(cases, adapter):
    sem = asyncio.Semaphore(4)

    async def bounded(case):
        async with sem:
            return await run_case(case, adapter)

    return list(await asyncio.gather(*(bounded(c) for c in cases)))


def aggregate(trials: list) -> dict:
    by_category = defaultdict(list)
    violation_counter = Counter()
    recalls, precisions = [], []
    for trial in trials:
        for r in trial:
            s = r["scores"]
            by_category[r["category"]].append(s)
            recalls.append(s["recall"])
            precisions.append(s["precision"])
            for v in s["violations"]:
                violation_counter[v["violation"]] += 1

    categories = {}
    for cat, scores in sorted(by_category.items()):
        categories[cat] = {
            "n": len(scores),
            "recall": round(statistics.mean(x["recall"] for x in scores), 3),
            "precision": round(statistics.mean(x["precision"] for x in scores), 3),
            "violations": sum(len(x["violations"]) for x in scores),
        }
    return {
        "categories": categories,
        "violations_by_type": dict(violation_counter.most_common()),
        "overall": {
            "recall": round(statistics.mean(recalls), 3),
            "recall_std": round(statistics.stdev(recalls), 3) if len(recalls) > 1 else 0.0,
            "precision": round(statistics.mean(precisions), 3),
            "precision_std": round(statistics.stdev(precisions), 3) if len(precisions) > 1 else 0.0,
            "violations_per_trial": round(
                sum(violation_counter.values()) / len(trials), 2
            ),
        },
    }


def build_manifest(args, adapter_name: str, dataset_raw: str, n_cases: int) -> dict:
    try:
        git_rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=ROOT,
        ).stdout.strip() or "uncommitted"
    except OSError:
        git_rev = "unknown"
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_rev": git_rev,
        "adapter": adapter_name,
        "judge": f"deepseek:{JUDGE_MODEL}",
        "dataset_sha256": hashlib.sha256(dataset_raw.encode()).hexdigest()[:12],
        "n_cases": n_cases,
        "trials": args.trials,
    }


def render_report(manifest: dict, agg: dict) -> str:
    lines = [
        "# Epoché report",
        "",
        f"{manifest['timestamp']} · system **{manifest['adapter']}** · judge "
        f"`{manifest['judge']}` · {manifest['trials']} trial(s) × {manifest['n_cases']} "
        f"cases · dataset `{manifest['dataset_sha256']}` · rev `{manifest['git_rev']}`",
        "",
        "| category | n | recall | precision | boundary violations |",
        "|---|---|---|---|---|",
    ]
    for cat, s in agg["categories"].items():
        lines.append(
            f"| {cat} | {s['n']} | {s['recall']:.3f} | {s['precision']:.3f} | {s['violations']} |"
        )
    o = agg["overall"]
    lines += [
        "",
        f"**Overall:** recall {o['recall']:.3f}±{o['recall_std']:.3f} · precision "
        f"{o['precision']:.3f}±{o['precision_std']:.3f} · "
        f"{o['violations_per_trial']} boundary violations/trial",
        "",
        "**Violations by type:** "
        + (", ".join(f"{k}: {v}" for k, v in agg["violations_by_type"].items()) or "none"),
        "",
    ]
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run Epoché")
    parser.add_argument("--adapter", default="reference")
    parser.add_argument("--model", default="gpt-5.2", help="Model for the reference adapter")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--case", help="Run a single case by id")
    args = parser.parse_args()

    dataset_raw = (ROOT / "data" / "cases.json").read_text()
    cases = json.loads(dataset_raw)["cases"]
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            raise SystemExit(f"No case with id {args.case!r}")

    adapter = (
        load_adapter(args.adapter, model=args.model)
        if args.adapter == "reference"
        else load_adapter(args.adapter)
    )
    manifest = build_manifest(args, adapter.name, dataset_raw, len(cases))
    logger.info("Manifest: %s", json.dumps(manifest))

    trials = []
    for t in range(args.trials):
        logger.info("Trial %d/%d", t + 1, args.trials)
        trials.append(await run_trial(cases, adapter))

    agg = aggregate(trials)
    report = render_report(manifest, agg)

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = args.adapter
    (RESULTS_DIR / f"{slug}-{stamp}.json").write_text(json.dumps(
        {"manifest": manifest, "aggregate": agg, "trials": trials},
        ensure_ascii=False, indent=2,
    ))
    (RESULTS_DIR / f"{slug}-{stamp}.md").write_text(report)
    print(report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(main())
