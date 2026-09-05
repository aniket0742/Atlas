"""Freeze the answer-model comparison as a committed evaluation artifact.

Reads the most recent `answers-<model>` report per model out of eval/results/,
copies them into eval/baselines/answer-models/, and writes a README recording
what was measured and which model was selected.

It also does the part a single-run table cannot: **decide whether the
differences are real.** The unverified-quote count moved 5 -> 8 for the same
model across two runs of the same configuration, so ranking models by a single
run of that metric would be reading noise. Per-query counts are compared with
the same paired bootstrap used for the retrieval decisions in ADR-0021.

Run:  python scripts/freeze_answer_baselines.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.eval import metrics  # noqa: E402

RESULTS = Path("eval/results")
BASELINES = Path("eval/baselines/answer-models")
REFERENCE = "gemini-3.5-flash-lite"  # the selected default

PRICING = {
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.7-flash": (0.75, 3.75),
    "gemini-3.8-flash": (0.75, 3.75),
}


def newest_reports() -> dict[str, tuple[Path, dict]]:
    found: dict[str, tuple[Path, dict]] = {}
    for path in sorted(RESULTS.glob("*answers-*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        label = data.get("label") or ""
        model = label.removeprefix("answers-")
        if model in PRICING and "input" in data["summary"].get("tokens", {}):
            found[model] = (path, data)  # later file wins
    return found


def unverified_by_query(report: dict) -> dict[str, float]:
    return {
        q["query_id"]: float(q.get("unverified_quotes") or 0)
        for q in report["queries"]
        if q.get("refused") is not None
    }


def cost_per_1k(model: str, report: dict) -> float:
    tokens = report["summary"]["tokens"]
    in_price, out_price = PRICING[model]
    n = report["dataset"]["queries"]
    return (tokens["input"] * in_price + tokens["output"] * out_price) / 1e6 / n * 1000


def main() -> int:
    reports = newest_reports()
    if REFERENCE not in reports:
        print(f"No report with token split found for {REFERENCE}. Run "
              "scripts/compare_answer_model.py first.")
        return 1

    BASELINES.mkdir(parents=True, exist_ok=True)
    rows = []
    for model, (path, report) in sorted(reports.items()):
        shutil.copy2(path, BASELINES / f"{model}.json")
        s = report["summary"]
        r, c = s["refusal"], s["citations"]
        rows.append({
            "model": model,
            "refused": f"{r['correctly_refused']}/{r['unanswerable_queries']}",
            "wrong_refuse": r["incorrectly_refused"],
            "cited": f"{c['answers_with_citations']}/{c['answers_scored']}",
            "unverified": c["unverified_quotes"],
            "in_tok": s["tokens"]["input"],
            "out_tok": s["tokens"]["output"],
            "cost": cost_per_1k(model, report),
            "failures": s.get("answer_failures", {}).get("count", 0),
        })

    print(f"froze {len(rows)} report(s) into {BASELINES}\n")
    header = (f"{'model':<24}{'refused':>10}{'wrong':>7}{'cited':>10}"
              f"{'unverif':>9}{'in tok':>9}{'out tok':>9}{'$/1k Q':>9}")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['model']:<24}{row['refused']:>10}{row['wrong_refuse']:>7}"
              f"{row['cited']:>10}{row['unverified']:>9}{row['in_tok']:>9}"
              f"{row['out_tok']:>9}{row['cost']:>9.2f}")

    # Are the unverified-quote differences real, or run-to-run noise?
    print(f"\nPaired comparison of unverified quotes vs {REFERENCE}")
    print("(negative delta = fewer paraphrased quotes than the reference)")
    ref = unverified_by_query(reports[REFERENCE][1])
    for model, (_, report) in sorted(reports.items()):
        if model == REFERENCE:
            continue
        cand = unverified_by_query(report)
        ids = sorted(set(ref) & set(cand))
        delta, (low, high) = metrics.paired_bootstrap_delta(
            [ref[i] for i in ids], [cand[i] for i in ids]
        )
        verdict = "DIFFERENT" if (low > 0 or high < 0) else "no measured difference"
        print(f"  {model:<24}{delta:>+8.3f}  CI [{low:+.3f}, {high:+.3f}]  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
