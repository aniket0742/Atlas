"""Evaluation dataset format and loading.

The important decision here is what a relevance label points at.

The obvious choice -- label the chunk ids that answer each question -- is a trap.
Chunk ids are derived from (document, version, ordinal), so any change to
chunking, overlap, or the token budget renumbers every chunk and silently
invalidates the entire dataset. Since the whole purpose of the harness is to
compare chunking and retrieval strategies, labels anchored to chunk ids would be
destroyed by the first experiment they are meant to evaluate.

So a label names a **document** (by its stable external id) and, optionally, a
**snippet** of text that must appear in the retrieved chunk. That survives
re-chunking, re-embedding and re-ingestion, and it stays readable: a reviewer can
tell what a label means without querying the database.

    {"id": "refund-window",
     "question": "How long do customers have to request a refund?",
     "answerable": true,
     "labels": [{"document": "policies/billing.md", "contains": "within 30 days"}]}

`answerable: false` marks a question the corpus genuinely cannot answer. Those
carry no labels and are scored on a different axis -- whether the system refuses
-- because a system that answers them confidently is broken no matter how good
its retrieval metrics are.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_WHITESPACE = re.compile(r"\s+")


def canonical(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip().casefold()


@dataclass(slots=True)
class Label:
    document: str
    contains: str | None = None

    def matches(self, document_external_id: str, chunk_text: str) -> bool:
        if document_external_id != self.document:
            return False
        if self.contains is None:
            return True
        return canonical(self.contains) in canonical(chunk_text)


@dataclass(slots=True)
class EvalQuery:
    id: str
    question: str
    answerable: bool = True
    labels: list[Label] = field(default_factory=list)
    # Query type: lookup, paraphrase, identifier, conceptual, multi-doc,
    # distractor, unanswerable. Carried so metrics can be broken down per kind
    # -- an aggregate can easily hide that lexical retrieval helps identifier
    # queries substantially while changing nothing else.
    kind: str = "unclassified"
    notes: str | None = None


def load(path: Path) -> list[EvalQuery]:
    """Load a JSONL eval set, validating as we go.

    Validation is strict: a dataset with a malformed label produces silently
    wrong metrics, which is worse than no metrics.
    """
    queries: list[EvalQuery] = []
    seen: set[str] = set()

    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc

        for key in ("id", "question"):
            if key not in record:
                raise ValueError(f"{path}:{line_number}: missing required field {key!r}")

        query_id = str(record["id"])
        if query_id in seen:
            raise ValueError(f"{path}:{line_number}: duplicate query id {query_id!r}")
        seen.add(query_id)

        answerable = bool(record.get("answerable", True))
        labels = [
            Label(document=str(item["document"]), contains=item.get("contains"))
            for item in record.get("labels", [])
        ]

        if answerable and not labels:
            raise ValueError(
                f"{path}:{line_number}: query {query_id!r} is marked answerable but has "
                "no labels. Add labels, or set answerable to false."
            )
        if not answerable and labels:
            raise ValueError(
                f"{path}:{line_number}: query {query_id!r} is marked unanswerable but "
                "has labels."
            )

        queries.append(
            EvalQuery(
                id=query_id,
                question=str(record["question"]),
                answerable=answerable,
                labels=labels,
                kind=str(record.get("kind", "unclassified")),
                notes=record.get("notes"),
            )
        )

    if not queries:
        raise ValueError(f"{path} contains no queries")
    return queries
