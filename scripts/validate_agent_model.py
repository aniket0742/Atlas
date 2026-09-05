"""Phase 4 step 1: does the candidate agent model actually route tools well?

The SDK question is already settled -- these models accept `tools` alongside a
response schema. What is NOT settled is *behaviour*: whether a small, cheap model
decides sensibly when to search, what to search for, and when to stop. A single
successful function call proves the wiring, not the judgement.

So this drives a real multi-turn loop against real retrieval over the real
corpus, on a question set whose correct behaviour is known in advance, and scores
two things that matter more than "did it emit a call":

  * **Correct decision.** Did it search when the answer is in the corpus, and
    NOT search when the question needs no lookup? An agent that searches for
    "what is 2 + 2" burns quota and latency for nothing; one that answers a
    policy question without searching is ungrounded.
  * **Iteration.** Multi-document questions were stuck at Recall@1 = 0.400 for
    every retrieval configuration in Phase 2. If iterative search is going to
    help them, the model has to issue more than one distinct query. Whether it
    does is the whole premise of the agentic path.

No production code lives here. This is an experiment that informs which model
takes the agent role before any framework is built around it.

Run:  python scripts/validate_agent_model.py
"""

from __future__ import annotations

import asyncio
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from google import genai  # noqa: E402
from google.genai import errors, types  # noqa: E402

import atlas  # noqa: F401,E402  (installs the Windows event loop policy)
from atlas.config import get_settings  # noqa: E402
from atlas.db import repository as repo  # noqa: E402
from atlas.db.pool import Database  # noqa: E402
from atlas.providers.factory import get_embedder  # noqa: E402
from atlas.retrieval.service import Retriever  # noqa: E402

MAX_TURNS = 4

# Paid-tier USD per 1M tokens, from the pricing page. The Flash models carry
# promotional pricing that doubles on 2027-01-01, so cost rankings here are not
# permanent -- recorded with the expiry so a future reader knows to recheck.
PRICING = {
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.6-flash": (0.75, 3.75),
    "gemini-3.7-flash": (0.75, 3.75),
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.5-flash": (1.50, 9.00),
}

# gemini-3.5-flash is included only as the incumbent reference. It is strictly
# dominated on price -- 2x input and 2.4x output against gemini-3.8-flash, which
# the vendor describes as a newer and more capable model.
# Narrowed to the plausible agent models. gemini-3.5-flash is excluded as
# strictly dominated (ADR-0024); 3.6 and 3.8 behaved like 3.7 on the first run
# at similar or worse cost.
CANDIDATES = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.7-flash",
]

SYSTEM = """\
You are Atlas, a retrieval assistant for an organisation's internal knowledge base.

You have one tool: search_knowledge_base.

Use it whenever the answer could plausibly be in the organisation's documents --
policies, engineering docs, runbooks, onboarding, security, release notes.

Do NOT use it for general knowledge, arithmetic, or conversational replies; those
need no lookup.

If one search does not surface everything a question needs, search again with a
different query. Questions that compare two topics usually need more than one
search.

When you have enough, answer from what you found. If the knowledge base does not
contain the answer, say so plainly."""

SEARCH_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_knowledge_base",
            description=(
                "Search the organisation's indexed documents. Returns the most "
                "relevant passages with their source document."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="A focused search phrase, not the whole question.",
                    )
                },
                required=["query"],
            ),
        )
    ]
)


@dataclass
class Case:
    id: str
    question: str
    expects_search: bool
    # Documents a complete answer must reach. Used only to report coverage, not
    # to score the routing decision.
    wants_documents: tuple[str, ...] = ()
    note: str = ""


CASES = [
    Case("refund-window", "How long do customers have to request a refund?", True,
         ("policies/billing.md",), "single lookup"),
    Case("atl-codes", "What is the difference between ATL-5002 and ATL-5004?", True,
         ("engineering/api-errors.md",), "identifier lookup"),
    Case("retention-multi",
         "How long is customer data kept after an account closes, and how does that "
         "interact with backups?", True,
         ("engineering/data-retention.md", "policies/privacy.md"), "multi-document"),
    Case("revocation-multi",
         "A revoked session still worked for a while. Is that by design, and did any "
         "release change it?", True,
         ("engineering/authentication.md", "product/release-notes-2026-q1.md"),
         "multi-document"),
    Case("refund-by-plan",
         "What is our refund window, and does it depend on the plan?", True,
         ("policies/billing.md", "policies/refunds-enterprise.md"), "multi-document"),
    Case("unanswerable-leave", "What is our parental leave policy?", True, (),
         "should search, then report nothing found"),
    Case("arithmetic", "What is 37 multiplied by 4?", False, (), "no lookup needed"),
    Case("greeting", "Hello, what are you?", False, (), "no lookup needed"),

    # --- harder cases -------------------------------------------------------
    # The first eight saturated: every model scored 8/8. A benchmark that
    # cannot separate candidates cannot justify choosing one, so these probe
    # the failure modes routing actually has -- vocabulary mismatch between the
    # question and the corpus, questions that merely look like lookups, and
    # comparisons that genuinely need two different searches.

    # Corpus says "chargeback"; the question never uses the word. Naive query
    # formulation (echoing the question) retrieves the wrong section.
    Case("vocab-chargeback",
         "A buyer went to their bank to reverse a payment instead of contacting us. "
         "What happens to their account?", True,
         ("policies/billing.md",), "vocabulary mismatch"),

    # Corpus says "expand and contract"; question uses none of those words.
    Case("vocab-migration",
         "How do we ship a database change that the currently running code would "
         "choke on?", True,
         ("engineering/database-migrations.md",), "vocabulary mismatch"),

    # General knowledge dressed up in company phrasing. Searching is waste.
    Case("general-knowledge",
         "In general, what does the acronym HMAC stand for?", False, (),
         "looks like a lookup, is not"),

    # The answer is contained in the question itself.
    Case("self-answering",
         "If a runbook says to wait five minutes before failing over, how long "
         "should I wait?", False, (), "answer is in the question"),

    # Two unrelated areas; one query cannot cover both.
    Case("cross-domain",
         "Compare how long we keep audit logs with how long we keep query text.",
         True,
         ("engineering/data-retention.md", "policies/privacy.md"),
         "multi-document"),

    # Requires connecting a symptom to a cause across two documents.
    Case("symptom-to-cause",
         "Our webhook signatures started failing after we added emoji support. "
         "Why, and what is the retry schedule we are now hitting?", True,
         ("engineering/webhooks.md",), "multi-hop within a document"),

    # Plausible-sounding but absent; the near-miss is topically adjacent.
    Case("unanswerable-sso",
         "Do we support SAML single sign-on?", True, (),
         "should search, then report nothing found"),

    # Terse, underspecified. Tests whether it asks the corpus something useful.
    Case("terse", "token lifetime?", True,
         ("engineering/authentication.md",), "terse query formulation"),
]


@dataclass
class Result:
    case: Case
    searched: bool = False
    queries: list[str] = field(default_factory=list)
    documents: set[str] = field(default_factory=set)
    turns: int = 0
    latency_ms: float = 0.0
    tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    final: str = ""
    error: str | None = None
    rate_limit_waits: int = 0

    @property
    def decision_correct(self) -> bool:
        return self.searched == self.case.expects_search


async def run_case(client, model, case, retriever, tenant) -> Result:
    result = Result(case=case)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM,
        tools=[SEARCH_TOOL],
        temperature=0.0,
        max_output_tokens=1024,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        http_options=types.HttpOptions(timeout=60000),
    )
    contents: list[types.Content] = [
        types.Content(role="user", parts=[types.Part(text=case.question)])
    ]

    started = time.perf_counter()
    turn = 0
    waited_ms = 0.0
    while turn < MAX_TURNS:
        try:
            response = await asyncio.to_thread(
                client.models.generate_content, model=model, contents=contents, config=config
            )
        except errors.APIError as exc:
            if getattr(exc, "code", None) == 429:
                # A quota wait is not a reasoning turn. Counting it against the
                # turn budget makes a rate-limited model look like one that
                # refuses to use its tools -- which is exactly how the first run
                # scored gemini-3.5-flash 0/8 with zero tokens spent.
                delay = 20.0
                match = re.search(r"'retryDelay':\s*'(\d+)s'", str(exc))
                if match:
                    delay = float(match.group(1)) + 2
                result.rate_limit_waits += 1
                waited_ms += delay * 1000
                await asyncio.sleep(delay)
                continue
            result.error = f"{getattr(exc, 'code', None)}: {str(exc)[:80]}"
            break

        if response.usage_metadata:
            meta = response.usage_metadata
            result.tokens += meta.total_token_count or 0
            result.input_tokens += meta.prompt_token_count or 0
            # Thinking tokens are billed as output and are invisible in the
            # text, so folding them in is required for the cost to be honest.
            result.output_tokens += (meta.candidates_token_count or 0) + (
                meta.thoughts_token_count or 0
            )

        candidate = (response.candidates or [None])[0]
        if candidate is None or candidate.content is None:
            result.error = "empty candidate"
            break

        turn += 1
        result.turns = turn
        parts = candidate.content.parts or []
        calls = [p.function_call for p in parts if getattr(p, "function_call", None)]

        if not calls:
            result.final = "".join(p.text or "" for p in parts if getattr(p, "text", None))
            break

        contents.append(candidate.content)
        for call in calls:
            query = (call.args or {}).get("query", "")
            result.searched = True
            result.queries.append(str(query))

            found = await retriever.retrieve(tenant, str(query), top_k=5)
            for chunk in found.chunks:
                result.documents.add(chunk.document_external_id)
            payload = [
                {
                    "document": c.document_external_id,
                    "section": " / ".join(c.heading_path),
                    "text": c.text[:700],
                }
                for c in found.chunks
            ]
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_function_response(
                            name=call.name, response={"results": payload}
                        )
                    ],
                )
            )

    # Report latency net of quota waiting, so the number describes the model
    # rather than the free tier.
    result.latency_ms = (time.perf_counter() - started) * 1000 - waited_ms
    return result


async def main() -> None:
    settings = get_settings()
    client = genai.Client(api_key=settings.gemini_api_key)

    db = Database(settings.database_url)
    await db.open()
    try:
        async with db.transaction() as conn:
            tenant = await repo.ensure_tenant(conn, settings.default_tenant_slug)
        retriever = Retriever(db, get_embedder(settings), settings)

        for model in CANDIDATES:
            print("=" * 96)
            print(f"agent model: {model}")
            print("=" * 96)
            results = []
            for case in CASES:
                res = await run_case(client, model, case, retriever, tenant)
                results.append(res)
                mark = "ok " if res.decision_correct else "MISS"
                expect = "search" if case.expects_search else "no search"
                got = f"{len(res.queries)} search(es)" if res.searched else "no search"
                print(f"  [{mark}] {case.id:<20} expect {expect:<10} got {got:<14} "
                      f"turns={res.turns} {res.latency_ms:>6.0f}ms tok={res.tokens}")
                if res.queries:
                    for q in res.queries:
                        print(f"           query: {q[:78]}")
                if res.case.wants_documents:
                    hit = set(res.case.wants_documents) & res.documents
                    print(f"           needed docs reached: "
                          f"{len(hit)}/{len(res.case.wants_documents)}")
                if res.error:
                    print(f"           ERROR {res.error}")

            ok = sum(r.decision_correct for r in results)
            multi = [r for r in results if r.case.note == "multi-document"]
            multi_searched = sum(1 for r in multi if len(r.queries) > 1)
            covered = sum(
                1 for r in multi
                if set(r.case.wants_documents).issubset(r.documents)
            )
            unnecessary = sum(1 for r in results if r.searched and not r.case.expects_search)
            errored = sum(1 for r in results if r.error)
            waits = sum(r.rate_limit_waits for r in results)

            print(f"\n  tool-selection accuracy : {ok}/{len(results)}")
            print(f"  unnecessary searches    : {unnecessary}")
            print(f"  multi-doc cases issuing >1 query : {multi_searched}/{len(multi)}")
            print(f"  multi-doc cases reaching all needed docs : {covered}/{len(multi)}")
            print(f"  mean latency (net of quota waits) : "
                  f"{sum(r.latency_ms for r in results) / len(results):.0f}ms")
            in_tok = sum(r.input_tokens for r in results)
            out_tok = sum(r.output_tokens for r in results)
            in_price, out_price = PRICING.get(model, (0.0, 0.0))
            cost_per_case = (in_tok * in_price + out_tok * out_price) / 1e6 / len(results)
            print(f"  total tokens            : {sum(r.tokens for r in results)} "
                  f"(in {in_tok} / out {out_tok})")
            print(f"  cost per 1000 questions : ${cost_per_case * 1000:.2f}")
            print(f"  errored cases           : {errored}")
            print(f"  rate-limit waits        : {waits} (excluded from latency)\n")
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
