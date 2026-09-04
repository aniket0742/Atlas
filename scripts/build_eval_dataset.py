"""Build eval/datasets/main.jsonl and validate every label against the corpus.

The expanded Phase 2 evaluation set. Questions were written from plausible user
need against the corpus as written; the `kind` tag on each was assigned
afterwards, so the mix reflects what people would ask rather than a quota
designed to flatter a particular retrieval method.

Validation is the point of this being a script rather than a hand-edited file: a
label whose snippet is not actually in the corpus scores zero forever and looks
like a retrieval failure.

Run:  python scripts/build_eval_dataset.py
"""

from __future__ import annotations

import json
import pathlib
import re
from collections import Counter

CORPUS = pathlib.Path("eval/corpus")
SMOKE = pathlib.Path("eval/datasets/smoke.jsonl")
OUT = pathlib.Path("eval/datasets/main.jsonl")


def canon(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


# (id, question, [(document, snippet)], kind)
NEW: list[tuple[str, str, list[tuple[str, str]], str]] = [
    # --- policy / support ---------------------------------------------------
    (
        "enterprise-refund-window",
        "Does the 30 day refund window apply to enterprise contracts too?",
        [("policies/refunds-enterprise.md", "within 90 days of an invoice date")],
        "distractor",
    ),
    (
        "enterprise-refund-approval",
        "Who has to sign off on a large enterprise refund?",
        [("policies/refunds-enterprise.md", "require sign-off from both the account executive")],
        "lookup",
    ),
    (
        "seat-downgrade",
        "If a customer reduces their seat count mid-contract do they get money back?",
        [("policies/refunds-enterprise.md", "does not generate a refund")],
        "paraphrase",
    ),
    (
        "enterprise-uptime",
        "What uptime do we commit to for Enterprise customers?",
        [("policies/sla.md", "99.95% for Enterprise")],
        "lookup",
    ),
    (
        "sla-credit-deadline",
        "How long does a customer have to claim an SLA credit?",
        [("policies/sla.md", "within 30 days of the end of the affected month")],
        "lookup",
    ),
    (
        "maintenance-uptime",
        "Does planned maintenance count against our uptime number?",
        [("policies/sla.md", "does not count against the availability commitment")],
        "paraphrase",
    ),
    (
        "region-change",
        "Can a customer move their data to a different region later?",
        [("policies/privacy.md", "cannot be changed after creation")],
        "paraphrase",
    ),
    (
        "query-retention",
        "How long do we keep the text of queries people run?",
        [("policies/privacy.md", "stored for 30 days")],
        "lookup",
    ),
    (
        "subprocessor-notice",
        "How much notice do customers get before a new subprocessor?",
        [("policies/privacy.md", "at least 30 days before a new subprocessor")],
        "lookup",
    ),
    (
        "engineer-doc-access",
        "Can our engineers read customer documents?",
        [("policies/privacy.md", "do not have standing access")],
        "paraphrase",
    ),
    (
        "console-scraping",
        "Is it OK for a customer to drive the web console with a headless browser?",
        [("policies/acceptable-use.md", "Scraping the web console with headless browsers is not")],
        "paraphrase",
    ),
    (
        "aup-enforcement",
        "Do we suspend an account immediately when someone breaks the usage rules?",
        [("policies/acceptable-use.md", "first response to a violation is always contact")],
        "paraphrase",
    ),
    (
        "tenant-capacity-share",
        "How much of a shared cluster can one customer use?",
        [("policies/acceptable-use.md", "more than 40% of a shared cluster")],
        "lookup",
    ),
    # --- API / identifiers --------------------------------------------------
    (
        "atl-4022",
        "What does error ATL-4022 mean?",
        [("engineering/api-errors.md", "Document parsed to no extractable text")],
        "identifier",
    ),
    (
        "atl-5002-vs-5004",
        "What is the difference between ATL-5002 and ATL-5004?",
        [("engineering/api-errors.md", "ATL-5002 means the provider answered")],
        "identifier",
    ),
    (
        "unsupported-type-code",
        "Which error code comes back when we cannot parse a document type?",
        [("engineering/api-errors.md", "ATL-4015")],
        "identifier",
    ),
    (
        "ratelimit-reset-header",
        "Which response header tells me when my rate limit window resets?",
        [("engineering/rate-limits.md", "X-Atlas-RateLimit-Reset")],
        "identifier",
    ),
    (
        "worker-concurrency-default",
        "What is the default for ATLAS_WORKER_CONCURRENCY?",
        [("engineering/environments.md", "default 4")],
        "identifier",
    ),
    (
        "topk-env-var",
        "Which environment variable sets how many chunks we retrieve?",
        [("engineering/environments.md", "ATLAS_RETRIEVAL_TOP_K")],
        "identifier",
    ),
    (
        "request-id-header",
        "Which header carries the request trace id?",
        [("engineering/observability.md", "X-Atlas-Request-Id")],
        "identifier",
    ),
    (
        "hybrid-flag-name",
        "What is the feature flag for hybrid search called?",
        [("engineering/feature-flags.md", "feature.retrieval.hybrid_search")],
        "identifier",
    ),
    (
        "ratelimit-per-what",
        "Does issuing more API tokens give a customer more throughput?",
        [("engineering/rate-limits.md", "Limits are applied per tenant, not per API token")],
        "paraphrase",
    ),
    (
        "retry-after",
        "Should a client use its own backoff or the value we send back?",
        [("engineering/rate-limits.md", "should honour")],
        "paraphrase",
    ),
    # --- conceptual / engineering ------------------------------------------
    (
        "why-floor",
        "Why do we discard low scoring chunks instead of passing them to the model?",
        [("engineering/search-architecture.md", "still produces k confident-looking chunks")],
        "conceptual",
    ),
    (
        "cache-key-tenant",
        "What goes wrong if a cache key leaves out the tenant?",
        [("engineering/caching.md", "serve one customer's results to another")],
        "conceptual",
    ),
    (
        "cache-retrieval-config",
        "Why does the retrieval cache key include the configuration?",
        [("engineering/caching.md", "serves stale results computed under the old settings")],
        "conceptual",
    ),
    (
        "answers-not-cached",
        "Do we cache generated answers?",
        [("engineering/caching.md", "Generated answers are not cached")],
        "lookup",
    ),
    (
        "refusal-label",
        "Why do we break refusals down by reason instead of counting them together?",
        [("engineering/observability.md", "throws away the distinction")],
        "conceptual",
    ),
    (
        "log-content",
        "Is it OK to log the text of a chunk when debugging?",
        [("engineering/observability.md", "never logged")],
        "paraphrase",
    ),
    (
        "column-default-lock",
        "What is the risk of adding a column with a default to a large table?",
        [("engineering/database-migrations.md", "rewrites the table")],
        "conceptual",
    ),
    (
        "expand-contract",
        "How do we make a schema change that would break the running code?",
        [("engineering/database-migrations.md", "Expand and contract")],
        "paraphrase",
    ),
    (
        "index-concurrently",
        "How should we add an index to a big table without blocking writes?",
        [("engineering/database-migrations.md", "CREATE INDEX CONCURRENTLY")],
        "identifier",
    ),
    (
        "hnsw-filter-gap",
        "Why might a filtered vector search return fewer results than we asked for?",
        [
            (
                "engineering/search-architecture.md",
                "searched before tenant and source predicates are applied",
            )
        ],
        "conceptual",
    ),
    (
        "chunk-section-boundary",
        "Do chunks ever cross a heading boundary?",
        [("engineering/search-architecture.md", "Chunks never span a section heading")],
        "lookup",
    ),
    (
        "flags-vs-config",
        "Should a tunable number be a feature flag?",
        [("engineering/feature-flags.md", "belongs in an environment variable")],
        "paraphrase",
    ),
    (
        "stale-flags",
        "Why do we chase feature flags that have been around a long time?",
        [("engineering/feature-flags.md", "main source of untested code paths")],
        "conceptual",
    ),
    (
        "secrets-not-env",
        "Are secrets passed to production as environment variables?",
        [("engineering/environments.md", "never environment variables in production")],
        "paraphrase",
    ),
    (
        "prod-config-edit",
        "Can I just change an env var on the production host?",
        [("engineering/environments.md", "is not permitted")],
        "paraphrase",
    ),
    # --- runbooks -----------------------------------------------------------
    (
        "queue-first-checks",
        "The ingestion queue is backing up, what do I check first?",
        [("runbooks/runbook-queue-backlog.md", "Are workers alive?")],
        "paraphrase",
    ),
    (
        "poison-message",
        "One document keeps failing and retrying, what do I do with it?",
        [("runbooks/runbook-queue-backlog.md", "move it to the dead letter queue by hand")],
        "paraphrase",
    ),
    (
        "concurrency-vs-workers",
        "Should I raise worker concurrency or add more worker processes?",
        [("runbooks/runbook-queue-backlog.md", "before adding worker processes")],
        "lookup",
    ),
    (
        "queue-shedding",
        "Does a backed up ingestion queue slow down search?",
        [("runbooks/runbook-queue-backlog.md", "Query traffic is unaffected")],
        "paraphrase",
    ),
    (
        "when-failover",
        "When should we actually fail the database over?",
        [("runbooks/runbook-database-failover.md", "unreachable for more than five")],
        "lookup",
    ),
    (
        "failover-data-loss",
        "How much data can we lose if we fail over?",
        [("runbooks/runbook-database-failover.md", "typically under 2 seconds")],
        "lookup",
    ),
    (
        "old-primary",
        "Can we bring the old primary back after a failover?",
        [("runbooks/runbook-database-failover.md", "without being rebuilt")],
        "paraphrase",
    ),
    (
        "cert-14-days",
        "We got a certificate expiry warning at 14 days, what does that tell us?",
        [("runbooks/runbook-cert-expiry.md", "automated renewal has already failed")],
        "conceptual",
    ),
    (
        "internal-cert-life",
        "How long do internal service certificates last?",
        [("runbooks/runbook-cert-expiry.md", "90 day lifetime")],
        "lookup",
    ),
    (
        "sev1-status-page",
        "How fast do we have to put something on the status page for a SEV1?",
        [("runbooks/incident-response.md", "within 15 minutes")],
        "lookup",
    ),
    (
        "incident-resolved",
        "When do we call an incident resolved?",
        [("runbooks/incident-response.md", "when customer impact has ended")],
        "paraphrase",
    ),
    (
        "who-declares",
        "Who is allowed to declare an incident?",
        [("runbooks/incident-response.md", "Anyone may declare an incident")],
        "lookup",
    ),
    (
        "commander-debugging",
        "Should the incident commander be debugging?",
        [("runbooks/incident-response.md", "decides, delegates, does not debug")],
        "paraphrase",
    ),
    # --- webhooks / SDK / integrations -------------------------------------
    (
        "webhook-delivery",
        "Are webhooks delivered exactly once?",
        [("engineering/webhooks.md", "at-least-once")],
        "lookup",
    ),
    (
        "webhook-retries",
        "What is the retry schedule for a failed webhook?",
        [("engineering/webhooks.md", "1, 5, 25 and 125 minutes")],
        "lookup",
    ),
    (
        "webhook-signature",
        "Why does my webhook signature check keep failing?",
        [("engineering/webhooks.md", "not a re-serialised JSON object")],
        "conceptual",
    ),
    (
        "webhook-timeout",
        "How long does my endpoint have to respond to a webhook?",
        [("engineering/webhooks.md", "does not return a 2xx within 10 seconds")],
        "lookup",
    ),
    (
        "github-issues",
        "Does the GitHub connector pull in issues and pull requests?",
        [("product/integrations.md", "does not index issues or pull requests")],
        "lookup",
    ),
    (
        "github-app",
        "Why do we use a GitHub App rather than a personal access token?",
        [("product/integrations.md", "survives the departure")],
        "conceptual",
    ),
    (
        "incremental-sync",
        "What makes an hourly sync affordable?",
        [("product/integrations.md", "does no work and costs nothing")],
        "conceptual",
    ),
    (
        "sdk-python-version",
        "What Python version does the SDK need?",
        [("engineering/sdk-python.md", "Python 3.10 or later")],
        "lookup",
    ),
    (
        "sdk-retries",
        "Does the SDK retry rate limited requests on its own?",
        [("engineering/sdk-python.md", "retries 429 and 5xx")],
        "lookup",
    ),
    (
        "sdk-refused",
        "How do I tell from the SDK that the system declined to answer?",
        [("engineering/sdk-python.md", "is still a successful HTTP response")],
        "paraphrase",
    ),
    # --- access / security --------------------------------------------------
    (
        "prod-db-access",
        "What do I need to get access to the production database?",
        [("onboarding/access-requests.md", "second approver from the security")],
        "lookup",
    ),
    (
        "elevated-access-duration",
        "How long does elevated access last before I have to ask again?",
        [("onboarding/access-requests.md", "expires after 8 hours")],
        "lookup",
    ),
    (
        "breakglass",
        "Am I in trouble for using break-glass access?",
        [("onboarding/access-requests.md", "using it is never itself a problem")],
        "paraphrase",
    ),
    (
        "access-in-chat",
        "Can I just ask someone in chat to grant me access?",
        [("onboarding/access-requests.md", "never by asking someone")],
        "paraphrase",
    ),
    (
        "committed-secret",
        "I committed a secret by accident, what is the order of operations?",
        [("security/secrets-management.md", "Rotate first, then remove")],
        "paraphrase",
    ),
    (
        "history-rewrite",
        "Does rewriting git history fix a leaked credential?",
        [("security/secrets-management.md", "assume anything pushed has been fetched")],
        "conceptual",
    ),
    (
        "db-cred-rotation",
        "How often do database credentials rotate?",
        [("security/secrets-management.md", "every 90 days")],
        "lookup",
    ),
    (
        "dos-testing",
        "Is denial of service testing allowed under our disclosure policy?",
        [("security/vulnerability-disclosure.md", "Denial of service testing against production")],
        "lookup",
    ),
    (
        "critical-fix-target",
        "How quickly do we have to fix a critical reported vulnerability?",
        [("security/vulnerability-disclosure.md", "7 days")],
        "lookup",
    ),
    (
        "pentest-high",
        "What was the high severity finding in the last penetration test?",
        [("security/pentest-2026.md", "cache key omitted the tenant identifier")],
        "lookup",
    ),
    # --- onboarding ---------------------------------------------------------
    (
        "first-week-goal",
        "What is a new engineer supposed to achieve in their first week?",
        [("onboarding/new-engineer.md", "one merged change")],
        "paraphrase",
    ),
    (
        "shadow-rotations",
        "How many rotations do you shadow before going on call?",
        [("onboarding/new-engineer.md", "two full rotations")],
        "lookup",
    ),
    (
        "mentor-not-manager",
        "Why is the onboarding mentor not the person's manager?",
        [("onboarding/new-engineer.md", "carries no performance implication")],
        "conceptual",
    ),
    (
        "port-5432",
        "Docker compose fails because port 5432 is already in use, why?",
        [("onboarding/development-setup.md", "a local PostgreSQL install is bound to it")],
        "paraphrase",
    ),
    (
        "model-redownload",
        "The embedding model downloads again every time I rebuild, what did I do wrong?",
        [("onboarding/development-setup.md", "The model volume was removed")],
        "paraphrase",
    ),
    # --- release notes / versions ------------------------------------------
    (
        "token-lifetime-release",
        "Which release cut the access token lifetime?",
        [("product/release-notes-2026-q1.md", "reduced from 60 minutes to 15 minutes")],
        "identifier",
    ),
    (
        "search-endpoint-removed",
        "When was the old /v1/search endpoint actually removed?",
        [("product/release-notes-2026-q2.md", "deprecated in v2.4.0")],
        "identifier",
    ),
    (
        "ingestion-202",
        "What changed about document ingestion in v2.5.0?",
        [("product/release-notes-2026-q2.md", "returns 202")],
        "identifier",
    ),
    (
        "windows-backslash-fix",
        "There was a bug about filenames with backslashes, which release fixed it?",
        [("product/release-notes-2026-q1.md", "contained a backslash created a")],
        "identifier",
    ),
    # --- multi document -----------------------------------------------------
    (
        "revocation-and-release",
        "A revoked session still worked for a while. Is that by design, and did any release change it?",  # noqa: E501 (question text; wrapping hurts readability)
        [
            ("engineering/authentication.md", "remains usable until it expires"),
            ("product/release-notes-2026-q1.md", "revoked session could survive up to 15 minutes"),
        ],
        "multi-doc",
    ),
    (
        "refund-window-by-plan",
        "What is our refund window, and does it depend on the plan?",
        [
            ("policies/billing.md", "within 30 days"),
            ("policies/refunds-enterprise.md", "within 90 days of an invoice date"),
        ],
        "multi-doc",
    ),
    (
        "retention-privacy-vs-eng",
        "How long is customer data kept after an account closes, and how does that interact with backups?",  # noqa: E501 (question text; wrapping hurts readability)
        [
            ("engineering/data-retention.md", "After account closure"),
            ("policies/privacy.md", "Backups are not selectively edited"),
        ],
        "multi-doc",
    ),
    (
        "ratelimit-error-and-header",
        "A customer is getting 429s. Which error code is that and what should they do?",
        [("engineering/api-errors.md", "ATL-4029"), ("engineering/rate-limits.md", "Retry-After")],
        "multi-doc",
    ),
    # --- unanswerable -------------------------------------------------------
    ("un-parental-leave", "What is our parental leave policy?", [], "unanswerable"),
    ("un-cloud-provider", "Which cloud provider do we run on?", [], "unanswerable"),
    (
        "un-max-doc-size",
        "What is the maximum document size a customer can upload?",
        [],
        "unanswerable",
    ),
    ("un-bug-bounty", "Do we pay cash rewards for reported vulnerabilities?", [], "unanswerable"),
    ("un-customer-count", "How many customers do we have?", [], "unanswerable"),
    (
        "un-oncall-pay",
        "How much extra do engineers get paid for being on call?",
        [],
        "unanswerable",
    ),
    (
        "un-contract-notice",
        "How much notice is required to terminate a contract?",
        [],
        "unanswerable",
    ),
    ("un-password-reset", "How does a user reset their password?", [], "unanswerable"),
    ("un-free-trial", "Do we offer a free trial?", [], "unanswerable"),
]

# The 19 inherited smoke queries predate this taxonomy. Classified here on the
# same scale so a per-kind breakdown covers the whole set; their original prose
# notes are left untouched.
SMOKE_KINDS = {
    "refund-window": "lookup",
    "refund-settle-time": "lookup",
    "annual-proration": "paraphrase",
    "failed-payment-grace": "paraphrase",
    "chargeback-consequence": "paraphrase",
    "access-token-lifetime": "lookup",
    "refresh-token-reuse": "lookup",
    "revocation-delay": "paraphrase",
    "service-account-scope": "lookup",
    "rollback-after-migration": "paraphrase",
    "release-days": "lookup",
    "audit-log-retention": "lookup",
    "retention-after-closure": "distractor",
    "priority-response-target": "lookup",
    "escalation-criteria": "lookup",
    "downgrade-and-closure-retention": "multi-doc",
    "unanswerable-vacation": "unanswerable",
    "unanswerable-pricing": "unanswerable",
    "unanswerable-sso": "unanswerable",
}

HEADER = """\
// Atlas main evaluation set.
//
// Built by scripts/build_eval_dataset.py -- edit that, not this file, so every
// label stays validated against the corpus.
//
// Supersedes smoke.jsonl for measurement. smoke.jsonl is retained as a fast
// regression set and its 19 queries are included here unchanged.
//
// Questions were written from plausible user need against the corpus as
// written. The `notes` field records the query kind, assigned AFTER the
// questions existed, so the mix is not a quota designed to favour any
// particular retrieval method.
//
// `kind` is the classification used for per-kind metric breakdowns. `notes`,
// where present, is free prose inherited from smoke.jsonl.
"""


def build() -> list[dict]:
    existing = [
        json.loads(line)
        for line in SMOKE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("//")
    ]
    records = list(existing)
    for qid, question, labels, kind in NEW:
        record: dict = {"id": qid, "question": question, "answerable": bool(labels)}
        if labels:
            record["labels"] = [{"document": d, "contains": s} for d, s in labels]
        record["kind"] = kind
        records.append(record)

    # Inherited queries carry prose in `notes`; give them a `kind` too.
    for record in records:
        record.setdefault("kind", SMOKE_KINDS.get(record["id"], "unclassified"))
    return records


def validate(records: list[dict]) -> list[str]:
    problems: list[str] = []
    seen: set[str] = set()
    for record in records:
        if record["id"] in seen:
            problems.append(f"duplicate id {record['id']}")
        seen.add(record["id"])
        for label in record.get("labels", []):
            path = CORPUS / label["document"]
            if not path.exists():
                problems.append(f"{record['id']}: missing file {label['document']}")
                continue
            snippet = label.get("contains")
            if snippet and canon(snippet) not in canon(path.read_text(encoding="utf-8")):
                problems.append(f"{record['id']}: snippet not in {label['document']}: {snippet!r}")
    return problems


def main() -> int:
    records = build()
    problems = validate(records)
    print(f"records: {len(records)}  ({len(records) - len(NEW)} existing + {len(NEW)} new)")

    if problems:
        print(f"\n*** {len(problems)} PROBLEM(S) ***")
        for problem in problems:
            print("   ", problem)
        return 1

    print("all labels validated against the corpus")
    OUT.write_text(
        HEADER + "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT}")

    print("\nquery kinds:")
    for kind, count in Counter(r["kind"] for r in records).most_common():
        print(f"  {kind:<28} {count}")
    print(f"\nanswerable   {sum(r['answerable'] for r in records)}")
    print(f"unanswerable {sum(not r['answerable'] for r in records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
