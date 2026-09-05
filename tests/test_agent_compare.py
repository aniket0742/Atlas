"""The paired plain-vs-agent evaluation harness.

No network and no database: `citation_recall` and `summarize` are pure
functions over `Answer`/`SystemOutcome` objects, and that is deliberate --
whether the scoring logic is correct must be checkable without spending API
budget, and the paid comparison run should be the only thing in this project
that costs money to verify.
"""

from __future__ import annotations

import uuid

from tests.conftest import make_chunk

from atlas.core.models import Citation
from atlas.eval.agent_compare import PairedReport, SystemOutcome, citation_recall, summarize
from atlas.eval.dataset import Label


def citation_for(chunk, quote: str | None = None) -> Citation:
    return Citation(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        document_external_id=chunk.document_external_id,
        document_title=chunk.document_title,
        document_uri=chunk.document_uri,
        char_start=chunk.char_start,
        char_end=chunk.char_end,
        quote=quote or chunk.text,
    )


# ---------------------------------------------------------------------------
# citation_recall
# ---------------------------------------------------------------------------


def test_an_unanswerable_query_is_not_scored():
    """None, not 0.0 -- a 0 would be misread as a failed answer."""
    matched, total, recall = citation_recall([], [], [])
    assert (matched, total, recall) == (0, 0, None)


def test_a_citation_pointing_at_a_satisfying_chunk_matches_its_label():
    chunk = make_chunk("Refunds within 30 days.", document_external_id="policies/billing.md")
    labels = [Label(document="policies/billing.md", contains="within 30 days")]

    matched, total, recall = citation_recall(labels, [citation_for(chunk)], [chunk])

    assert (matched, total, recall) == (1, 1, 1.0)


def test_a_retrieved_but_uncited_chunk_does_not_satisfy_the_label():
    """The point of this metric: presence in evidence is not enough."""
    satisfying = make_chunk("Refunds within 30 days.", document_external_id="policies/billing.md")
    cited_instead = make_chunk("Unrelated passage.", document_external_id="policies/billing.md")
    labels = [Label(document="policies/billing.md", contains="within 30 days")]

    matched, total, recall = citation_recall(
        labels, [citation_for(cited_instead)], [satisfying, cited_instead]
    )

    assert (matched, total, recall) == (0, 1, 0.0)


def test_the_right_document_but_missing_snippet_does_not_match():
    chunk = make_chunk("Some other clause entirely.", document_external_id="policies/billing.md")
    labels = [Label(document="policies/billing.md", contains="within 30 days")]

    matched, total, recall = citation_recall(labels, [citation_for(chunk)], [chunk])

    assert (matched, total, recall) == (0, 1, 0.0)


def test_a_multi_doc_question_needs_both_labels_cited_for_full_recall():
    a = make_chunk("Downgrade retention is 30 days.", document_external_id="policies/billing.md")
    b = make_chunk(
        "Closure retention is 90 days.", document_external_id="engineering/data-retention.md"
    )
    labels = [
        Label(document="policies/billing.md", contains="30 days"),
        Label(document="engineering/data-retention.md", contains="90 days"),
    ]

    both = citation_recall(labels, [citation_for(a), citation_for(b)], [a, b])
    only_one = citation_recall(labels, [citation_for(a)], [a, b])

    assert both == (2, 2, 1.0)
    assert only_one == (1, 2, 0.5)


def test_one_citation_can_satisfy_more_than_one_label():
    """Overlapping labels on one chunk must not be double-penalised."""
    chunk = make_chunk(
        "Refunds are 30 days. Exceptions need finance approval.",
        document_external_id="policies/billing.md",
    )
    labels = [
        Label(document="policies/billing.md", contains="30 days"),
        Label(document="policies/billing.md", contains="finance approval"),
    ]

    matched, total, recall = citation_recall(labels, [citation_for(chunk)], [chunk])

    assert (matched, total, recall) == (2, 2, 1.0)


def test_a_citation_id_absent_from_retrieved_is_ignored_not_crashing():
    """Defensive: an id that does not resolve should not raise a KeyError."""
    chunk = make_chunk("x", document_external_id="a.md")
    phantom = Citation(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_external_id="a.md",
        document_title=None,
        document_uri=None,
        char_start=0,
        char_end=1,
        quote="x",
    )
    labels = [Label(document="a.md")]

    matched, total, recall = citation_recall(labels, [phantom], [chunk])

    assert (matched, total, recall) == (0, 1, 0.0)


def test_no_citations_at_all_scores_zero_not_none():
    """A refused or uncited answer should count against recall, not be excluded."""
    labels = [Label(document="a.md")]
    matched, total, recall = citation_recall(labels, [], [])
    assert (matched, total, recall) == (0, 1, 0.0)


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


def outcome(
    *,
    refused=False,
    citation_recall=1.0,
    total_labels=1,
    unverified=0,
    tool_calls=None,
    degraded=False,
    stop_reason="finished",
    latency_ms=100.0,
    input_tokens=100,
    output_tokens=20,
    agent_prompt_tokens=None,
    agent_output_tokens=None,
    answer_prompt_tokens=None,
    answer_output_tokens=None,
) -> SystemOutcome:
    matched = round(citation_recall * total_labels) if citation_recall is not None else 0
    return SystemOutcome(
        refused=refused,
        refusal_reason="x" if refused else None,
        citation_count=1,
        unverified_quotes=unverified,
        matched_labels=matched,
        total_labels=total_labels,
        citation_recall=citation_recall,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        tool_calls=tool_calls,
        degraded=degraded,
        stop_reason=stop_reason,
        agent_prompt_tokens=agent_prompt_tokens,
        agent_output_tokens=agent_output_tokens,
        answer_prompt_tokens=answer_prompt_tokens,
        answer_output_tokens=answer_output_tokens,
    )


def report(qid, *, kind="lookup", answerable=True, plain=None, agent=None) -> PairedReport:
    return PairedReport(
        query_id=qid,
        question=f"question {qid}",
        answerable=answerable,
        kind=kind,
        plain=plain or outcome(),
        agent=agent or outcome(tool_calls=1),
    )


def test_overall_recall_is_averaged_across_answerable_queries_only():
    rows = [
        report("q1", plain=outcome(citation_recall=1.0), agent=outcome(citation_recall=1.0)),
        report("q2", plain=outcome(citation_recall=0.0), agent=outcome(citation_recall=1.0)),
        report("q3", answerable=False, plain=outcome(refused=True), agent=outcome(refused=True)),
    ]

    summary = summarize(rows)

    assert summary["overall"]["n"] == 2
    assert summary["overall"]["plain_citation_recall"]["mean"] == 0.5
    assert summary["overall"]["agent_citation_recall"]["mean"] == 1.0


def test_a_paired_delta_is_reported_when_both_systems_are_scored():
    rows = [
        report("q1", plain=outcome(citation_recall=0.0), agent=outcome(citation_recall=1.0)),
        report("q2", plain=outcome(citation_recall=0.0), agent=outcome(citation_recall=1.0)),
    ]

    summary = summarize(rows)

    delta = summary["overall"]["paired_delta_agent_minus_plain"]
    assert delta["mean"] == 1.0
    assert delta["ci95"][0] > 0, "a consistent improvement should show a CI excluding zero"


def test_refusal_correctness_is_tracked_per_system_in_both_directions():
    rows = [
        report("q1", answerable=False, plain=outcome(refused=True), agent=outcome(refused=False)),
        report("q2", answerable=True, plain=outcome(refused=False), agent=outcome(refused=True)),
    ]

    summary = summarize(rows)

    assert summary["refusal"]["plain"]["correctly_refused"] == 1
    assert summary["refusal"]["agent"]["correctly_refused"] == 0
    assert summary["refusal"]["agent"]["incorrectly_refused"] == 1
    assert summary["refusal"]["plain"]["incorrectly_refused"] == 0


def test_by_kind_breakdown_isolates_multi_doc_performance():
    rows = [
        report("q1", kind="multi-doc", plain=outcome(citation_recall=0.5)),
        report("q2", kind="lookup", plain=outcome(citation_recall=1.0)),
    ]

    summary = summarize(rows)

    assert summary["by_kind"]["multi-doc"]["n"] == 1
    assert summary["by_kind"]["multi-doc"]["plain_citation_recall"]["mean"] == 0.5
    assert summary["by_kind"]["lookup"]["plain_citation_recall"]["mean"] == 1.0


def test_agent_behaviour_reports_degraded_and_bound_hit_rates():
    rows = [
        report("q1", agent=outcome(tool_calls=2, degraded=False, stop_reason="finished")),
        report("q2", agent=outcome(tool_calls=8, degraded=False, stop_reason="max_tool_calls")),
        report("q3", agent=outcome(tool_calls=1, degraded=True, stop_reason="llm_error")),
    ]

    summary = summarize(rows)
    behaviour = summary["agent_behaviour"]

    assert behaviour["degraded_count"] == 1
    assert behaviour["bound_hit_count"] == 1
    assert behaviour["tool_calls_max"] == 8
    assert behaviour["tool_calls_mean"] == round((2 + 8 + 1) / 3, 2)


def test_token_accounting_splits_agent_and_answer_model_shares():
    rows = [
        report(
            "q1",
            plain=outcome(input_tokens=500, output_tokens=100),
            agent=outcome(
                input_tokens=900,
                output_tokens=150,
                agent_prompt_tokens=600,
                agent_output_tokens=50,
                answer_prompt_tokens=300,
                answer_output_tokens=100,
            ),
        ),
    ]

    summary = summarize(rows)
    tokens = summary["tokens"]

    assert tokens["plain"] == {"input": 500, "output": 100}
    assert tokens["agent"]["agent_model_input"] == 600
    assert tokens["agent"]["answer_model_input"] == 300
    # The split must reconstruct the combined total, or the cost figure lies.
    assert tokens["agent"]["agent_model_input"] + tokens["agent"]["answer_model_input"] == 900


def test_latency_percentiles_are_reported_per_system():
    rows = [
        report("q1", plain=outcome(latency_ms=100), agent=outcome(latency_ms=400)),
        report("q2", plain=outcome(latency_ms=200), agent=outcome(latency_ms=500)),
    ]

    summary = summarize(rows)

    assert summary["latency_ms"]["plain"]["p50"] in (100, 200)
    assert summary["latency_ms"]["agent"]["max"] == 500


def test_summarize_on_an_empty_report_list_does_not_crash():
    summary = summarize([])
    assert summary["overall"]["n"] == 0
    assert summary["agent_behaviour"]["tool_calls_mean"] == 0.0


def test_unverified_quotes_and_errors_are_summed_per_system():
    rows = [
        report("q1", plain=outcome(unverified=2), agent=outcome(unverified=0)),
        report("q2", plain=outcome(unverified=1), agent=outcome(unverified=3)),
    ]

    summary = summarize(rows)

    assert summary["unverified_quotes"]["plain"] == 3
    assert summary["unverified_quotes"]["agent"] == 3
