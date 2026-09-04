"""Atlas command line.

Exists so that the system can be driven without the HTTP API: migrations, bulk
ingestion of a directory, one-off queries, and eval runs. The eval harness in
particular has to be runnable without a server, because it sweeps configuration
and a running server has fixed configuration.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Annotated

import typer

from atlas.config import get_settings

app = typer.Typer(add_completion=False, help="Atlas - AI knowledge and retrieval platform")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)s %(message)s",
    )


@app.command()
def migrate(
    status_only: Annotated[
        bool, typer.Option("--status", help="Show status, apply nothing.")
    ] = False,
) -> None:
    """Apply pending database migrations."""
    from atlas.db import migrate as migrations

    settings = get_settings()

    async def run() -> None:
        if status_only:
            for version, applied in await migrations.status(settings.database_url):
                mark = "applied" if applied else "PENDING"
                typer.echo(f"  {mark:>8}  {version}")
            return
        applied = await migrations.apply_all(settings.database_url)
        if applied:
            for version in applied:
                typer.echo(f"applied {version}")
        else:
            typer.echo("database is up to date")

    asyncio.run(run())


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(help="File or directory to index.")],
    source: Annotated[str, typer.Option(help="Source name to file documents under.")] = "default",
    pattern: Annotated[str, typer.Option(help="Glob used when PATH is a directory.")] = "**/*",
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Index a file or a directory tree."""
    _setup_logging(verbose)
    from atlas.db import repository as repo
    from atlas.db.pool import Database
    from atlas.ingest.parsers import UnparseableDocument, UnsupportedDocument
    from atlas.ingest.pipeline import Ingestor, IngestRequest
    from atlas.providers.factory import get_embedder

    settings = get_settings()

    if path.is_dir():
        files = sorted(p for p in path.glob(pattern) if p.is_file())
    elif path.is_file():
        files = [path]
    else:
        raise typer.BadParameter(f"{path} does not exist")

    if not files:
        typer.echo(f"no files matched {pattern!r} under {path}")
        raise typer.Exit(code=1)

    async def run() -> None:
        db = Database(settings.database_url)
        await db.open()
        try:
            async with db.transaction() as conn:
                tenant_id = await repo.ensure_tenant(conn, settings.default_tenant_slug)

            ingestor = Ingestor(db, get_embedder(settings), settings)
            indexed = skipped = failed = 0

            for file_path in files:
                # Relative path as the stable id: re-running over the same tree
                # updates documents in place instead of duplicating them.
                #
                # as_posix() rather than str(): on Windows str() yields
                # "policies\billing.md" while every other platform yields
                # "policies/billing.md". Since the external id is the document's
                # stable identity, that would make the same corpus produce
                # different ids per OS -- and would silently break eval labels,
                # which are written with forward slashes.
                try:
                    external_id = (
                        file_path.relative_to(path).as_posix() if path.is_dir() else file_path.name
                    )
                except ValueError:
                    external_id = file_path.name
                try:
                    result = await ingestor.ingest(
                        tenant_id,
                        IngestRequest(
                            data=file_path.read_bytes(),
                            external_id=external_id,
                            source_name=source,
                            filename=file_path.name,
                            uri=file_path.resolve().as_uri(),
                        ),
                    )
                except (UnsupportedDocument, UnparseableDocument) as exc:
                    typer.echo(f"  skip  {external_id}: {exc}")
                    skipped += 1
                    continue
                except Exception as exc:
                    typer.echo(f"  FAIL  {external_id}: {exc}")
                    failed += 1
                    continue

                if result.changed:
                    typer.echo(
                        f"  ok    {external_id} v{result.version} "
                        f"({result.chunk_count} chunks)"
                    )
                    indexed += 1
                else:
                    typer.echo(f"  same  {external_id} (unchanged)")
                    skipped += 1

            typer.echo(f"\nindexed={indexed} skipped={skipped} failed={failed}")
            if failed:
                raise typer.Exit(code=1)
        finally:
            await db.close()

    asyncio.run(run())


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="The question to ask.")],
    top_k: Annotated[int | None, typer.Option(help="Chunks to retrieve.")] = None,
    show_evidence: Annotated[
        bool, typer.Option("--evidence", help="Print retrieved chunks.")
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Emit the raw response.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Ask a question against the indexed corpus."""
    _setup_logging(verbose)
    from atlas.answer.service import AnswerService
    from atlas.db import repository as repo
    from atlas.db.pool import Database
    from atlas.providers.factory import get_embedder, get_llm
    from atlas.retrieval.service import Retriever

    settings = get_settings()

    async def run() -> None:
        db = Database(settings.database_url)
        await db.open()
        try:
            async with db.transaction() as conn:
                tenant_id = await repo.ensure_tenant(conn, settings.default_tenant_slug)

            embedder = get_embedder(settings)
            retriever = Retriever(db, embedder, settings)
            service = AnswerService(db, retriever, get_llm(settings), settings)
            result = await service.answer(tenant_id, question, top_k=top_k)

            if as_json:
                typer.echo(
                    json.dumps(
                        {
                            "answer": result.text,
                            "refused": result.refused,
                            "refusal_reason": result.refusal_reason,
                            "citations": [
                                {
                                    "chunk_id": str(c.chunk_id),
                                    "document_title": c.document_title,
                                    "page": c.page,
                                    "quote": c.quote,
                                    "quote_verified": c.quote_verified,
                                }
                                for c in result.citations
                            ],
                            "usage": {
                                "prompt_tokens": result.usage.prompt_tokens,
                                "output_tokens": result.usage.output_tokens,
                                "total_tokens": result.usage.total_tokens,
                            },
                            "timings_ms": result.timings_ms,
                        },
                        indent=2,
                    )
                )
                return

            typer.echo(f"\n{result.text}\n")
            if result.refused:
                typer.echo(f"[refused: {result.refusal_reason}]")
            for i, citation in enumerate(result.citations, 1):
                where = citation.document_title or str(citation.document_id)
                if citation.page:
                    where += f" p.{citation.page}"
                flag = "" if citation.quote_verified else "  (quote not verbatim)"
                typer.echo(f"  [{i}] {where}{flag}")
                typer.echo(f"      \"{citation.quote[:160].strip()}\"")

            if show_evidence:
                typer.echo("\nretrieved:")
                for chunk in result.retrieved:
                    path = " > ".join(chunk.heading_path) or "-"
                    typer.echo(f"  {chunk.score:.3f}  {chunk.document_title} :: {path}")

            typer.echo(
                f"\ntokens: {result.usage.total_tokens}  "
                + "  ".join(f"{k}={v:.0f}ms" for k, v in result.timings_ms.items())
            )
        finally:
            await db.close()

    asyncio.run(run())


@app.command()
def stats() -> None:
    """Show corpus counts."""
    from atlas.db import repository as repo
    from atlas.db.pool import Database

    settings = get_settings()

    async def run() -> None:
        db = Database(settings.database_url)
        await db.open()
        try:
            async with db.transaction() as conn:
                tenant_id = await repo.ensure_tenant(conn, settings.default_tenant_slug)
                row = await repo.corpus_stats(conn, tenant_id)
            for key, value in row.items():
                typer.echo(f"  {key:<20} {value}")
        finally:
            await db.close()

    asyncio.run(run())


@app.command()
def models() -> None:
    """List Gemini models this API key can actually reach.

    Free-tier availability is not reliably documented per model, so resolve it
    against the key rather than trusting a table.
    """
    from atlas.providers.gemini import list_available_models

    settings = get_settings()
    if not settings.gemini_api_key:
        typer.echo("GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey")
        raise typer.Exit(code=1)

    for name in list_available_models(settings.gemini_api_key):
        marker = " <- ATLAS_LLM_MODEL" if name == settings.llm_model else ""
        typer.echo(f"  {name}{marker}")


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: Annotated[bool, typer.Option("--reload")] = False,
) -> None:
    """Run the HTTP API."""
    import uvicorn

    uvicorn.run("atlas.api.app:app", host=host, port=port, reload=reload)


@app.command(name="eval")
def run_eval(
    dataset: Annotated[Path, typer.Argument(help="JSONL eval set.")] = Path(
        "eval/datasets/smoke.jsonl"
    ),
    k: Annotated[int | None, typer.Option(help="Retrieval depth to score at.")] = None,
    with_answers: Annotated[
        bool, typer.Option("--with-answers", help="Also generate answers (uses LLM quota).")
    ] = False,
    label: Annotated[str | None, typer.Option(help="Name for this run in the report.")] = None,
    out: Annotated[Path, typer.Option(help="Directory for report JSON.")] = Path("eval/results"),
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Score retrieval (and optionally answering) against a labelled dataset."""
    _setup_logging(verbose)
    from atlas.answer.service import AnswerService
    from atlas.db import repository as repo
    from atlas.db.pool import Database
    from atlas.eval.runner import EvalRunner, write_report
    from atlas.providers.factory import get_embedder, get_llm
    from atlas.retrieval.service import Retriever

    settings = get_settings()

    async def run() -> None:
        db = Database(settings.database_url)
        await db.open()
        try:
            async with db.transaction() as conn:
                tenant_id = await repo.ensure_tenant(conn, settings.default_tenant_slug)

            retriever = Retriever(db, get_embedder(settings), settings)
            answerer = (
                AnswerService(db, retriever, get_llm(settings), settings) if with_answers else None
            )
            report = await EvalRunner(retriever, settings, answerer).run(
                tenant_id, dataset, k=k, with_answers=with_answers, label=label
            )

            summary = report["summary"]
            typer.echo("")
            typer.echo(
                f"dataset: {report['dataset']['queries']} queries "
                f"({report['dataset']['answerable']} answerable) "
                f"k={report['config']['k']}"
            )
            typer.echo(f"embedding: {report['config']['embedding_model']}")
            typer.echo("")
            for name in ("recall_at_k", "precision_at_k", "mrr", "ndcg_at_k"):
                entry = summary[name]
                low, high = entry["ci95"]
                typer.echo(f"  {name:<16} {entry['mean']:.4f}   95% CI [{low:.4f}, {high:.4f}]")

            if "refusal" in summary:
                r = summary["refusal"]
                typer.echo("")
                typer.echo(
                    f"  refusal          "
                    f"{r['correctly_refused']}/{r['unanswerable_queries']} "
                    f"unanswerable refused, {r['incorrectly_refused']} answerable wrongly refused"
                )
                c = summary["citations"]
                typer.echo(
                    f"  citations        {c['answers_with_citations']}/{c['answers_scored']} "
                    f"answers cited, {c['unverified_quotes']} unverified quotes"
                )
                typer.echo(f"  tokens           {summary['tokens']['total']}")

            path = write_report(report, out)
            typer.echo("")
            typer.echo(f"report written to {path}")
            typer.echo(
                "note: with a small dataset the confidence intervals are wide; "
                "treat overlapping intervals as no measured difference."
            )
        finally:
            await db.close()

    asyncio.run(run())


@app.command()
def worker(
    once: Annotated[
        bool, typer.Option("--once", help="Drain the queue and exit, for scripts and tests.")
    ] = False,
    concurrency: Annotated[int | None, typer.Option(help="Jobs run in parallel.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run an ingestion worker."""
    _setup_logging(verbose)
    import signal

    from atlas.db.pool import Database
    from atlas.ingest.pipeline import Ingestor
    from atlas.ingest.worker import Worker
    from atlas.providers.factory import get_embedder

    settings = get_settings()
    if concurrency is not None:
        settings = settings.model_copy(update={"worker_concurrency": concurrency})

    async def run() -> None:
        db = Database(settings.database_url)
        await db.open()
        try:
            worker_ = Worker(db, Ingestor(db, get_embedder(settings), settings), settings)

            if once:
                processed = 0
                while await worker_.run_once():
                    processed += 1
                typer.echo(f"processed {processed} job(s)")
                return

            stop = asyncio.Event()

            # Graceful shutdown: stop claiming new work, let the job in flight
            # finish. Killing mid-job is survivable -- the lease expires and the
            # reaper requeues it -- but it wastes the work already done.
            #
            # loop.add_signal_handler is not implemented on Windows, so fall back
            # to signal.signal there. SIGTERM is what `docker stop` sends.
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, stop.set)
                except NotImplementedError:
                    signal.signal(sig, lambda *_: stop.set())

            await worker_.run_forever(stop)
        finally:
            await db.close()

    asyncio.run(run())


@app.command()
def jobs(
    status: Annotated[
        str | None, typer.Option(help="Filter: pending/running/succeeded/dead.")
    ] = None,
    limit: Annotated[int, typer.Option()] = 20,
) -> None:
    """Show the ingestion queue."""
    from atlas.db import jobs as jobq
    from atlas.db import repository as repo
    from atlas.db.pool import Database

    settings = get_settings()

    async def run() -> None:
        db = Database(settings.database_url)
        await db.open()
        try:
            async with db.transaction() as conn:
                tenant_id = await repo.ensure_tenant(conn, settings.default_tenant_slug)
                stats = await jobq.queue_stats(conn, tenant_id)
                rows = await jobq.list_jobs(conn, tenant_id, status=status, limit=limit)

            typer.echo(
                f"pending={stats['pending']} running={stats['running']} "
                f"succeeded={stats['succeeded']} dead={stats['dead']} "
                f"oldest_pending={stats['oldest_pending_seconds']:.0f}s"
            )
            typer.echo("")
            for row in rows:
                line = (
                    f"  {row['status']:<10} {str(row['id'])[:8]}  "
                    f"attempt {row['attempts']}/{row['max_attempts']}  {row['external_id']}"
                )
                typer.echo(line)
                if row["last_error"]:
                    typer.echo(f"             {row['last_error'][:100]}")
        finally:
            await db.close()

    asyncio.run(run())


if __name__ == "__main__":
    app()
