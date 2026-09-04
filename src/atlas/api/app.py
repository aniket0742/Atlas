"""FastAPI application.

Phase 1 has no authentication. Every request is attributed to a single
configured tenant, resolved by a dependency that Phase 5 replaces with one that
reads the caller's identity from a token. Because every downstream call already
takes a tenant_id, that swap is one function -- which is the entire reason the
tenant plumbing exists this early.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from atlas.answer.service import AnswerService
from atlas.api import schemas
from atlas.config import Settings, get_settings
from atlas.core.models import Answer
from atlas.db import repository as repo
from atlas.db.pool import Database
from atlas.ingest.parsers import UnparseableDocument, UnsupportedDocument
from atlas.ingest.pipeline import Ingestor, IngestRequest
from atlas.providers.base import LLMError, LLMTimeout
from atlas.providers.factory import get_embedder, get_llm, get_reranker
from atlas.retrieval.service import Retriever

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )

    db = Database(
        settings.database_url,
        min_size=settings.db_pool_min_size,
        max_size=settings.db_pool_max_size,
    )
    await db.open()

    embedder = get_embedder(settings)
    llm = get_llm(settings)
    reranker = get_reranker(settings)
    retriever = Retriever(db, embedder, settings, reranker=reranker)

    app.state.settings = settings
    app.state.db = db
    app.state.embedder = embedder
    app.state.ingestor = Ingestor(db, embedder, settings)
    app.state.retriever = retriever
    app.state.answerer = AnswerService(db, retriever, llm, settings)
    app.state.llm = llm
    app.state.reranker = reranker

    # Resolve (and create on first boot) the tenant every request is attributed
    # to, so request handling never has to branch on "does the tenant exist".
    async with db.transaction() as conn:
        app.state.tenant_id = await repo.ensure_tenant(conn, settings.default_tenant_slug)

    logger.info(
        "atlas ready embedding=%s llm=%s retrieval=%s rerank=%s tenant=%s",
        embedder.model_id,
        llm.model_id,
        settings.retrieval_mode,
        reranker.model_id if reranker else "off",
        app.state.tenant_id,
    )
    try:
        yield
    finally:
        await db.close()


app = FastAPI(
    title="Atlas",
    description="AI knowledge and retrieval platform",
    version="0.1.0",
    lifespan=lifespan,
)

# The inspection console is served by this app rather than by a separate
# frontend service. Same-origin means no CORS configuration and no second port,
# and it keeps the UI a plain client of the documented API -- it cannot show
# anything the API does not already return.
_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def console() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


# --------------------------------------------------------------------------
# Dependencies
# --------------------------------------------------------------------------


def get_db(request: Request) -> Database:
    return request.app.state.db


def get_config(request: Request) -> Settings:
    return request.app.state.settings


def current_tenant(request: Request) -> uuid.UUID:
    """The tenant this request acts on behalf of.

    Phase 1: always the configured default. Phase 5: derived from the caller's
    credentials. Nothing downstream changes.
    """
    return request.app.state.tenant_id


DbDep = Annotated[Database, Depends(get_db)]
TenantDep = Annotated[uuid.UUID, Depends(current_tenant)]


# --------------------------------------------------------------------------
# Error handling
# --------------------------------------------------------------------------


@app.exception_handler(UnsupportedDocument)
async def _unsupported(_: Request, exc: UnsupportedDocument) -> JSONResponse:
    return JSONResponse(status_code=415, content={"detail": str(exc)})


@app.exception_handler(UnparseableDocument)
async def _unparseable(_: Request, exc: UnparseableDocument) -> JSONResponse:
    # 422 rather than 500: the request was well-formed, the document was not.
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(LLMTimeout)
async def _llm_timeout(_: Request, exc: LLMTimeout) -> JSONResponse:
    return JSONResponse(status_code=504, content={"detail": f"Model timed out: {exc}"})


@app.exception_handler(LLMError)
async def _llm_error(_: Request, exc: LLMError) -> JSONResponse:
    # 502: Atlas is healthy, its upstream model provider is not.
    return JSONResponse(status_code=502, content={"detail": f"Model call failed: {exc}"})


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@app.get("/health", response_model=schemas.HealthResponse, tags=["ops"])
async def health(request: Request, db: DbDep) -> schemas.HealthResponse:
    database_ok = await db.healthy()
    return schemas.HealthResponse(
        status="ok" if database_ok else "degraded",
        database=database_ok,
        embedding_model=request.app.state.embedder.model_id,
        llm_model=request.app.state.llm.model_id,
        retrieval_mode=request.app.state.settings.retrieval_mode,
        rerank_model=(
            request.app.state.reranker.model_id if request.app.state.reranker else None
        ),
    )


@app.get("/v1/stats", response_model=schemas.StatsResponse, tags=["ops"])
async def stats(db: DbDep, tenant_id: TenantDep) -> schemas.StatsResponse:
    async with db.connection() as conn:
        row = await repo.corpus_stats(conn, tenant_id)
    return schemas.StatsResponse(**row)


@app.get("/v1/sources", response_model=list[schemas.SourceSummary], tags=["sources"])
async def list_sources(db: DbDep, tenant_id: TenantDep) -> list[schemas.SourceSummary]:
    async with db.connection() as conn:
        rows = await repo.list_sources(conn, tenant_id)
    return [schemas.SourceSummary(**row) for row in rows]


@app.post(
    "/v1/documents",
    response_model=schemas.IngestResponse,
    status_code=201,
    tags=["documents"],
    responses={
        415: {"model": schemas.ErrorResponse, "description": "Unsupported document type"},
        422: {"model": schemas.ErrorResponse, "description": "Document could not be parsed"},
    },
)
async def ingest_document(
    request: Request,
    tenant_id: TenantDep,
    settings: Annotated[Settings, Depends(get_config)],
    file: Annotated[UploadFile, File(description="The document to index.")],
    source: Annotated[str, Form()] = "default",
    external_id: Annotated[str | None, Form()] = None,
    uri: Annotated[str | None, Form()] = None,
) -> schemas.IngestResponse:
    """Index a document.

    Synchronous in Phase 1: the response is returned after the document is
    queryable. Phase 3 replaces this with an enqueue that returns 202 and a job
    id. Both are honest about what has happened, which matters more than which
    one is used.
    """
    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Document is {len(data)} bytes, limit is {settings.max_upload_bytes}.",
        )
    if not data:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    ingestor: Ingestor = request.app.state.ingestor
    result = await ingestor.ingest(
        tenant_id,
        IngestRequest(
            data=data,
            # Default the stable id to the filename: re-uploading the same
            # filename updates that document rather than creating a duplicate.
            external_id=external_id or file.filename or "untitled",
            source_name=source,
            filename=file.filename,
            mime_type=file.content_type,
            uri=uri,
        ),
    )
    return schemas.IngestResponse(
        document_id=result.document_id,
        version=result.version,
        chunk_count=result.chunk_count,
        changed=result.changed,
    )


@app.get("/v1/documents", response_model=list[schemas.DocumentSummary], tags=["documents"])
async def list_documents(
    db: DbDep,
    tenant_id: TenantDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[schemas.DocumentSummary]:
    async with db.connection() as conn:
        rows = await repo.list_documents(conn, tenant_id, limit=limit, offset=offset)
    return [schemas.DocumentSummary(**row) for row in rows]


@app.get(
    "/v1/documents/{document_id}",
    response_model=schemas.DocumentDetail,
    tags=["documents"],
    responses={404: {"model": schemas.ErrorResponse}},
)
async def get_document(
    document_id: uuid.UUID, db: DbDep, tenant_id: TenantDep
) -> schemas.DocumentDetail:
    async with db.connection() as conn:
        row = await repo.get_document(conn, tenant_id, document_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return schemas.DocumentDetail(**row)


@app.post(
    "/v1/query",
    response_model=schemas.QueryResponse,
    tags=["query"],
    responses={
        502: {"model": schemas.ErrorResponse, "description": "Model provider failed"},
        504: {"model": schemas.ErrorResponse, "description": "Model provider timed out"},
    },
)
async def query(
    request: Request, body: schemas.QueryRequest, tenant_id: TenantDep
) -> schemas.QueryResponse:
    answerer: AnswerService = request.app.state.answerer
    result = await answerer.answer(
        tenant_id,
        body.question,
        top_k=body.top_k,
        min_similarity=body.min_similarity,
        source_ids=body.source_ids,
        mode=body.mode,
        rerank=body.rerank,
    )
    return _to_query_response(result, include_evidence=body.include_evidence)


def _to_query_response(result: Answer, *, include_evidence: bool) -> schemas.QueryResponse:
    return schemas.QueryResponse(
        answer=result.text,
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        retrieval=schemas.RetrievalInfo(
            mode=result.retrieval_mode,
            reranked=result.reranked,
            best_dense_score=result.best_dense_score,
            candidates_per_component=result.per_component,
        ),
        citations=[
            schemas.CitationOut(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                document_external_id=c.document_external_id,
                document_title=c.document_title,
                document_uri=c.document_uri,
                page=c.page,
                char_start=c.char_start,
                char_end=c.char_end,
                quote=c.quote,
                quote_verified=c.quote_verified,
            )
            for c in result.citations
        ],
        evidence=(
            [
                schemas.EvidenceOut(
                    chunk_id=e.chunk_id,
                    document_id=e.document_id,
                    document_external_id=e.document_external_id,
                    document_title=e.document_title,
                    source_name=e.source_name,
                    heading_path=e.heading_path,
                    score=e.score,
                    component_scores=e.component_scores,
                    text=e.text,
                )
                for e in result.retrieved
            ]
            if include_evidence
            else None
        ),
        usage=schemas.UsageOut(
            prompt_tokens=result.usage.prompt_tokens,
            output_tokens=result.usage.output_tokens,
            thinking_tokens=result.usage.thinking_tokens,
            total_tokens=result.usage.total_tokens,
        ),
        timings_ms={k: round(v, 2) for k, v in result.timings_ms.items()},
    )
