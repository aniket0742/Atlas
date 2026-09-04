/* Atlas inspection console.
 *
 * Plain browser JS against the same-origin API. No framework and no build step,
 * so this file is also the whole client: what it can display is exactly what
 * /v1/query, /v1/documents and /v1/stats already return.
 *
 * All rendering goes through textContent, never innerHTML with response data.
 * Document titles and chunk text come from uploaded files, so they are
 * untrusted input; building markup by concatenation would make stored XSS a
 * one-line mistake.
 */

"use strict";

const $ = (id) => document.getElementById(id);

const el = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
};

async function readError(response) {
  try {
    const body = await response.json();
    if (body && body.detail) {
      return typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail, null, 2);
    }
    return JSON.stringify(body, null, 2);
  } catch {
    return `${response.status} ${response.statusText}`;
  }
}

/* ── health ─────────────────────────────────────────────────────────── */

async function loadHealth() {
  const box = $("health");
  const dot = box.querySelector(".dot");
  const text = box.querySelector(".health-text");
  try {
    const response = await fetch("/health");
    const data = await response.json();
    dot.className = `dot ${data.database ? "dot-ok" : "dot-bad"}`;
    text.textContent = `db ${data.database ? "up" : "down"} · ${data.llm_model} · ${data.embedding_model}`;
  } catch {
    dot.className = "dot dot-bad";
    text.textContent = "api unreachable";
  }
}

/* ── corpus ─────────────────────────────────────────────────────────── */

async function loadCorpus() {
  const stats = $("stats");
  const docs = $("documents");
  try {
    const [statsResponse, docsResponse] = await Promise.all([
      fetch("/v1/stats"),
      fetch("/v1/documents?limit=50"),
    ]);
    const s = await statsResponse.json();
    const documents = await docsResponse.json();

    stats.replaceChildren();
    for (const [label, value] of [
      ["documents", s.documents],
      ["indexed", s.indexed_documents],
      ["failed", s.failed_documents],
      ["chunks", s.chunks],
      ["embeddings", s.embeddings],
      ["sources", s.sources],
    ]) {
      stats.append(el("dt", null, label), el("dd", null, value));
    }

    docs.replaceChildren();
    if (!documents.length) {
      docs.append(el("li", "muted", "no documents indexed"));
      return;
    }
    for (const d of documents) {
      const row = el("li");
      row.append(
        el("span", `status status-${d.status}`, d.status),
        el("span", "doc-name", d.external_id),
      );
      docs.append(row);
    }
  } catch {
    stats.replaceChildren(el("dd", "muted", "unavailable"));
    docs.replaceChildren(el("li", "muted", "unavailable"));
  }
}

/* ── ask ────────────────────────────────────────────────────────────── */

function renderAnswer(data) {
  const badge = $("answer-badge");
  badge.textContent = data.refused ? "refused" : "answered";
  badge.className = `badge ${data.refused ? "badge-refused" : "badge-ok"}`;

  const reason = $("refusal-reason");
  if (data.refused && data.refusal_reason) {
    reason.textContent = `refusal_reason: ${data.refusal_reason}`;
    reason.hidden = false;
  } else {
    reason.hidden = true;
  }

  $("answer-text").textContent = data.answer;
  $("answer-panel").hidden = false;
}

function renderCitations(citations) {
  const list = $("citations");
  list.replaceChildren();
  $("citation-count").textContent = `${citations.length}`;

  if (!citations.length) {
    list.append(el("li", "muted", "none — nothing in the answer is attributed to a retrieved chunk"));
    $("citations-panel").hidden = false;
    return;
  }

  for (const c of citations) {
    const item = el("li");
    item.append(el("p", "cite-quote", `“${c.quote}”`));

    const meta = el("div", "cite-meta");
    meta.append(
      el("span", c.quote_verified ? "tag tag-ok" : "tag tag-warn",
        c.quote_verified ? "verbatim" : "not verbatim"),
      el("span", "tag", c.document_external_id),
    );
    if (c.document_title) meta.append(el("span", null, c.document_title));
    if (c.page !== null && c.page !== undefined) meta.append(el("span", null, `p.${c.page}`));
    meta.append(el("span", null, `chars ${c.char_start}–${c.char_end}`));

    item.append(meta);
    list.append(item);
  }
  $("citations-panel").hidden = false;
}

function renderEvidence(evidence) {
  const list = $("evidence");
  list.replaceChildren();
  $("evidence-count").textContent = `${evidence.length}`;

  if (!evidence.length) {
    list.append(el("li", "muted", "nothing passed the similarity floor — the model was never called"));
    $("evidence-panel").hidden = false;
    return;
  }

  // The bar is normalised against this result set, not against an absolute
  // scale. `score` means cosine in dense mode, a small RRF sum in hybrid mode,
  // and an unbounded cross-encoder logit after reranking -- three incomparable
  // scales, none of which map onto a 0-100% bar on its own.
  const values = evidence.map((c) => c.score);
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo || 1;

  evidence.forEach((chunk, index) => {
    const item = el("li");

    const head = el("div", "ev-head");
    head.append(el("span", "ev-rank", `#${index + 1}`));
    head.append(el("span", null, chunk.document_external_id || chunk.document_title || ""));
    if (chunk.heading_path && chunk.heading_path.length) {
      head.append(el("span", null, `· ${chunk.heading_path.join(" / ")}`));
    }
    head.append(el("span", "ev-score", chunk.score.toFixed(4)));
    item.append(head);

    const bar = el("div", "bar");
    const fill = el("span");
    // Relative within this list: the top result is always full width, so the
    // bar shows the gap between results rather than an absolute score.
    fill.style.width = `${(15 + 85 * ((chunk.score - lo) / span)).toFixed(1)}%`;
    bar.append(fill);
    item.append(bar);

    // Every component that contributed, so a rank can be explained rather than
    // just displayed. This is the panel that shows hybrid retrieval working.
    const cs = chunk.component_scores || {};
    const chips = el("div", "ev-components");
    const add = (label, value, cls) => {
      const chip = el("span", cls ? `chip ${cls}` : "chip");
      chip.append(el("b", null, label), document.createTextNode(` ${value}`));
      chips.append(chip);
    };
    if (cs.dense !== undefined) {
      add("dense", cs.dense.toFixed(4) + (cs.dense_rank ? ` #${cs.dense_rank}` : ""), "chip-dense");
    }
    if (cs.lexical !== undefined) {
      add("lex", cs.lexical.toFixed(4) + (cs.lexical_rank ? ` #${cs.lexical_rank}` : ""), "chip-lexical");
    }
    if (cs.rrf !== undefined) add("rrf", cs.rrf.toFixed(5));
    if (cs.rerank !== undefined) add("rerank", cs.rerank.toFixed(3), "chip-rerank");
    if (chips.childElementCount) item.append(chips);

    const text = el("p", "ev-text", chunk.text);
    item.append(text);

    const toggle = el("button", "ev-toggle", "show full chunk");
    toggle.type = "button";
    toggle.addEventListener("click", () => {
      const open = text.classList.toggle("open");
      toggle.textContent = open ? "collapse" : "show full chunk";
    });
    item.append(toggle);

    list.append(item);
  });
  $("evidence-panel").hidden = false;
}

function renderTimings(data) {
  const info = $("retrieval-info");
  info.replaceChildren();
  const r = data.retrieval;
  if (r) {
    const chip = (label, value, cls) => {
      const c = el("span", cls ? `chip ${cls}` : "chip");
      c.append(el("b", null, label), document.createTextNode(` ${value}`));
      info.append(c);
    };
    chip("mode", r.mode, "chip-dense");
    chip("rerank", r.reranked ? "on" : "off", r.reranked ? "chip-rerank" : "");
    if (r.best_dense_score !== null && r.best_dense_score !== undefined) {
      chip("best dense", r.best_dense_score.toFixed(4), "chip-lexical");
    }
    for (const [component, n] of Object.entries(r.candidates_per_component || {})) {
      chip(component, `${n} cand`);
    }
  }

  const list = $("timings");
  list.replaceChildren();

  const order = [
    "embed_query_ms", "dense_search_ms", "lexical_search_ms",
    "fuse_ms", "rerank_ms", "llm_ms", "total_ms",
  ];
  const timings = data.timings_ms || {};
  for (const key of order) {
    if (timings[key] === undefined) continue;
    list.append(el("dt", null, key.replace(/_ms$/, "")), el("dd", null, `${Math.round(timings[key])} ms`));
  }
  for (const [key, value] of Object.entries(timings)) {
    if (order.includes(key)) continue;
    list.append(el("dt", null, key.replace(/_ms$/, "")), el("dd", null, `${Math.round(value)} ms`));
  }

  const u = data.usage || {};
  list.append(el("dt", null, "prompt tok"), el("dd", null, u.prompt_tokens ?? 0));
  list.append(el("dt", null, "output tok"), el("dd", null, u.output_tokens ?? 0));
  if (u.thinking_tokens) list.append(el("dt", null, "thinking tok"), el("dd", null, u.thinking_tokens));
  list.append(el("dt", null, "total tok"), el("dd", null, u.total_tokens ?? 0));

  $("metrics-panel").hidden = false;
}

function hideResults() {
  for (const id of ["answer-panel", "citations-panel", "evidence-panel", "metrics-panel", "error"]) {
    $(id).hidden = true;
  }
}

async function ask(event) {
  event.preventDefault();
  const button = $("ask-btn");
  const question = $("question").value.trim();
  if (!question) return;

  hideResults();
  button.disabled = true;
  button.textContent = "Asking…";

  const payload = { question, include_evidence: true };
  const topK = $("top-k").value;
  const minSim = $("min-sim").value;
  const mode = $("mode").value;
  if (topK !== "") payload.top_k = Number(topK);
  if (minSim !== "") payload.min_similarity = Number(minSim);
  if (mode !== "") payload.mode = mode;
  if ($("rerank").checked) payload.rerank = true;

  try {
    const response = await fetch("/v1/query", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      $("error-body").textContent = `HTTP ${response.status}\n\n${await readError(response)}`;
      $("error").hidden = false;
      return;
    }
    const data = await response.json();
    renderAnswer(data);
    renderCitations(data.citations || []);
    renderEvidence(data.evidence || []);
    renderTimings(data);
  } catch (err) {
    $("error-body").textContent = `Could not reach the API.\n\n${err}`;
    $("error").hidden = false;
  } finally {
    button.disabled = false;
    button.textContent = "Ask";
  }
}

/* ── upload ─────────────────────────────────────────────────────────── */

async function upload(event) {
  event.preventDefault();
  const button = $("upload-btn");
  const input = $("file");
  const status = $("upload-status");
  const file = input.files[0];
  if (!file) return;

  // Ingestion is synchronous: parse, chunk and embed all happen inside this
  // request. Saying so beats a spinner that looks like it has hung.
  status.className = "upload-status upload-working";
  status.replaceChildren(
    el("span", "spinner"),
    el("span", null, `indexing ${file.name} — parsing, chunking and embedding…`),
  );
  status.hidden = false;
  button.disabled = true;

  const form = new FormData();
  form.append("file", file);
  form.append("source", $("source").value || "default");

  try {
    const response = await fetch("/v1/documents", { method: "POST", body: form });
    if (!response.ok) {
      status.className = "upload-status upload-bad";
      status.textContent = `HTTP ${response.status}: ${await readError(response)}`;
      return;
    }
    const data = await response.json();
    status.className = "upload-status upload-ok";
    status.textContent = data.changed
      ? `indexed v${data.version} — ${data.chunk_count} chunks`
      : "unchanged — content hash matched, indexing skipped";
    input.value = "";
    await loadCorpus();
  } catch (err) {
    status.className = "upload-status upload-bad";
    status.textContent = `upload failed: ${err}`;
  } finally {
    button.disabled = false;
  }
}

/* ── init ───────────────────────────────────────────────────────────── */

$("ask-form").addEventListener("submit", ask);
$("upload-form").addEventListener("submit", upload);
loadHealth();
loadCorpus();
