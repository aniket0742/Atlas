# Release Notes — 2026 Q1

## v2.3.0 — 14 January 2026

**Added.** Source-scoped queries. A query may now be restricted to named
sources, so a support agent can search policy documents without engineering
documentation competing for the top results.

**Added.** `document.failed` webhook event, carrying the parse error.

**Changed.** Default retrieval depth raised from 5 to 8 chunks.

**Fixed.** PDF page numbers were off by one for documents whose first page had
no extractable text.

## v2.3.1 — 29 January 2026

**Fixed.** Uploading a document whose filename contained a backslash created a
duplicate rather than updating the existing document on Windows clients.

## v2.4.0 — 3 March 2026

**Added.** Per-tenant rate limit headers on every response.

**Changed.** Access token lifetime reduced from 60 minutes to 15 minutes.
Refresh tokens now rotate on every use, and reusing a spent refresh token
revokes the whole token family.

**Deprecated.** The `/v1/search` endpoint. Use `/v1/query`. Removal no earlier
than v3.0.

## v2.4.1 — 20 March 2026

**Fixed.** Session revocation did not invalidate the token family in cache, so a
revoked session could survive up to 15 minutes. Revocation is now immediate for
refresh, though access tokens remain valid until expiry by design.
