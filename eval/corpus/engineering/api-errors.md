# API Error Codes

Every error response carries a stable `code` field. The HTTP status tells a
client how to react; the code tells a human which condition occurred.

## Client errors

| Code | HTTP | Meaning |
|---|---|---|
| ATL-4001 | 400 | Request body failed schema validation |
| ATL-4004 | 404 | Document or source not found in this tenant |
| ATL-4013 | 403 | Token valid but lacks the required scope |
| ATL-4015 | 415 | Document type is not one Atlas can parse |
| ATL-4022 | 422 | Document parsed to no extractable text |
| ATL-4029 | 429 | Rate limit exceeded, see the retry headers |

## Server and upstream errors

| Code | HTTP | Meaning |
|---|---|---|
| ATL-5002 | 502 | The model provider returned an error |
| ATL-5004 | 504 | The model provider exceeded its timeout budget |
| ATL-5031 | 503 | Ingestion queue is saturated and shedding load |

## ATL-4022 in practice

ATL-4022 is most often a scanned PDF with no text layer. Atlas does not perform
OCR, so the document must be OCRed before upload. This is deliberate: silently
indexing an empty document is worse than refusing it, because nothing downstream
can detect the omission.

## ATL-5002 versus ATL-5004

ATL-5002 means the provider answered and the answer was an error. ATL-5004 means
the provider did not answer in time. The second is retryable at the same
settings; the first usually is not.
