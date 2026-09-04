# Python SDK

## Installation

    pip install atlas-client

Requires Python 3.10 or later. The SDK vendors no HTTP client of its own; it
uses `httpx` and will use an existing client if one is passed.

## Client construction

    from atlas_client import AtlasClient
    client = AtlasClient(api_key="...", base_url="https://api.example.com")

The API key is read from `ATLAS_API_KEY` if not supplied. The base URL defaults
to the production endpoint.

## Querying

    result = client.query("What is the refund window?")
    print(result.answer)
    for citation in result.citations:
        print(citation.document_title, citation.quote)

`result.refused` is True when the system declined to answer. Check it before
using `result.answer`, because a refusal is still a successful HTTP response.

## Uploading

    client.upload_document(path="policy.pdf", source="handbook")

Uploads are synchronous and block until the document is queryable.

## Retries

The SDK retries 429 and 5xx responses with exponential backoff, honouring
`Retry-After` when present. It does not retry 4xx responses other than 429,
because retrying a malformed request only wastes quota.

## Timeouts

The default request timeout is 90 seconds, chosen to exceed the server-side
generation budget. Setting it lower than the server's timeout means the client
gives up on requests the server would have completed.
