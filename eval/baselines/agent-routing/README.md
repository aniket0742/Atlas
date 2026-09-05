# Agent tool-routing benchmark (frozen)

Which model decides *when* to call a tool and *what* to search for. A different
job from writing the final answer, benchmarked separately
(`eval/baselines/answer-models/`).

Produced by `scripts/validate_agent_model.py`: a real multi-turn tool-calling
loop against real retrieval over the 33-document corpus, on 16 cases whose
correct behaviour is known in advance. Raw run output in
`routing-benchmark.txt`.

## Results (16 cases)

| model | tool-selection accuracy | unnecessary searches | multi-doc: all needed docs reached | mean latency | $/1k questions |
|---|---|---|---|---|---|
| **gemini-3.1-flash-lite** | 16/16 | 0 | **4/4** | **2,568 ms** | **$0.36** |
| gemini-3.5-flash-lite | 16/16 | 0 | 3/4 | 2,724 ms | $0.58 |
| gemini-3.7-flash | 16/16 | 0 | 3/4 | 4,684 ms | $1.83 |

## The first benchmark was saturated, and that was the finding

An initial 8-case set scored **8/8 for all six candidates** including
`gemini-3.5-flash`. A benchmark that cannot separate candidates cannot justify
choosing one, so it was rebuilt around the failure modes routing actually has:

- **vocabulary mismatch** — the question says "went to their bank to reverse a
  payment"; the corpus says "chargeback". A model that echoes the question into
  the search box retrieves the wrong section.
- **looks like a lookup but is not** — "what does HMAC stand for" is general
  knowledge; searching wastes a call.
- **the answer is inside the question** — tests reflexive searching.
- **cross-domain comparison** — audit-log retention and query-text retention are
  in different documents; one query cannot cover both.
- **terse input** — "token lifetime?" tests query formulation from nearly nothing.

Selection accuracy *still* saturated at 16/16. Routing over a single tool is
genuinely easy, and that is worth stating plainly rather than manufacturing a
difference. What the harder set did surface is **document coverage**.

## What actually separated them

`gemini-3.1-flash-lite` was the only model to reach both documents on the
retention comparison, issuing two genuinely different queries:

```
gemini-3.1-flash-lite   "customer data retention policy account closure"
                        "data retention backups account closure"     -> 2/2 docs
gemini-3.5-flash-lite   "customer data retention account closure backups" -> 1/2
gemini-3.7-flash        "customer data retention account closure backup"  -> 1/2
```

The other two collapsed the comparison into one query and missed the second
document.

The unanswerable case cuts the other way on efficiency. `gemini-3.7-flash`
persisted through four searches and 10.5 s — "parental leave policy", "leave",
"time off", "benefits" — before reporting nothing found. The lite models reached
the same correct conclusion in one or two searches. More persistence did not
change the outcome; it only cost time.

## Selected: `gemini-3.1-flash-lite`

Equal selection accuracy, **better** multi-document coverage, fastest, and 5x
cheaper than `gemini-3.7-flash`. See ADR-0025.

## Caveat

16 cases, single run, one corpus. Selection accuracy is saturated, so this
benchmark can only discriminate on coverage, latency and cost. The real test is
the Phase 4 agent evaluation over the full 112-query set with the actual tool
set, which will revisit this choice against evidence this benchmark cannot
provide.
