# Frozen retrieval baselines

The Phase 1 dense-retrieval numbers that Phase 2 must beat. Unlike
`eval/results/`, which is regenerable scratch output and is gitignored, these
two reports are committed as fixed reference evidence: every retrieval claim in
`docs/evaluation.md` and `README.md` traces back to one of them.

| file | k | Recall@k | MRR | nDCG@k |
|---|---|---|---|---|
| `dense-k1.json` | 1 | 0.906 | 0.938 | 0.938 |
| `dense-k2.json` | 2 | 1.000 | 0.969 | 0.977 |

Both: 19-query smoke set, `BAAI/bge-small-en-v1.5`, structure-aware chunking at
320/64 tokens, dense retrieval only. Each file embeds the full configuration
that produced it, so a later run is only comparable if that block matches.

**Compare at k=1.** k=2 is included to document where saturation begins: the
sample corpus is 17 chunks, so Recall@k pins at 1.000 from k=2 onward and cannot
discriminate between retrieval strategies. A Phase 2 result quoting Recall@5 or
Recall@8 against these files would be measuring nothing.

Regenerate with:

    atlas eval eval/datasets/smoke.jsonl --k 1 --label dense-baseline-k1

Replace these files only when the baseline itself legitimately changes (a
different embedding model or chunking strategy), and say so in `Decision.md`.
