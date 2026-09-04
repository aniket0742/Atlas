# Feature Flags

## Lifecycle

New behaviour ships behind a flag defaulted to off, is enabled for internal
tenants, then for a percentage of external tenants, then fully. The flag is
removed within two releases of reaching full rollout.

## Naming

Flags are named `feature.<area>.<behaviour>`, for example
`feature.retrieval.hybrid_search`. The prefix makes it possible to find every
flag owned by a team.

## Evaluation order

Flags evaluate in order: explicit tenant override, then percentage rollout, then
default. A tenant override always wins, which is how a customer can be excluded
from a rollout that is causing them trouble.

## Stale flags

A flag older than 90 days without a rollout change is reported to its owning
team weekly. Long-lived flags are the main source of untested code paths,
because the combination of flag states is never exercised together.

## Flags are not configuration

A flag answers "is this behaviour on". A value that needs tuning is
configuration and belongs in an environment variable. Using flags for values
produces a flag system that nobody can reason about.
