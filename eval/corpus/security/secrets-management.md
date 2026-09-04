# Secrets Management

## Where secrets live

In the secret store. Not in environment variables in production, not in
configuration files, and never in the repository.

## Detection

A pre-commit hook and a CI job scan for credential patterns. The CI job is the
one that matters; a hook can be bypassed with a flag and will be, eventually.

## If a secret is committed

Rotate first, then remove. Rewriting history does not help: assume anything
pushed has been fetched, and any clone still holds it. Removing the secret from
history without rotating it produces a false sense of resolution.

## Rotation

Database credentials rotate every 90 days. API signing keys rotate every 180
days with an overlap window so that in-flight tokens remain valid.

## Third-party keys

Keys for third-party providers are stored per-environment. A staging key must
never be able to reach production data, which is why they are separate
credentials rather than the same credential with different configuration.
