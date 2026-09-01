# Deployments

## Release cadence

The platform ships on Tuesdays and Thursdays. Hotfixes ship on any day with
approval from the on-call engineer and one reviewer.

## Rollback

Every deployment records the previous image digest. Rollback redeploys that
digest and is expected to complete within 10 minutes. Database migrations are
forward-only, so a rollback after a migration requires a compensating migration
rather than reverting the schema.

## Feature flags

New behaviour ships behind a flag defaulted to off. Flags are removed within two
releases of reaching full rollout.
