# Incident Response

## Severity levels

| Severity | Definition | Response |
|---|---|---|
| SEV1 | Complete outage or data loss risk | Page immediately, all hands |
| SEV2 | Major feature unavailable, no workaround | Page primary |
| SEV3 | Degraded performance or a workaround exists | Ticket, next business day |

## Declaring

Anyone may declare an incident. Over-declaring is cheap and under-declaring is
expensive, so err towards declaring.

## Roles

- Incident commander — decides, delegates, does not debug
- Communications lead — customer and internal updates
- Subject matter experts — investigate and fix

For SEV3 one person holds all three roles. Separating them matters at SEV1,
where the commander debugging is the most common way an incident drifts.

## Customer communication

The first status page update goes out within 15 minutes of declaring a SEV1,
even if it only says the problem is confirmed and being investigated. Silence is
read as absence, and customers escalate to fill it.

## Resolution

An incident is resolved when customer impact has ended, not when the root cause
is understood. Root cause work continues afterwards under a normal ticket.
