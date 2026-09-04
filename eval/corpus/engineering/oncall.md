# On-Call

## Rotation

One primary and one secondary, rotating weekly, handing over on Wednesday
mornings. Wednesday rather than Monday so that a handover never lands next to a
weekend.

## Paging thresholds

An alert pages only if a human must act within the hour. Everything else is a
ticket. An alert that pages and is routinely acknowledged without action is a
bug in the alert, and is fixed rather than tolerated.

## Escalation

If the primary does not acknowledge within 10 minutes the page escalates to the
secondary. After a further 10 minutes it escalates to the engineering manager.

## During an incident

The primary owns coordination, not necessarily the fix. Their first job is to
decide whether to mitigate or to diagnose; mitigation comes first when customer
traffic is affected.

## After an incident

A written review is produced within five working days. Reviews are blameless and
focus on the conditions that allowed the failure, not on the individual who
triggered it. Action items have owners and dates or they are not action items.
