# Privacy and Data Handling

## What we store

Customer documents, their extracted text, and the derived embeddings. Query text
is stored for 30 days to support debugging and is then deleted.

## Data residency

Data is stored in the region selected at account creation. The region cannot be
changed after creation; moving regions requires creating a new account and
re-ingesting. This is a deliberate constraint, not a limitation we intend to
lift.

## Subprocessor changes

Customers are notified at least 30 days before a new subprocessor begins
processing customer data. Objections are handled through the account team.

## Deletion requests

A verified deletion request is completed within 30 days. Deletion removes
documents, extracted text and embeddings. Backups are not selectively edited;
deleted data ages out of backups on the normal backup retention schedule.

## Employee access

Engineers do not have standing access to customer document content. Access is
granted per-incident, requires a second approver, and is logged to the audit
trail.
