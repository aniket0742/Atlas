# Runbook: Certificate Expiry

## Symptom

TLS handshake failures, or monitoring warning that a certificate expires within
14 days.

## Automated renewal

Certificates renew automatically 30 days before expiry. A warning at 14 days
means automated renewal has already failed at least twice and needs a human.

## Common causes

- The DNS validation record was removed or changed
- The renewal service lost its credentials to the DNS provider
- The domain was transferred and the validation delegation was not recreated

## Manual renewal

1. Confirm the DNS validation record resolves from an external resolver.
2. Trigger renewal from the certificate manager.
3. Verify the new certificate's expiry and its chain.
4. Confirm the load balancer picked it up; some terminators cache until reload.

## Internal certificates

Internal service-to-service certificates are issued by the internal CA with a 90
day lifetime and rotate weekly. They are deliberately short-lived so that
rotation failures surface in days rather than at expiry.
