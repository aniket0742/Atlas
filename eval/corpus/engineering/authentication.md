# Authentication

## Overview

The platform authenticates API clients with bearer tokens issued by the auth
service. Tokens are signed with an asymmetric key and validated at the edge.

## Token lifetime

Access tokens are valid for 15 minutes. Refresh tokens are valid for 30 days and
are rotated on every use; presenting a previously used refresh token revokes the
entire token family, which is how token theft is detected.

## Session revocation

Revoking a session marks the token family as invalid in Redis. Because access
tokens are validated statelessly at the edge, a revoked access token remains
usable until it expires, up to 15 minutes.

## Service accounts

Service accounts authenticate with a client credentials grant. They are scoped
to a single tenant and cannot be granted cross-tenant permissions.
