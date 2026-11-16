"""
Verify Cloudflare Access login tokens at the origin.

When an app sits behind Cloudflare Access, every request that passed the login
carries a signed JWT in the ``Cf-Access-Jwt-Assertion`` header. Trusting the
request just because it arrived is not enough: a client that reaches the origin
another way could send any headers it likes. This module checks the token itself:
RS256 signature against the team's published keys, plus audience, issuer, expiry
and (optionally) an allow-list of emails.

Usage:
    verifier = AccessVerifier(team_domain="<team>.cloudflareaccess.com", audience="<application AUD tag>")
    email = verifier.verify(request.headers.get("Cf-Access-Jwt-Assertion"))
    if email is None:
        ...  # refuse with 403
"""
from __future__ import annotations

import logging
from typing import Iterable, Mapping, Optional

import jwt

log = logging.getLogger(__name__)

TOKEN_HEADER = "Cf-Access-Jwt-Assertion"
# Headers the Cloudflare edge always adds; their presence means "this came through Cloudflare".
EDGE_HEADERS = (TOKEN_HEADER, "Cf-Ray", "Cf-Connecting-Ip")


def came_through_cloudflare(headers: Mapping[str, str]) -> bool:
    """True when the request carries headers that only the Cloudflare edge adds.

    A direct client adding them only locks itself out of any non-Cloudflare login path.
    """
    return any(headers.get(h) for h in EDGE_HEADERS)


class AccessVerifier:
    """Verifies Cloudflare Access JWTs for one Access application."""

    def __init__(
        self,
        team_domain: str,
        audience: str,
        allowed_emails: Optional[Iterable[str]] = None,
        *,
        leeway_seconds: int = 30,
        jwks_client: Optional[jwt.PyJWKClient] = None,
    ) -> None:
        if not team_domain or not audience:
            raise ValueError("team_domain and audience are required")
        self.issuer = f"https://{team_domain.strip().strip('/')}"
        self.audience = audience.strip()
        self.allowed_emails = {e.strip().lower() for e in (allowed_emails or []) if e.strip()}
        self.leeway_seconds = leeway_seconds
        # Keys are fetched once and cached; Cloudflare rotates them periodically.
        self._jwks = jwks_client or jwt.PyJWKClient(f"{self.issuer}/cdn-cgi/access/certs", cache_keys=True, timeout=10)

    def verify(self, token: Optional[str]) -> Optional[str]:
        """Return the verified email for ``token``, or None if it is missing or invalid."""
        if not token:
            return None
        try:
            key = self._jwks.get_signing_key_from_jwt(token).key
            claims = jwt.decode(
                token,
                key,
                algorithms=["RS256"],
                audience=self.audience,
                issuer=self.issuer,
                leeway=self.leeway_seconds,
                options={"require": ["exp", "iat", "aud", "iss"]},
            )
        except Exception as exc:  # bad signature, expired, wrong audience/issuer, key fetch failure
            log.warning("Access token rejected: %s", exc)
            return None

        email = str(claims.get("email") or "").strip().lower()
        if not email:
            return None
        if self.allowed_emails and email not in self.allowed_emails:
            log.warning("Access token for a non-allowed identity")
            return None
        return email

    def verify_headers(self, headers: Mapping[str, str]) -> Optional[str]:
        """Convenience wrapper: verify the token carried in request headers."""
        return self.verify(headers.get(TOKEN_HEADER))
