import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from mlsystem.security.cloudflare_access import AccessVerifier, came_through_cloudflare

TEAM = "example-team.cloudflareaccess.com"
AUD = "0" * 64


class FakeJWKS:
    """Stands in for PyJWKClient: always returns the test public key."""

    def __init__(self, public_key):
        self.public_key = public_key

    def get_signing_key_from_jwt(self, token):
        return type("Key", (), {"key": self.public_key})()


@pytest.fixture(scope="module")
def keys():
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private, private.public_key()


def token(private, **overrides):
    now = int(time.time())
    claims = {"aud": [AUD], "iss": f"https://{TEAM}", "email": "Someone@Example.com",
              "iat": now, "exp": now + 300} | overrides
    return jwt.encode(claims, private, algorithm="RS256")


def verifier(public, **kw):
    return AccessVerifier(TEAM, AUD, jwks_client=FakeJWKS(public), **kw)


def test_valid_token_returns_lowercased_email(keys):
    private, public = keys
    assert verifier(public).verify(token(private)) == "someone@example.com"


@pytest.mark.parametrize("override", [
    {"aud": ["another-application"]},
    {"iss": "https://someone-else.cloudflareaccess.com"},
    {"exp": int(time.time()) - 3600},
])
def test_wrong_audience_issuer_or_expired_is_rejected(keys, override):
    private, public = keys
    assert verifier(public).verify(token(private, **override)) is None


def test_token_signed_by_another_key_is_rejected(keys):
    _, public = keys
    attacker = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    assert verifier(public).verify(token(attacker)) is None


def test_allow_list_is_enforced(keys):
    private, public = keys
    assert verifier(public, allowed_emails=["owner@example.com"]).verify(token(private)) is None
    assert verifier(public, allowed_emails=["someone@example.com"]).verify(token(private)) == "someone@example.com"


def test_missing_or_garbage_token_is_rejected(keys):
    _, public = keys
    assert verifier(public).verify(None) is None
    assert verifier(public).verify("not-a-jwt") is None


def test_edge_header_detection():
    assert came_through_cloudflare({"Cf-Ray": "abc"})
    assert not came_through_cloudflare({"X-Forwarded-For": "203.0.113.5"})
