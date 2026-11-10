"""Negative security tests: requests a real OpenID provider must reject.

These tests ensure the provider does not let through requests that violate
OAuth 2.0 / OpenID Connect security requirements, so that clients can
verify their own error handling against it:

* unregistered ``redirect_uri`` values must never be redirected back to
  (open redirect), while other authorization request errors must redirect
  with an ``error`` parameter per RFC 6749 §4.1.2.1,
* authorization codes are single-use and bound to the client,
  ``redirect_uri`` and ``nonce`` (RFC 6749 §4.1.2, §4.1.3),
* replaying an authorization code revokes the tokens issued for it, and
  tampered or expired access tokens are rejected with 401 (RFC 6750 §3.1).
"""

import concurrent.futures
import urllib.parse
from datetime import timedelta
from http import HTTPStatus

import httpx2
from faker import Faker
from freezegun.api import TickingDateTimeFactory

from oidc_provider_mock._client_lib import OidcClient

from .conftest import use_provider_config

faker = Faker()


def _register_client(oidc_server: str) -> tuple[OidcClient, str]:
    redirect_uri = faker.uri(schemes=["https"])
    client = OidcClient.register(oidc_server, redirect_uri=redirect_uri)
    return client, redirect_uri


def _authorize(client: OidcClient, *, nonce: str | None = None) -> str:
    """Complete the authorization form and return the authorization code."""
    state = faker.password()
    response = httpx2.post(
        client.authorization_url(state=state, nonce=nonce),
        data={"sub": faker.email()},
    )
    assert response.status_code == HTTPStatus.FOUND
    params = urllib.parse.parse_qs(
        urllib.parse.urlsplit(response.headers["location"]).query
    )
    assert params["state"] == [state]
    return params["code"][0]


def _exchange_code(
    oidc_server: str,
    client: OidcClient,
    code: str,
    *,
    redirect_uri: str | None = None,
    client_id: str | None = None,
    secret: str | None = None,
) -> httpx2.Response:
    data = {"grant_type": "authorization_code", "code": code}
    if redirect_uri is not None:
        data["redirect_uri"] = redirect_uri
    return httpx2.post(
        urllib.parse.urljoin(oidc_server, "oauth2/token"),
        data=data,
        auth=(client_id or client.id, secret if secret is not None else client.secret),
    )


def _userinfo(oidc_server: str, access_token: str) -> httpx2.Response:
    return httpx2.get(
        urllib.parse.urljoin(oidc_server, "userinfo"),
        headers={"authorization": f"Bearer {access_token}"},
    )


def test_unregistered_redirect_uri_is_not_redirected_to(oidc_server: str):
    """An unregistered redirect_uri must yield an error page, never a 302."""

    client, redirect_uri = _register_client(oidc_server)
    attacker_uri = faker.uri(schemes=["https"])

    url = client.authorization_url(state=faker.password())
    url = url.replace(
        urllib.parse.quote(redirect_uri, safe=""),
        urllib.parse.quote(attacker_uri, safe=""),
    )
    assert urllib.parse.quote(attacker_uri, safe="") in url

    for make_request in (
        lambda: httpx2.get(url),
        lambda: httpx2.post(url, data={"sub": faker.email()}),
    ):
        response = make_request()
        assert response.status_code == HTTPStatus.BAD_REQUEST
        assert "location" not in response.headers
        # The error is rendered as a page, not redirected to the attacker URI
        assert response.headers["content-type"].startswith("text/html")
        assert "invalid_client" in response.text


def test_unsupported_scope_redirects_with_error(oidc_server: str):
    """A valid request with an unsupported scope redirects with `error`."""

    client, redirect_uri = _register_client(oidc_server)
    state = faker.password()

    response = httpx2.post(
        client.authorization_url(state=state, scope="openid admin"),
        data={"sub": faker.email()},
    )

    assert response.status_code == HTTPStatus.FOUND
    location = response.headers["location"]
    assert location.startswith(redirect_uri)
    params = urllib.parse.parse_qs(urllib.parse.urlsplit(location).query)
    assert params["error"] == ["invalid_scope"]
    assert params["state"] == [state]
    assert "code" not in params


def test_authorization_code_cannot_be_exchanged_twice(oidc_server: str):
    client, redirect_uri = _register_client(oidc_server)
    code = _authorize(client)

    response = _exchange_code(oidc_server, client, code, redirect_uri=redirect_uri)
    assert response.status_code == HTTPStatus.OK

    response = _exchange_code(oidc_server, client, code, redirect_uri=redirect_uri)
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.json()["error"] == "invalid_grant"


def test_authorization_code_replay_revokes_issued_tokens(oidc_server: str):
    """Replaying a code revokes the access token issued by the first exchange."""

    client, redirect_uri = _register_client(oidc_server)
    code = _authorize(client)

    response = _exchange_code(oidc_server, client, code, redirect_uri=redirect_uri)
    assert response.status_code == HTTPStatus.OK
    access_token = response.json()["access_token"]

    assert _userinfo(oidc_server, access_token).status_code == HTTPStatus.OK

    replay = _exchange_code(oidc_server, client, code, redirect_uri=redirect_uri)
    assert replay.status_code == HTTPStatus.BAD_REQUEST
    assert replay.json()["error"] == "invalid_grant"

    response = _userinfo(oidc_server, access_token)
    assert response.status_code == HTTPStatus.UNAUTHORIZED
    assert response.json()["error"] == "invalid_token"


def test_concurrent_code_exchange_succeeds_only_once(oidc_server: str):
    """Racing exchanges of the same code: exactly one may succeed."""

    client, redirect_uri = _register_client(oidc_server)
    code = _authorize(client)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        responses = list(
            executor.map(
                lambda _: _exchange_code(
                    oidc_server, client, code, redirect_uri=redirect_uri
                ),
                range(8),
            )
        )

    statuses = [response.status_code for response in responses]
    assert statuses.count(HTTPStatus.OK) == 1
    assert statuses.count(HTTPStatus.BAD_REQUEST) == 7


def test_authorization_code_is_bound_to_client(oidc_server: str):
    """A code issued to one client cannot be redeemed by another client."""

    client, redirect_uri = _register_client(oidc_server)
    other_client, _ = _register_client(oidc_server)
    code = _authorize(client)

    response = _exchange_code(
        oidc_server,
        client,
        code,
        redirect_uri=redirect_uri,
        client_id=other_client.id,
        secret=other_client.secret,
    )
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.json()["error"] == "invalid_grant"


def test_authorization_code_is_bound_to_redirect_uri(oidc_server: str):
    """The redirect_uri at the token endpoint must match the one from the
    authorization request."""

    client, _ = _register_client(oidc_server)

    # Mismatched redirect_uri
    code = _authorize(client)
    response = _exchange_code(
        oidc_server, client, code, redirect_uri=faker.uri(schemes=["https"])
    )
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.json()["error"] == "invalid_grant"

    # Missing redirect_uri
    code = _authorize(client)
    response = _exchange_code(oidc_server, client, code)
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.json()["error"] == "invalid_grant"


def test_id_token_nonce_matches_authorization_request(oidc_server: str):
    """The nonce in the id_token is the one from its own authorization request."""

    client, _ = _register_client(oidc_server)

    for nonce in (faker.password(), faker.password()):
        state = faker.password()
        response = httpx2.post(
            client.authorization_url(state=state, nonce=nonce),
            data={"sub": faker.email()},
        )
        token_data = client.fetch_token(response.headers["location"], state=state)
        assert token_data.claims["nonce"] == nonce


def test_tampered_access_token_is_rejected(oidc_server: str):
    client, redirect_uri = _register_client(oidc_server)
    code = _authorize(client)
    response = _exchange_code(oidc_server, client, code, redirect_uri=redirect_uri)
    access_token = response.json()["access_token"]

    for token in (
        faker.password(),
        # Well-formed but unknown token
        access_token[:-1] + ("A" if access_token[-1] != "A" else "B"),
    ):
        response = _userinfo(oidc_server, token)
        assert response.status_code == HTTPStatus.UNAUTHORIZED
        assert response.json()["error"] == "invalid_token"


@use_provider_config(access_token_max_age=timedelta(minutes=5))
def test_expired_access_token_is_rejected(
    oidc_server: str, ticking_datetime: TickingDateTimeFactory
):
    client, redirect_uri = _register_client(oidc_server)
    code = _authorize(client)
    response = _exchange_code(oidc_server, client, code, redirect_uri=redirect_uri)
    access_token = response.json()["access_token"]

    assert _userinfo(oidc_server, access_token).status_code == HTTPStatus.OK

    ticking_datetime.tick(timedelta(minutes=6))

    response = _userinfo(oidc_server, access_token)
    assert response.status_code == HTTPStatus.UNAUTHORIZED
    assert response.json()["error"] == "invalid_token"
