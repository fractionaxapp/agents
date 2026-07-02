import pytest
from fastapi.testclient import TestClient

from fractionax_agents.config import Settings
from fractionax_agents.server import app

client = TestClient(app)


def _no_provider_settings() -> Settings:
    # Ignore any local .env so the test is hermetic regardless of the dev's keys.
    return Settings(_env_file=None, anthropic_api_key=None, minimax_api_key=None)


def test_health() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_requires_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    # With NO provider configured (neither Anthropic nor MiniMax), /chat returns 503
    # rather than calling an LLM.
    monkeypatch.setattr("fractionax_agents.server.get_settings", _no_provider_settings)
    resp = client.post("/chat", json={"message": "hello"})
    assert resp.status_code == 503


def _investor(jurisdiction: str, accredited: bool = False, id_: str = "inv_1") -> dict:
    return {
        "id": id_,
        "display_name": "Test Investor",
        "jurisdiction": jurisdiction,
        "accredited": accredited,
        "risk_appetite": "medium",
    }


def test_compliance_rules_endpoint() -> None:
    resp = client.get("/compliance/rules")
    assert resp.status_code == 200
    regimes = {r["regime"] for r in resp.json()}
    assert regimes == {"reg_d", "reg_s", "reg_a"}


def test_compliance_verify_allows_eligible(
    monkeypatch: pytest.MonkeyPatch, populated_db: list
) -> None:
    # No key configured -> deterministic rationale, no LLM call. deal_rwa_1 is offshore (SV).
    monkeypatch.setattr("fractionax_agents.compliance.get_settings", _no_provider_settings)
    resp = client.post(
        "/compliance/verify",
        json={"investor": _investor("MY"), "deal_id": "deal_rwa_1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["outcome"] == "allow"
    assert body["regime"] == "reg_s"


def test_compliance_verify_denies_us_person_offshore(
    monkeypatch: pytest.MonkeyPatch, populated_db: list
) -> None:
    monkeypatch.setattr("fractionax_agents.compliance.get_settings", _no_provider_settings)
    resp = client.post(
        "/compliance/verify",
        json={"investor": _investor("US", accredited=True), "deal_id": "deal_rwa_1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["outcome"] == "deny"
    assert any(r["code"] == "jurisdiction_blocked" for r in body["reasons"])


def test_compliance_verify_unknown_deal_404() -> None:
    resp = client.post(
        "/compliance/verify",
        json={"investor": _investor("MY"), "deal_id": "does_not_exist"},
    )
    assert resp.status_code == 404


def test_compliance_transfer_endpoint(monkeypatch: pytest.MonkeyPatch, populated_db: list) -> None:
    monkeypatch.setattr("fractionax_agents.compliance.get_settings", _no_provider_settings)
    resp = client.post(
        "/compliance/transfer",
        json={
            "deal_id": "deal_rwa_1",
            "seller": _investor("MY", accredited=True, id_="seller"),
            "buyer": _investor("US", accredited=True, id_="buyer"),
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["allowed"] is False
    assert any(r["code"] == "jurisdiction_blocked" for r in body["reasons"])


# --- Admin endpoints --------------------------------------------------------


def _admin_settings(**over: object) -> Settings:
    return Settings(_env_file=None, admin_api_key="test-key", **over)  # type: ignore[arg-type]


def test_admin_disabled_when_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "fractionax_agents.server.get_settings",
        lambda: Settings(_env_file=None, admin_api_key=None),
    )
    resp = client.get("/admin/investors")
    assert resp.status_code == 503


def test_admin_wrong_key_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fractionax_agents.server.get_settings", _admin_settings)
    resp = client.get("/admin/investors", headers={"X-Admin-Key": "nope"})
    assert resp.status_code == 401


def test_admin_investors_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from fractionax_agents.admin_store import AdminStore

    store = AdminStore(tmp_path / "s.json")
    monkeypatch.setattr("fractionax_agents.server.get_settings", _admin_settings)
    monkeypatch.setattr("fractionax_agents.server.get_admin_store", lambda: store)
    headers = {"X-Admin-Key": "test-key"}

    # Empty to start.
    assert client.get("/admin/investors", headers=headers).json() == []

    # Add/screen an investor.
    add = client.post("/admin/investors", headers=headers, json=_investor("MY", id_="inv_a"))
    assert add.status_code == 200
    assert add.json()["kyc_status"] == "verified"

    listed = client.get("/admin/investors", headers=headers).json()
    assert len(listed) == 1 and listed[0]["investor"]["id"] == "inv_a"

    # Mirror an on-chain credential issue.
    cred = client.post(
        "/admin/investors/inv_a/credential",
        headers=headers,
        json={"status": "issued", "tx": "sig123", "updated_at": "t"},
    )
    assert cred.status_code == 200 and cred.json()["credential_status"] == "issued"

    # Unknown investor -> 404.
    assert (
        client.post(
            "/admin/investors/ghost/credential",
            headers=headers,
            json={"status": "issued"},
        ).status_code
        == 404
    )


def test_admin_catalogue_import_and_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    # DB isolation (fresh temp SQLite) comes from the autouse conftest fixture.
    monkeypatch.setattr("fractionax_agents.server.get_settings", _admin_settings)
    headers = {"X-Admin-Key": "test-key"}

    asset = {
        "id": 7,
        "name": "Imported Fund",
        "asset_class": {"slug": "real-estate"},
        "jurisdiction": {"name": "Singapore"},
        "minInvestment": {"amount": 5000, "currency": "USD"},
        "stats": {"return": 8.0, "aum": 1000000.0, "inception": "1/2/2024"},
    }

    # Starts empty (no seed fallback).
    assert client.get("/admin/deals/catalogue", headers=headers).json()["source"] == "empty"

    # Import an inline payload (as an uploaded file would deliver).
    res = client.post(
        "/admin/deals/import",
        headers=headers,
        json={"payload": {"pageProps": {"assets": [asset]}}},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["imported"] == 1 and body["source"] == "database"
    assert body["sample"][0]["id"] == "deal_rwa_7"

    # The imported deal is now discoverable via /deals.
    ids = [d["id"] for d in client.get("/deals").json()]
    assert ids == ["deal_rwa_7"]

    # A bad payload is rejected.
    assert (
        client.post("/admin/deals/import", headers=headers, json={"payload": {"x": 1}}).status_code
        == 422
    )

    # Reset empties the catalogue (no seed to fall back to).
    reset = client.post("/admin/deals/reset", headers=headers)
    assert reset.status_code == 200 and reset.json()["source"] == "empty"
