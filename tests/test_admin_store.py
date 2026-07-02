"""The file-backed admin store: investor upsert, decision log, credential mirror."""

from __future__ import annotations

from pathlib import Path

from fractionax_core.domain import ComplianceDecision, Investor

from fractionax_agents.admin_store import AdminStore


def _investor(id_: str = "inv_1", jurisdiction: str = "MY") -> Investor:
    return Investor(
        id=id_,
        display_name="Test",
        jurisdiction=jurisdiction,
        accredited=False,
        risk_appetite="medium",
    )


def _decision(investor_id: str = "inv_1") -> ComplianceDecision:
    return ComplianceDecision(
        investor_id=investor_id,
        deal_id="deal_x",
        outcome="allow",
        kyc_status="verified",
        accreditation_tier="retail",
        regime="reg_s",
        reasons=[],
        rationale="ok",
        decided_at="2026-07-02T00:00:00Z",
    )


def test_upsert_and_list(tmp_path: Path) -> None:
    store = AdminStore(tmp_path / "s.json")
    store.upsert_investor(
        _investor(), kyc_status="verified", accreditation_tier="retail", updated_at="t1"
    )
    records = store.list_investors()
    assert len(records) == 1
    assert records[0].investor.id == "inv_1"
    assert records[0].kyc_status == "verified"
    assert records[0].credential_status == "none"


def test_upsert_preserves_credential_state(tmp_path: Path) -> None:
    store = AdminStore(tmp_path / "s.json")
    store.upsert_investor(
        _investor(), kyc_status="verified", accreditation_tier="retail", updated_at="t1"
    )
    store.set_credential_status("inv_1", "issued", tx="sig123", updated_at="t2")
    # A later re-screen (upsert) must not wipe the on-chain credential mirror.
    store.upsert_investor(
        _investor(), kyc_status="verified", accreditation_tier="accredited", updated_at="t3"
    )
    rec = store.get_investor("inv_1")
    assert rec is not None
    assert rec.credential_status == "issued"
    assert rec.credential_tx == "sig123"
    assert rec.accreditation_tier == "accredited"


def test_set_credential_status_unknown_investor(tmp_path: Path) -> None:
    store = AdminStore(tmp_path / "s.json")
    assert store.set_credential_status("missing", "issued", tx=None, updated_at="t") is None


def test_decision_log_orders_recent_first_and_caps(tmp_path: Path) -> None:
    store = AdminStore(tmp_path / "s.json")
    for i in range(5):
        d = _decision()
        store.log_decision(d.model_copy(update={"deal_id": f"deal_{i}"}))
    recent = store.list_decisions(limit=3)
    assert len(recent) == 3
    assert recent[0].deal_id == "deal_4"  # most recent first


def test_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    AdminStore(path).upsert_investor(
        _investor(), kyc_status="verified", accreditation_tier="retail", updated_at="t1"
    )
    # A fresh instance reads the same file.
    assert AdminStore(path).get_investor("inv_1") is not None
