"""A lightweight, file-backed store for the super-admin dashboard's off-chain
data: the investor directory and the compliance decision log.

This is deliberately minimal — a JSON file guarded by a process lock — because no
database is wired yet. It gives the admin dashboard a real, persistent data source
for the off-chain side (investors, their KYC/credential state, and a history of
decisions) with a clean seam to swap in Postgres/Redis later (the root env already
anticipates DATABASE_URL / REDIS_URL). It is not built for high concurrency or
multi-process deployment.

On-chain credential state (`credential_status`) is mirrored here when the admin
issues/revokes a credential, so the directory can show it without a chain read per
row; the chain remains the source of truth.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Literal

from fractionax_core import ComplianceDecision, Investor
from pydantic import BaseModel

from .config import get_settings

# On-chain credential mirror: whether the admin has issued a credential for the
# investor, revoked it, or neither.
CredentialStatus = Literal["none", "issued", "revoked"]

_MAX_DECISIONS = 500  # cap the log so the file stays small


class InvestorRecord(BaseModel):
    """An investor as tracked by the admin directory: their profile plus the
    latest screening result and on-chain credential state."""

    investor: Investor
    kyc_status: str = "unverified"
    accreditation_tier: str = "retail"
    credential_status: CredentialStatus = "none"
    # Signature of the last on-chain credential tx (issue/revoke), if any.
    credential_tx: str | None = None
    updated_at: str = "1970-01-01T00:00:00Z"


class _StoreData(BaseModel):
    investors: dict[str, InvestorRecord] = {}
    decisions: list[ComplianceDecision] = []


def _store_path() -> Path:
    configured = get_settings().admin_store_path
    if configured:
        return Path(configured)
    # Default to the working directory (not the source tree) so the data file is
    # never committed; gitignored as admin_store.json.
    return Path("admin_store.json")


class AdminStore:
    """Process-wide JSON-backed store. Reads/writes are serialized by a lock."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

    def _read(self) -> _StoreData:
        if not self._path.exists():
            return _StoreData()
        try:
            return _StoreData.model_validate_json(self._path.read_text())
        except (ValueError, OSError):
            return _StoreData()

    def _write(self, data: _StoreData) -> None:
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(data.model_dump_json(indent=2))
        tmp.replace(self._path)  # atomic on POSIX

    # --- Investors ----------------------------------------------------------

    def list_investors(self) -> list[InvestorRecord]:
        with self._lock:
            data = self._read()
        return sorted(data.investors.values(), key=lambda r: r.updated_at, reverse=True)

    def get_investor(self, investor_id: str) -> InvestorRecord | None:
        with self._lock:
            return self._read().investors.get(investor_id)

    def upsert_investor(
        self,
        investor: Investor,
        *,
        kyc_status: str,
        accreditation_tier: str,
        updated_at: str,
    ) -> InvestorRecord:
        """Insert or update an investor's directory record, preserving any existing
        credential state."""
        with self._lock:
            data = self._read()
            existing = data.investors.get(investor.id)
            record = InvestorRecord(
                investor=investor,
                kyc_status=kyc_status,
                accreditation_tier=accreditation_tier,
                credential_status=existing.credential_status if existing else "none",
                credential_tx=existing.credential_tx if existing else None,
                updated_at=updated_at,
            )
            data.investors[investor.id] = record
            self._write(data)
            return record

    def set_credential_status(
        self, investor_id: str, status: CredentialStatus, *, tx: str | None, updated_at: str
    ) -> InvestorRecord | None:
        """Mirror an on-chain credential issue/revoke onto the investor record."""
        with self._lock:
            data = self._read()
            record = data.investors.get(investor_id)
            if record is None:
                return None
            record = record.model_copy(
                update={
                    "credential_status": status,
                    "credential_tx": tx,
                    "updated_at": updated_at,
                }
            )
            data.investors[investor_id] = record
            self._write(data)
            return record

    # --- Decisions ----------------------------------------------------------

    def log_decision(self, decision: ComplianceDecision) -> None:
        with self._lock:
            data = self._read()
            data.decisions.append(decision)
            if len(data.decisions) > _MAX_DECISIONS:
                data.decisions = data.decisions[-_MAX_DECISIONS:]
            self._write(data)

    def list_decisions(self, limit: int = 100) -> list[ComplianceDecision]:
        with self._lock:
            data = self._read()
        return list(reversed(data.decisions[-limit:]))


_store: AdminStore | None = None


def get_admin_store() -> AdminStore:
    global _store
    if _store is None:
        _store = AdminStore(_store_path())
    return _store
