"""Compliance Agent: KYC/AML screening, jurisdiction-aware rules, accreditation
gating, and transfer restrictions for the M3 "compliant investing" vertical.

Design principle (mirrors the underwriting agent): the ALLOW/DENY decision is
**deterministic and authoritative** — the LLM never decides compliance, it only
writes the human-readable rationale, and even that has a deterministic fallback
so the gate works with no API key configured. KYC runs through a pluggable
provider (:class:`KycProvider`); the default is a deterministic mock that a real
vendor (e.g. Sumsub/Persona) drops in behind later.

The decision this agent emits (:class:`ComplianceDecision`) is the same gate the
web invest CTA and the on-chain ``assert_compliant`` instruction enforce.
"""

from __future__ import annotations

from typing import Protocol

from fractionax_core import (
    ComplianceDecision,
    ComplianceProfile,
    ComplianceReason,
    Deal,
    Investor,
    JurisdictionRule,
    TransferCheck,
)
from fractionax_core.domain import (
    AccreditationTier,
    ComplianceOutcome,
    KycStatus,
    OfferingRegime,
)

from .config import get_settings

# --- Static compliance policy ------------------------------------------------
# Jurisdictions under global sanctions/embargo — blocked everywhere regardless
# of KYC or accreditation. A coarse demo stand-in for a real sanctions feed
# (OFAC/UN/EU consolidated lists) wired in behind the KYC provider later.
SANCTIONED_JURISDICTIONS: frozenset[str] = frozenset({"IR", "KP", "SY", "CU", "RU"})

# Accreditation ordering, low → high, so tier gates are a simple index compare.
_TIER_ORDER: tuple[AccreditationTier, ...] = ("retail", "accredited", "institutional")

# The jurisdiction-aware rules engine, one rule per securities-exemption regime.
# Kept as data so it is inspectable (`GET /compliance/rules`) and testable.
JURISDICTION_RULES: dict[OfferingRegime, JurisdictionRule] = {
    # US private placement: US accredited (or institutional) investors only.
    "reg_d": JurisdictionRule(
        regime="reg_d",
        allowed_jurisdictions=["US"],
        blocked_jurisdictions=[],
        min_tier="accredited",
        asset_kinds=[],
        accredited_only_risk_tiers=["high"],
    ),
    # Regulation S offshore offering: non-US persons only, open to retail.
    "reg_s": JurisdictionRule(
        regime="reg_s",
        allowed_jurisdictions=[],  # any not blocked
        blocked_jurisdictions=["US"],
        min_tier="retail",
        asset_kinds=[],
        accredited_only_risk_tiers=["high"],
    ),
    # Regulation A mini-IPO: open to retail within permitted jurisdictions.
    "reg_a": JurisdictionRule(
        regime="reg_a",
        allowed_jurisdictions=[],
        blocked_jurisdictions=[],
        min_tier="retail",
        asset_kinds=[],
        accredited_only_risk_tiers=["high"],
    ),
}


def _tier_rank(tier: AccreditationTier) -> int:
    return _TIER_ORDER.index(tier)


def deal_regime(deal: Deal) -> OfferingRegime:
    """Infer the securities-exemption regime a deal is offered under.

    Deterministic and documented: US-domiciled deals are treated as Reg D private
    placements (accredited US investors); non-US deals as Reg S offshore offerings
    (non-US persons, retail-friendly). A future onboarding flow can carry an
    explicit regime on the deal and override this.
    """
    return "reg_d" if deal.jurisdiction == "US" else "reg_s"


# --- KYC/AML provider (pluggable) --------------------------------------------


class KycProvider(Protocol):
    """Screens an investor into a :class:`ComplianceProfile`. Swap the mock for a
    real vendor by implementing this and injecting it into :func:`verify_investor`."""

    def screen(self, investor: Investor) -> ComplianceProfile: ...


class MockKycProvider:
    """Deterministic KYC/AML screening for devnet/demo.

    Rules: an investor in a sanctioned jurisdiction fails the sanctions screen; an
    investor whose id/name is on the watchlist is rejected; everyone else is
    verified. Accreditation tier is derived from the investor's ``accredited`` flag
    (with an optional institutional override set).
    """

    def __init__(
        self,
        *,
        watchlist: frozenset[str] = frozenset(),
        institutional_ids: frozenset[str] = frozenset(),
        screened_at: str = "1970-01-01T00:00:00Z",
    ) -> None:
        self._watchlist = {v.lower() for v in watchlist}
        self._institutional = institutional_ids
        self._screened_at = screened_at

    def _tier(self, investor: Investor) -> AccreditationTier:
        if investor.id in self._institutional:
            return "institutional"
        return "accredited" if investor.accredited else "retail"

    def screen(self, investor: Investor) -> ComplianceProfile:
        sanctions_clear = investor.jurisdiction not in SANCTIONED_JURISDICTIONS
        flagged = {investor.id.lower(), investor.display_name.lower()} & self._watchlist
        kyc_status: KycStatus = "rejected" if (not sanctions_clear or flagged) else "verified"
        return ComplianceProfile(
            investor_id=investor.id,
            jurisdiction=investor.jurisdiction,
            kyc_status=kyc_status,
            accreditation_tier=self._tier(investor),
            sanctions_clear=sanctions_clear,
            screened_at=self._screened_at,
        )


_DEFAULT_PROVIDER = MockKycProvider()


# --- Deterministic rules engine (authoritative) ------------------------------


def _holding_reasons(
    profile: ComplianceProfile, deal: Deal, regime: OfferingRegime
) -> list[ComplianceReason]:
    """The authoritative gate: every reason an investor may NOT hold ``deal``.

    An empty list means eligible. Shared by the invest decision and the transfer
    check so a secondary-market recipient is held to the same bar as a primary buyer.
    """
    rule = JURISDICTION_RULES[regime]
    reasons: list[ComplianceReason] = []

    if not profile.sanctions_clear:
        reasons.append(
            ComplianceReason(
                code="sanctions_hit",
                detail=f"Jurisdiction {profile.jurisdiction} is under sanctions/embargo.",
            )
        )
    if profile.kyc_status != "verified":
        reasons.append(
            ComplianceReason(
                code="kyc_not_verified",
                detail=f"KYC status is '{profile.kyc_status}', not 'verified'.",
            )
        )

    j = profile.jurisdiction
    if j in rule.blocked_jurisdictions:
        reasons.append(
            ComplianceReason(
                code="jurisdiction_blocked",
                detail=f"{regime}: investors in {j} may not participate.",
            )
        )
    elif rule.allowed_jurisdictions and j not in rule.allowed_jurisdictions:
        allowed = ", ".join(rule.allowed_jurisdictions)
        reasons.append(
            ComplianceReason(
                code="jurisdiction_not_permitted",
                detail=f"{regime}: only {allowed} permitted; investor is in {j}.",
            )
        )

    if _tier_rank(profile.accreditation_tier) < _tier_rank(rule.min_tier):
        reasons.append(
            ComplianceReason(
                code="accreditation_below_minimum",
                detail=(
                    f"{regime} requires at least '{rule.min_tier}'; investor is "
                    f"'{profile.accreditation_tier}'."
                ),
            )
        )
    elif (
        deal.risk_tier in rule.accredited_only_risk_tiers and profile.accreditation_tier == "retail"
    ):
        reasons.append(
            ComplianceReason(
                code="risk_tier_requires_accreditation",
                detail=f"{deal.risk_tier}-risk deals require an accredited investor.",
            )
        )

    return reasons


# --- Rationale narrative (LLM-optional) --------------------------------------

_RATIONALE_SYSTEM = (
    "You are the FractionAX compliance agent. You are given a pre-computed, "
    "authoritative ALLOW/DENY decision and the exact reason codes behind it. Write "
    "one or two plain-language sentences explaining the decision to the investor. Do "
    "NOT change the decision, invent new reasons, or cite rules not in the input."
)


def _deterministic_rationale(
    outcome: str, reasons: list[ComplianceReason], regime: OfferingRegime
) -> str:
    if outcome == "allow":
        return (
            f"Eligible to invest under {regime}: KYC verified and all jurisdiction "
            "and accreditation checks passed."
        )
    lead = "; ".join(r.detail for r in reasons)
    return f"Not eligible under {regime}. {lead}"


def _llm_rationale(outcome: str, reasons: list[ComplianceReason], regime: OfferingRegime) -> str:
    """Best-effort LLM narrative; falls back to the deterministic sentence on any
    error or when no provider key is configured (so the gate never depends on the LLM)."""
    settings = get_settings()
    if not (settings.anthropic_api_key or settings.minimax_api_key):
        return _deterministic_rationale(outcome, reasons, regime)
    try:
        from pydantic import BaseModel

        from .structured import extract

        class _Rationale(BaseModel):
            rationale: str

        reason_lines = "\n".join(f"- {r.code}: {r.detail}" for r in reasons) or "- none"
        result = extract(
            system=_RATIONALE_SYSTEM,
            user=(
                f"Decision: {outcome.upper()} under {regime}.\n"
                f"Reason codes:\n{reason_lines}\n\n"
                "Write the investor-facing rationale."
            ),
            model_cls=_Rationale,
            tool_name="record_rationale",
            tool_description="Record the plain-language compliance rationale.",
        )
        return result.rationale.strip() or _deterministic_rationale(outcome, reasons, regime)
    except Exception:
        return _deterministic_rationale(outcome, reasons, regime)


# --- Public API --------------------------------------------------------------


def verify_investor(
    investor: Investor,
    deal: Deal,
    *,
    provider: KycProvider | None = None,
    decided_at: str = "1970-01-01T00:00:00Z",
    with_rationale: bool = True,
) -> ComplianceDecision:
    """Screen ``investor`` and decide whether they may invest in ``deal``.

    The outcome is fully deterministic (KYC + sanctions + jurisdiction + tier +
    risk gating). ``with_rationale`` adds an LLM narrative when a provider key is
    set; otherwise a deterministic sentence is used.
    """
    regime = deal_regime(deal)
    profile = (provider or _DEFAULT_PROVIDER).screen(investor)
    reasons = _holding_reasons(profile, deal, regime)
    outcome: ComplianceOutcome = "allow" if not reasons else "deny"
    rationale = (
        _llm_rationale(outcome, reasons, regime)
        if with_rationale
        else _deterministic_rationale(outcome, reasons, regime)
    )
    return ComplianceDecision(
        investor_id=investor.id,
        deal_id=deal.id,
        outcome=outcome,
        kyc_status=profile.kyc_status,
        accreditation_tier=profile.accreditation_tier,
        regime=regime,
        reasons=reasons,
        rationale=rationale,
        decided_at=decided_at,
    )


def check_transfer(
    deal: Deal,
    seller: Investor,
    buyer: Investor,
    *,
    provider: KycProvider | None = None,
) -> TransferCheck:
    """Transfer-restriction predicate for the secondary market (M3): may ``buyer``
    receive/hold ``deal``'s fractional token from ``seller``?

    The recipient is held to the same holding bar as a primary investor. The seller
    only needs to be a verified holder (not blocked by sanctions/KYC)."""
    p = provider or _DEFAULT_PROVIDER
    regime = deal_regime(deal)
    buyer_reasons = _holding_reasons(p.screen(buyer), deal, regime)

    seller_profile = p.screen(seller)
    reasons = list(buyer_reasons)
    if not seller_profile.sanctions_clear or seller_profile.kyc_status != "verified":
        reasons.append(
            ComplianceReason(
                code="seller_not_in_good_standing",
                detail="Seller is not a KYC-verified, sanctions-clear holder.",
            )
        )
    return TransferCheck(
        deal_id=deal.id,
        from_investor_id=seller.id,
        to_investor_id=buyer.id,
        allowed=not reasons,
        reasons=reasons,
    )


def jurisdiction_rules() -> list[JurisdictionRule]:
    """The active jurisdiction rules — exposed by ``GET /compliance/rules``."""
    return list(JURISDICTION_RULES.values())
