from __future__ import annotations

from dataclasses import dataclass


MOBILE_PARTNER_DOMAIN_SCHEMA = "mobile_partner_residue_domain_v1"


@dataclass(frozen=True)
class MobilePartnerResidueDomain:
    """Single operational residue-count contract for the moving partner.

    This contract controls input acceptance only.  It deliberately makes no
    claim that an existing confidence artifact is calibrated across the whole
    accepted range.
    """

    minimum_residues: int
    maximum_residues: int

    def __post_init__(self) -> None:
        for field, value in (
            ("minimum_residues", self.minimum_residues),
            ("maximum_residues", self.maximum_residues),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{field} must be an integer")
            if value < 1:
                raise ValueError(f"{field} must be positive")
        if self.minimum_residues > self.maximum_residues:
            raise ValueError(
                "minimum_residues cannot exceed maximum_residues"
            )

    @property
    def label(self) -> str:
        return f"{self.minimum_residues}–{self.maximum_residues} aa"

    def contains(self, residue_count: int) -> bool:
        if isinstance(residue_count, bool) or not isinstance(residue_count, int):
            raise ValueError("residue_count must be an integer")
        return self.minimum_residues <= residue_count <= self.maximum_residues

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": MOBILE_PARTNER_DOMAIN_SCHEMA,
            "role": "operational_mobile_partner",
            "minimum_residues": self.minimum_residues,
            "maximum_residues": self.maximum_residues,
            "unit": "residue",
            "scope": "input_acceptance_not_calibration_guarantee",
        }


# The local PPI4DOCK corpus currently tops out at 1,150 residues.  A fixed
# ceiling of 1,200 covers that corpus while retaining a fail-closed bound.
DEFAULT_MOBILE_PARTNER_DOMAIN = MobilePartnerResidueDomain(
    minimum_residues=5,
    maximum_residues=1200,
)


__all__ = [
    "DEFAULT_MOBILE_PARTNER_DOMAIN",
    "MOBILE_PARTNER_DOMAIN_SCHEMA",
    "MobilePartnerResidueDomain",
]
