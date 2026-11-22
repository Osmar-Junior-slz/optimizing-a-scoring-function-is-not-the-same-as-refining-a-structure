from dataclasses import dataclass


@dataclass(frozen=True)
class PreservationRule:
    minimum_reduced_dockq: float = 0.23
    maximum_dockq_loss: float = 0.05

    def label(self, reduced_dockq: float, full_dockq: float) -> int:
        return int(
            reduced_dockq >= self.minimum_reduced_dockq
            and (full_dockq - reduced_dockq) <= self.maximum_dockq_loss
        )
