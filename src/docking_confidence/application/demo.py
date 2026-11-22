from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd


_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def _synthetic_sequence(identifier: str, length: int) -> str:
    """Create deterministic, explicitly non-scientific demo sequences."""
    digest = hashlib.sha256(identifier.encode("ascii")).digest()
    return "".join(
        _AMINO_ACIDS[digest[index % len(digest)] % len(_AMINO_ACIDS)]
        for index in range(length)
    )


def make_demo(path: str, *, seed: int = 1234, n_complexes: int = 30, runs_per_complex: int = 12) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_complexes):
        complex_id = f"C{c+1:03d}"
        receptor_sequence = _synthetic_sequence(f"receptor-{complex_id}", 60)
        ligand_sequence = _synthetic_sequence(f"ligand-{complex_id}", 12)
        complex_shift = rng.normal(0, 0.65)
        for r in range(runs_per_complex):
            f1 = rng.normal(complex_shift, 1)
            f2 = np.clip(rng.beta(5, 2), 0, 1)
            f3 = abs(rng.normal(0.5 - 0.1 * complex_shift, 0.2))
            f4 = max(0, int(rng.poisson(2.0 - min(complex_shift, 1.0) * 0.4)))
            f5 = rng.normal(-4.0 - complex_shift * 0.5, 0.8)
            latent = 1.0 * f1 + 1.4 * (f2 - 0.5) - 0.7 * f3 - 0.25 * f4 - 0.35 * (f5 + 4)
            preserve_prob = 1 / (1 + np.exp(-latent))
            preserved = rng.random() < preserve_prob

            full_dockq = float(np.clip(rng.normal(0.48 + 0.05 * complex_shift, 0.07), 0.05, 0.95))
            if preserved:
                loss = float(rng.uniform(-0.02, 0.045))
                reduced_dockq = max(0.23, full_dockq - loss)
            else:
                if rng.random() < 0.5:
                    reduced_dockq = float(rng.uniform(0.08, 0.22))
                else:
                    reduced_dockq = max(0.0, full_dockq - float(rng.uniform(0.06, 0.18)))
            reduced_dockq = float(np.clip(reduced_dockq, 0, 1))

            full_rt = float(rng.normal(240, 20))
            reduced_rt = float(rng.normal(165, 18))
            rows.append({
                "complex_id": complex_id,
                "run_id": f"R{r+1:02d}",
                "algorithm": ["ag", "pso", "hybrid"][r % 3],
                "seed": r + 1,
                "receptor_sequence": receptor_sequence,
                "ligand_sequence": ligand_sequence,
                "engine_reduced": "synthetic_demo_non_scientific",
                "engine_full": "synthetic_demo_non_scientific",
                "scientific_reduced": False,
                "scientific_full": False,
                "engine": "synthetic_demo_non_scientific",
                "scientific": False,
                "dockq_reduced": reduced_dockq,
                "dockq_full": full_dockq,
                "runtime_reduced_sec": max(1.0, reduced_rt),
                "runtime_full_sec": max(1.0, full_rt),
                "scan_gap_norm": f1,
                "viable_fraction_mean": f2,
                "elite_dispersion": f3,
                "uncertainty_count": f4,
                "energy_q20_norm": f5,
            })
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
