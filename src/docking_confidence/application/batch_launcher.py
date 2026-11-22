from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Iterable

import yaml

from docking_confidence.domain.execution import DockingTask

DOCKINGPP_ARCHIVE_CONTRACT_FILENAME = "dockingpp_v5_archive.yaml"
DOCKINGPP_ARCHIVE_CONTRACT_SCHEMA = "dockingpp_archive_pin_v1"
DOCKINGPP_ARCHIVE_FILENAME = "DockingReduce_V5.zip"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REAL_ALGORITHMS = frozenset({"ag", "pso", "hybrid"})
_REAL_PROTOCOLS = frozenset({"full", "reduced", "reduced_vs_full", "factorial_2x2", "operational_cascade"})
_CONTRACT_FIELDS = frozenset(
    {
        "schema_version",
        "engine_name",
        "archive_filename",
        "archive_sha256",
        "config_entry",
        "config_sha256",
    }
)


class EngineArchiveDiscoveryError(RuntimeError):
    """A real DockingPP archive could not be selected unambiguously."""


@dataclass(frozen=True)
class DockingPPArchiveContract:
    engine_name: str
    archive_filename: str
    archive_sha256: str
    config_entry: str
    config_sha256: str
    source_path: Path


@dataclass(frozen=True)
class BatchProcess:
    pid: int
    log_path: Path
    command: tuple[str, ...]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _contract_path() -> Path:
    return _project_root() / "configs" / DOCKINGPP_ARCHIVE_CONTRACT_FILENAME


def default_dockingpp_snapshot_dir() -> Path:
    """Return a stable, short cache root suitable for Windows CreateProcess.

    ``LOCALAPPDATA`` can be virtualized differently for the desktop app and a
    child process, which would also change an immutable batch manifest.  The
    account home is stable across those execution boundaries.
    """

    return Path.home() / ".cache" / "DockingConfidence" / "engine_snapshots"


def load_dockingpp_archive_contract(
    path: str | Path | None = None,
) -> DockingPPArchiveContract:
    """Load the versioned V5 code/config pins with a closed schema."""

    original = Path(path).expanduser() if path is not None else _contract_path()
    if original.is_symlink():
        raise EngineArchiveDiscoveryError("engine pin contract must not be a symbolic link")
    try:
        resolved = original.resolve(strict=True)
    except OSError as exc:
        raise EngineArchiveDiscoveryError(f"engine pin contract does not exist: {original}") from exc
    if not resolved.is_file():
        raise EngineArchiveDiscoveryError(f"engine pin contract is not a file: {resolved}")
    try:
        payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise EngineArchiveDiscoveryError(
            f"engine pin contract is not valid UTF-8 YAML: {resolved}"
        ) from exc
    if not isinstance(payload, Mapping) or not all(isinstance(key, str) for key in payload):
        raise EngineArchiveDiscoveryError("engine pin contract must be a string-keyed mapping")
    fields = frozenset(payload)
    if fields != _CONTRACT_FIELDS:
        missing = sorted(_CONTRACT_FIELDS - fields)
        extra = sorted(fields - _CONTRACT_FIELDS)
        raise EngineArchiveDiscoveryError(
            f"engine pin contract fields do not match the schema; missing={missing}, extra={extra}"
        )
    if payload["schema_version"] != DOCKINGPP_ARCHIVE_CONTRACT_SCHEMA:
        raise EngineArchiveDiscoveryError("unsupported engine pin contract schema_version")
    if payload["engine_name"] != "dockingpp_archive":
        raise EngineArchiveDiscoveryError("engine pin contract has an unexpected engine_name")

    archive_filename = str(payload["archive_filename"]).strip()
    if archive_filename != DOCKINGPP_ARCHIVE_FILENAME:
        raise EngineArchiveDiscoveryError("engine pin contract has an unexpected archive_filename")
    config_entry = str(payload["config_entry"]).strip()
    config_path = PurePosixPath(config_entry)
    if (
        not config_entry
        or "\\" in config_entry
        or config_path.is_absolute()
        or any(part in {"", ".", ".."} for part in config_path.parts)
        or config_path.suffix.lower() not in {".yaml", ".yml"}
    ):
        raise EngineArchiveDiscoveryError("engine pin contract has an unsafe config_entry")

    return DockingPPArchiveContract(
        engine_name="dockingpp_archive",
        archive_filename=archive_filename,
        archive_sha256=_sha256_pin(
            str(payload["archive_sha256"]), field="contract.archive_sha256"
        ),
        config_entry=config_path.as_posix(),
        config_sha256=_sha256_pin(
            str(payload["config_sha256"]), field="contract.config_sha256"
        ),
        source_path=resolved,
    )


def _validated_archive(path: str | Path) -> Path:
    original = Path(path).expanduser()
    if original.is_symlink():
        raise EngineArchiveDiscoveryError("engine archive must not be a symbolic link")
    try:
        resolved = original.resolve(strict=True)
    except OSError as exc:
        raise EngineArchiveDiscoveryError(f"engine archive does not exist: {original}") from exc
    if not resolved.is_file() or resolved.suffix.lower() != ".zip":
        raise EngineArchiveDiscoveryError(f"engine archive must be a ZIP file: {resolved}")
    return resolved


def _default_archive_search_roots() -> tuple[Path, ...]:
    project_root = _project_root()
    return (project_root, project_root.parent, project_root.parent.parent)


def discover_dockingpp_archive(
    engine_archive: str | Path | None = None,
    *,
    search_roots: Iterable[str | Path] | None = None,
) -> Path:
    """Select the exact V5 archive without recursive or first-match discovery.

    Explicit selection always wins. Automatic discovery checks only the project
    root and its two packaging parents. More than one exact filename is an error
    so a different executable ZIP is never selected by search order.
    ``search_roots`` exists to make the deterministic policy directly testable.
    """

    if engine_archive is not None:
        return _validated_archive(engine_archive)

    contract = load_dockingpp_archive_contract()
    roots = (
        tuple(Path(value).expanduser() for value in search_roots)
        if search_roots is not None
        else _default_archive_search_roots()
    )
    candidates: dict[str, Path] = {}
    for root in roots:
        candidate = root / contract.archive_filename
        if not candidate.exists() and not candidate.is_symlink():
            continue
        resolved = _validated_archive(candidate)
        candidates[str(resolved).casefold()] = resolved

    if not candidates:
        raise EngineArchiveDiscoveryError(
            f"{contract.archive_filename} was not found in the trusted project locations; "
            "provide --engine-archive explicitly"
        )
    if len(candidates) != 1:
        rendered = ", ".join(str(path) for path in sorted(candidates.values()))
        raise EngineArchiveDiscoveryError(
            f"multiple {contract.archive_filename} files were found ({rendered}); "
            "provide --engine-archive explicitly"
        )
    return next(iter(candidates.values()))


def _canonical_algorithms(values: Iterable[str]) -> tuple[str, ...]:
    algorithms = tuple(dict.fromkeys(str(value).strip().lower() for value in values))
    if not algorithms:
        raise ValueError("select at least one algorithm")
    invalid = [value for value in algorithms if value not in _REAL_ALGORITHMS]
    if invalid:
        raise ValueError(f"unsupported algorithms: {invalid}")
    return algorithms


def _positive_int(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be >= 1")
    return value


def _nonnegative_int(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be >= 0")
    return value


def _positive_float(value: float, *, field: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field} must be finite and > 0")
    return number


def _sha256_pin(value: str, *, field: str) -> str:
    normalized = str(value).strip().lower()
    if not _SHA256.fullmatch(normalized):
        raise ValueError(f"{field} must contain 64 hexadecimal characters")
    return normalized


def real_batch_command(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    engine_archive: str | Path | None = None,
    algorithms: Iterable[str],
    protocol: str,
    generations: int,
    population: int,
    top_pockets: int,
    seed_start: int,
    seed_count: int,
    timeout_seconds: float,
    docking_task: str = DockingTask.LOCAL_REFINEMENT.value,
    engine_config_entry: str | None = None,
    engine_archive_sha256: str | None = None,
    engine_config_sha256: str | None = None,
    resume_after_cancel: bool = False,
    confidence_artifact: str | Path | None = None,
    evaluation_budget: int | None = None,
    reduced_search_scale: float | None = None,
    max_translation: float | None = None,
    max_rotation_degrees: float | None = None,
    pose_source: str | None = None,
    execution_order_seed: int | None = None,
    experimental_protocol: str | Path | None = None,
    operational_config: str | Path | None = None,
    collect_operational_pairs: bool = False,
) -> tuple[str, ...]:
    """Build an argv-only invocation of the hash-pinned real batch worker."""

    archive = discover_dockingpp_archive(engine_archive) if engine_archive is not None else None
    canonical_algorithms = _canonical_algorithms(algorithms)
    canonical_protocol = str(protocol).strip().lower()
    canonical_docking_task = DockingTask(docking_task).value
    if canonical_protocol not in _REAL_PROTOCOLS:
        raise ValueError(
            "protocol must be full, reduced, reduced_vs_full or factorial_2x2"
        )
    if confidence_artifact is not None and canonical_protocol not in {"reduced_vs_full", "operational_cascade"}:
        raise ValueError("confidence_artifact requires protocol='reduced_vs_full'")
    if operational_config is not None and canonical_protocol != "operational_cascade":
        raise ValueError("operational_config requires operational_cascade")
    if not isinstance(collect_operational_pairs, bool) or (collect_operational_pairs and (
            canonical_protocol != "operational_cascade" or confidence_artifact is not None)):
        raise ValueError("article pair collection requires operational_cascade without an inference artifact")
    if canonical_protocol == "operational_cascade":
        from docking_confidence.application.operational import load_operational_config
        operational_base = load_operational_config(operational_config)
        if canonical_docking_task != "local_refinement" or top_pockets != 1:
            raise ValueError("operational cascade requires local_refinement and one pocket")
        if population != operational_base.population or generations != operational_base.generations:
            raise ValueError("operational population/generations must match the operational configuration")
    if canonical_protocol == "factorial_2x2":
        from docking_confidence.domain.execution import FACTORIAL_EVALUATION_BUDGET, FactorialExperimentConfig
        controls = FactorialExperimentConfig(
            evaluation_budget=evaluation_budget if evaluation_budget is not None else FACTORIAL_EVALUATION_BUDGET,
            reduced_search_scale=reduced_search_scale if reduced_search_scale is not None else 0.5,
            max_translation=max_translation if max_translation is not None else 5.0,
            max_rotation_degrees=max_rotation_degrees if max_rotation_degrees is not None else 25.0,
            pose_source=pose_source or "",
            execution_order_seed=execution_order_seed if execution_order_seed is not None else 20260925,
        )
        controls.validate(population)
        if canonical_docking_task != "local_refinement" or experimental_protocol is None:
            raise ValueError("factorial_2x2 requires local_refinement and a prepared experimental protocol")
    elif any(value is not None for value in (evaluation_budget, pose_source, experimental_protocol,
        reduced_search_scale, max_translation, max_rotation_degrees, execution_order_seed)):
        raise ValueError("experimental controls require factorial_2x2")
    if (engine_archive_sha256 is None) != (engine_config_sha256 is None):
        raise ValueError("alternate engine pins must be provided together")
    if archive is None and any(value is not None for value in (
        engine_config_entry, engine_archive_sha256, engine_config_sha256,
    )):
        raise ValueError("archive options require engine_archive")
    first_seed = _nonnegative_int(seed_start, field="seed_start")
    number_of_seeds = _positive_int(seed_count, field="seed_count")
    seeds = tuple(range(first_seed, first_seed + number_of_seeds))

    command = [
        sys.executable,
        "-m",
        "docking_confidence",
        "batch-run",
        "--input-dir",
        str(Path(input_dir).expanduser()),
        "--output-dir",
        str(Path(output_dir).expanduser()),
        "--algorithms",
        *canonical_algorithms,
        "--protocol",
        canonical_protocol,
        "--docking-task",
        canonical_docking_task,
        "--seed",
        *[str(value) for value in seeds],
        "--generations",
        str(_positive_int(generations, field="generations")),
        "--pop-size",
        str(_positive_int(population, field="population")),
        "--top-pockets",
        str(_positive_int(top_pockets, field="top_pockets")),
        "--timeout-seconds",
        str(_positive_float(timeout_seconds, field="timeout_seconds")),
    ]
    if archive is not None:
        contract = load_dockingpp_archive_contract()
        canonical_config_entry = str(engine_config_entry or contract.config_entry).strip()
        if not canonical_config_entry:
            raise ValueError("engine_config_entry must not be empty")
        archive_pin = engine_archive_sha256 or contract.archive_sha256
        config_pin = engine_config_sha256 or contract.config_sha256
        command.extend([
            "--engine-archive", str(archive),
            "--engine-config-entry", canonical_config_entry,
            "--engine-archive-sha256", _sha256_pin(archive_pin, field="engine_archive_sha256"),
            "--engine-config-sha256", _sha256_pin(config_pin, field="engine_config_sha256"),
        ])
    if canonical_protocol == "factorial_2x2":
        protocol_path = Path(experimental_protocol).expanduser().resolve(strict=True)
        if not protocol_path.is_file():
            raise ValueError("experimental_protocol must be a JSON file")
        command.extend([
            "--evaluation-budget", str(controls.evaluation_budget),
            "--reduced-search-scale", str(controls.reduced_search_scale),
            "--max-translation", str(controls.max_translation),
            "--max-rotation-degrees", str(controls.max_rotation_degrees),
            "--pose-source", controls.pose_source,
            "--execution-order-seed", str(controls.execution_order_seed),
            "--experimental-protocol", str(protocol_path),
        ])
    if resume_after_cancel:
        command.append("--resume-after-cancel")
    if operational_config is not None:
        command.extend(["--operational-config", str(Path(operational_config).expanduser().resolve(strict=True))])
    if collect_operational_pairs:
        command.append("--collect-operational-pairs")
    if confidence_artifact is not None:
        artifact_path = Path(confidence_artifact).expanduser().resolve(strict=canonical_protocol != "operational_cascade")
        if canonical_protocol != "operational_cascade" and not artifact_path.is_file():
            raise ValueError("confidence_artifact must be an artifact.json file")
        command.extend(["--confidence-artifact", str(artifact_path)])
    return tuple(command)


def demo_batch_command(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    algorithms: Iterable[str],
    protocol: str,
    generations: int,
    population: int,
    top_pockets: int,
    seed_start: int,
    seed_count: int,
    resume_after_cancel: bool = False,
) -> tuple[str, ...]:
    command = [
        sys.executable,
        "-m",
        "docking_confidence",
        "batch-run-demo",
        "--input-dir",
        str(Path(input_dir).expanduser()),
        "--output-dir",
        str(Path(output_dir).expanduser()),
        "--algorithms",
        *[str(value) for value in algorithms],
        "--protocol",
        str(protocol),
        "--generations",
        str(int(generations)),
        "--population",
        str(int(population)),
        "--top-pockets",
        str(int(top_pockets)),
        "--seed-start",
        str(int(seed_start)),
        "--seed-count",
        str(int(seed_count)),
        "--acknowledge-non-scientific-demo",
    ]
    if resume_after_cancel:
        command.append("--resume-after-cancel")
    return tuple(command)


def _launch_batch_process(command: tuple[str, ...], *, log_kind: str) -> BatchProcess:
    output_index = command.index("--output-dir") + 1
    output = Path(command[output_index]).resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.parent / f".{output.name}.batch-{log_kind}.log"
    creationflags = 0
    if sys.platform == "win32":
        creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) | int(
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
    with log_path.open("ab") as log_handle:
        process = subprocess.Popen(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            close_fds=True,
            creationflags=creationflags,
            shell=False,
        )
    return BatchProcess(pid=int(process.pid), log_path=log_path, command=command)


def launch_real_batch(**command_options: object) -> BatchProcess:
    """Start the hash-pinned DockingPP worker outside the caller process."""

    command = real_batch_command(**command_options)  # type: ignore[arg-type]
    return _launch_batch_process(command, log_kind="real")


def launch_demo_batch(**command_options: object) -> BatchProcess:
    """Start the demo worker outside the Streamlit rerun process."""

    command = demo_batch_command(**command_options)  # type: ignore[arg-type]
    return _launch_batch_process(command, log_kind="demo")
