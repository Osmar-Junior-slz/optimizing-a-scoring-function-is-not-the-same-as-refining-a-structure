"""Camada estrutural de calibracao de confianca para uso futuro."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable


CALIBRATION_INTERFACE_VERSION = "1.0"


def _to_json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _to_json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(v) for v in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _to_json_compatible(tolist())
        except Exception:
            return str(value)
    return str(value)


@runtime_checkable
class ConfidenceCalibrator(Protocol):
    """Contrato minimo para calibradores futuros de confianca."""

    name: str
    version: str
    calibrated: bool
    decision_capable: bool

    def predict(self, *, confidence_features: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """Retorna payload de confianca calibrada, se disponivel."""


def build_confidence_calibration_metadata(
    *,
    confidence_features: Mapping[str, Any] | None,
    confidence_features_version: str | None = None,
    ready_for_calibration: bool,
    calibrator: ConfidenceCalibrator | None = None,
) -> dict[str, Any]:
    """Produz metadados estruturais de calibracao sem alterar decisao do pipeline."""

    if calibrator is None:
        return {
            "confidence_model_available": False,
            "confidence_calibrated": False,
            "confidence_decision_capable": False,
            "confidence_calibration_version": CALIBRATION_INTERFACE_VERSION,
            "confidence_output_available": False,
            "confidence_output_semantics": "no_calibrator_available",
            "confidence_output": None,
            "confidence_feature_schema_compatible": None,
            "confidence_runtime_feature_schema_version": confidence_features_version,
            "confidence_calibrator_feature_schema_version": None,
        }

    model_available = True
    calibrated = bool(getattr(calibrator, "calibrated", False))
    decision_capable = bool(
        getattr(calibrator, "decision_capable", False)
    )
    model_name = getattr(calibrator, "name", None)
    model_version = getattr(calibrator, "version", None)
    artifact_sha256 = getattr(calibrator, "artifact_sha256", None)
    calibrator_feature_schema_version = getattr(
        calibrator,
        "feature_schema_version",
        None,
    )
    feature_schema_compatible = not (
        confidence_features_version is not None
        and calibrator_feature_schema_version is not None
        and str(confidence_features_version) != str(calibrator_feature_schema_version)
    )
    calibration_version = (
        str(model_version) if model_version is not None else CALIBRATION_INTERFACE_VERSION
    )

    output = None
    semantics = "model_available_output_unavailable"
    error_type = None
    error_message = None
    if not feature_schema_compatible:
        semantics = "incompatible_feature_schema"
        error_type = "FeatureSchemaVersionMismatch"
        error_message = (
            "runtime confidence feature schema "
            f"{confidence_features_version!r} does not match calibrator schema "
            f"{calibrator_feature_schema_version!r}"
        )
    elif ready_for_calibration:
        try:
            raw_output = calibrator.predict(
                confidence_features=(
                    confidence_features if isinstance(confidence_features, Mapping) else {}
                )
            )
            output = _to_json_compatible(raw_output)
        except Exception as exc:  # noqa: BLE001 - persisted as auditable metadata
            output = None
            semantics = "calibrator_error"
            error_type = type(exc).__name__
            error_message = str(exc)
    else:
        semantics = "features_not_ready_for_calibration"

    output_available = isinstance(output, Mapping) and bool(output)
    if output_available:
        output = dict(output)
        output.setdefault("calibrated", calibrated)
        output.setdefault("decision_capable", decision_capable)
        output.setdefault("model_name", model_name)
        output.setdefault("model_version", model_version)
        output.setdefault("artifact_sha256", artifact_sha256)
        decision_capable = bool(output.get("decision_capable") is True)
        semantics = (
            "calibrated_output_available_for_policy"
            if calibrated and decision_capable
            else (
                "calibrated_output_observation_only"
                if calibrated
                else "uncalibrated_output_not_for_decision"
            )
        )

    return {
        "confidence_model_available": model_available,
        "confidence_model_name": model_name,
        "confidence_model_artifact_sha256": artifact_sha256,
        "confidence_calibrated": calibrated,
        "confidence_decision_capable": decision_capable,
        "confidence_calibration_version": calibration_version,
        "confidence_output_available": bool(output_available),
        "confidence_output_semantics": semantics,
        "confidence_output": output if output_available else None,
        "confidence_calibrator_error_type": error_type,
        "confidence_calibrator_error_message": error_message,
        "confidence_feature_schema_compatible": bool(feature_schema_compatible),
        "confidence_runtime_feature_schema_version": confidence_features_version,
        "confidence_calibrator_feature_schema_version": calibrator_feature_schema_version,
    }
