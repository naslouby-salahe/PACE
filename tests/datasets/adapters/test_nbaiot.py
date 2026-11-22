import csv
import shutil
from math import isclose, log1p
from pathlib import Path

import numpy as np
import pytest

from pace.datasets.adapters.nbaiot import load_nbaiot_device, nbaiot_source_files
from pace.types import (
    DatasetValidationError,
    NBaIoTAttackFamily,
    NBaIoTAttackVariation,
)

FEATURE_NAMES = tuple(f"feature_{index}" for index in range(115))


FAMILY_FILES = {
    "gafgyt_attacks": ("combo", "junk", "scan", "tcp", "udp"),
    "mirai_attacks": ("ack", "scan", "syn", "udp", "udpplain"),
}


def _write_capture(
    path: Path,
    feature_names: tuple[str, ...],
    first_feature_values: tuple[float, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(feature_names)
        for first_feature_value in first_feature_values:
            writer.writerow((first_feature_value, *(0.0 for _ in feature_names[1:])))


def _create_device_directory(device_directory: Path) -> tuple[tuple[Path, bytes], ...]:
    _write_capture(
        device_directory / "benign_traffic.csv",
        FEATURE_NAMES,
        (11.0, 12.0),
    )
    for family_index, (family_directory, variations) in enumerate(FAMILY_FILES.items()):
        for variation_index, variation in enumerate(variations):
            _write_capture(
                device_directory / family_directory / f"{variation}.csv",
                FEATURE_NAMES,
                (
                    100.0 + family_index * 10 + variation_index,
                    101.0 + family_index * 10 + variation_index,
                ),
            )
    return tuple((path, path.read_bytes()) for path in sorted(device_directory.rglob("*.csv")))


def test_nbaiot_adapter_preserves_rows_and_labels_attack_captures(
    tmp_path: Path,
) -> None:
    assert tuple(NBaIoTAttackFamily) == (
        NBaIoTAttackFamily.GAFGYT,
        NBaIoTAttackFamily.MIRAI,
    )
    assert tuple(NBaIoTAttackVariation) == (
        NBaIoTAttackVariation.GAFGYT_COMBO,
        NBaIoTAttackVariation.GAFGYT_JUNK,
        NBaIoTAttackVariation.GAFGYT_SCAN,
        NBaIoTAttackVariation.GAFGYT_TCP,
        NBaIoTAttackVariation.GAFGYT_UDP,
        NBaIoTAttackVariation.MIRAI_ACK,
        NBaIoTAttackVariation.MIRAI_SCAN,
        NBaIoTAttackVariation.MIRAI_SYN,
        NBaIoTAttackVariation.MIRAI_UDP,
        NBaIoTAttackVariation.MIRAI_UDPPLAIN,
    )
    device_directory = tmp_path / "N-BaIoT" / "Danmini_Doorbell"
    source_contents = _create_device_directory(device_directory)

    device = load_nbaiot_device(device_directory)

    assert device.client.name == "Danmini_Doorbell"
    assert device.client.dataset.name == "N_BAIOT"
    assert device.feature_names == FEATURE_NAMES
    assert isclose(device.benign_features[0, 0], log1p(11), rel_tol=1e-6)
    assert isclose(device.benign_features[1, 0], log1p(12), rel_tol=1e-6)
    assert device.benign_source_order.tolist() == [0, 1]
    assert len(device.attack_captures) == 10
    assert device.attack_captures[0].attack_type.identifier == "GAFGYT/COMBO"
    assert device.attack_captures[-1].attack_type.identifier == "MIRAI/UDPPLAIN"
    assert isclose(device.attack_captures[0].features[0, 0], log1p(100), rel_tol=1e-6)
    assert isclose(device.attack_captures[0].features[1, 0], log1p(101), rel_tol=1e-6)
    assert device.attack_captures[0].source_order.tolist() == [0, 1]
    assert all(not capture.source_order.flags.writeable for capture in device.attack_captures)
    assert all(not capture.features.flags.writeable for capture in device.attack_captures)
    assert not device.benign_source_order.flags.writeable
    assert not device.benign_features.flags.writeable
    assert tuple((path, path.read_bytes()) for path, _ in source_contents) == source_contents


def test_nbaiot_adapter_deduplicates_stably_within_each_source_file(tmp_path: Path) -> None:
    device_directory = tmp_path / "N-BaIoT" / "Danmini_Doorbell"
    _write_capture(device_directory / "benign_traffic.csv", FEATURE_NAMES, (11.0, 12.0, 11.0))
    _write_capture(
        device_directory / "gafgyt_attacks" / "combo.csv",
        FEATURE_NAMES,
        (100.0, 101.0, 100.0),
    )
    device = load_nbaiot_device(device_directory)

    assert isclose(device.benign_features[0, 0], log1p(11), rel_tol=1e-6)
    assert isclose(device.benign_features[1, 0], log1p(12), rel_tol=1e-6)
    assert device.benign_source_order.tolist() == [0, 1]
    assert isclose(device.attack_captures[0].features[0, 0], log1p(100), rel_tol=1e-6)
    assert isclose(device.attack_captures[0].features[1, 0], log1p(101), rel_tol=1e-6)
    assert device.attack_captures[0].source_order.tolist() == [0, 1]


def test_nbaiot_deduplication_treats_signed_zero_as_equal(tmp_path: Path) -> None:
    device_directory = tmp_path / "N-BaIoT" / "Danmini_Doorbell"
    _write_capture(device_directory / "benign_traffic.csv", FEATURE_NAMES, (-0.0, 0.0, 1.0))
    _write_capture(
        device_directory / "gafgyt_attacks" / "combo.csv",
        FEATURE_NAMES,
        (100.0,),
    )

    device = load_nbaiot_device(device_directory)

    assert np.allclose(device.benign_features[:, 0], [0.0, log1p(1)], rtol=1e-6)
    assert device.benign_source_order.tolist() == [0, 1]


def test_nbaiot_adapter_rejects_misaligned_feature_headers(tmp_path: Path) -> None:
    device_directory = tmp_path / "N-BaIoT" / "Danmini_Doorbell"
    _create_device_directory(device_directory)
    mismatched_names = (*FEATURE_NAMES[:-1], "different_feature")
    _write_capture(
        device_directory / "mirai_attacks" / "ack.csv",
        mismatched_names,
        (1.0,),
    )

    with pytest.raises(DatasetValidationError, match="feature schema"):
        load_nbaiot_device(device_directory)


def test_nbaiot_adapter_allows_device_without_one_attack_family(tmp_path: Path) -> None:
    device_directory = tmp_path / "N-BaIoT" / "Ennio_Doorbell"
    _create_device_directory(device_directory)
    shutil.rmtree(device_directory / "mirai_attacks")

    device = load_nbaiot_device(device_directory)

    assert len(device.attack_captures) == 5
    assert all(
        capture.attack_type.identifier.startswith("GAFGYT/") for capture in device.attack_captures
    )


def test_nbaiot_adapter_rejects_non_finite_data_and_accepts_device_specific_coverage(
    tmp_path: Path,
) -> None:
    device_directory = tmp_path / "N-BaIoT" / "Danmini_Doorbell"
    _create_device_directory(device_directory)
    non_finite_capture = device_directory / "gafgyt_attacks" / "combo.csv"
    with non_finite_capture.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(FEATURE_NAMES)
        writer.writerow(("nan", *(0.0 for _ in FEATURE_NAMES[1:])))
    with pytest.raises(DatasetValidationError, match="finite"):
        load_nbaiot_device(device_directory)

    non_finite_capture.unlink()
    _write_capture(non_finite_capture, FEATURE_NAMES, (100.0, 101.0))
    (device_directory / "mirai_attacks" / "ack.csv").unlink()
    device = load_nbaiot_device(device_directory)
    assert len(device.attack_captures) == 9
    assert all(capture.attack_type.identifier != "MIRAI/ACK" for capture in device.attack_captures)

    _write_capture(
        device_directory / "mirai_attacks" / "unexpected.csv",
        FEATURE_NAMES,
        (1.0,),
    )
    with pytest.raises(DatasetValidationError, match="unsupported attack capture"):
        load_nbaiot_device(device_directory)


def _write_rows(path: Path, header: tuple[str, ...], rows: tuple[tuple[object, ...], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(rows)


def _valid_row(value: float = 1.0) -> tuple[object, ...]:
    return (value, *(0.0 for _ in FEATURE_NAMES[1:]))


def _device(tmp_path: Path) -> Path:
    device = tmp_path / "Device"
    _write_rows(device / "benign_traffic.csv", FEATURE_NAMES, (_valid_row(1.0), _valid_row(2.0)))
    return device


def _add_attack(device: Path, family: str, name: str) -> None:
    _write_rows(device / family / f"{name}.csv", FEATURE_NAMES, (_valid_row(5.0),))


@pytest.mark.parametrize(
    ("header", "message"),
    [
        (FEATURE_NAMES[:-1], "exactly 115 named features"),
        (("", *FEATURE_NAMES[1:]), "exactly 115 named features"),
        ((FEATURE_NAMES[0], *FEATURE_NAMES[:-1]), "duplicate feature names"),
    ],
)
def test_benign_header_must_name_unique_features(
    tmp_path: Path, header: tuple[str, ...], message: str
) -> None:
    device = tmp_path / "Device"
    _write_rows(device / "benign_traffic.csv", header, (_valid_row(),))
    with pytest.raises(DatasetValidationError, match=message):
        load_nbaiot_device(device)


def test_benign_file_without_rows_is_rejected(tmp_path: Path) -> None:
    device = tmp_path / "Device"
    _write_rows(device / "benign_traffic.csv", FEATURE_NAMES, ())
    with pytest.raises(DatasetValidationError, match="contains no traffic records"):
        load_nbaiot_device(device)


def test_malformed_first_record_is_rejected(tmp_path: Path) -> None:
    device = tmp_path / "Device"
    _write_rows(device / "benign_traffic.csv", FEATURE_NAMES, ((1.0, 2.0),))
    with pytest.raises(DatasetValidationError, match="malformed first record"):
        load_nbaiot_device(device)


def test_missing_and_non_numeric_files_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(DatasetValidationError, match="Could not read numeric features"):
        load_nbaiot_device(tmp_path / "Absent")
    device = tmp_path / "Device"
    _write_rows(device / "benign_traffic.csv", FEATURE_NAMES, (("text", *[0.0] * 114),))
    with pytest.raises(DatasetValidationError, match="Could not read numeric features"):
        load_nbaiot_device(device)


def test_unsupported_and_empty_attack_directories_are_rejected(tmp_path: Path) -> None:
    unsupported = _device(tmp_path / "a")
    _add_attack(unsupported, "mirai_attacks", "not_a_variation")
    with pytest.raises(DatasetValidationError, match="unsupported attack capture files"):
        load_nbaiot_device(unsupported)
    empty = _device(tmp_path / "b")
    (empty / "gafgyt_attacks").mkdir()
    with pytest.raises(DatasetValidationError, match="contains no attack captures"):
        load_nbaiot_device(empty)


def test_device_without_any_attack_capture_is_rejected(tmp_path: Path) -> None:
    device_input = _device(tmp_path)
    with pytest.raises(DatasetValidationError, match="contains no attack captures"):
        load_nbaiot_device(device_input)


def test_attack_capture_must_share_the_benign_schema(tmp_path: Path) -> None:
    device = _device(tmp_path)
    renamed = ("renamed", *FEATURE_NAMES[1:])
    _write_rows(device / "mirai_attacks" / "ack.csv", renamed, (_valid_row(),))
    with pytest.raises(DatasetValidationError, match="does not match the device feature schema"):
        load_nbaiot_device(device)


def test_source_file_discovery_requires_a_dataset_directory_with_csv_files(tmp_path: Path) -> None:
    with pytest.raises(DatasetValidationError, match="data directory is unavailable"):
        nbaiot_source_files(tmp_path)
    (tmp_path / "N-BaIoT").mkdir()
    with pytest.raises(DatasetValidationError, match="No N-BaIoT CSV files"):
        nbaiot_source_files(tmp_path)
    _write_rows(tmp_path / "N-BaIoT" / "Device" / "benign_traffic.csv", FEATURE_NAMES, ())
    assert len(nbaiot_source_files(tmp_path)) == 1
