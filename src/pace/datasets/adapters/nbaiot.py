import csv
from pathlib import Path

import numpy as np
import polars as pl

from pace.datasets.common import signed_log_feature_transform, target_cohort_role
from pace.types import (
    AttackCapture,
    ClientName,
    CorpusPath,
    CsvNewline,
    DatasetClientData,
    DatasetClientId,
    DatasetName,
    DatasetValidationError,
    FeatureName,
    FeatureSchemaId,
    FloatMatrix,
    NBaIoTAttackFamily,
    NBaIoTAttackVariation,
    NBaIoTLayout,
    NBaIoTSourceFile,
    SourceColumn,
    StudyStratum,
    TextEncoding,
)


def _stable_unique_rows(features: FloatMatrix) -> FloatMatrix:
    features += 0.0
    row_bytes = features.view(np.dtype((np.void, features.dtype.itemsize * features.shape[1])))
    _, first_indices = np.unique(row_bytes.ravel(), return_index=True)
    return features[np.sort(first_indices)]


def _feature_names(header: list[SourceColumn], source_file: Path) -> tuple[FeatureName, ...]:
    names = tuple(FeatureName(name.strip()) for name in header)
    if len(names) != NBaIoTLayout.FEATURE_COUNT or any(not name for name in names):
        raise DatasetValidationError(
            f"{source_file} must contain exactly {NBaIoTLayout.FEATURE_COUNT} named features"
        )
    if len(set(names)) != len(names):
        raise DatasetValidationError(f"{source_file} contains duplicate feature names")
    return names


def _read_feature_matrix(
    source_file: Path, expected_names: tuple[FeatureName, ...] | None
) -> tuple[tuple[FeatureName, ...], FloatMatrix]:
    try:
        with source_file.open(
            newline=CsvNewline.UNIVERSAL, encoding=TextEncoding.UTF8_SIG
        ) as csv_file:
            reader = csv.reader(csv_file)
            header = [SourceColumn(name) for name in next(reader)]
            first_data_row = next(reader, None)
        feature_names = _feature_names(header, source_file)
        if expected_names is not None and feature_names != expected_names:
            raise DatasetValidationError(f"{source_file} does not match the device feature schema")
        if first_data_row is None:
            raise DatasetValidationError(f"{source_file} contains no traffic records")
        if len(first_data_row) != NBaIoTLayout.FEATURE_COUNT:
            raise DatasetValidationError(f"{source_file} contains a malformed first record")
        features = pl.read_csv(source_file, infer_schema=False).cast(pl.Float64).to_numpy(order="c")
    except (OSError, StopIteration, ValueError, pl.exceptions.PolarsError) as error:
        raise DatasetValidationError(
            f"Could not read numeric features from {source_file}"
        ) from error
    if (
        features.shape[1] != NBaIoTLayout.FEATURE_COUNT
        or features.shape[0] == 0
        or not np.isfinite(features).all()
    ):
        raise DatasetValidationError(
            f"{source_file} must contain finite rows with {NBaIoTLayout.FEATURE_COUNT} features"
        )
    features = _stable_unique_rows(features)
    features = signed_log_feature_transform(features)
    features.setflags(write=False)
    return feature_names, features


def _attack_files(
    device_directory: Path, family: NBaIoTAttackFamily
) -> tuple[tuple[NBaIoTAttackVariation, Path], ...]:
    family_variations = tuple(
        variation for variation in NBaIoTAttackVariation if variation.family is family
    )
    expected_files = tuple(
        (variation, device_directory / family / variation.filename)
        for variation in family_variations
    )
    attack_directory = device_directory / family
    try:
        actual_files = tuple(
            sorted(path for path in attack_directory.glob("*.csv") if path.is_file())
        )
    except OSError as error:
        raise DatasetValidationError(
            f"Could not inspect attack captures in {attack_directory}"
        ) from error
    unexpected_files = set(actual_files) - {path for _, path in expected_files}
    if unexpected_files:
        raise DatasetValidationError(
            f"{attack_directory} contains unsupported attack capture files"
        )
    if attack_directory.is_dir() and not actual_files:
        raise DatasetValidationError(f"{attack_directory} contains no attack captures")
    return tuple(
        (variation, source_file)
        for variation, source_file in expected_files
        if source_file in actual_files
    )


def load_nbaiot_device(device_directory: Path) -> DatasetClientData:
    benign_file = device_directory / NBaIoTSourceFile.BENIGN
    feature_names, benign_features = _read_feature_matrix(benign_file, None)
    captures: list[AttackCapture] = []
    for family in NBaIoTAttackFamily:
        for variation, source_file in _attack_files(device_directory, family):
            _, attack_features = _read_feature_matrix(source_file, feature_names)
            captures.append(
                AttackCapture(
                    variation.attack_type,
                    source_file,
                    attack_features,
                    np.arange(attack_features.shape[0], dtype=np.int64),
                )
            )
    if not captures:
        raise DatasetValidationError(f"{device_directory} contains no attack captures")
    client = DatasetClientId(DatasetName.N_BAIOT, ClientName(device_directory.name))
    return DatasetClientData(
        client,
        StudyStratum.N_BAIOT,
        FeatureSchemaId.N_BAIOT_KITSUNE_SIGNED_LOG_V1,
        feature_names,
        benign_features,
        np.arange(benign_features.shape[0], dtype=np.int64),
        tuple(captures),
        target_cohort_role(client),
    )


def nbaiot_source_files(data_root: Path) -> tuple[Path, ...]:
    dataset_root = data_root / CorpusPath.N_BAIOT
    if not dataset_root.is_dir():
        raise DatasetValidationError(f"N-BaIoT data directory is unavailable: {dataset_root}")
    try:
        source_files = tuple(sorted(path for path in dataset_root.rglob("*.csv") if path.is_file()))
    except OSError as error:
        raise DatasetValidationError(f"Could not inspect N-BaIoT data in {dataset_root}") from error
    if not source_files:
        raise DatasetValidationError(f"No N-BaIoT CSV files were found in {dataset_root}")
    return source_files
