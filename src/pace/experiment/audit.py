import numpy as np
import structlog

from pace.config import ResolvedConfiguration
from pace.datasets.cache import resolve_clients
from pace.datasets.registry import dataset_source_files, load_dataset
from pace.experiment.inputs import PreparedDevice
from pace.experiment.runner import prepare_clients
from pace.experiment.sources import digest_source_files
from pace.types import (
    ChronologicalIndices,
    CriterionSatisfied,
    DatasetClientData,
    LogEvent,
    RoleAudit,
    StudyStratum,
)

logger = structlog.get_logger()


def _benign_roles_hold(
    device: DatasetClientData, roles: ChronologicalIndices
) -> CriterionSatisfied:
    parts = tuple(indices for _, indices in roles.by_role())
    joined = np.concatenate(parts)
    covered = np.unique(joined).size == joined.size == device.benign_features.shape[0]
    order = device.benign_source_order
    chronological = all(
        np.max(order[earlier]) <= np.min(order[later])
        for earlier, later in zip(parts[:-1], parts[1:], strict=True)
        if earlier.size and later.size
    )
    return covered and chronological


def _attack_halves_hold(device: DatasetClientData, prepared: PreparedDevice) -> CriterionSatisfied:
    orders = {capture.attack_type: capture.source_order for capture in device.attack_captures}
    for split in prepared.split.attack_splits:
        development, held_out = split.indices.peer_development, split.indices.held_out
        if np.intersect1d(development, held_out).size:
            return False
        order = orders[split.attack_type]
        if np.max(order[development]) > np.min(order[held_out]):
            return False
    return True


def audit_roles(
    prepared: tuple[PreparedDevice, ...], clients: tuple[DatasetClientData, ...]
) -> tuple[RoleAudit, ...]:
    by_client = {client.client: client for client in clients}
    return tuple(
        RoleAudit(
            stratum=device.study_stratum,
            target=device.client,
            benign_rows=by_client[device.client].benign_features.shape[0],
            training_rows=device.training.shape[0],
            reference_rows=device.reference.shape[0],
            calibration_rows=device.calibration.shape[0],
            test_rows=device.test.shape[0],
            attack_types=len(device.split.attack_splits),
            development_rows=sum(features.shape[0] for _, features in device.development),
            held_out_rows=sum(features.shape[0] for _, features in device.held_out),
            roles_disjoint=_benign_roles_hold(by_client[device.client], device.split.benign_roles)
            and _attack_halves_hold(by_client[device.client], device),
        )
        for device in prepared
    )


def audit_registered_stratum(
    stratum: StudyStratum, configuration: ResolvedConfiguration
) -> tuple[RoleAudit, ...]:
    root = configuration.runtime.raw_data_root
    files = dataset_source_files(root, stratum.dataset, stratum)
    clients = resolve_clients(
        lambda: (
            load_dataset(root, configuration.runtime.lab_network, stratum.dataset, stratum).clients
        ),
        configuration,
        (stratum.dataset, stratum),
        digest_source_files(files, root),
        logger,
    )
    audits = audit_roles(prepare_clients(clients, configuration), clients)
    logger.info(LogEvent.AUDIT_RECORDED, study_stratum=stratum.name, client_count=len(audits))
    return audits
