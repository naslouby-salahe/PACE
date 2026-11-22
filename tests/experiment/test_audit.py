from dataclasses import replace

import numpy as np

from pace.experiment.audit import audit_roles
from pace.experiment.runner import prepare_clients
from pace.types import ChronologicalIndices
from tests.support import campaign_configuration, campaign_devices


def test_every_synthetic_client_satisfies_the_role_contract() -> None:
    configuration = campaign_configuration()
    clients = campaign_devices()
    audits = audit_roles(prepare_clients(clients, configuration), clients)
    assert len(audits) == len(clients)
    for audit in audits:
        assert audit.roles_disjoint
        assert (
            audit.training_rows + audit.reference_rows + audit.calibration_rows + audit.test_rows
            == audit.benign_rows
        )
        assert audit.attack_types == 2
        assert audit.development_rows > 0
        assert audit.held_out_rows > 0


def test_a_chronology_violation_is_reported() -> None:
    configuration = campaign_configuration()
    clients = campaign_devices()
    prepared = prepare_clients(clients, configuration)
    roles = prepared[0].split.benign_roles
    swapped = object.__new__(ChronologicalIndices)
    for name, value in (
        ("training", roles.test),
        ("reference", roles.reference),
        ("calibration", roles.calibration),
        ("test", roles.training),
    ):
        object.__setattr__(swapped, name, value)
    tampered = replace(prepared[0], split=replace(prepared[0].split, benign_roles=swapped))
    audits = audit_roles((tampered, *prepared[1:]), clients)
    assert not audits[0].roles_disjoint
    assert all(item.roles_disjoint for item in audits[1:])


def test_overlapping_roles_are_reported() -> None:
    configuration = campaign_configuration()
    clients = campaign_devices()
    prepared = prepare_clients(clients, configuration)
    roles = prepared[0].split.benign_roles
    overlapping = object.__new__(ChronologicalIndices)
    for name, value in (
        ("training", roles.training),
        ("reference", roles.training),
        ("calibration", roles.calibration),
        ("test", roles.test),
    ):
        object.__setattr__(overlapping, name, value)
    tampered = replace(prepared[0], split=replace(prepared[0].split, benign_roles=overlapping))
    assert not audit_roles((tampered,), clients)[0].roles_disjoint


def test_standardised_training_roles_do_not_depend_on_any_other_role() -> None:
    configuration = campaign_configuration()
    clients = campaign_devices()
    baseline = prepare_clients(clients, configuration)
    plan = baseline[0].split.benign_roles
    perturbed_features = clients[0].benign_features.copy()
    for indices in (plan.reference, plan.calibration, plan.test):
        perturbed_features[indices] += 1000.0
    perturbed_captures = tuple(
        replace(capture, features=capture.features + 1000.0)
        for capture in clients[0].attack_captures
    )
    perturbed = replace(
        clients[0], benign_features=perturbed_features, attack_captures=perturbed_captures
    )
    after = prepare_clients((perturbed, *clients[1:]), configuration)
    for before_device, after_device in zip(baseline, after, strict=True):
        np.testing.assert_array_equal(before_device.training, after_device.training)
