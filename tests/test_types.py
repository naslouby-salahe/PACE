import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from pace.types import (
    AtLeastOneFloat,
    AttackCapture,
    AttackType,
    AttackTypeName,
    AttackTypePerformance,
    BootstrapReplicateCount,
    BudgetSweepPerformance,
    ClientName,
    DatasetClientId,
    DatasetName,
    DatasetValidationError,
    FiniteFloat,
    LocalExpertFamily,
    LocalExpertKind,
    MinimumAttackRows,
    MinimumPeerCount,
    NBaIoTAttackVariation,
    NonBlankText,
    NonNegativeInt,
    OpenUnitFloat,
    PositiveFloat,
    PositiveInt,
    ScientificDigest,
    SeedDomain,
    SignedUnitFloat,
    StudyStratum,
    TargetPerformance,
    TonIotType,
    UnitFloat,
    UnswAttackFamily,
    default_study_stratum,
)
from tests.support import target_provenance


@pytest.mark.parametrize(
    ("annotation", "invalid"),
    [
        (NonBlankText, "  "),
        (ScientificDigest, "bad"),
        (ScientificDigest, "A" * 64),
        (OpenUnitFloat, 1.0),
        (OpenUnitFloat, 0.0),
        (UnitFloat, math.nan),
        (UnitFloat, math.inf),
        (UnitFloat, -0.1),
        (UnitFloat, 1.1),
        (SignedUnitFloat, 1.5),
        (PositiveFloat, 0.0),
        (AtLeastOneFloat, 0.5),
        (NonNegativeInt, -1),
        (NonNegativeInt, True),
        (PositiveInt, 0),
        (MinimumPeerCount, 3),
        (MinimumAttackRows, 1),
        (BootstrapReplicateCount, 999),
        (FiniteFloat, math.inf),
    ],
)
def test_scalar_aliases_reject_invalid_values(annotation: object, invalid: object) -> None:
    adapter: TypeAdapter[object] = TypeAdapter(annotation)
    with pytest.raises(ValidationError):
        adapter.validate_python(invalid, strict=True)


@pytest.mark.parametrize(
    ("annotation", "valid"),
    [
        (UnitFloat, 0.0),
        (UnitFloat, 1.0),
        (OpenUnitFloat, 0.5),
        (SignedUnitFloat, -1.0),
        (AtLeastOneFloat, 1.0),
        (PositiveInt, 1),
        (NonNegativeInt, 0),
        (MinimumPeerCount, 4),
        (ScientificDigest, "a" * 64),
    ],
)
def test_scalar_aliases_accept_boundary_values(annotation: object, valid: object) -> None:
    assert TypeAdapter(annotation).validate_python(valid, strict=True) == valid


def test_expert_family_follows_the_member_not_its_name() -> None:
    assert LocalExpertKind.FEATURE_ISOLATION_FOREST.family is LocalExpertFamily.FEATURE
    assert LocalExpertKind.FEATURE_HISTOGRAM.family is LocalExpertFamily.FEATURE
    assert LocalExpertKind.FEATURE_GAUSSIAN.family is LocalExpertFamily.FEATURE
    assert LocalExpertKind.RESPONSE_ISOLATION_FOREST.family is LocalExpertFamily.RESPONSE
    assert LocalExpertKind.RARITY_PAIR_COOCCURRENCE.family is LocalExpertFamily.RARITY
    assert LocalExpertKind.RARITY_ECOD_SUM.family is LocalExpertFamily.RARITY


def test_default_study_stratum_maps_each_dataset_by_identity() -> None:
    assert default_study_stratum(DatasetName.IOT_23) is StudyStratum.IOT_23
    assert default_study_stratum(DatasetName.TON_IOT) is StudyStratum.TON_IOT
    assert default_study_stratum(DatasetName.GOTHAM_2025) is StudyStratum.GOTHAM_2025
    assert default_study_stratum(DatasetName.CTU_13) is StudyStratum.CTU_13
    assert default_study_stratum(DatasetName.N_BAIOT) is StudyStratum.N_BAIOT
    with pytest.raises(DatasetValidationError):
        default_study_stratum(DatasetName.UNSW_IOT_ATTACK_FLOWS)


def test_attack_identifiers_are_member_data_not_derived_from_the_python_name() -> None:
    assert NBaIoTAttackVariation.GAFGYT_COMBO.attack_type.identifier == "GAFGYT/COMBO"
    assert NBaIoTAttackVariation.MIRAI_UDPPLAIN.attack_type.identifier == "MIRAI/UDPPLAIN"
    assert UnswAttackFamily.TCP_SYN_DEVICE.attack_type.identifier == "TCP_SYN_DEVICE"
    assert UnswAttackFamily.ARP_SPOOF.attack_type.identifier == "ARP_SPOOF"
    assert TonIotType.DDOS.attack_type_name == "DDOS"
    assert TonIotType.INJECTION.attack_type_name == "INJECTION"
    assert TonIotType.MITM.attack_type_name == "MAN_IN_THE_MIDDLE"
    assert TonIotType.XSS.attack_type_name == "CROSS_SITE_SCRIPTING"
    assert TonIotType.NORMAL.attack_type_name is None


def test_attack_type_rejects_blank_identifiers() -> None:
    name = AttackTypeName(" ")
    with pytest.raises(ValueError):
        AttackType(name)


def test_attack_capture_rejects_non_increasing_source_order() -> None:
    parsed = AttackType(AttackTypeName("GAFGYT/COMBO"))
    Path_input = Path("capture.csv")
    matrix = np.zeros((2, 1), dtype=np.float64)
    array = np.asarray([0, 0], dtype=np.int64)
    with pytest.raises(ValueError, match="strictly increasing"):
        AttackCapture(
            parsed,
            Path_input,
            matrix,
            array,
        )


def _target_performance() -> TargetPerformance:
    target = DatasetClientId(DatasetName.N_BAIOT, ClientName("target"))
    seed = 7
    split_digest, peer_seeds, local_seed = target_provenance(target, seed)
    attack_type = AttackType(AttackTypeName("GAFGYT/COMBO"))
    return TargetPerformance(
        target=target,
        seed=seed,
        split_digest=split_digest,
        peer_model_seeds=peer_seeds,
        local_model_seed=local_seed,
        benign_test_rows=10,
        realized_false_alert_rate=0.02,
        attack_types=(
            AttackTypePerformance(
                attack_type=attack_type,
                test_rows=10,
                detection_rate=0.8,
                local_detection_rate=0.7,
                max_fusion_detection_rate=0.6,
                peer_union_detection_rate=0.5,
                bonferroni_detection_rate=0.4,
            ),
        ),
        local_false_alert_rate=0.01,
        max_fusion_false_alert_rate=0.03,
        peer_union_false_alert_rate=0.02,
        bonferroni_false_alert_rate=0.025,
        peer_count=4,
        budget_sweep=(
            BudgetSweepPerformance(multiplier=0.4, detection_rate=0.4, false_alert_rate=0.0),
            BudgetSweepPerformance(multiplier=1.4, detection_rate=0.8, false_alert_rate=0.2),
        ),
        study_stratum=StudyStratum.N_BAIOT,
    )


def test_target_performance_requires_complete_dataset_local_and_peer_seed_provenance() -> None:
    performance = _target_performance()

    with pytest.raises(ValueError, match="cover every contributing expert"):
        replace(performance, peer_model_seeds=performance.peer_model_seeds[:-1])
    with pytest.raises(ValueError, match="distinct non-target peers"):
        replace(
            performance,
            peer_model_seeds=(
                performance.peer_model_seeds[0],
                *performance.peer_model_seeds[:3],
            ),
        )
    foreign_peer = replace(
        performance.peer_model_seeds[0],
        owner=DatasetClientId(DatasetName.TON_IOT, ClientName("foreign")),
    )
    with pytest.raises(ValueError, match="target's dataset"):
        replace(
            performance,
            peer_model_seeds=(foreign_peer, *performance.peer_model_seeds[1:]),
        )
    wrong_peer_domain = replace(performance.peer_model_seeds[0], domain=SeedDomain.LOCAL_MODEL)
    with pytest.raises(ValueError, match="peer-model seed domain"):
        replace(
            performance,
            peer_model_seeds=(wrong_peer_domain, *performance.peer_model_seeds[1:]),
        )
    wrong_local_owner = replace(
        performance.local_model_seed,
        owner=performance.peer_model_seeds[0].owner,
    )
    with pytest.raises(ValueError, match="must belong to the target"):
        replace(performance, local_model_seed=wrong_local_owner)
