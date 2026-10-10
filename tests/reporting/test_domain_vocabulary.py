from math import isclose

from pace.reporting.records import (
    DetectionShares,
    FamilyRates,
    MethodRates,
    TargetRecord,
    decomposition_of,
    verdict_text,
)
from pace.types import (
    AcceptanceDesign,
    AcceptanceRule,
    AlphaLevel,
    AttackType,
    AttackTypeName,
    BudgetGuide,
    ClientName,
    ConfigurationSection,
    DatasetClientId,
    DatasetName,
    DetectionContribution,
    DisplayText,
    FalseAlertGuide,
    LeaveOneOutScope,
    SatisfiedAnswer,
    SeedPanel,
    StudyStratum,
    SummaryQuantile,
)


def _record(pace: float, local: float, peers: float) -> TargetRecord:
    rates = MethodRates(pace, 0.0)
    return TargetRecord(
        stratum=StudyStratum.N_BAIOT,
        target=DatasetClientId(DatasetName.N_BAIOT, ClientName("device")),
        alpha=AlphaLevel.ALPHA_005,
        rows=10,
        benign_test_rows=1,
        peer_count=1,
        pace=rates,
        mean_ensemble=rates,
        local=rates,
        peer_union=rates,
        max_fusion=rates,
        bonferroni=rates,
        arms=(),
        families=(
            FamilyRates(
                AttackType(AttackTypeName("SYN")),
                pace,
                pace,
                local,
                peers,
            ),
        ),
        sweep=(),
        seed_gains=(),
    )


def test_detection_shares_follow_the_attack_traffic_partition() -> None:
    shares = decomposition_of(_record(0.8, 0.3, 0.6))
    expected = DetectionShares(peers_only=0.5, local_only=0.2, both=0.1, neither=0.2)
    assert isclose(shares.peers_only, expected.peers_only)
    assert isclose(shares.local_only, expected.local_only)
    assert isclose(shares.both, expected.both)
    assert isclose(shares.neither, expected.neither)
    assert shares.share(DetectionContribution.PEERS_ONLY) == shares.peers_only
    assert shares.share(DetectionContribution.NEITHER) == shares.neither
    assert tuple(part.label for part in DetectionContribution.stack_order()) == (
        DisplayText("peers only"),
        DisplayText("both"),
        DisplayText("local only"),
        DisplayText("neither"),
    )


def test_verdict_text_uses_the_satisfied_answer_enum() -> None:
    assert verdict_text(True) == DisplayText(SatisfiedAnswer.YES)
    assert verdict_text(False) == DisplayText(SatisfiedAnswer.NO)


def test_reporting_quantities_keep_their_scientific_values() -> None:
    assert SummaryQuantile.LOWER_DECILE.level == 0.1
    assert SummaryQuantile.UPPER_DECILE.level == 0.9
    assert FalseAlertGuide.TWICE.factor == AcceptanceRule.EXCEEDANCE_MULTIPLE.threshold
    assert FalseAlertGuide.NOMINAL.factor == 1.0
    reserve = 0.5
    alpha = AlphaLevel.ALPHA_005
    assert BudgetGuide.HALF_LOCAL.budget(alpha, reserve) == alpha.fraction * reserve / 2.0
    assert BudgetGuide.LOCAL.budget(alpha, reserve) == alpha.fraction * reserve
    assert SeedPanel.WITHOUT_EXCLUDED_DATASET.excluded_dataset is (
        AcceptanceDesign.PROTOCOL.excluded_dataset
    )
    assert SeedPanel.WITHOUT_EXCLUDED_DATASET.label == DisplayText("without TON_IoT")
    assert SeedPanel.ALL_TARGETS.excluded_dataset is None
    assert {scope.column_label for scope in LeaveOneOutScope} == {
        DisplayText("target"),
        DisplayText("stratum"),
    }
    assert {section for section in ConfigurationSection} == {
        ConfigurationSection.SCIENTIFIC,
        ConfigurationSection.SPLITTING,
        ConfigurationSection.REPORTING,
        ConfigurationSection.PROTOCOL_MATRIX,
    }
