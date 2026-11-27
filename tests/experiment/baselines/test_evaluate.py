from pace.config import ResolvedConfiguration
from pace.experiment.baselines.evaluate import BaselineCache, baseline_curves
from pace.experiment.inputs import build_target_frame
from pace.experiment.runner import prepare_clients
from pace.types import ExternalBaseline, OperatingCurve
from tests.support import campaign_configuration, campaign_devices


def _diot_curves(
    configuration: ResolvedConfiguration, rows: int, cache: BaselineCache
) -> tuple[OperatingCurve, ...]:
    scientific = configuration.scientific
    selected = scientific.model_copy(
        update={
            "local_detectors": scientific.local_detectors.with_training_rows(rows),
            "baselines": scientific.baselines.model_copy(
                update={"methods": (ExternalBaseline.DIOT,)}
            ),
        }
    )
    prepared = prepare_clients(campaign_devices(), configuration)
    target, *peers = prepared
    return baseline_curves(target, tuple(peers), build_target_frame(target), selected, 3, cache)


def test_diot_scores_ignore_the_order_of_local_sizes() -> None:
    configuration = campaign_configuration()
    shared = BaselineCache()
    _diot_curves(configuration, 16, shared)
    after_other_size = _diot_curves(configuration, 32, shared)
    assert after_other_size == _diot_curves(configuration, 32, BaselineCache())
