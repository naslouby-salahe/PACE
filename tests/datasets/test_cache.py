import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import pace.datasets.cache as cache_module
from pace.datasets.cache import cache_key, load_cached_clients, store_clients
from pace.types import CacheFileName, DatasetName, LabNetwork, LockMode, StudyStratum
from tests.support import campaign_devices

LAN = LabNetwork("192.168.1.0/24")
OTHER_LAN = LabNetwork("10.0.0.0/24")


def test_cached_clients_round_trip_bit_identically(tmp_path: Path) -> None:
    clients = campaign_devices()
    key = cache_key(DatasetName.N_BAIOT, StudyStratum.N_BAIOT, "1" * 64, LAN)
    assert load_cached_clients(tmp_path, key) is None
    store_clients(tmp_path, key, clients)
    restored = load_cached_clients(tmp_path, key)
    assert restored is not None
    assert len(restored) == len(clients)
    for original, loaded in zip(clients, restored, strict=True):
        assert (original.client, original.study_stratum, original.feature_schema) == (
            loaded.client,
            loaded.study_stratum,
            loaded.feature_schema,
        )
        assert original.feature_names == loaded.feature_names
        assert original.target_cohort_role is loaded.target_cohort_role
        np.testing.assert_array_equal(original.benign_features, loaded.benign_features)
        np.testing.assert_array_equal(original.benign_source_order, loaded.benign_source_order)
        assert len(original.attack_captures) == len(loaded.attack_captures)
        for source, copy in zip(original.attack_captures, loaded.attack_captures, strict=True):
            assert source.attack_type == copy.attack_type
            assert source.source_file == copy.source_file
            assert source.source_population_rows == copy.source_population_rows
            np.testing.assert_array_equal(source.features, copy.features)
            np.testing.assert_array_equal(source.source_order, copy.source_order)


def test_cache_key_tracks_inputs_and_corruption_is_a_miss(tmp_path: Path) -> None:
    first = cache_key(DatasetName.N_BAIOT, StudyStratum.N_BAIOT, "1" * 64, LAN)
    assert first != cache_key(DatasetName.N_BAIOT, StudyStratum.N_BAIOT, "2" * 64, LAN)
    assert first != cache_key(DatasetName.IOT_23, StudyStratum.IOT_23, "1" * 64, LAN)
    assert first != cache_key(DatasetName.N_BAIOT, StudyStratum.N_BAIOT, "1" * 64, OTHER_LAN)
    store_clients(tmp_path, first, campaign_devices())
    (tmp_path / first / "manifest.json").write_text("{broken", encoding="utf-8")
    assert load_cached_clients(tmp_path, first) is None


def test_storing_a_cache_prunes_superseded_copies_of_the_same_stratum_only(
    tmp_path: Path,
) -> None:
    clients = campaign_devices()
    old = cache_key(DatasetName.N_BAIOT, StudyStratum.N_BAIOT, "1" * 64, LAN)
    new = cache_key(DatasetName.N_BAIOT, StudyStratum.N_BAIOT, "2" * 64, LAN)
    foreign = cache_key(DatasetName.IOT_23, StudyStratum.IOT_23, "3" * 64, LAN)
    store_clients(tmp_path, old, clients)
    store_clients(tmp_path, foreign, clients)
    manifest = tmp_path / foreign / "manifest.json"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace(
            f'"study_stratum":"{StudyStratum.N_BAIOT.value}"',
            f'"study_stratum":"{StudyStratum.IOT_23.value}"',
        ),
        encoding="utf-8",
    )
    (tmp_path / f"{CacheFileName.STAGING}crashed").mkdir()
    (tmp_path / "unreadable").mkdir()

    store_clients(tmp_path, new, clients)

    names = sorted(path.name for path in tmp_path.iterdir())
    assert [name for name in names if not name.endswith(".lock")] == sorted(
        (new, foreign, f"{CacheFileName.STAGING}crashed", "unreadable")
    )
    assert f".{old}.lock" not in names
    assert load_cached_clients(tmp_path, new) is not None
    assert (tmp_path / foreign / "manifest.json").is_file()


def test_simultaneous_stores_of_one_key_always_leave_a_complete_loadable_entry(
    tmp_path: Path,
) -> None:
    clients = campaign_devices()
    key = cache_key(DatasetName.N_BAIOT, StudyStratum.N_BAIOT, "4" * 64, LAN)
    gate = threading.Barrier(5)

    def store(_worker: int) -> None:
        gate.wait()
        store_clients(tmp_path, key, clients)

    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(store, range(5)))

    restored = load_cached_clients(tmp_path, key)
    assert restored is not None
    for original, loaded in zip(clients, restored, strict=True):
        np.testing.assert_array_equal(original.benign_features, loaded.benign_features)


def test_a_reader_holding_an_entry_blocks_pruning_until_it_is_done(tmp_path: Path) -> None:
    clients = campaign_devices()
    old = cache_key(DatasetName.N_BAIOT, StudyStratum.N_BAIOT, "1" * 64, LAN)
    new = cache_key(DatasetName.N_BAIOT, StudyStratum.N_BAIOT, "2" * 64, LAN)
    newest = cache_key(DatasetName.N_BAIOT, StudyStratum.N_BAIOT, "3" * 64, LAN)
    store_clients(tmp_path, old, clients)

    with cache_module._entry_lock(tmp_path, old, LockMode.SHARED):  # pyright: ignore[reportPrivateUsage]
        store_clients(tmp_path, new, clients)
        assert load_cached_clients(tmp_path, old) is not None

    store_clients(tmp_path, newest, clients)

    assert load_cached_clients(tmp_path, old) is None
    assert load_cached_clients(tmp_path, new) is None
    assert load_cached_clients(tmp_path, newest) is not None
