"""Unit tests for the xdist_compat module.

These tests verify the utility functions for pytest-xdist compatibility,
including worker detection, serialization, and data merging.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pytest_test_categories.distribution.stats import TestCounts
from pytest_test_categories.reporting import TestSizeReport
from pytest_test_categories.types import TestSize
from pytest_test_categories.suggestion import ResourceType, SuggestionCollector
from pytest_test_categories.xdist_compat import (
    WORKEROUTPUT_DISTRIBUTION_KEY,
    WORKEROUTPUT_REPORT_KEY,
    WORKEROUTPUT_SUGGESTION_KEY,
    XDIST_WORKER_ENV,
    deserialize_distribution_counts,
    is_xdist_controller,
    is_xdist_worker,
    merge_report_data,
    merge_suggestion_data,
    serialize_distribution_counts,
    serialize_report_data,
    serialize_suggestion_data,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.mark.small
class DescribeXdistWorkerDetection:
    """Tests for is_xdist_worker function."""

    def it_returns_false_when_env_var_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns False when PYTEST_XDIST_WORKER env var is not set."""
        monkeypatch.delenv(XDIST_WORKER_ENV, raising=False)

        assert is_xdist_worker() is False

    def it_returns_true_when_env_var_is_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns True when PYTEST_XDIST_WORKER env var is set."""
        monkeypatch.setenv(XDIST_WORKER_ENV, 'gw0')

        assert is_xdist_worker() is True


@pytest.mark.small
class DescribeXdistControllerDetection:
    """Tests for is_xdist_controller function."""

    def it_returns_false_when_xdist_plugin_not_loaded(self, mocker: MockerFixture) -> None:
        """Returns False when xdist plugin is not registered."""
        mock_config = mocker.Mock()
        mock_config.pluginmanager.hasplugin.return_value = False

        assert is_xdist_controller(mock_config) is False

    def it_returns_false_when_numprocesses_is_zero(
        self, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Returns False when numprocesses is 0 (xdist not active)."""
        monkeypatch.delenv(XDIST_WORKER_ENV, raising=False)

        mock_config = mocker.Mock()
        mock_config.pluginmanager.hasplugin.return_value = True
        mock_config.getoption.return_value = 0

        assert is_xdist_controller(mock_config) is False

    def it_returns_true_when_controller_with_workers(
        self, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Returns True when xdist is active and we're not a worker."""
        monkeypatch.delenv(XDIST_WORKER_ENV, raising=False)

        mock_config = mocker.Mock()
        mock_config.pluginmanager.hasplugin.return_value = True
        mock_config.getoption.return_value = 2

        assert is_xdist_controller(mock_config) is True

    def it_returns_true_when_numprocesses_is_auto(self, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns True when numprocesses is 'auto' and we're not a worker."""
        monkeypatch.delenv(XDIST_WORKER_ENV, raising=False)

        mock_config = mocker.Mock()
        mock_config.pluginmanager.hasplugin.return_value = True
        mock_config.getoption.return_value = 'auto'

        assert is_xdist_controller(mock_config) is True

    def it_returns_false_when_running_as_worker(self, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns False when we're running as a worker."""
        monkeypatch.setenv(XDIST_WORKER_ENV, 'gw0')

        mock_config = mocker.Mock()
        mock_config.pluginmanager.hasplugin.return_value = True
        mock_config.getoption.return_value = 2

        assert is_xdist_controller(mock_config) is False

    def it_handles_getoption_value_error(self, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns False when getoption raises ValueError."""
        monkeypatch.delenv(XDIST_WORKER_ENV, raising=False)

        mock_config = mocker.Mock()
        mock_config.pluginmanager.hasplugin.return_value = True
        mock_config.getoption.side_effect = ValueError('Invalid option')

        assert is_xdist_controller(mock_config) is False

    def it_handles_getoption_type_error(self, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns False when getoption raises TypeError."""
        monkeypatch.delenv(XDIST_WORKER_ENV, raising=False)

        mock_config = mocker.Mock()
        mock_config.pluginmanager.hasplugin.return_value = True
        mock_config.getoption.side_effect = TypeError('Type error')

        assert is_xdist_controller(mock_config) is False

    def it_handles_negative_numprocesses(self, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns False when numprocesses is negative (invalid config)."""
        monkeypatch.delenv(XDIST_WORKER_ENV, raising=False)

        mock_config = mocker.Mock()
        mock_config.pluginmanager.hasplugin.return_value = True
        mock_config.getoption.return_value = -1

        assert is_xdist_controller(mock_config) is False


@pytest.mark.small
class DescribeDistributionCountsSerialization:
    """Tests for distribution counts serialization."""

    def it_serializes_test_counts_to_dict(self) -> None:
        """Serializes TestCounts to a plain dict."""
        counts = TestCounts(small=10, medium=5, large=2, xlarge=1)

        result = serialize_distribution_counts(counts)

        assert result == {'small': 10, 'medium': 5, 'large': 2, 'xlarge': 1}

    def it_deserializes_dict_to_counts(self) -> None:
        """Deserializes dict back to counts format."""
        data = {'small': 10, 'medium': 5, 'large': 2, 'xlarge': 1}

        result = deserialize_distribution_counts(data)

        assert result == {'small': 10, 'medium': 5, 'large': 2, 'xlarge': 1}

    def it_handles_missing_keys_in_deserialization(self) -> None:
        """Deserializes dict with missing keys using defaults."""
        data = {'small': 10}

        result = deserialize_distribution_counts(data)

        assert result == {'small': 10, 'medium': 0, 'large': 0, 'xlarge': 0}


@pytest.mark.small
class DescribeReportDataSerialization:
    """Tests for report data serialization."""

    def it_serializes_empty_report(self) -> None:
        """Serializes an empty TestSizeReport."""
        report = TestSizeReport()

        result = serialize_report_data(report)

        assert result['sized_tests'] == {
            'small': [],
            'medium': [],
            'large': [],
            'xlarge': [],
        }
        assert result['unsized_tests'] == []
        assert result['test_durations'] == {}
        assert result['test_outcomes'] == {}

    def it_serializes_report_with_tests(self) -> None:
        """Serializes a TestSizeReport with test data."""
        report = TestSizeReport()
        report.add_test('test_one', TestSize.SMALL, duration=0.1, outcome='passed')
        report.add_test('test_two', TestSize.MEDIUM, duration=0.5, outcome='failed')

        result = serialize_report_data(report)

        sized_tests = result['sized_tests']
        assert isinstance(sized_tests, dict)
        assert 'test_one' in sized_tests['small']
        assert 'test_two' in sized_tests['medium']

        test_durations = result['test_durations']
        assert isinstance(test_durations, dict)
        assert test_durations['test_one'] == 0.1

        test_outcomes = result['test_outcomes']
        assert isinstance(test_outcomes, dict)
        assert test_outcomes['test_one'] == 'passed'
        assert test_outcomes['test_two'] == 'failed'


@pytest.mark.small
class DescribeReportDataMerging:
    """Tests for report data merging."""

    def it_merges_sized_tests_from_worker(self) -> None:
        """Merges sized tests from worker data into target report."""
        target = TestSizeReport()
        worker_data: dict[str, object] = {
            'sized_tests': {'small': ['test_a', 'test_b'], 'medium': ['test_c']},
            'unsized_tests': [],
            'test_durations': {},
            'test_outcomes': {},
        }

        merge_report_data(target, worker_data)

        assert 'test_a' in target.sized_tests[TestSize.SMALL]
        assert 'test_b' in target.sized_tests[TestSize.SMALL]
        assert 'test_c' in target.sized_tests[TestSize.MEDIUM]

    def it_merges_unsized_tests(self) -> None:
        """Merges unsized tests from worker data."""
        target = TestSizeReport()
        worker_data: dict[str, object] = {
            'sized_tests': {},
            'unsized_tests': ['test_unmarked'],
            'test_durations': {},
            'test_outcomes': {},
        }

        merge_report_data(target, worker_data)

        assert 'test_unmarked' in target.unsized_tests

    def it_merges_durations(self) -> None:
        """Merges test durations from worker data."""
        target = TestSizeReport()
        worker_data: dict[str, object] = {
            'sized_tests': {},
            'unsized_tests': [],
            'test_durations': {'test_a': 0.5, 'test_b': 1.2},
            'test_outcomes': {},
        }

        merge_report_data(target, worker_data)

        assert target.test_durations['test_a'] == 0.5
        assert target.test_durations['test_b'] == 1.2

    def it_merges_outcomes(self) -> None:
        """Merges test outcomes from worker data."""
        target = TestSizeReport()
        worker_data: dict[str, object] = {
            'sized_tests': {},
            'unsized_tests': [],
            'test_durations': {},
            'test_outcomes': {'test_a': 'passed', 'test_b': 'failed'},
        }

        merge_report_data(target, worker_data)

        assert target.test_outcomes['test_a'] == 'passed'
        assert target.test_outcomes['test_b'] == 'failed'

    def it_avoids_duplicate_tests(self) -> None:
        """Does not add duplicate tests when merging."""
        target = TestSizeReport()
        target.add_test('test_a', TestSize.SMALL)

        worker_data: dict[str, object] = {
            'sized_tests': {'small': ['test_a', 'test_b']},
            'unsized_tests': [],
            'test_durations': {},
            'test_outcomes': {},
        }

        merge_report_data(target, worker_data)

        # Should have test_a only once
        assert target.sized_tests[TestSize.SMALL].count('test_a') == 1
        assert 'test_b' in target.sized_tests[TestSize.SMALL]

    def it_handles_invalid_size_strings(self) -> None:
        """Ignores invalid size strings in worker data."""
        target = TestSizeReport()
        worker_data: dict[str, object] = {
            'sized_tests': {'invalid_size': ['test_a'], 'small': ['test_b']},
            'unsized_tests': [],
            'test_durations': {},
            'test_outcomes': {},
        }

        merge_report_data(target, worker_data)

        # test_a with invalid size should be skipped
        assert 'test_a' not in target.sized_tests[TestSize.SMALL]
        assert 'test_b' in target.sized_tests[TestSize.SMALL]

    def it_handles_non_list_tests_in_sized_tests(self) -> None:
        """Skips non-list values in sized_tests dict."""
        target = TestSizeReport()
        worker_data: dict[str, object] = {
            'sized_tests': {
                'small': 'not a list',
                'medium': ['test_b'],
            },
            'unsized_tests': [],
            'test_durations': {},
            'test_outcomes': {},
        }

        merge_report_data(target, worker_data)

        # small with non-list value should be skipped
        assert target.sized_tests[TestSize.SMALL] == []
        assert 'test_b' in target.sized_tests[TestSize.MEDIUM]

    def it_handles_missing_data_keys(self) -> None:
        """Handles missing keys in worker data gracefully."""
        target = TestSizeReport()
        worker_data: dict[str, object] = {}

        merge_report_data(target, worker_data)

        # Should not raise, target should be unchanged
        assert target.get_total_tests() == 0

    def it_handles_invalid_data_types(self) -> None:
        """Handles invalid data types gracefully."""
        target = TestSizeReport()
        worker_data: dict[str, object] = {
            'sized_tests': 'not a dict',
            'unsized_tests': 123,
            'test_durations': ['not', 'a', 'dict'],
            'test_outcomes': None,
        }

        merge_report_data(target, worker_data)

        # Should not raise, target should be unchanged
        assert target.get_total_tests() == 0


@pytest.mark.small
class DescribeWorkerOutputKeys:
    """Tests for worker output key constants."""

    def it_has_distribution_key(self) -> None:
        """Has a constant for distribution data key."""
        assert WORKEROUTPUT_DISTRIBUTION_KEY == 'test_categories_distribution'

    def it_has_report_key(self) -> None:
        """Has a constant for report data key."""
        assert WORKEROUTPUT_REPORT_KEY == 'test_categories_report'

    def it_has_worker_env_var(self) -> None:
        """Has a constant for worker env var name."""
        assert XDIST_WORKER_ENV == 'PYTEST_XDIST_WORKER'


@pytest.mark.small
class DescribeSuggestionAggregation:
    """Suggestion observations must survive the trip from an xdist worker to the controller.

    Without this the --test-categories-suggest flag produces NOTHING under -n, silently: workers
    record observations the controller never receives, and there is no error to notice.
    """

    def it_uses_a_distinct_workeroutput_key(self) -> None:
        """Its key does not collide with the distribution or report keys."""
        assert WORKEROUTPUT_SUGGESTION_KEY not in {WORKEROUTPUT_DISTRIBUTION_KEY, WORKEROUTPUT_REPORT_KEY}

    def it_round_trips_observations_and_durations(self) -> None:
        """A worker's collector serializes and merges into a controller's with nothing lost."""
        worker = SuggestionCollector()
        worker.record_observation('test_api.py::test_fetch', ResourceType.NETWORK, 'connect example.com:443')
        worker.record_observation('test_api.py::test_fetch', ResourceType.FILESYSTEM, 'open /tmp/x')
        worker.record_execution_time('test_api.py::test_fetch', 2.5)

        controller = SuggestionCollector()
        merge_suggestion_data(controller, serialize_suggestion_data(worker))

        assert controller.get_execution_time('test_api.py::test_fetch') == 2.5
        merged = controller.get_observations('test_api.py::test_fetch')
        assert {o.resource_type for o in merged} == {ResourceType.NETWORK, ResourceType.FILESYSTEM}
        assert {o.details for o in merged} == {'connect example.com:443', 'open /tmp/x'}

    def it_merges_two_workers_without_collision(self) -> None:
        """Each test belongs to exactly one worker, so merges are additive by node ID."""
        first, second = SuggestionCollector(), SuggestionCollector()
        first.record_observation('test_a.py::test_one', ResourceType.NETWORK, 'a')
        first.record_execution_time('test_a.py::test_one', 1.0)
        second.record_observation('test_b.py::test_two', ResourceType.SUBPROCESS, 'b')
        second.record_execution_time('test_b.py::test_two', 2.0)

        controller = SuggestionCollector()
        merge_suggestion_data(controller, serialize_suggestion_data(first))
        merge_suggestion_data(controller, serialize_suggestion_data(second))

        assert controller.get_execution_time('test_a.py::test_one') == 1.0
        assert controller.get_execution_time('test_b.py::test_two') == 2.0
        assert controller.observation_count == 2

    def it_omits_tests_that_recorded_no_observations(self) -> None:
        """A test with only a duration does not create an empty observation list on the wire."""
        worker = SuggestionCollector()
        worker.record_execution_time('test_c.py::test_quiet', 0.01)

        payload = serialize_suggestion_data(worker)

        assert payload['observations'] == {}
        assert payload['execution_times'] == {'test_c.py::test_quiet': 0.01}

    def it_does_not_carry_current_sizes_across_the_wire(self) -> None:
        """Current sizes stay on the controller, which collects every test itself.

        Re-sending them would reintroduce the double-count the distribution path guards against.
        """
        worker = SuggestionCollector()
        worker.record_current_size('test_d.py::test_marked', TestSize.SMALL)

        assert 'current_sizes' not in serialize_suggestion_data(worker)

    def it_tolerates_an_empty_payload(self) -> None:
        """A worker that ran no tests contributes nothing rather than raising."""
        controller = SuggestionCollector()
        merge_suggestion_data(controller, {})

        assert controller.observation_count == 0
