"""Test the network blocker adapters.

This module tests both the FakeNetworkBlocker (test adapter) and
SocketPatchingNetworkBlocker (production adapter) implementations.

The network blockers follow hexagonal architecture:
- NetworkBlockerPort is the Port (interface)
- FakeNetworkBlocker is a Test Adapter (test double)
- SocketPatchingNetworkBlocker is a Production Adapter (real implementation)

This follows the same pattern as the timer module (TestTimer/FakeTimer/WallTimer).
"""

from __future__ import annotations

import socket
from typing import TYPE_CHECKING

import pytest
from icontract import ViolationError

from pytest_test_categories.adapters.fake_network import FakeNetworkBlocker
from pytest_test_categories.adapters.network import SocketPatchingNetworkBlocker
from pytest_test_categories.exceptions import NetworkAccessViolationError
from pytest_test_categories.ports.network import (
    BlockerState,
    ConnectionAttempt,
    EnforcementMode,
    is_localhost,
)
from pytest_test_categories.types import (
    NetworkMode,
    TestSize,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.mark.small
class DescribeFakeNetworkBlocker:
    """Tests for the FakeNetworkBlocker test double."""

    def it_starts_in_inactive_state(self) -> None:
        """Verify the blocker initializes in INACTIVE state."""
        blocker = FakeNetworkBlocker()

        assert blocker.state == BlockerState.INACTIVE

    def it_transitions_to_active_on_activate(self) -> None:
        """Verify activate() transitions from INACTIVE to ACTIVE."""
        blocker = FakeNetworkBlocker()

        blocker.activate(TestSize.SMALL, EnforcementMode.STRICT)

        assert blocker.state == BlockerState.ACTIVE

    def it_transitions_to_inactive_on_deactivate(self) -> None:
        """Verify deactivate() transitions from ACTIVE to INACTIVE."""
        blocker = FakeNetworkBlocker()
        blocker.activate(TestSize.SMALL, EnforcementMode.STRICT)

        blocker.deactivate()

        assert blocker.state == BlockerState.INACTIVE

    def it_fails_to_activate_when_already_active(self) -> None:
        """Verify activate() raises when already ACTIVE."""
        blocker = FakeNetworkBlocker()
        blocker.activate(TestSize.SMALL, EnforcementMode.STRICT)

        with pytest.raises(ViolationError, match='INACTIVE'):
            blocker.activate(TestSize.SMALL, EnforcementMode.STRICT)

    def it_fails_to_deactivate_when_inactive(self) -> None:
        """Verify deactivate() raises when already INACTIVE."""
        blocker = FakeNetworkBlocker()

        with pytest.raises(ViolationError, match='ACTIVE'):
            blocker.deactivate()

    def it_records_activation_parameters(self) -> None:
        """Verify the blocker records test size and enforcement mode."""
        blocker = FakeNetworkBlocker()

        blocker.activate(TestSize.MEDIUM, EnforcementMode.WARN)

        assert blocker.current_test_size == TestSize.MEDIUM
        assert blocker.current_enforcement_mode == EnforcementMode.WARN

    def it_blocks_all_connections_for_small_tests(self) -> None:
        """Verify small tests cannot access any network."""
        blocker = FakeNetworkBlocker()
        blocker.activate(TestSize.SMALL, EnforcementMode.STRICT)

        assert blocker.check_connection_allowed('localhost', 8080) is False
        assert blocker.check_connection_allowed('127.0.0.1', 80) is False
        assert blocker.check_connection_allowed('api.example.com', 443) is False

    def it_allows_localhost_for_medium_tests(self) -> None:
        """Verify medium tests can access localhost only."""
        blocker = FakeNetworkBlocker()
        blocker.activate(TestSize.MEDIUM, EnforcementMode.STRICT)

        assert blocker.check_connection_allowed('localhost', 8080) is True
        assert blocker.check_connection_allowed('127.0.0.1', 80) is True
        assert blocker.check_connection_allowed('::1', 443) is True
        assert blocker.check_connection_allowed('api.example.com', 443) is False

    def it_allows_all_connections_for_large_tests(self) -> None:
        """Verify large tests can access any network."""
        blocker = FakeNetworkBlocker()
        blocker.activate(TestSize.LARGE, EnforcementMode.STRICT)

        assert blocker.check_connection_allowed('localhost', 8080) is True
        assert blocker.check_connection_allowed('api.example.com', 443) is True
        assert blocker.check_connection_allowed('external.service.io', 9000) is True

    def it_allows_all_connections_for_xlarge_tests(self) -> None:
        """Verify xlarge tests can access any network."""
        blocker = FakeNetworkBlocker()
        blocker.activate(TestSize.XLARGE, EnforcementMode.STRICT)

        assert blocker.check_connection_allowed('localhost', 8080) is True
        assert blocker.check_connection_allowed('api.example.com', 443) is True

    def it_records_connection_attempts(self) -> None:
        """Verify the blocker tracks connection attempts."""
        blocker = FakeNetworkBlocker()
        blocker.activate(TestSize.SMALL, EnforcementMode.STRICT)

        blocker.check_connection_allowed('api.example.com', 443)
        blocker.check_connection_allowed('localhost', 8080)

        assert len(blocker.connection_attempts) == 2
        assert blocker.connection_attempts[0] == ConnectionAttempt(
            host='api.example.com',
            port=443,
            test_nodeid='',
            allowed=False,
        )
        assert blocker.connection_attempts[1] == ConnectionAttempt(
            host='localhost',
            port=8080,
            test_nodeid='',
            allowed=False,
        )

    def it_raises_on_violation_in_strict_mode(self) -> None:
        """Verify on_violation raises NetworkAccessViolationError in STRICT mode."""
        blocker = FakeNetworkBlocker()
        blocker.activate(TestSize.SMALL, EnforcementMode.STRICT)

        with pytest.raises(NetworkAccessViolationError) as exc_info:
            blocker.on_violation('api.example.com', 443, 'test_module.py::test_fn')

        assert exc_info.value.host == 'api.example.com'
        assert exc_info.value.port == 443
        assert exc_info.value.test_size == TestSize.SMALL

    def it_records_warning_in_warn_mode(self) -> None:
        """Verify on_violation records warning in WARN mode."""
        blocker = FakeNetworkBlocker()
        blocker.activate(TestSize.SMALL, EnforcementMode.WARN)

        blocker.on_violation('api.example.com', 443, 'test_module.py::test_fn')

        assert len(blocker.warnings) == 1
        assert 'api.example.com:443' in blocker.warnings[0]

    def it_does_nothing_in_off_mode(self) -> None:
        """Verify on_violation does nothing in OFF mode."""
        blocker = FakeNetworkBlocker()
        blocker.activate(TestSize.SMALL, EnforcementMode.OFF)

        # This should not raise
        blocker.on_violation('api.example.com', 443, 'test_module.py::test_fn')

        assert len(blocker.warnings) == 0

    def it_fails_check_connection_when_inactive(self) -> None:
        """Verify check_connection_allowed raises when INACTIVE."""
        blocker = FakeNetworkBlocker()

        with pytest.raises(ViolationError, match='ACTIVE'):
            blocker.check_connection_allowed('localhost', 8080)

    def it_fails_on_violation_when_inactive(self) -> None:
        """Verify on_violation raises when INACTIVE."""
        blocker = FakeNetworkBlocker()

        with pytest.raises(ViolationError, match='ACTIVE'):
            blocker.on_violation('localhost', 8080, 'test::fn')

    def it_resets_to_initial_state(self) -> None:
        """Verify reset() returns blocker to initial state."""
        blocker = FakeNetworkBlocker()
        blocker.activate(TestSize.SMALL, EnforcementMode.STRICT)
        blocker.check_connection_allowed('api.example.com', 443)

        blocker.reset()

        assert blocker.state == BlockerState.INACTIVE
        assert blocker.current_test_size is None
        assert blocker.current_enforcement_mode is None
        assert len(blocker.connection_attempts) == 0
        assert len(blocker.warnings) == 0

    def it_resets_even_when_active(self) -> None:
        """Verify reset() works regardless of current state."""
        blocker = FakeNetworkBlocker()
        blocker.activate(TestSize.SMALL, EnforcementMode.STRICT)

        # reset() should work even when ACTIVE (unlike deactivate)
        blocker.reset()

        assert blocker.state == BlockerState.INACTIVE

    def it_tracks_call_counts(self) -> None:
        """Verify the blocker tracks method invocation counts."""
        blocker = FakeNetworkBlocker()

        blocker.activate(TestSize.SMALL, EnforcementMode.STRICT)
        blocker.check_connection_allowed('host1', 80)
        blocker.check_connection_allowed('host2', 443)
        blocker.deactivate()

        assert blocker.activate_count == 1
        assert blocker.deactivate_count == 1
        assert blocker.check_count == 2


@pytest.mark.small
class DescribeSocketPatchingNetworkBlocker:
    """Tests for the SocketPatchingNetworkBlocker production adapter."""

    def it_starts_in_inactive_state(self) -> None:
        """Verify the blocker initializes in INACTIVE state."""
        blocker = SocketPatchingNetworkBlocker()

        assert blocker.state == BlockerState.INACTIVE

    def it_transitions_to_active_on_activate(self) -> None:
        """Verify activate() transitions from INACTIVE to ACTIVE."""
        blocker = SocketPatchingNetworkBlocker()

        blocker.activate(TestSize.SMALL, EnforcementMode.STRICT)

        assert blocker.state == BlockerState.ACTIVE

        # Clean up
        blocker.deactivate()

    def it_transitions_to_inactive_on_deactivate(self) -> None:
        """Verify deactivate() transitions from ACTIVE to INACTIVE."""
        blocker = SocketPatchingNetworkBlocker()
        blocker.activate(TestSize.SMALL, EnforcementMode.STRICT)

        blocker.deactivate()

        assert blocker.state == BlockerState.INACTIVE

    def it_fails_to_activate_when_already_active(self) -> None:
        """Verify activate() raises when already ACTIVE."""
        blocker = SocketPatchingNetworkBlocker()
        blocker.activate(TestSize.SMALL, EnforcementMode.STRICT)

        try:
            with pytest.raises(ViolationError, match='INACTIVE'):
                blocker.activate(TestSize.SMALL, EnforcementMode.STRICT)
        finally:
            blocker.reset()

    def it_fails_to_deactivate_when_inactive(self) -> None:
        """Verify deactivate() raises when already INACTIVE."""
        blocker = SocketPatchingNetworkBlocker()

        with pytest.raises(ViolationError, match='ACTIVE'):
            blocker.deactivate()

    def it_stores_activation_parameters(self) -> None:
        """Verify the blocker stores test size and enforcement mode."""
        blocker = SocketPatchingNetworkBlocker()

        blocker.activate(TestSize.MEDIUM, EnforcementMode.WARN)

        assert blocker.current_test_size == TestSize.MEDIUM
        assert blocker.current_enforcement_mode == EnforcementMode.WARN

        blocker.deactivate()

    def it_blocks_all_connections_for_small_tests(self) -> None:
        """Verify small tests cannot access any network."""
        blocker = SocketPatchingNetworkBlocker()
        blocker.activate(TestSize.SMALL, EnforcementMode.STRICT)

        assert blocker.check_connection_allowed('localhost', 8080) is False
        assert blocker.check_connection_allowed('127.0.0.1', 80) is False
        assert blocker.check_connection_allowed('api.example.com', 443) is False

        blocker.deactivate()

    def it_allows_localhost_for_medium_tests(self) -> None:
        """Verify medium tests can access localhost only."""
        blocker = SocketPatchingNetworkBlocker()
        blocker.activate(TestSize.MEDIUM, EnforcementMode.STRICT)

        assert blocker.check_connection_allowed('localhost', 8080) is True
        assert blocker.check_connection_allowed('127.0.0.1', 80) is True
        assert blocker.check_connection_allowed('::1', 443) is True
        assert blocker.check_connection_allowed('api.example.com', 443) is False

        blocker.deactivate()

    def it_allows_all_connections_for_large_tests(self) -> None:
        """Verify large tests can access any network."""
        blocker = SocketPatchingNetworkBlocker()
        blocker.activate(TestSize.LARGE, EnforcementMode.STRICT)

        assert blocker.check_connection_allowed('localhost', 8080) is True
        assert blocker.check_connection_allowed('api.example.com', 443) is True

        blocker.deactivate()

    def it_raises_on_violation_in_strict_mode(self) -> None:
        """Verify on_violation raises NetworkAccessViolationError in STRICT mode."""
        blocker = SocketPatchingNetworkBlocker()
        blocker.activate(TestSize.SMALL, EnforcementMode.STRICT)

        with pytest.raises(NetworkAccessViolationError) as exc_info:
            blocker.on_violation('api.example.com', 443, 'test_module.py::test_fn')

        assert exc_info.value.host == 'api.example.com'
        assert exc_info.value.port == 443

        blocker.deactivate()

    def it_resets_to_initial_state(self) -> None:
        """Verify reset() returns blocker to initial state."""
        blocker = SocketPatchingNetworkBlocker()
        blocker.activate(TestSize.SMALL, EnforcementMode.STRICT)

        blocker.reset()

        assert blocker.state == BlockerState.INACTIVE
        assert blocker.current_test_size is None
        assert blocker.current_enforcement_mode is None

    def it_patches_socket_on_activate(self) -> None:
        """Verify socket.socket is patched when activated."""
        import socket

        original_socket = socket.socket
        blocker = SocketPatchingNetworkBlocker()

        blocker.activate(TestSize.SMALL, EnforcementMode.STRICT)

        # The socket class should be patched
        assert socket.socket is not original_socket

        blocker.deactivate()

        # Should be restored after deactivate
        assert socket.socket is original_socket

    def it_restores_socket_on_deactivate(self) -> None:
        """Verify socket.socket is restored when deactivated."""
        import socket

        original_socket = socket.socket
        blocker = SocketPatchingNetworkBlocker()

        blocker.activate(TestSize.SMALL, EnforcementMode.STRICT)
        blocker.deactivate()

        assert socket.socket is original_socket

    def it_restores_socket_on_reset(self) -> None:
        """Verify socket.socket is restored on reset."""
        import socket

        original_socket = socket.socket
        blocker = SocketPatchingNetworkBlocker()

        blocker.activate(TestSize.SMALL, EnforcementMode.STRICT)
        blocker.reset()

        assert socket.socket is original_socket


@pytest.mark.small
class DescribeLocalhostDetection:
    """Tests for localhost detection logic."""

    @pytest.mark.parametrize(
        'host',
        [
            'localhost',
            'LOCALHOST',
            'LocalHost',
            '127.0.0.1',
            '127.0.0.255',
            '127.255.255.255',
            '::1',
            '0:0:0:0:0:0:0:1',
            '127.1.2.3',
            '::ffff:127.0.0.1',
        ],
    )
    def it_recognizes_localhost_variants(self, host: str) -> None:
        """Verify various localhost representations are recognized."""
        blocker = FakeNetworkBlocker()
        blocker.activate(TestSize.MEDIUM, EnforcementMode.STRICT)

        assert blocker.check_connection_allowed(host, 80) is True

    @pytest.mark.parametrize(
        'host',
        [
            'example.com',
            '192.168.1.1',
            '10.0.0.1',
            '8.8.8.8',
            '::2',
            'localhost.localdomain',  # Not a standard localhost alias
        ],
    )
    def it_rejects_non_localhost_for_medium_tests(self, host: str) -> None:
        """Verify non-localhost hosts are blocked for medium tests."""
        blocker = FakeNetworkBlocker()
        blocker.activate(TestSize.MEDIUM, EnforcementMode.STRICT)

        assert blocker.check_connection_allowed(host, 80) is False


# DNS names that merely start with '127.' resolve wherever their owner points them (#14).
LOOPBACK_LOOKALIKE_HOSTS = [
    '127.evil.attacker.com',
    '127.0.0.1.attacker.com',
    '127.anything',
    '127.',
]


@pytest.mark.small
class DescribeLocalhostAllowlistBypass:
    """The medium-test allowlist accepts real loopback addresses, never look-alike names."""

    @pytest.mark.parametrize('host', LOOPBACK_LOOKALIKE_HOSTS)
    def it_rejects_hostnames_that_start_with_a_loopback_prefix(self, host: str) -> None:
        """Verify a DNS name is not mistaken for a 127.0.0.0/8 address."""
        assert is_localhost(host) is False

    @pytest.mark.parametrize('host', [b'\x7f\x00\x00\x01', 2130706433, None])
    def it_fails_closed_on_hosts_that_are_not_strings(self, host: object) -> None:
        """Verify packed bytes, integers and None are refused, though ipaddress would parse the first two."""
        assert is_localhost(host) is False

    @pytest.mark.parametrize('host', [' 127.0.0.1', '[::1]', '::ffff:8.8.8.8', 'localhost.attacker.com'])
    def it_still_refuses_hosts_the_prefix_match_refused__control(self, host: str) -> None:
        """Verify padded, bracketed, non-loopback-mapped and localhost-prefixed hosts stay refused.

        PASSES ON BASE: the prefix match refused these too; they pin that parsing the address did not let them in.
        """
        assert is_localhost(host) is False

    @pytest.mark.parametrize('host', LOOPBACK_LOOKALIKE_HOSTS)
    def it_blocks_lookalike_hosts_for_medium_tests_in_both_adapters(self, host: str) -> None:
        """Verify neither adapter lets a look-alike name through the MEDIUM allowlist."""
        fake = FakeNetworkBlocker()
        real = SocketPatchingNetworkBlocker()
        fake.activate(TestSize.MEDIUM, EnforcementMode.STRICT)
        real.activate(TestSize.MEDIUM, EnforcementMode.STRICT)

        try:
            assert fake.check_connection_allowed(host, 443) is False
            assert real.check_connection_allowed(host, 443) is False
        finally:
            real.deactivate()

    # These two tests swap the socket module's attributes by hand and restore them before returning.
    # mocker/monkeypatch restore at fixture teardown, which runs after the plugin's own per-test
    # blocker has been deactivated, so they would reinstall that blocker's patched socket for good.

    @pytest.mark.parametrize('method_name', ['connect', 'connect_ex'])
    def it_stops_a_medium_test_connecting_to_a_lookalike_host(self, method_name: str) -> None:
        """Verify the patched socket raises before the real connect resolves the name off-box."""
        connected: list[object] = []

        class RecordingSocket:
            """Stands in for the real socket class so nothing can leave the machine."""

            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def connect(self, address: object) -> None:
                connected.append(address)

            def connect_ex(self, address: object) -> int:
                connected.append(address)
                return 0

        socket_class = socket.socket
        socket.socket = RecordingSocket  # type: ignore[misc,assignment]
        blocker = SocketPatchingNetworkBlocker()
        blocker.activate(TestSize.MEDIUM, EnforcementMode.STRICT)

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            with pytest.raises(NetworkAccessViolationError) as exc_info:
                getattr(sock, method_name)(('127.evil.attacker.com', 443))
        finally:
            blocker.deactivate()
            socket.socket = socket_class  # type: ignore[misc]

        assert exc_info.value.host == '127.evil.attacker.com'
        assert connected == []

    @pytest.mark.parametrize('resolver_name', ['getaddrinfo', 'gethostbyname', 'gethostbyname_ex'])
    def it_stops_a_medium_test_resolving_a_lookalike_host(self, mocker: MockerFixture, resolver_name: str) -> None:
        """Verify the patched resolver raises before the lookup can leak the name to its nameserver."""
        resolver = getattr(socket, resolver_name)
        real_resolver = mocker.Mock()
        setattr(socket, resolver_name, real_resolver)
        blocker = SocketPatchingNetworkBlocker()
        blocker.activate(TestSize.MEDIUM, EnforcementMode.STRICT)

        try:
            with pytest.raises(NetworkAccessViolationError):
                getattr(socket, resolver_name)('127.secret-token.attacker.com', 0)
        finally:
            blocker.deactivate()
            setattr(socket, resolver_name, resolver)

        real_resolver.assert_not_called()


@pytest.mark.small
class DescribeNetworkModeEnum:
    """Tests for the NetworkMode enum."""

    def it_has_block_all_value(self) -> None:
        """Verify BLOCK_ALL value is correct."""
        assert NetworkMode.BLOCK_ALL.value == 'block_all'

    def it_has_localhost_only_value(self) -> None:
        """Verify LOCALHOST_ONLY value is correct."""
        assert NetworkMode.LOCALHOST_ONLY.value == 'localhost'

    def it_has_allow_all_value(self) -> None:
        """Verify ALLOW_ALL value is correct."""
        assert NetworkMode.ALLOW_ALL.value == 'allow_all'


@pytest.mark.small
class DescribeTestSizeNetworkMode:
    """Tests for TestSize.network_mode property."""

    def it_maps_small_to_block_all(self) -> None:
        """Verify small tests map to BLOCK_ALL network mode."""
        assert TestSize.SMALL.network_mode == NetworkMode.BLOCK_ALL

    def it_maps_medium_to_localhost_only(self) -> None:
        """Verify medium tests map to LOCALHOST_ONLY network mode."""
        assert TestSize.MEDIUM.network_mode == NetworkMode.LOCALHOST_ONLY

    def it_maps_large_to_allow_all(self) -> None:
        """Verify large tests map to ALLOW_ALL network mode."""
        assert TestSize.LARGE.network_mode == NetworkMode.ALLOW_ALL

    def it_maps_xlarge_to_allow_all(self) -> None:
        """Verify xlarge tests map to ALLOW_ALL network mode."""
        assert TestSize.XLARGE.network_mode == NetworkMode.ALLOW_ALL
