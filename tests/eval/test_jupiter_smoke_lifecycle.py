"""Queue admission regressions for the isolated Jupiter smoke."""

import pytest

from scripts.smoke.jupiter32k.lifecycle import (
    proxy_window,
    select_bind,
    validate_lifetime,
)


def test_bind_uses_only_internal_interface():
    assert (
        select_bind("10.128.1.3 10.99.0.3 134.94.0.133 2001:638:404::85")
        == "10.128.1.3"
    )


@pytest.mark.parametrize("addresses", ["127.0.0.1 ::1", "10.128.1.3 10.128.1.9", ""])
def test_ambiguous_or_absent_interface_rejects(addresses):
    with pytest.raises(RuntimeError):
        select_bind(addresses)


@pytest.mark.parametrize("hours", [0, 6.5, 7.99])
def test_proxy_covers_execution_after_long_queue(hours):
    window = proxy_window(1000)
    validate_lifetime(window, 1000 + hours * 3600)
    assert window["expires_epoch"] - (1000 + hours * 3600) > 2400


@pytest.mark.parametrize("hours", [8, 9, 10])
def test_late_allocation_never_starts_work(hours):
    with pytest.raises(RuntimeError, match="Queue allowance"):
        validate_lifetime(proxy_window(1000), 1000 + hours * 3600)


def test_old_two_hour_proxy_cannot_be_admitted():
    window = proxy_window(1000)
    window["expires_epoch"] = 1000 + 7200
    with pytest.raises(ValueError, match="approved bounds"):
        validate_lifetime(window, 1000 + 6.5 * 3600)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "9999999999", True])
def test_invalid_expiry_fails_closed(value):
    window = proxy_window(1000)
    window["expires_epoch"] = value
    with pytest.raises(ValueError):
        validate_lifetime(window, 1001)


def test_clock_regression_fails_closed():
    with pytest.raises(RuntimeError, match="clock"):
        validate_lifetime(proxy_window(1000), 999)
