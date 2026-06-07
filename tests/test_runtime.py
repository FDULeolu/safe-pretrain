from __future__ import annotations

import pytest

from safe_pretrain.utils.runtime import resolve_main_process_port


@pytest.mark.parametrize("value", [None, "", "none", "null", "auto"])
def test_resolve_main_process_port_allows_empty_values(value: object) -> None:
    assert resolve_main_process_port(value) is None


def test_resolve_main_process_port_returns_normalized_port() -> None:
    assert resolve_main_process_port("29510") == "29510"
    assert resolve_main_process_port(29520) == "29520"


@pytest.mark.parametrize("value", ["0", "65536"])
def test_resolve_main_process_port_rejects_out_of_range_values(value: object) -> None:
    with pytest.raises(ValueError):
        resolve_main_process_port(value)
