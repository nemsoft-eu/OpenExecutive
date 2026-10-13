"""Tests for the shared public-address classifier (utils/public_address.py)."""
from __future__ import annotations

import ipaddress

import pytest

from openexecutive.utils.public_address import is_public_address, is_public_ip

_NON_PUBLIC = [
    "127.0.0.1",
    "10.1.2.3",
    "172.16.0.1",
    "192.168.1.1",
    "169.254.169.254",  # cloud metadata
    "100.100.1.1",  # CGNAT / Tailscale
    "0.0.0.0",
    "224.0.0.1",  # multicast
    "ff02::1",  # IPv6 multicast, which Python calls global
    "ff0e::1",
    "::1",
    "fe80::1%eth0",
    "fd00::1",
    "::ffff:127.0.0.1",  # IPv4-mapped
    "2002:7f00:1::",  # 6to4 around 127.0.0.1
    "2001:0:4136:e378:8000:63bf:f5ff:fffe",  # Teredo, client 10.0.0.1
    "64:ff9b::7f00:1",  # NAT64 around 127.0.0.1
    "64:ff9b::a00:1",  # NAT64 around 10.0.0.1
    "::7f00:1",  # IPv4-compatible around 127.0.0.1
    "::a00:1",  # IPv4-compatible around 10.0.0.1
    "::ffff:0:a9fe:a9fe",  # IPv4-translated (SIIT) around 169.254.169.254
    "fec0::1",  # deprecated site-local, which Python calls global
    # Local-use NAT64 (RFC 8215), refused whole: the /48 form of
    # 127.0.0.1 (bits 48-63 and 72-87), and a public-looking tail.
    "64:ff9b:1:7f00:0:100::",
    "64:ff9b:1::5db8:d822",
    # The /48 form of 127.0.0.1 with 93.184.216.34 in the last 32 bits:
    # unwrapping it there, as the Confluence copy of this check did, saw a
    # public address.
    "64:ff9b:1:7f00:0:100:5db8:d822",
]
_PUBLIC = [
    "93.184.216.34",
    "2606:2800:220:1:248:1893:25c8:1946",
    "2606:2800:220:1:248:1893:25c8:1946%eth0",  # the zone id is dropped
    # Each embedded-IPv4 form around a public address is let through.
    "64:ff9b::5db8:d822",  # NAT64
    "::ffff:93.184.216.34",  # IPv4-mapped
    "::ffff:0:5db8:d822",  # IPv4-translated (SIIT)
]


@pytest.mark.parametrize("addr", [*_NON_PUBLIC, "not-an-ip", "", "%", "1.2.3"])
def test_non_public_addresses_are_refused(addr: str) -> None:
    assert not is_public_address(addr)


@pytest.mark.parametrize("addr", _PUBLIC)
def test_public_addresses_are_allowed(addr: str) -> None:
    assert is_public_address(addr)


@pytest.mark.parametrize("addr", [*_NON_PUBLIC, *_PUBLIC])
def test_a_parsed_address_is_judged_alike(addr: str) -> None:
    """The Confluence client parses each address itself and asks
    ``is_public_ip``, so check that entry point against the same lists."""
    parsed = ipaddress.ip_address(addr.split("%", 1)[0])
    assert is_public_ip(parsed) is (addr in _PUBLIC)
