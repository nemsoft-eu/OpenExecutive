"""One classifier for "is this an address an outbound request may reach".

Shared by every client that connects only to public addresses: the chat's
``web_fetch`` (``orchestrator.web_fetch``) and the Confluence sync under
``CONFLUENCE_SYNC_PUBLIC_HOSTS_ONLY`` (``knowledge.confluence_client``). Each
used to keep its own copy, and the copies drifted. The Confluence one unwrapped
the local-use NAT64 prefix by its last 32 bits, which is not where RFC 6052
puts the IPv4 address for a /48; it checked only the Teredo client address,
not the server's; and it knew neither the IPv4-translated form nor deprecated
site-local space, both of which Python calls global. The policy here is the
union of the two: an address either copy refused is refused.
"""
from __future__ import annotations

import ipaddress

# The well-known NAT64 prefix (RFC 6052), SIIT's IPv4-translated
# ``::ffff:0:a.b.c.d`` (RFC 6145) and the deprecated IPv4-compatible
# ``::a.b.c.d``. All three are /96, so the last 32 bits are an IPv4 address a
# gateway or tunnel may deliver to, and Python reports ``64:ff9b::7f00:1``,
# ``::ffff:0:7f00:1`` and ``::7f00:1`` (127.0.0.1) as global.
_NAT64_PREFIXES = (
    ipaddress.IPv6Network("64:ff9b::/96"),
    ipaddress.IPv6Network("::ffff:0:0:0/96"),  # not ::ffff:0:0/96, the mapped form
    ipaddress.IPv6Network("::/96"),
)
# Refused whole, whatever Python's version calls global:
# - deprecated site-local space (RFC 3879), which a network that still routes
#   it routes internally;
# - the local-use NAT64 prefix (RFC 8215), which an operator subdivides at a
#   length of their choosing, so the embedded IPv4 sits at bits RFC 6052
#   places differently per length (/48 puts it at bits 48-63 and 72-87), not
#   reliably in the last 32. Local use means no public site lives there.
_REFUSED_V6 = (
    ipaddress.IPv6Network("fec0::/10"),
    ipaddress.IPv6Network("64:ff9b:1::/48"),
)


def _embedded_ipv4(
    ip: ipaddress.IPv6Address,
) -> list[ipaddress.IPv4Address]:
    """IPv4 addresses an IPv6 address carries and a gateway would reach."""
    found: list[ipaddress.IPv4Address] = []
    if ip.ipv4_mapped is not None:
        found.append(ip.ipv4_mapped)
    if ip.sixtofour is not None:
        found.append(ip.sixtofour)
    if ip.teredo is not None:
        found.extend(ip.teredo)
    if any(ip in net for net in _NAT64_PREFIXES):
        found.append(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF))
    return found


def is_public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Whether ``ip`` is an address an outbound request may connect to.

    ``is_global`` already excludes private, loopback, link-local, reserved,
    unspecified and shared (CGNAT, which Tailscale uses) space. Multicast is
    excluded separately because ``is_global`` does not cover it, and an IPv6
    address that embeds an IPv4 one is judged by the IPv4 address too.
    """
    candidates: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = [ip]
    if isinstance(ip, ipaddress.IPv6Address):
        if any(ip in net for net in _REFUSED_V6):
            return False
        candidates.extend(_embedded_ipv4(ip))
    return all(c.is_global and not c.is_multicast for c in candidates)


def is_public_address(raw: str) -> bool:
    """``is_public_ip`` for an address string, such as a resolver returns.

    A zone id (``fe80::1%eth0``) is dropped before parsing, and a string that
    is not an IP address is not public.
    """
    try:
        ip = ipaddress.ip_address(raw.split("%", 1)[0])
    except ValueError:
        return False
    return is_public_ip(ip)
