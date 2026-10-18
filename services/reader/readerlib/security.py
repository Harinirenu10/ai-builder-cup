"""URL checks. The reader fetches caller-supplied URLs, so it must not be pointed at internal addresses."""

import ipaddress
import os
import socket
from urllib.parse import urlparse

BLOCKED_HOSTS = {"localhost", "metadata.google.internal", "metadata"}


class UnsafeURL(ValueError):
    pass


def validate_url(url: str, *, allow_private: bool | None = None) -> str:
    """Return the URL if it is a public http(s) URL; raise UnsafeURL otherwise."""
    if allow_private is None:
        allow_private = os.environ.get("READER_ALLOW_PRIVATE") == "1"
    url = url.strip()
    parts = urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise UnsafeURL("Only http(s) URLs are allowed")
    if allow_private:
        return url
    host = parts.hostname.lower()
    if host in BLOCKED_HOSTS or host.endswith((".local", ".internal", ".localhost")):
        raise UnsafeURL("Internal hostnames are not allowed")
    try:
        infos = socket.getaddrinfo(host, parts.port or (443 if parts.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise UnsafeURL(f"Cannot resolve host: {exc}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            raise UnsafeURL("The host resolves to a non-public address")
    return url
