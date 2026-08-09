# Copyright 2026 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SSRF guards for remote media URL fetches (image/audio/video helpers)."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


def validate_remote_url(url: str) -> None:
    """Raise ``ValueError`` if ``url`` resolves to a non-global address.

    Blocks loopback, RFC1918/ULA private, link-local (cloud metadata),
    CGNAT, and other non-global ranges before ``httpx.get`` runs. Used by
    video/audio (and sibling of open ``load_image`` hardening).
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"URL scheme must be http or https for remote media fetch, got {parsed.scheme!r}."
        )
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Remote media URL is missing a hostname.")

    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        # Let httpx surface resolution failures on the actual request.
        return

    for info in infos:
        ip = info[4][0]
        try:
            ip_obj = ipaddress.ip_address(ip)
        except ValueError:
            continue
        if not ip_obj.is_global:
            raise ValueError(
                f"URL hostname '{hostname}' resolves to a non-global address ({ip}). "
                "Requests to internal addresses are blocked to prevent SSRF. "
                "If this is a legitimate local media server, download the file and "
                "pass a local path or bytes instead."
            )
