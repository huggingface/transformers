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

import socket
import unittest
from unittest.mock import patch

from transformers.utils.remote_url import validate_remote_url


def _fake_addrinfo(ip: str):
    # (family, type, proto, canonname, sockaddr)
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]


class ValidateRemoteUrlTester(unittest.TestCase):
    def test_rejects_non_http_schemes(self):
        with self.assertRaises(ValueError):
            validate_remote_url("file:///etc/passwd")
        with self.assertRaises(ValueError):
            validate_remote_url("javascript:alert(1)")

    def test_rejects_loopback(self):
        with patch("socket.getaddrinfo", return_value=_fake_addrinfo("127.0.0.1")):
            with self.assertRaises(ValueError):
                validate_remote_url("http://metadata.local/latest")

    def test_rejects_link_local_metadata(self):
        with patch("socket.getaddrinfo", return_value=_fake_addrinfo("169.254.169.254")):
            with self.assertRaises(ValueError):
                validate_remote_url("http://169.254.169.254/latest/meta-data/")

    def test_rejects_cgnat(self):
        with patch("socket.getaddrinfo", return_value=_fake_addrinfo("100.64.0.1")):
            with self.assertRaises(ValueError):
                validate_remote_url("http://cgnat.example/audio.wav")

    def test_allows_public_ip(self):
        with patch("socket.getaddrinfo", return_value=_fake_addrinfo("8.8.8.8")):
            validate_remote_url("https://cdn.example.com/video.mp4")
