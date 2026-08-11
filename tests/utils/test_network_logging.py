# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

from transformers.utils.network_logging import (
    _clear_network_debug_report,
    _disable_network_debug_report,
    _enable_network_debug_report,
    _format_network_debug_report,
    _get_network_debug_report,
)


class _SlowHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        time.sleep(0.01)
        response = b"ok"
        self.send_response(200)
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format, *args):
        return


class NetworkLoggingTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._server = ThreadingHTTPServer(("127.0.0.1", 0), _SlowHandler)
        cls._thread = threading.Thread(target=cls._server.serve_forever, daemon=True)
        cls._thread.start()
        cls._base_url = f"http://127.0.0.1:{cls._server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls._server.shutdown()
        cls._thread.join()
        cls._server.server_close()

    def tearDown(self):
        _disable_network_debug_report()

    def test_network_debug_report_records_httpx_requests(self):
        _enable_network_debug_report()
        _clear_network_debug_report()

        response = httpx.get(f"{self._base_url}/slow")
        self.assertEqual(response.text, "ok")

        report = _get_network_debug_report()
        matching_requests = [request for request in report["requests"] if request["url"].endswith("/slow")]
        self.assertEqual(len(matching_requests), 1)

        request = matching_requests[0]
        self.assertEqual(request["method"], "GET")
        self.assertEqual(request["status_code"], 200)
        self.assertEqual(request["path"], "/slow")
        self.assertGreater(request["total_ms"], 0)
        self.assertIn("receive_response_headers", request["phases_ms"])

        summary = _format_network_debug_report()
        self.assertIn("Network debug report", summary)
        self.assertIn("Slowest requests:", summary)
        self.assertIn("/slow", summary)
