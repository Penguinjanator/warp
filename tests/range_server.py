#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
range_server.py — a throwaway HTTP server for testing fetch_weights.sh.

The standard library's http.server answers 200 to a Range request and
sends the whole file, which is exactly the case the download script must
not silently accept. This serves a directory with real 206 responses, or
with Range support switched off, so both branches of the worker's resume
logic can be exercised without touching the network.

  python3 tests/range_server.py DIR PORT [--no-range]
"""

import http.server
import os
import re
import sys


def make_handler(root, ranges):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _path(self):
            # No traversal guard: this serves one temp dir to one test on
            # loopback and is killed straight after.
            return os.path.join(root, self.path.lstrip("/"))

        def do_HEAD(self):
            try:
                n = os.path.getsize(self._path())
            except OSError:
                return self.send_error(404)
            self.send_response(200)
            self.send_header("Content-Length", str(n))
            if ranges:
                self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

        def do_GET(self):
            try:
                n = os.path.getsize(self._path())
            except OSError:
                return self.send_error(404)
            m = ranges and re.match(r"bytes=(\d+)-", self.headers.get("Range", ""))
            start = int(m.group(1)) if m else 0
            self.send_response(206 if m else 200)
            if m:
                self.send_header("Content-Range", f"bytes {start}-{n-1}/{n}")
            self.send_header("Content-Length", str(n - start))
            if ranges:
                self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with open(self._path(), "rb") as f:
                f.seek(start)
                self.wfile.write(f.read())

    return H


def main():
    root, port = sys.argv[1], int(sys.argv[2])
    ranges = "--no-range" not in sys.argv
    srv = http.server.HTTPServer(("127.0.0.1", port), make_handler(root, ranges))
    srv.serve_forever()


if __name__ == "__main__":
    main()
