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

  python3 tests/range_server.py DIR PORT [--no-range] [--port-file F]

PORT 0 asks the kernel for a free one, which is what the caller should
use: a hardcoded port is a check that fails on any machine already running
something there, and it fails as "resume" rather than as "that port is
taken". --port-file is how the caller learns which port it got.
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
    argv = sys.argv[1:]
    port_file = None
    if "--port-file" in argv:
        i = argv.index("--port-file")
        port_file = argv[i + 1]
        del argv[i:i + 2]
    ranges = "--no-range" not in argv
    root, port = [a for a in argv if not a.startswith("--")][:2]
    srv = http.server.HTTPServer(("127.0.0.1", int(port)), make_handler(root, ranges))

    # Hand the real port back only after the constructor has bound and
    # listened, so a caller that sees this file can connect immediately —
    # anything arriving before serve_forever() waits in the backlog rather
    # than being refused. Written under a temp name and renamed, so a
    # caller polling for the file never reads half a number.
    if port_file:
        with open(port_file + ".tmp", "w") as f:
            f.write(str(srv.server_address[1]))
        os.replace(port_file + ".tmp", port_file)

    srv.serve_forever()


if __name__ == "__main__":
    main()
