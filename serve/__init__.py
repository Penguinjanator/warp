# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
serve — an OpenAI-compatible server for the WASTE engine.

    python3 -m serve ~/models/k3.waste --port 8000

The engine is a C library with a public header; this package is one of its
hosts, reaching it through ctypes rather than keeping a second copy of the
model code in Python. What lives here is everything that is *not* inference:
K3's prompt format, the parser that reads its replies back, and the HTTP
surface clients expect.

    xtml.py     the K3 prompt format (port of the release's encoding_k3.py)
    regions.py  its replies, back into reasoning / answer / tool calls
    engine.py   ctypes binding to libwaste, and the request queue
    server.py   /v1/chat/completions and friends

Stdlib only, like the rest of the repo's tooling: a server that needs a
package index to start is one more thing between a downloaded model and an
answer.
"""

__all__ = ["xtml", "regions", "engine", "server"]
