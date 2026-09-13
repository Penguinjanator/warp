#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""serve/glmtools.py's tool protocol against the GLM template that defines it.

`test_chatfmt_upstream` diffs `kimitools.py`'s rendering against Kimi K2's
published chat_template. This is the same check for the GLM half, and it
exists for the same reason: chat.json cannot express a tool declaration, an
argument list, or a result turn, so the ground truth for how GLM spells those
lives in the model's own `chat_template.jinja`, not anywhere this repo owns.

The assertion is narrower than the Kimi one because there is no single
surface to diff: the template's tool declaration is prose with embedded JSON,
and its tool/tool-response turns carry GLM's own whitespace around them,
which the declarative chat.json path spreads across segments. What is
compared is the XML the grammar is named for — the `<tool_call>…</tool_call>`
list and the `<|observation|><tool_response>…</tool_response>` block — each
extracted from a full template render and compared exactly against what this
repo renders.

    GLM_DIR=/path/to/glm-5.3-flash python3 -m unittest \\
        tests.serve.test_glm_upstream -t .

Only a `chat_template.jinja` is needed; no weights. The template is looked
up in this order:

    1. $GLM_DIR, if set — a GLM release directory
    2. tests/serve/glm_upstream/ — the vendored upstream template (the
       default; see that directory's README.md for provenance)

So the default path needs nothing on disk beyond this repo. Without it the
suite skips, loudly; with CI_GLM_ORACLE_STRICT=1 the designated CI oracle
job runs it instead, and a missing template, missing jinja2, or unresolved
markers is a failure rather than a skip — that job must never be green
because it did nothing.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from serve import glmtools                                   # noqa: E402
from serve.chatfmt import ChatFormat                         # noqa: E402
from tests.serve.fake_engine import FakeEngine               # noqa: E402
from tests.serve.test_glmtools import GLM_MARKERS, TOOLS     # noqa: E402

VENDORED_DIR = REPO / "tests" / "serve" / "glm_upstream"
GLM_DIR = os.environ.get("GLM_DIR", str(VENDORED_DIR))
GLM_CHAT = REPO / "examples" / "chat-glm53.json"

# The designated CI oracle job sets this: skips become failures, so a green
# run means the three checks below actually executed.
STRICT = os.environ.get("CI_GLM_ORACLE_STRICT") == "1"

TOOL_CALL_ARGS = {"city": "Rome", "days": 3}


def load_template():
    """GLM's own chat template, or None if it is not on disk."""
    jinja = Path(GLM_DIR) / "chat_template.jinja"
    if jinja.exists():
        return jinja.read_text(encoding="utf-8")
    cfg = Path(GLM_DIR) / "tokenizer_config.json"
    if cfg.exists():
        return json.loads(cfg.read_text(encoding="utf-8")).get("chat_template")
    return None


def render_upstream(template, messages, tools):
    from jinja2 import Environment

    def tojson(x, ensure_ascii=False, indent=None, separators=None,
               sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                          separators=separators, sort_keys=sort_keys)

    env = Environment(extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = tojson
    return env.from_string(template).render(
        messages=messages, tools=tools, add_generation_prompt=True)


def _skip_or_fail(reason):
    """Skip in local development, fail in the strict CI oracle job."""
    if STRICT:
        raise AssertionError(f"CI_GLM_ORACLE_STRICT: {reason}")
    raise unittest.SkipTest(reason)


class TestAgainstGlmTemplate(unittest.TestCase):
    n_ran = 0

    @classmethod
    def setUpClass(cls):
        try:
            import jinja2                                     # noqa: F401
        except ImportError:
            _skip_or_fail("jinja2 not installed; the template cannot be "
                          "rendered (pip install -r requirements-test.txt)")
        cls.template = load_template()
        if not cls.template:
            _skip_or_fail(f"no chat_template.jinja at {GLM_DIR} (set GLM_DIR "
                          "to a GLM release directory)")
        cls._tmp = tempfile.mkdtemp()
        shutil.copyfile(GLM_CHAT, os.path.join(cls._tmp, "chat.json"))
        eng = FakeEngine(no_markers=True, model_path=cls._tmp,
                         markers=dict(GLM_MARKERS))
        cls.fmt = ChatFormat.load(eng)
        if cls.fmt.tool_protocol != "glm":
            _skip_or_fail(
                "the GLM tool markers did not resolve; there is nothing "
                "to compare against the template")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(getattr(cls, "_tmp", ""), ignore_errors=True)
        # Strict is a promise about what ran, not merely what passed: if a
        # registration mistake left a check out of the class, a green job
        # would quietly cover one fewer oracle.
        if STRICT:
            assert cls.n_ran == 3, (
                f"strict oracle ran {cls.n_ran} of 3 checks; the rest "
                "never executed")

    def tearDown(self):
        TestAgainstGlmTemplate.n_ran += 1

    def render_ours(self, messages, tools=None):
        segs = self.fmt.build_chat_segments(messages, tools=tools,
                                            thinking=True)
        return "".join(s.text for s in segs)

    def test_tool_call_matches_the_template(self):
        """Our flat `<tool_call>…</tool_call>` is byte-for-byte the call the
        template writes, name and arguments in the same order."""
        import re
        msgs = [{"role": "assistant", "content": "",
                 "tool_calls": [{
                     "id": "call_1", "type": "function",
                     "function": {"name": "get_weather",
                                  "arguments": dict(TOOL_CALL_ARGS)}}]},
                {"role": "tool", "tool_call_id": "call_1",
                 "content": "18C"}]
        up = render_upstream(self.template, msgs, TOOLS)
        upstream_calls = re.findall(r"<tool_call>.*?</tool_call>", up, re.S)
        # Drop the template's format example; keep the model's actual call.
        real = [c for c in upstream_calls
                if "get_weather" in c]
        self.assertEqual(len(real), 1)
        ours = "".join(s.text for s in
                       glmtools.call_section(msgs[0]["tool_calls"], 0))
        self.assertEqual(ours, real[0])

    def test_tool_result_matches_the_template(self):
        """`<|observation|><tool_response>…</tool_response>` matches the
        template's authored result turn."""
        msgs = [{"role": "assistant", "content": "",
                 "tool_calls": [{
                     "id": "call_1", "type": "function",
                     "function": {"name": "get_weather",
                                  "arguments": dict(TOOL_CALL_ARGS)}}]},
                {"role": "tool", "tool_call_id": "call_1",
                 "content": "18C"},
                {"role": "user", "content": "thanks"}]
        up = render_upstream(self.template, msgs, TOOLS)
        i = up.rfind("<|observation|>")
        upstream_result = up[i:up.find("<|user|>", i)]
        ours = self.render_ours(msgs, tools=TOOLS)
        oi = ours.rfind("<|observation|>")
        our_result = ours[oi:ours.find("<|user|>", oi)]
        self.assertEqual(our_result, upstream_result)

    def test_declaration_embeds_the_same_signatures(self):
        """Every signature we render also appears in the template's
        `<tools>…</tools>` declaration block."""
        msgs = [{"role": "user", "content": "hi"}]
        up = render_upstream(self.template, msgs, TOOLS)
        ours = self.render_ours(msgs, tools=TOOLS)
        # Compare the complete declaration, including the instruction
        # example and whitespace, rather than only finding a function name.
        start = up.rfind("<|system|>", 0, up.index("# Tools"))
        end = up.index("<|user|>", start)
        our_start = ours.rfind("<|system|>", 0, ours.index("# Tools"))
        self.assertEqual(ours[our_start:ours.index("<|user|>", our_start)],
                         up[start:end])
        segs = glmtools.declaration(TOOLS)
        self.assertTrue(next(s for s in segs if "{function-name}" in s.text).markup)


if __name__ == "__main__":
    unittest.main()
