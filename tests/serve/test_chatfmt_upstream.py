#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""serve/chatfmt.py's tool protocol against the template that defines it.

`test_xtml.TestAgainstUpstream` checks our port of K3's encoder against the
`encoding_k3.py` the release ships. This is the same check for the other
protocol `serve/` renders, and it exists because that one had nowhere to
look: Kimi-Linear's tokenizer carries the five tool-call control tokens and
its release ships **no chat_template at all**, so the vocabulary is stated
and the grammar is not. The grammar is Kimi K2's, and K2 does publish it.

Without this, the tool rendering is checked only against a parser that
reads back what the renderer wrote — which agrees with itself whatever the
format is. That is how the turn a tool result opens came to say `system`
where the template says the tool's own name: every self-consistent test
passed, and one diff against the release did not.

  K2_DIR=/path/to/Kimi-K2-Instruct python3 -m unittest \\
      tests.serve.test_chatfmt_upstream -t .

The directory needs one file — `chat_template.jinja`, or a
`tokenizer_config.json` carrying a `chat_template` key. No weights: the
template plus `examples/chat-kimi-linear.json` is the whole input, which
makes this the cheapest oracle in the repo and the only one that needs no
container.

Rendered with transformers' `tojson`, not Jinja's: a HF chat template is
evaluated with a filter that takes `separators` and defaults `ensure_ascii`
off, and the builtin does neither. Rendering with the wrong filter fails
loudly here (an unexpected keyword) rather than quietly producing spaced
JSON, but the default matters and is set.
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

from serve.chatfmt import ChatFormat                          # noqa: E402
from tests.serve.fake_engine import FakeEngine                # noqa: E402
from tests.serve.test_chatfmt import KIMI_K2_MARKERS          # noqa: E402

K2_DIR = os.environ.get("K2_DIR", "/Volumes/WasteDisk/kimi-k2")
LINEAR_CHAT = REPO / "examples" / "chat-kimi-linear.json"

TOOLS = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Current weather for a city",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]

CALL = {"role": "assistant", "content": "", "tool_calls": [{
    "id": "functions.get_weather:0", "type": "function",
    "function": {"name": "get_weather",
                 "arguments": '{"city":"Rome"}'}}]}


def load_template():
    """The release's own chat template, or None if it is not on disk."""
    d = Path(K2_DIR)
    jinja = d / "chat_template.jinja"
    if jinja.exists():
        return jinja.read_text(encoding="utf-8")
    cfg = d / "tokenizer_config.json"
    if cfg.exists():
        return json.loads(cfg.read_text(encoding="utf-8")).get("chat_template")
    return None


def render_upstream(template, messages, tools):
    from jinja2 import Environment

    def tojson(x, ensure_ascii=False, indent=None, separators=None,
               sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                          separators=separators, sort_keys=sort_keys)

    env = Environment()
    env.filters["tojson"] = tojson
    return env.from_string(template).render(
        messages=messages, tools=tools, add_generation_prompt=True)


class TestAgainstK2Template(unittest.TestCase):
    """chat.json rendering against Kimi K2's published chat_template."""

    @classmethod
    def setUpClass(cls):
        try:
            import jinja2                                     # noqa: F401
        except ImportError:
            raise unittest.SkipTest(
                "jinja2 not installed; the template cannot be rendered")
        cls.template = load_template()
        if not cls.template:
            raise unittest.SkipTest(
                f"no chat_template at {K2_DIR} (set K2_DIR to a Kimi-K2 "
                f"release directory; only chat_template.jinja is needed)")
        # ChatFormat.load reads <model_path>/chat.json, so the format under
        # test is Kimi-Linear's own file under that name -- the container
        # that actually carries these five tokens.
        cls._tmp = tempfile.mkdtemp()
        shutil.copyfile(LINEAR_CHAT, os.path.join(cls._tmp, "chat.json"))
        eng = FakeEngine(no_markers=True, model_path=cls._tmp,
                         markers=dict(KIMI_K2_MARKERS))
        cls.fmt = ChatFormat.load(eng)
        if not cls.fmt.tool_markers:
            raise unittest.SkipTest(
                "the five tool markers did not resolve; there is nothing "
                "to compare against the template")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(getattr(cls, "_tmp", ""), ignore_errors=True)

    def rendered(self, messages, tools=None):
        segs = self.fmt.build_chat_segments(messages, tools=tools,
                                            thinking=False)
        return "".join(s.text for s in segs)

    def same(self, messages, tools=None):
        want = render_upstream(self.template, messages, tools)
        self.assertEqual(want, self.rendered(messages, tools))

    # -- the cases ---------------------------------------------------------

    def test_plain_conversation(self):
        """No tools at all: the part that was already right, pinned."""
        self.same([{"role": "system", "content": "You are terse."},
                   {"role": "user", "content": "Weather in Rome?"},
                   {"role": "assistant", "content": "Ask a tool."},
                   {"role": "user", "content": "Do it."}])

    def test_tool_declaration(self):
        """`<|im_system|>tool_declare<|im_middle|>` and compact JSON."""
        self.same([{"role": "system", "content": "S"},
                   {"role": "user", "content": "U"}], TOOLS)

    def test_tool_call(self):
        """The section, the id after tool_call_begin, the arguments."""
        self.same([{"role": "system", "content": "S"},
                   {"role": "user", "content": "U"}, CALL], TOOLS)

    def test_tool_result_named(self):
        """A result carrying `name` opens a turn named for the function."""
        self.same([{"role": "system", "content": "S"},
                   {"role": "user", "content": "U"}, CALL,
                   {"role": "tool", "name": "get_weather",
                    "tool_call_id": "functions.get_weather:0",
                    "content": "18C, clear"}], TOOLS)

    def test_tool_result_unnamed(self):
        """Without `name` the turn is `tool` — and never `system`, which is
        what chat.json's own system prefix carries."""
        self.same([{"role": "system", "content": "S"},
                   {"role": "user", "content": "U"}, CALL,
                   {"role": "tool",
                    "tool_call_id": "functions.get_weather:0",
                    "content": "18C, clear"}], TOOLS)

    def test_two_calls_in_one_turn(self):
        """Both calls inside one section, not a section each."""
        two = dict(CALL)
        two["tool_calls"] = CALL["tool_calls"] + [{
            "id": "functions.get_weather:1", "type": "function",
            "function": {"name": "get_weather",
                         "arguments": '{"city":"Milan"}'}}]
        self.same([{"role": "system", "content": "S"},
                   {"role": "user", "content": "U"}, two], TOOLS)

    def test_default_system_turn_is_deliberately_not_ours(self):
        """The one divergence this file asserts rather than forbids.

        With no system turn first, K2's template inserts Moonshot's own —
        "You are Kimi, an AI assistant created by Moonshot AI." A server
        rendering an arbitrary container's chat.json has no business
        inventing that: chat.json says how turns are shaped, not who the
        model is, and the same code serves Kimi-Linear and anything else
        that ships one. Asserted so the difference stays a decision.
        """
        msgs = [{"role": "user", "content": "U"}]
        want = render_upstream(self.template, msgs, None)
        got = self.rendered(msgs)
        self.assertIn("You are Kimi", want)
        self.assertNotIn("You are Kimi", got)
        self.assertEqual(want.split("<|im_end|>", 1)[1].strip(), got.strip())


if __name__ == "__main__":
    unittest.main()
