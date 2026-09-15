#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""test_dsml_upstream.py — serve/dsml.py against DeepSeek's own encoder.

`serve/dsml.py` is a port of `encoding/encoding.py`, and a port is a second
implementation of something whose only specification is the first one. So it
is diffed against that file rather than against a set of strings this repo
wrote down — which would be the same thing compared to itself.

Two diffs, and the first is the stronger one:

  - **the release's five golden cases**, `encoding/tests/test_input_N.json`
    against `test_output_N.txt`. They are the release's own regression
    corpus, they cover tools, a mid-conversation system turn, an internal
    task token, a latest_reminder and a two-image vision turn, and they are
    checked-in output rather than output this test generated.
  - **conversations this file writes**, encoded both ways, for the shapes
    the goldens do not reach — every reasoning effort, thinking off, a
    tool call and its result.

And the reply reader against `parse_message_from_completion_text`.

Needs the release on disk:

  DS41_DIR=/path/to/DeepSeek-V4.1-Flash \\
      python3 -m unittest discover -s tests/serve -t . -p 'test_dsml_upstream.py'

Without it every test skips, loudly, the way test_xtml.py does for K3.
"""
import importlib.util
import json
import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from serve import dsml                                        # noqa: E402

DS41_DIR = os.environ.get("DS41_DIR", "")
_ENC = os.path.join(DS41_DIR, "encoding", "encoding.py") if DS41_DIR else ""
_FIXTURES = os.path.join(DS41_DIR, "encoding", "tests") if DS41_DIR else ""

SKIP = ""
UP = None
if not DS41_DIR:
    SKIP = "DS41_DIR is not set"
elif not os.path.exists(_ENC):
    SKIP = f"no encoding/encoding.py under {DS41_DIR}"
else:
    spec = importlib.util.spec_from_file_location("ds41_encoding", _ENC)
    UP = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(UP)
    except Exception as e:                                    # noqa: BLE001
        SKIP = f"{_ENC} did not import: {type(e).__name__}: {e}"
        UP = None


def render(messages, tools=None, **kw):
    return "".join(s.text for s in dsml.build_chat_segments(
        messages, tools, **kw))


@unittest.skipIf(SKIP, SKIP)
class TestGoldenCases(unittest.TestCase):
    """The release's own checked-in outputs."""

    def _case(self, n):
        ip = os.path.join(_FIXTURES, f"test_input_{n}.json")
        op = os.path.join(_FIXTURES, f"test_output_{n}.txt")
        if not (os.path.exists(ip) and os.path.exists(op)):
            self.skipTest(f"no golden case {n} under {_FIXTURES}")
        with open(op, encoding="utf-8") as f:
            want = f.read()
        cases = UP.load_cases(ip)
        self.assertEqual(len(cases), 1, "one case per fixture")
        case = cases[0]
        if case.get("context"):
            self.skipTest("this case carries a `context`, which the server "
                          "has no concept of")
        msgs = [dict(m) for m in case["messages"]]
        # Tools stay on the message that carries them: load_cases puts them
        # on message 0, but a case can attach them to a mid-conversation
        # system turn and moving them renders the block in the wrong place.
        n_img = sum(
            1 for m in case["messages"]
            for b in (m["content"] if isinstance(m.get("content"), list) else [])
            if isinstance(b, dict) and b.get("type") in ("image_url", "image"))
        # test_encoding.py drives these with thinking_mode="chat" unless the
        # case states one.
        thinking = (case.get("thinking_mode") or "chat") == "thinking"
        got = render(msgs, None, thinking=thinking,
                     image_prompts=[dsml.IMAGE] * n_img,
                     thinking_effort=case.get("reasoning_effort"))
        self.assertEqual(got, want)

    def test_case_1_tools_and_a_tool_result(self):
        self._case(1)

    def test_case_2_plain_chat(self):
        self._case(2)

    def test_case_3_mid_conversation_system_with_tools(self):
        self._case(3)

    def test_case_4_an_internal_task_token(self):
        self._case(4)

    def test_case_5_two_images(self):
        self._case(5)


@unittest.skipIf(SKIP, SKIP)
class TestAgainstUpstream(unittest.TestCase):
    """Shapes the goldens do not reach, encoded both ways."""

    def both(self, messages, thinking, effort=None, tools=None):
        want = UP.encode_messages(
            [dict(m) for m in messages],
            thinking_mode="thinking" if thinking else "chat",
            reasoning_effort=effort)
        got = render([dict(m) for m in messages], tools,
                     thinking=thinking, thinking_effort=effort)
        self.assertEqual(got, want)

    def test_plain(self):
        self.both([{"role": "user", "content": "What is 2+2?"}], False)
        self.both([{"role": "user", "content": "What is 2+2?"}], True)

    def test_every_effort_name(self):
        for name in dsml.EFFORT_NAMES:
            self.both([{"role": "user", "content": "hi"}], True, name)

    def test_effort_numbers(self):
        for n in (1, 2, 50, 75, 99, 100):
            self.both([{"role": "user", "content": "hi"}], True, n)

    def test_system_first(self):
        self.both([{"role": "system", "content": "You are terse."},
                   {"role": "user", "content": "hi"}], True)
        self.both([{"role": "system", "content": "You are terse."},
                   {"role": "user", "content": "hi"}], False)

    def test_multi_turn_drops_old_reasoning(self):
        self.both([
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two",
             "reasoning_content": "dropped"},
            {"role": "user", "content": "three"}], True)

    def test_mid_conversation_system(self):
        self.both([
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "system", "content": "be terse"}], True)

    def test_latest_reminder(self):
        self.both([
            {"role": "user", "content": "one"},
            {"role": "latest_reminder", "content": "2026-09-15, Rome"}], False)

    def test_tool_call_and_result(self):
        tools = [{"type": "function", "function": {
            "name": "get_weather", "description": "Get the weather",
            "parameters": {"type": "object",
                           "properties": {"city": {"type": "string"}}}}}]
        msgs = [
            {"role": "system", "content": "sys", "tools": tools},
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "",
             "reasoning_content": "need a tool",
             "tool_calls": [{"id": "c1", "type": "function", "function": {
                 "name": "get_weather",
                 "arguments": '{"city": "Rome", "days": 3}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
        ]
        # tools ride on the message here, as the release's own cases do
        self.both(msgs, True)

    def test_response_format(self):
        msgs = [{"role": "system", "content": "sys",
                 "response_format": {"type": "json_object"}},
                {"role": "user", "content": "go"}]
        self.both(msgs, False)


@unittest.skipIf(SKIP, SKIP)
class TestReplyReader(unittest.TestCase):
    """The parser against parse_message_from_completion_text."""

    MARKER_ID = {m: 1000 + i for i, m in enumerate(dsml.MARKERS)}

    def drive(self, text, thinking):
        ids = {v: k for k, v in self.MARKER_ID.items()}
        p = dsml.DSMLParser(thinking=thinking, markers=ids)
        i = 0
        while i < len(text):
            hit, at = None, len(text)
            for m in dsml.MARKERS:
                q = text.find(m, i)
                if q != -1 and q < at:
                    at, hit = q, m
            for ch in text[i:at]:
                p.feed_token(-1, ch)
            if hit is None:
                break
            p.feed_token(self.MARKER_ID[hit], hit)
            i = at + len(hit)
        p.finish()
        return p

    def both(self, text, thinking):
        want = UP.parse_message_from_completion_text(
            text, "thinking" if thinking else "chat")
        got = self.drive(text, thinking).openai_message()
        self.assertEqual(got.get("content") or "", want["content"])
        self.assertEqual(got.get("reasoning_content") or "",
                         want["reasoning_content"])
        self.assertEqual(
            [(c["function"]["name"],
              json.loads(c["function"]["arguments"]))
             for c in got.get("tool_calls", [])],
            [(c["function"]["name"],
              json.loads(c["function"]["arguments"]))
             for c in want["tool_calls"]])

    def test_plain(self):
        self.both("Hi there!" + dsml.EOS, False)

    def test_thinking(self):
        self.both("let me think</think>The answer is 4." + dsml.EOS, True)

    def test_empty_reasoning(self):
        self.both("</think>Four." + dsml.EOS, True)

    def test_one_call(self):
        d = dsml.DSML
        self.both(
            f"I'll look it up.\n\n<{d} calls>\n<{d} invoke name=\"get\">\n"
            f"<{d} parameter name=\"city\" string=\"true\">Rome</{d} parameter>\n"
            f"</{d} invoke>\n</{d} calls>" + dsml.EOS, False)

    def test_two_calls_and_json_arguments(self):
        d = dsml.DSML
        self.both(
            f"\n\n<{d} calls>\n<{d} invoke name=\"a\">\n"
            f"<{d} parameter name=\"x\" string=\"false\">1</{d} parameter>\n"
            f"</{d} invoke>\n<{d} invoke name=\"b\">\n"
            f"<{d} parameter name=\"y\" string=\"true\">two</{d} parameter>\n"
            f"</{d} invoke>\n</{d} calls>" + dsml.EOS, False)

    def test_namespaced_tool_name(self):
        d = dsml.DSML
        self.both(
            f"\n\n<{d} calls>\n<{d} invoke name=\"ns::f\">\n"
            f"<{d} parameter name=\"x\" string=\"true\">v</{d} parameter>\n"
            f"</{d} invoke>\n</{d} calls>" + dsml.EOS, False)

    def test_thinking_then_tools(self):
        d = dsml.DSML
        self.both(
            f"reasoning</think>a word\n\n<{d} calls>\n<{d} invoke name=\"a\">\n"
            f"<{d} parameter name=\"x\" string=\"true\">v</{d} parameter>\n"
            f"</{d} invoke>\n</{d} calls>" + dsml.EOS, True)


if __name__ == "__main__":
    unittest.main()
