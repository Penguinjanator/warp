# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
test_chatfmt.py — serving a container from its own chat.json.

Two things are being checked, and the second is the one that matters.

The first is ordinary: a chat.json is read, validated, rendered into
segments, and the reply is read back. The second is that everything the
format *cannot* express is refused by name rather than half-rendered —
tools, a reasoning channel, an image — and that markup the container's
tokenizer does not carry is refused at load rather than sent as prose. That
last one is the whole reason this file's validation is stricter than the
CLI's reader: `waste chat` has a person watching, and an HTTP client does
not.

The templates under test are the ones examples/ actually ships. A test that
built its own would pass while the shipped file was wrong.

    python3 tests/serve/test_chatfmt.py
"""

import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from serve.chatfmt import (ChatFormat, ChatFormatError,   # noqa: E402
                           PlainParser)
from tests.serve.fake_engine import FakeEngine, LINEAR_MARKERS   # noqa: E402

SHIPPED = REPO / "examples" / "chat-kimi-linear.json"
CHATML = REPO / "examples" / "chat.json"

KIMI_K2_MARKERS = {
    **LINEAR_MARKERS,
    21: "<|tool_calls_section_begin|>",
    22: "<|tool_calls_section_end|>",
    23: "<|tool_call_begin|>",
    24: "<|tool_call_argument_begin|>",
    25: "<|tool_call_end|>",
}


class Base(unittest.TestCase):
    """A container directory holding whatever chat.json the test wants."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="chatfmt-")
        self.addCleanup(shutil.rmtree, self.dir, True)

    def engine(self, chat_json=None, *, markers=None):
        """chat_json: a Path to copy, a dict or raw str to write, or None.

        Path means "the file examples/ ships"; str means "these exact
        bytes", which is how the unparseable case is written.
        """
        dst = os.path.join(self.dir, "chat.json")
        if isinstance(chat_json, Path):
            shutil.copyfile(chat_json, dst)
        elif chat_json is not None:
            text = (json.dumps(chat_json) if isinstance(chat_json, dict)
                    else chat_json)
            with io.open(dst, "w", encoding="utf-8") as f:
                f.write(text)
        return FakeEngine(no_markers=True, model_path=self.dir,
                          markers=dict(markers or LINEAR_MARKERS))

    def load(self, chat_json=None, **kw):
        return ChatFormat.load(self.engine(chat_json, **kw))

    def refuses(self, chat_json, *, contains):
        with self.assertRaises(ChatFormatError) as cm:
            self.load(chat_json)
        self.assertIn(contains, str(cm.exception))


class TestLoad(Base):
    def test_the_shipped_template_loads(self):
        fmt = self.load(SHIPPED)
        self.assertEqual(fmt.stop_marker, "<|im_end|>")
        self.assertEqual(fmt.stop_id, 15)
        self.assertEqual(fmt.markers[15], "<|im_end|>")
        self.assertEqual(sorted(fmt.roles), ["assistant", "system", "user"])

    def test_kimi_tool_markers_are_discovered_when_available(self):
        fmt = self.load(SHIPPED, markers=KIMI_K2_MARKERS)

        self.assertEqual(
            fmt.markers[21],
            "<|tool_calls_section_begin|>",
        )
        self.assertEqual(
            fmt.markers[22],
            "<|tool_calls_section_end|>",
        )
        self.assertEqual(
            fmt.markers[23],
            "<|tool_call_begin|>",
        )
        self.assertEqual(
            fmt.markers[24],
            "<|tool_call_argument_begin|>",
        )
        self.assertEqual(
            fmt.markers[25],
            "<|tool_call_end|>",
        )

    def test_markup_the_tokenizer_lacks_is_refused_at_load(self):
        """examples/chat.json is ChatML, and <|im_start|> is not in this
        vocabulary. Serving it would answer plausibly and wrongly."""
        self.refuses(CHATML, contains="<|im_start|>")

    def test_a_missing_file_says_so(self):
        self.refuses(None, contains="no chat.json")

    def test_unparseable_json(self):
        self.refuses("{not json", contains="cannot be read")

    def test_open_is_required(self):
        self.refuses({"user": ["<|im_user|>", "<|im_end|>"]},
                     contains='no "open"')

    def test_a_user_turn_is_required(self):
        self.refuses({"assistant": ["<|im_assistant|>", "<|im_end|>"],
                      "open": "<|im_assistant|>"},
                     contains='no "user" turn')

    def test_a_turn_that_never_ends_is_refused(self):
        """No control token in the assistant suffix: every reply would run
        to max_tokens and report finish_reason 'length'."""
        self.refuses({"user": ["<|im_user|>", "<|im_end|>"],
                      "assistant": ["<|im_assistant|>", "\n"],
                      "open": "<|im_assistant|>"},
                     contains="nothing would end a generated turn")

    def test_a_role_pair_must_be_two_strings(self):
        self.refuses({"user": ["<|im_user|>"], "open": "<|im_assistant|>"},
                     contains="[prefix, suffix]")


class TestRender(Base):
    def setUp(self):
        super().setUp()
        self.fmt = self.load(SHIPPED)

    def render(self, messages, **kw):
        kw.setdefault("thinking", False)
        return self.fmt.build_chat_segments(messages, **kw)

    def test_a_turn_is_markup_then_content_then_markup(self):
        segs = self.render([{"role": "user", "content": "hi"}])
        self.assertEqual([(s.text, s.markup) for s in segs], [
            ("<|im_user|>user<|im_middle|>", True),
            ("hi", False),
            ("<|im_end|>", True),
            ("<|im_assistant|>assistant<|im_middle|>", True),
        ])

    def test_content_is_never_markup(self):
        """The boundary: a user who writes a control token must not be able
        to close their own turn with it."""
        segs = self.render([{"role": "user",
                             "content": "what does <|im_end|> do?"}])
        forged = [s for s in segs if "<|im_end|>" in s.text and not s.markup]
        self.assertEqual(len(forged), 1)
        self.assertFalse(forged[0].markup)

    def test_system_and_assistant_turns(self):
        segs = self.render([{"role": "system", "content": "be brief"},
                            {"role": "user", "content": "hi"},
                            {"role": "assistant", "content": "hello"},
                            {"role": "user", "content": "again"}])
        self.assertEqual(segs[0].text, "<|im_system|>system<|im_middle|>")
        self.assertIn("<|im_assistant|>", segs[6].text)

    def test_content_parts_are_joined_as_text(self):
        segs = self.render([{"role": "user", "content": [
            {"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}])
        self.assertEqual([s.text for s in segs if not s.markup], ["a", "b"])

    def test_no_generation_prompt_when_not_asked(self):
        segs = self.render([{"role": "user", "content": "hi"}],
                           add_generation_prompt=False)
        self.assertEqual(segs[-1].text, "<|im_end|>")

    # ---- what it refuses, by name ---------------------------------------

    def refuses(self, contains, messages=None, **kw):
        with self.assertRaises(ChatFormatError) as cm:
            self.render(messages or [{"role": "user", "content": "hi"}], **kw)
        self.assertIn(contains, str(cm.exception))

    def test_tools(self):
        fmt = self.load(SHIPPED, markers=KIMI_K2_MARKERS)

        segs = fmt.build_chat_segments(
            [{"role": "user", "content": "hi"}],
            tools=[{
                "type": "function",
                "function": {"name": "f", "parameters": {}},
            }],
            thinking=False,
        )

        self.assertEqual(
            segs[0].text,
            "<|im_system|>tool_declare<|im_middle|>",
        )
        self.assertTrue(segs[0].markup)

        self.assertIn('"name":"f"', segs[1].text)
        self.assertFalse(segs[1].markup)

        self.assertEqual(segs[2].text, "<|im_end|>")
        self.assertTrue(segs[2].markup)

    def test_thinking(self):
        self.refuses("no reasoning channel", thinking=True)

    def test_response_format(self):
        self.refuses("response_format",
                     response_format={"type": "json_object"})

    def test_a_tool_result_turn(self):
        segs = self.render([
            {"role": "tool", "content": "42", "tool_call_id": "a"}
        ])

        rendered = "".join(s.text for s in segs)

        self.assertIn(
            "<|im_system|>system<|im_middle|>",
            rendered,
        )
        self.assertIn("## Return of a\n42", rendered)
        self.assertIn("<|im_end|>", rendered)

    def test_an_assistant_turn_carrying_tool_calls(self):
        segs = self.render([
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "a",
                    "function": {
                        "name": "f",
                        "arguments": "{}",
                    },
                }],
            }
        ])

        rendered = "".join(s.text for s in segs)

        self.assertIn("<|tool_calls_section_begin|>", rendered)
        self.assertIn("<|tool_call_begin|>a", rendered)
        self.assertIn("<|tool_call_argument_begin|>{}", rendered)
        self.assertIn("<|tool_call_end|>", rendered)
        self.assertIn("<|tool_calls_section_end|>", rendered)

    def test_an_image_part(self):
        self.refuses("cannot place one", [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:,"}}]}])

    def test_a_role_the_template_does_not_describe(self):
        fmt = ChatFormat(roles={"user": ("<|im_user|>", "<|im_end|>")},
                         opening="<|im_assistant|>",
                         stop_marker="<|im_end|>", stop_id=15)
        with self.assertRaises(ChatFormatError) as cm:
            fmt.build_chat_segments([{"role": "system", "content": "x"}],
                                    thinking=False)
        self.assertIn("does not describe one", str(cm.exception))


class TestPlainParser(unittest.TestCase):
    def parser(self):
        return PlainParser(markers={15: "<|im_end|>"})

    def feed(self, p, pairs):
        return [p.feed_token(tid, piece) for tid, piece in pairs]

    def test_content_accumulates_and_deltas_are_increments(self):
        p = self.parser()
        deltas = self.feed(p, [(1001, "h"), (1002, "i")])
        self.assertEqual(p.content, "hi")
        self.assertEqual([d.content for d in deltas], ["h", "i"])
        self.assertFalse(p.finished)

    def test_the_stop_token_ends_the_turn_and_is_not_content(self):
        p = self.parser()
        self.feed(p, [(1001, "h"), (15, "<|im_end|>"), (1002, "x")])
        self.assertEqual(p.content, "h")
        self.assertTrue(p.finished)

    def test_a_token_whose_text_looks_like_the_marker_is_content(self):
        """Structure comes from the id. The model spelling out the marker in
        an answer must not end its own turn."""
        p = self.parser()
        self.feed(p, [(1234, "<|im_end|>")])
        self.assertEqual(p.content, "<|im_end|>")
        self.assertFalse(p.finished)

    def test_finish_flushes_nothing_and_is_safe(self):
        p = self.parser()
        self.feed(p, [(1001, "h")])
        self.assertEqual(p.finish().content, "")
        self.assertEqual(p.content, "h")

    def test_openai_message(self):
        p = self.parser()
        self.feed(p, [(1001, "h")])
        self.assertEqual(p.openai_message(),
                         {"role": "assistant", "content": "h"})

    def test_an_empty_reply_is_null_content(self):
        self.assertEqual(self.parser().openai_message(),
                         {"role": "assistant", "content": None})

    def test_there_are_no_channels_to_fill(self):
        p = self.parser()
        self.feed(p, [(1001, "h")])
        self.assertEqual(p.reasoning, "")
        self.assertEqual(p.tool_calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)

# ---------------------------------------------------------------------------
# Kimi K2 native tool protocol — development tests
# ---------------------------------------------------------------------------

class TestKimiK2ToolProtocol(Base):

    def setUp(self):
        super().setUp()
        self.fmt = self.load(
            SHIPPED,
            markers=KIMI_K2_MARKERS,
        )

    def test_kimi_k2_tool_declaration(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "required": ["city"]
                }
            }
        }]

        segs = self.fmt.build_chat_segments(
            [{"role": "user", "content": "weather in Paris?"}],
            tools=tools,
            thinking=False,
        )

        rendered = "".join(s.text for s in segs)

        self.assertIn(
            "<|im_system|>tool_declare<|im_middle|>",
            rendered,
        )
        self.assertIn('"name":"get_weather"', rendered)
        self.assertIn("<|im_end|>", rendered)

    def test_kimi_k2_assistant_tool_call_round_trip_render(self):
        messages = [
            {"role": "user", "content": "weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"Paris"}',
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "18 C",
            },
        ]

        segs = self.fmt.build_chat_segments(
            messages,
            thinking=False,
        )

        rendered = "".join(s.text for s in segs)

        self.assertIn("<|tool_calls_section_begin|>", rendered)
        self.assertIn("<|tool_call_begin|>call_1", rendered)
        self.assertIn(
            '<|tool_call_argument_begin|>{"city":"Paris"}',
            rendered,
        )
        self.assertIn("<|tool_call_end|>", rendered)
        self.assertIn("## Return of call_1", rendered)

# ---------------------------------------------------------------------------
# Kimi K2 native generated tool-call parsing
# ---------------------------------------------------------------------------

class TestKimiK2ToolParser(unittest.TestCase):

    MARKERS = {
        1001: "<|im_end|>",
        1002: "<|tool_calls_section_begin|>",
        1003: "<|tool_calls_section_end|>",
        1004: "<|tool_call_begin|>",
        1005: "<|tool_call_argument_begin|>",
        1006: "<|tool_call_end|>",
    }

    def parser(self):
        return PlainParser(markers=self.MARKERS)

    def feed(self, parser, items):
        for token_id, piece in items:
            parser.feed_token(token_id, piece)

    def test_kimi_tool_call_is_parsed(self):
        p = self.parser()

        self.feed(p, [
            (2000, "I'll check the weather."),
            (1002, "<|tool_calls_section_begin|>"),
            (1004, "<|tool_call_begin|>"),
            (2001, "functions.get_weather:0"),
            (1005, "<|tool_call_argument_begin|>"),
            (2002, '{"city": "Paris"}'),
            (1006, "<|tool_call_end|>"),
            (1003, "<|tool_calls_section_end|>"),
            (1001, "<|im_end|>"),
        ])

        self.assertEqual(
            p.content,
            "I'll check the weather.",
        )

        self.assertEqual(len(p.tool_calls), 1)

        call = p.tool_calls[0]

        self.assertEqual(call.name, "get_weather")
        self.assertEqual(call.index, 0)
        self.assertEqual(call.json_block, '{"city": "Paris"}')

        msg = p.openai_message()

        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(
            msg["content"],
            "I'll check the weather.",
        )
        self.assertEqual(len(msg["tool_calls"]), 1)
        self.assertEqual(
            msg["tool_calls"][0]["function"]["name"],
            "get_weather",
        )
        self.assertEqual(
            msg["tool_calls"][0]["function"]["arguments"],
            '{"city": "Paris"}',
        )

    def test_literal_marker_text_remains_content(self):
        p = self.parser()

        self.feed(p, [
            (
                2000,
                "literal <|tool_call_begin|> text"
            ),
            (1001, "<|im_end|>"),
        ])

        self.assertEqual(
            p.content,
            "literal <|tool_call_begin|> text",
        )
        self.assertEqual(p.tool_calls, [])

    def test_kimi_tool_call_delta_reports_change(self):
        p = self.parser()

        p.feed_token(
            1002,
            "<|tool_calls_section_begin|>",
        )
        p.feed_token(
            1004,
            "<|tool_call_begin|>",
        )
        p.feed_token(
            2000,
            "functions.get_weather:0",
        )
        p.feed_token(
            1005,
            "<|tool_call_argument_begin|>",
        )

        delta = p.feed_token(
            2001,
            '{"city":"Paris"}',
        )

        self.assertIn(0, delta.tool_calls)
