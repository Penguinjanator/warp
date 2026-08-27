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
        self.assertEqual(fmt.markers, {15: "<|im_end|>"})
        self.assertEqual(sorted(fmt.roles), ["assistant", "system", "user"])

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
        """No control token in the assistant suffix and no "stop": every
        reply would run to max_tokens and report finish_reason 'length'."""
        self.refuses({"user": ["<|im_user|>", "<|im_end|>"],
                      "assistant": ["<|im_assistant|>", "\n"],
                      "open": "<|im_assistant|>"},
                     contains="ends a generated turn")

    def test_a_stop_of_its_own_ends_the_turn(self):
        """A format whose turns end because the next role marker begins —
        GLM writes `<|assistant|>answer` then `<|user|>next question` — has
        no suffix to close them with, and says so in "stop"."""
        fmt = self.load({"user": ["<|im_user|>", ""],
                         "assistant": ["<|im_assistant|>", ""],
                         "open": "<|im_assistant|>",
                         "stop": "<|im_user|>"})
        self.assertEqual(fmt.stop_marker, "<|im_user|>")
        self.assertIn(fmt.stop_id, fmt.markers)

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
        self.refuses("tool definitions",
                     tools=[{"type": "function",
                             "function": {"name": "f", "parameters": {}}}])

    def test_thinking(self):
        self.refuses("no reasoning channel", thinking=True)

    def test_response_format(self):
        self.refuses("response_format",
                     response_format={"type": "json_object"})

    def test_a_tool_result_turn(self):
        self.refuses("tool call",
                     [{"role": "tool", "content": "42", "tool_call_id": "a"}])

    def test_an_assistant_turn_carrying_tool_calls(self):
        self.refuses("tool call",
                     [{"role": "assistant", "content": None, "tool_calls": [
                         {"id": "a", "function": {"name": "f",
                                                  "arguments": "{}"}}]}])

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


# The three things a Kimi format does not need and GLM-5.3-Flash does. Each
# is here because the format could not otherwise be written down, and each
# fails quietly when it is missing: no prelude and the model is addressed in
# a format it was not trained on; no stop and the reply runs into the next
# turn; no think pair and the model's scratch work is returned as the answer.
GLM_JSON = {
    "prelude": "<|im_system|>",
    "system": ["<|im_system|>", ""],
    "user": ["<|im_user|>", ""],
    "assistant": ["<|im_assistant|>", ""],
    "open": "<|im_assistant|>",
    "think": ["<|im_middle|>", "<|im_end|>"],
    "stop": "<|im_user|>",
    "effort": "<|im_system|>Reasoning Effort: {}",
}


class TestThinkChannel(Base):
    """A chat.json that carries a prelude, a reasoning channel and a stop.

    The markers are the fake tokenizer's rather than GLM's, because what is
    under test is the format machinery and not the vocabulary; GLM's own
    file is examples/chat-glm53.json and the converter installs it.
    """

    def fmt(self, **over):
        return self.load({**GLM_JSON, **over})

    def rendered(self, fmt, **kw):
        return "".join(seg.text for seg in fmt.build_chat_segments(
            [{"role": "user", "content": "hi"}], **kw))

    def test_the_prelude_opens_the_conversation(self):
        out = self.rendered(self.fmt(), thinking=True)
        self.assertTrue(out.startswith("<|im_system|><|im_user|>hi"), out)

    def test_the_generation_prompt_opens_the_channel(self):
        out = self.rendered(self.fmt(), thinking=True)
        self.assertTrue(out.endswith("<|im_assistant|><|im_middle|>"), out)

    def test_the_effort_is_a_turn_of_its_own(self):
        out = self.rendered(self.fmt(), thinking=True, thinking_effort="high")
        self.assertIn("Reasoning Effort: High", out)

    def test_an_effort_the_format_cannot_express_is_refused(self):
        fmt = self.fmt(effort="")
        with self.assertRaises(ChatFormatError) as cm:
            self.rendered(fmt, thinking=True, thinking_effort="high")
        self.assertIn("reasoning effort", str(cm.exception))

    def test_a_format_that_always_thinks_cannot_be_asked_not_to(self):
        """GLM's template has no path that leaves the channel closed, and
        answering with it closed puts a stray close marker in the reply."""
        with self.assertRaises(ChatFormatError) as cm:
            self.rendered(self.fmt(), thinking=False)
        self.assertIn("always opens a reasoning channel", str(cm.exception))

    def test_the_stop_is_the_next_role_marker(self):
        fmt = self.fmt()
        self.assertEqual(fmt.stop_marker, "<|im_user|>")

    def test_markup_the_tokenizer_lacks_is_refused_in_every_field(self):
        for field, value in (("prelude", "<|nope|>"),
                             ("think", ["<|nope|>", "<|im_end|>"]),
                             ("effort", "<|nope|>{}")):
            with self.assertRaises(ChatFormatError, msg=field):
                self.fmt(**{field: value})

    def test_think_must_be_a_pair(self):
        self.refuses({**GLM_JSON, "think": ["<|im_middle|>"]},
                     contains="[open, close]")

    def test_the_reply_splits_into_reasoning_and_content(self):
        fmt = self.fmt()
        p = PlainParser(markers=fmt.markers,
                        think_close_id=fmt.think_close_id, in_think=True)
        for tid, piece in [(1001, "weigh"), (1002, "ing"),
                           (fmt.think_close_id, ""), (1003, "Rome")]:
            p.feed_token(tid, piece)
        self.assertEqual(p.reasoning, "weighing")
        self.assertEqual(p.content, "Rome")
        self.assertEqual(p.openai_message(),
                         {"role": "assistant", "content": "Rome",
                          "reasoning_content": "weighing"})

    def test_the_close_marker_belongs_to_neither_channel(self):
        fmt = self.fmt()
        p = PlainParser(markers=fmt.markers,
                        think_close_id=fmt.think_close_id, in_think=True)
        d = p.feed_token(fmt.think_close_id, "<|im_end|>")
        self.assertEqual((d.reasoning, d.content), ("", ""))
        self.assertFalse(p.finished)

    def test_the_stop_still_ends_the_turn_from_inside_the_channel(self):
        """A reply that hits the stop before closing its reasoning is
        truncated, not an exception: what it managed to think is returned."""
        fmt = self.fmt()
        p = PlainParser(markers=fmt.markers,
                        think_close_id=fmt.think_close_id, in_think=True)
        p.feed_token(1001, "half a thou")
        p.feed_token(fmt.stop_id, "<|im_user|>")
        self.assertTrue(p.finished)
        self.assertEqual(p.reasoning, "half a thou")
        self.assertIsNone(p.openai_message()["content"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
