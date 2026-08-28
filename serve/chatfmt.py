# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
chatfmt.py — serving a container that describes its own chat format.

xtml.py renders K3's format and regions.py reads it back, and until this
file existed that was the only conversation `serve/` could hold: a
container whose tokenizer has no XTML markers got a 400 on
/v1/chat/completions and nothing else. Kimi-Linear is such a container, and
it is not exotic — it is the second model this engine ships numbers for.

The format such a container *does* carry is `chat.json`, the four
prefix/suffix strings `waste chat` has always read. This file serves from
the same file, so a container is addressed the same way over HTTP as it is
on the command line, and a hand-edited chat.json is honoured by both.

What that buys, and what it does not:

- **Plain conversation, streaming included.** system / user / assistant
  turns, and a stop that comes from the format rather than from a guess.
- **No tools.** Four strings cannot express a tool declaration, an argument
  list, or a result turn. K3's encoder needs 647 lines for that, and the
  markup Kimi-Linear's tokenizer carries for it is not transcribed anywhere
  in this repo. Refused, by name, rather than half-rendered.
- **A reasoning channel, when the format names one.** `chat.json` may
  carry `think: ["<think>", "</think>"]`, and GLM-5.3-Flash's does: its
  generation prompt opens the channel and the model closes it before the
  answer. The reply reader then fills `reasoning_content` and `content`
  separately instead of returning the model's scratch work as the reply.
  A container whose specials carry no think markup still refuses a request
  that asks for one, rather than answering without it — a server that
  silently drops `reasoning_effort` reports a different amount of reasoning
  than it did.
- **Images, when the format names the block.** `chat.json` may carry
  `image: "<|begin_of_image|><|image|><|end_of_image|>"`, and GLM's does.
  The block holds exactly one placeholder, which the engine repeats into as
  many positions as the tower produced. A container whose format does not
  name one still refuses an image part rather than dropping it.

Everything refused here is refused with a message naming what to use
instead. See https://github.com/sqliteai/waste/issues/34.

The security boundary survives intact, and gets it for free: the
prefix/suffix strings are the template's, so they go out as markup
segments, and message content is the caller's, so it goes out as plain
text. That is the same split xtml.py makes, for the same reason — a user
who writes `<|im_end|>` in a question must not thereby close their own
turn.
"""

from __future__ import annotations

import io
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .regions import Delta, ToolCall
from .xtml import Segment

# The markup that has to exist in the tokenizer. Anything of this shape in
# chat.json is checked against the container before the format is used.
# Both spellings this family uses for a control token. K3 and Kimi write
# <|...|>; GLM's reasoning markers are bare <think> and </think>, and a
# format checked for one and not the other passes while carrying markup the
# tokenizer will emit as ordinary text.
_MARKER_RE = re.compile(r"<\|[^|>]*\|>|</?think>")

# `developer` is OpenAI's newer spelling of a system turn; chat.json has no
# separate slot for it, and neither does any model that reads this format.
_ROLE_ALIASES = {"developer": "system"}

_ROLES = ("system", "user", "assistant")

_KIMI_TOOL_MARKERS = (
    "<|tool_calls_section_begin|>",
    "<|tool_calls_section_end|>",
    "<|tool_call_begin|>",
    "<|tool_call_argument_begin|>",
    "<|tool_call_end|>",
)


class ChatFormatError(ValueError):
    """A conversation this container's chat.json cannot express.

    Raised both when the file cannot be used at all — which the server
    reports once, at startup — and per request, for a conversation that
    needs more than four strings can carry.

    `param` names the request field at fault, so the 400 points at `tools`
    rather than at `messages`. A client that highlights the offending field
    highlights the wrong one otherwise, and the whole value of refusing by
    name is that the caller can act on it.
    """

    def __init__(self, message: str, *, param: Optional[str] = None):
        super().__init__(message)
        self.param = param


@dataclass(frozen=True)
class ChatFormat:
    """A container's chat.json, checked against its tokenizer.

    Built by `load`, which is the only thing that should construct one: the
    invariants below are what the renderer relies on, and they are only
    true if every marker was resolved against the real vocabulary.
    """

    roles: dict[str, tuple[str, str]]
    opening: str
    stop_marker: str
    stop_id: int
    # Everything below is optional and absent on both Kimi formats.
    prelude: str = ""
    think: Optional[tuple[str, str]] = None
    think_close_id: int = -1
    effort: str = ""
    image: str = ""
    tool_markers: dict[int, str] = field(default_factory=dict)

    @property
    def markers(self) -> dict[int, str]:
        """Control-token ids understood by the reply parser."""
        out = {self.stop_id: self.stop_marker}
        out.update(self.tool_markers)
        return out

    # ---- loading --------------------------------------------------------

    @classmethod
    def load(cls, engine: Any) -> "ChatFormat":
        """Read chat.json from the container and validate it for serving.

        Stricter than the CLI's reader on purpose. `waste chat` has a person
        watching: a template that never terminates a turn shows up as a
        reply that keeps going, and they hit Ctrl-C. An HTTP client has no
        such person, so a format that cannot stop a generation is refused
        here rather than discovered one 4096-token response at a time.
        """
        path = os.path.join(engine.model_path, "chat.json")
        if not os.path.exists(path):
            raise ChatFormatError(
                "the container has no chat.json describing its conversation "
                "format (examples/ has one per supported architecture)")
        try:
            with io.open(path, encoding="utf-8") as f:
                raw = json.loads(f.read())
        except (OSError, ValueError) as e:
            raise ChatFormatError(f"chat.json cannot be read: {e}") from None
        if not isinstance(raw, dict):
            raise ChatFormatError("chat.json must be a JSON object")

        roles: dict[str, tuple[str, str]] = {}
        for key in _ROLES:
            pair = raw.get(key)
            if pair is None:
                continue
            if (not isinstance(pair, list) or len(pair) != 2
                    or not all(isinstance(x, str) for x in pair)):
                raise ChatFormatError(
                    f'chat.json "{key}" must be a [prefix, suffix] pair of '
                    f'strings')
            roles[key] = (pair[0], pair[1])

        opening = raw.get("open")
        if not isinstance(opening, str) or not opening:
            raise ChatFormatError(
                'chat.json has no "open", so there is nothing to hand the '
                'model the floor with')
        if "user" not in roles:
            raise ChatFormatError(
                'chat.json has no "user" turn, so a request cannot be put '
                'to the model')

        # The stop marker. Without one every reply runs to max_tokens and
        # then reports finish_reason "length", which reads as a broken
        # model rather than a broken template.
        #
        # Usually it is the assistant suffix. For a format where a turn ends
        # because the *next role marker* begins — GLM writes
        # `<|assistant|>answer` and then `<|user|>next question`, with no
        # suffix at all — "stop" names it instead, and the history must not
        # carry it.
        stop_src = raw.get("stop")
        if stop_src is not None and not isinstance(stop_src, str):
            raise ChatFormatError('chat.json "stop" must be a string')
        if not stop_src:
            stop_src = roles.get("assistant", ("", ""))[1]
        found = _MARKER_RE.search(stop_src or "")
        if found is None:
            raise ChatFormatError(
                'chat.json carries no control token that ends a generated '
                'turn: give it a "stop", or an "assistant" suffix that has '
                'one')
        stop_marker = found.group(0)

        prelude = raw.get("prelude") or ""
        if not isinstance(prelude, str):
            raise ChatFormatError('chat.json "prelude" must be a string')

        think = raw.get("think")
        if think is not None:
            if (not isinstance(think, list) or len(think) != 2
                    or not all(isinstance(x, str) and x for x in think)):
                raise ChatFormatError(
                    'chat.json "think" must be an [open, close] pair of '
                    'non-empty strings')
            think = (think[0], think[1])

        effort = raw.get("effort") or ""
        if not isinstance(effort, str):
            raise ChatFormatError('chat.json "effort" must be a string')

        image = raw.get("image") or ""
        if not isinstance(image, str):
            raise ChatFormatError('chat.json "image" must be a string')
        if effort and "{}" not in effort:
            raise ChatFormatError(
                'chat.json "effort" must contain {} where the level goes')

        # Every marker in the file, against the real vocabulary. This is
        # the check the whole file exists to make: markup the tokenizer
        # does not have encodes as ordinary text, and the model then reads
        # its own turn structure as prose and answers anyway.
        extra = [prelude, effort, image] + list(think or ())
        ids: dict[str, int] = {}
        for text in sorted({m for s in _strings(roles, opening) + extra
                            for m in _MARKER_RE.findall(s)}):
            got = engine.tokenize(text, markup=True)
            if len(got) != 1:
                raise ChatFormatError(
                    f"chat.json uses {text}, which is not a single token in "
                    f"this container (got {len(got)}): the tokenizer and the "
                    f"chat format disagree")
            ids[text] = got[0]

        # Kimi K2 carries a richer native tool-call protocol in reserved
        # tokenizer tokens even though chat.json itself only describes the
        # ordinary conversation turns. Discover those markers defensively:
        # they count as structure only when every required marker is a real
        # single special token in this container.
        discovered: dict[int, str] = {}
        candidates: list[tuple[int, str]] = []

        for text in _KIMI_TOOL_MARKERS:
            got = engine.tokenize(text, markup=True)
            if len(got) != 1:
                candidates = []
                break
            candidates.append((got[0], text))

        if len(candidates) == len(_KIMI_TOOL_MARKERS):
            discovered = dict(candidates)

        return cls(roles=roles, opening=opening, stop_marker=stop_marker,
                   stop_id=ids[stop_marker], prelude=prelude, think=think,
                   think_close_id=ids[think[1]] if think else -1,
                   effort=effort, image=image, tool_markers=discovered)

    # ---- rendering ------------------------------------------------------

    def image_prompt(self, width: int, height: int) -> str:
        """The block one image expands into.

        The dimensions are ignored: unlike K3's, GLM's block carries no
        text about the source resolution — its processor does not tell the
        model the size — so the same four markers serve every image."""
        if not self.image:
            raise ChatFormatError(
                "this container is served from its chat.json, which does not "
                "say how to place an image")
        return self.image

    def build_chat_segments(self, messages: list[Any],
                            tools: Optional[list[dict]] = None,
                            *,
                            add_generation_prompt: bool = True,
                            thinking: bool = True,
                            image_prompts: Optional[list[str]] = None,
                            **kwargs: Any) -> list[Segment]:
        """Render a conversation. Same signature as xtml.build_chat_segments.

        Deliberately the same, so api.build_prompt calls one or the other
        without knowing which — the two formats differ in what they can
        express, not in how they are asked.
        """
        segments: list[Segment] = []
        # The prelude belongs to no role and opens the conversation: GLM's
        # is `[gMASK]<sop>`. Before the tool declaration, which is a system
        # turn and so belongs inside the conversation rather than ahead of
        # it — no format carries both, and this is the order they would go
        # in if one did.
        if self.prelude:
            segments.append(Segment(self.prelude, markup=True))

        if tools:
            if not self.tool_markers:
                raise ChatFormatError(
                    "this container is served from its chat.json, which "
                    "cannot express tool definitions because its tokenizer "
                    "does not carry the Kimi K2 native tool markers",
                    param="tools",
                )

            # Kimi K2 native tool declaration format.  Keep structural
            # markers separate from the JSON payload so caller-controlled
            # strings cannot be interpreted as model markup.
            segments.append(
                Segment("<|im_system|>tool_declare<|im_middle|>",
                        markup=True)
            )
            segments.append(
                Segment(json.dumps(tools, separators=(",", ":")))
            )
            segments.append(Segment("<|im_end|>", markup=True))
        for name in ("tool_choice", "response_format", "response_schema"):
            if kwargs.get(name) is not None:
                raise ChatFormatError(
                    f"this container is served from its chat.json, which "
                    f"cannot express '{name}'", param=name)
        if thinking and not self.think:
            raise ChatFormatError(
                "this container has no reasoning channel in its tokenizer, "
                "so it cannot think on request; pass reasoning_effort "
                "'none' or thinking false", param="reasoning_effort")
        if self.think and not thinking:
            # GLM's generation prompt opens <think> unconditionally and its
            # template has no path that does not; answering with the channel
            # closed produces a stray </think> in the reply. Refused rather
            # than approximated, for the same reason the line above is.
            raise ChatFormatError(
                "this container's chat format always opens a reasoning "
                "channel, so it cannot answer without one; drop "
                "'thinking'/'reasoning_effort' or use a container whose "
                "format does not", param="reasoning_effort")
        effort = kwargs.get("thinking_effort")
        if effort and not self.effort:
            raise ChatFormatError(
                "this container's chat.json does not say how to ask for a "
                "reasoning effort", param="reasoning_effort")
        if image_prompts and not self.image:
            raise ChatFormatError(
                "this container is served from its chat.json, which does not "
                "say how to place an image")

        # One block per placeholder, in order, and every one consumed:
        # a request that encoded three images and rendered two would show
        # the model a prompt whose media queue does not line up with it.
        images = iter(image_prompts) if (image_prompts and self.image) else None
        # The effort, which GLM states as a system turn of its own
        # rather than as an attribute of the request.
        if effort and self.effort:
            segments.append(Segment(self.effort.format(effort.capitalize()),
                                    markup=True))
        for i, message in enumerate(messages):
            role = message.get("role")
            role = _ROLE_ALIASES.get(role, role)

            # Kimi K2 represents a tool result as a system-style turn whose
            # content begins with "## Return of <tool_call_id>".
            if role == "tool":
                pair = self.roles.get("system")
                if pair is None:
                    raise ChatFormatError(
                        f'messages[{i}] is a tool result, but this '
                        f'container\'s chat.json has no system turn',
                        param=f"messages[{i}].role")
                tool_call_id = message.get("tool_call_id")
                if not tool_call_id:
                    raise ChatFormatError(
                        f"messages[{i}] is a tool result without "
                        "tool_call_id",
                        param=f"messages[{i}].tool_call_id")
                prefix, suffix = pair
                # K2 names this turn for the tool, not for the system:
                # `message.get('name') or message['role']` in the release's
                # own template. chat.json's system prefix carries the literal
                # `system`, so the name is substituted into it.
                who = message.get("name") or "tool"
                sys_open = self.roles["system"][0]
                marker = _rename_turn(sys_open, who)
                segments.append(Segment(marker, markup=True))
                segments.append(Segment(f"## Return of {tool_call_id}\n"))
                segments.extend(_content_segments(message.get("content"), i))
                segments.append(Segment(suffix, markup=True))
                continue

            pair = self.roles.get(role)
            if pair is None:
                raise ChatFormatError(
                    f'messages[{i}] is a "{role}" turn, and this '
                    f'container\'s chat.json does not describe one',
                    param=f"messages[{i}].role")
            prefix, suffix = pair
            segments.append(Segment(prefix, markup=True))
            segments.extend(_content_segments(message.get("content"), i, images))

            tool_calls = message.get("tool_calls")
            if tool_calls:
                if role != "assistant":
                    raise ChatFormatError(
                        f"messages[{i}] carries tool_calls on a non-assistant "
                        "turn",
                        param=f"messages[{i}].tool_calls")

                segments.append(
                    Segment("<|tool_calls_section_begin|>", markup=True)
                )

                for j, call in enumerate(tool_calls):
                    call_id = call.get("id")
                    function = call.get("function") or {}
                    arguments = function.get("arguments", "")

                    if not call_id:
                        raise ChatFormatError(
                            f"messages[{i}].tool_calls[{j}] has no id",
                            param=f"messages[{i}].tool_calls[{j}].id")

                    if not isinstance(arguments, str):
                        arguments = json.dumps(
                            arguments, separators=(",", ":")
                        )

                    segments.append(
                        Segment("<|tool_call_begin|>", markup=True)
                    )
                    segments.append(Segment(str(call_id)))
                    segments.append(
                        Segment("<|tool_call_argument_begin|>", markup=True)
                    )
                    segments.append(Segment(arguments))
                    segments.append(
                        Segment("<|tool_call_end|>", markup=True)
                    )

                segments.append(
                    Segment("<|tool_calls_section_end|>", markup=True)
                )

            segments.append(Segment(suffix, markup=True))

        if images is not None and next(images, None) is not None:
            raise ChatFormatError(
                "more images were encoded than the conversation places",
                param="messages")
        if add_generation_prompt:
            segments.append(Segment(self.opening, markup=True))
            if self.think:
                segments.append(Segment(self.think[0], markup=True))
        return segments


def _rename_turn(opening: str, who: str) -> str:
    """`<|im_system|>system<|im_middle|>` -> `<|im_system|>{who}<|im_middle|>`.

    A Kimi turn opener is three parts: a role token, a free-text name, and a
    separator. chat.json spells the whole thing out because for an ordinary
    turn the name never varies; a tool result is the one turn whose name
    comes from the message. Rewriting the middle is the smallest thing that
    keeps chat.json the single source of the turn's shape.

    A format whose opener is not that shape gets the name appended to the
    role token instead of a silent substitution, so an opener this does not
    understand renders visibly wrong rather than plausibly wrong.
    """
    i = opening.rfind("<|")
    if i <= 0 or not opening.endswith("|>"):
        return opening + who
    j = opening.find("|>")
    if j < 0 or j + 2 > i:
        return opening + who
    return opening[:j + 2] + who + opening[i:]


def _strings(roles: dict[str, tuple[str, str]], opening: str) -> list[str]:
    return [s for pair in roles.values() for s in pair] + [opening]


def _content_segments(content: Any, index: int, images: Any = None) -> list[Segment]:
    """A message's content: a plain string, or OpenAI's list of parts.

    Text is never markup. This is the caller's — a user's question, a
    document, a tool's output — and encoding it in markup mode is what
    would let it forge a turn boundary. An image part is the one exception:
    it contributes the format's own block, which is markup by definition,
    and the caller's bytes went to the tower rather than to the tokenizer.

    `images` is an iterator over the rendered blocks, one per placeholder in
    order; None means this container cannot place one.
    """
    if content is None:
        return []
    if isinstance(content, str):
        return [Segment(content)]
    out: list[Segment] = []
    for part in content:
        if part.get("type") in ("image", "image_url"):
            if images is None:
                raise ChatFormatError(
                    f"messages[{index}] carries an image, and this container "
                    f"is served from its chat.json, which does not say how to "
                    f"place one", param=f"messages[{index}].content")
            try:
                out.append(Segment(next(images), markup=True))
            except StopIteration:
                raise ChatFormatError(
                    f"messages[{index}] carries more images than the request "
                    f"encoded", param=f"messages[{index}].content") from None
            continue
        out.append(Segment(part["text"]))
    return out


class PlainParser:
    """Read a chat.json reply, including Kimi K2 native tool calls.

    Structure is recognized only from tokenizer marker ids. Marker-looking
    text carried by an ordinary token remains ordinary model content.
    """

    def __init__(self, *, markers: Optional[dict[int, str]] = None,
                 think_close_id: int = -1, in_think: bool = False):
        self._markers = dict(markers or {})
        # The channel, when the format has one. `in_think` says the
        # generation prompt left it open — which for GLM it always does —
        # and `think_close_id` is the token that hands the floor back to
        # the answer. Everything before it is reasoning; the marker itself
        # belongs to neither and is dropped.
        self._think_close = think_close_id
        self._in_think = in_think and think_close_id >= 0
        self.reasoning = ""
        self.content = ""
        self.tool_calls: list[ToolCall] = []
        self._done = False

        # Kimi K2 tool-call parsing state.
        self._tool_state = "content"
        self._tool_header = ""
        self._tool_arguments = ""
        self._current_tool: Optional[ToolCall] = None

    @property
    def finished(self) -> bool:
        return self._done

    def _marker(self, token_id: int) -> Optional[str]:
        return self._markers.get(token_id)

    def _begin_tool_call(self) -> None:
        self._tool_header = ""
        self._tool_arguments = ""
        self._current_tool = None
        self._tool_state = "header"

    def _parse_tool_header(self) -> ToolCall:
        """Parse Kimi's `functions.<name>:<index>` call header."""
        raw = self._tool_header.strip()

        if raw.startswith("functions."):
            raw = raw[len("functions."):]

        name = raw
        index = len(self.tool_calls)

        if ":" in raw:
            maybe_name, maybe_index = raw.rsplit(":", 1)
            if maybe_name:
                name = maybe_name
            try:
                index = int(maybe_index)
            except ValueError:
                pass

        return ToolCall(name=name, index=index)

    def feed_token(self, token_id: int, piece: str) -> Delta:
        delta = Delta()

        if self._done:
            return delta

        marker = self._marker(token_id)

        if marker == "<|im_end|>":
            self._done = True
            return delta

        if self._in_think and token_id == self._think_close:
            self._in_think = False
            return delta

        if marker == "<|tool_calls_section_begin|>":
            self._tool_state = "section"
            return delta

        if marker == "<|tool_call_begin|>":
            self._begin_tool_call()
            return delta

        if marker == "<|tool_call_argument_begin|>":
            if self._tool_state == "header":
                self._current_tool = self._parse_tool_header()
                self.tool_calls.append(self._current_tool)
                self._tool_state = "arguments"
            return delta

        if marker == "<|tool_call_end|>":
            if self._current_tool is not None:
                self._current_tool.json_block = self._tool_arguments
            self._current_tool = None
            self._tool_state = "section"
            return delta

        if marker == "<|tool_calls_section_end|>":
            self._current_tool = None
            self._tool_state = "content"
            return delta

        # Unknown structural markers retain the old plain-parser behavior:
        # a real control-token id ends the generated turn.
        if marker is not None:
            self._done = True
            return delta

        if not piece:
            return delta

        if self._tool_state == "header":
            self._tool_header += piece
            return delta

        if self._tool_state == "arguments":
            self._tool_arguments += piece
            if self._current_tool is not None:
                self._current_tool.json_block = self._tool_arguments
                delta.tool_calls.append(self._current_tool.index)
            return delta

        if self._tool_state == "section":
            # Ignore stray ordinary text between Kimi tool-call structures.
            return delta

        if self._in_think:
            self.reasoning += piece
            delta.reasoning = piece
        else:
            self.content += piece
            delta.content = piece
        return delta

    def finish(self) -> Delta:
        """Flush any in-progress Kimi tool argument block."""
        if self._current_tool is not None:
            self._current_tool.json_block = self._tool_arguments
        return Delta()

    def openai_message(self) -> dict:
        message: dict = {
            "role": "assistant",
            "content": self.content if self.content else None,
        }

        if self.reasoning:
            message["reasoning_content"] = self.reasoning

        if self.tool_calls:
            message["tool_calls"] = [
                call.to_openai() for call in self.tool_calls
            ]

        return message
