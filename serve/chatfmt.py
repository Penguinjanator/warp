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
- **No reasoning channel.** There is no think markup in these containers'
  specials at all, so a request that asks for one is refused rather than
  answered without it — a server that silently drops `reasoning_effort`
  reports a different amount of reasoning than it did.
- **No images.** The media block is K3's, and its shape is part of the
  vision tower's contract, not something chat.json can place.

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
_MARKER_RE = re.compile(r"<\|[^|>]*\|>")

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
        suffix = roles.get("assistant", ("", ""))[1]
        found = _MARKER_RE.search(suffix)
        if found is None:
            raise ChatFormatError(
                'chat.json\'s "assistant" suffix carries no control token, '
                'so nothing would end a generated turn')
        stop_marker = found.group(0)

        # Every marker in the file, against the real vocabulary. This is
        # the check the whole file exists to make: markup the tokenizer
        # does not have encodes as ordinary text, and the model then reads
        # its own turn structure as prose and answers anyway.
        ids: dict[str, int] = {}
        for text in sorted({m for s in _strings(roles, opening)
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

        return cls(
            roles=roles,
            opening=opening,
            stop_marker=stop_marker,
            stop_id=ids[stop_marker],
            tool_markers=discovered,
        )

    # ---- rendering ------------------------------------------------------

    def image_prompt(self, width: int, height: int) -> str:
        raise ChatFormatError(
            "this container is served from its chat.json, which describes a "
            "text conversation and cannot place an image")

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
        if thinking:
            raise ChatFormatError(
                "this container has no reasoning channel in its tokenizer, "
                "so it cannot think on request; pass reasoning_effort "
                "'none' or thinking false", param="reasoning_effort")
        if image_prompts:
            raise ChatFormatError(
                "this container is served from its chat.json, which cannot "
                "place an image")

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
                segments.append(Segment(prefix, markup=True))
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
            segments.extend(_content_segments(message.get("content"), i))

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

        if add_generation_prompt:
            segments.append(Segment(self.opening, markup=True))
        return segments


def _strings(roles: dict[str, tuple[str, str]], opening: str) -> list[str]:
    return [s for pair in roles.values() for s in pair] + [opening]


def _content_segments(content: Any, index: int) -> list[Segment]:
    """A message's content: a plain string, or OpenAI's list of parts.

    Never markup. This is the caller's text — a user's question, a
    document, a tool's output — and encoding it in markup mode is what
    would let it forge a turn boundary.
    """
    if content is None:
        return []
    if isinstance(content, str):
        return [Segment(content)]
    out: list[Segment] = []
    for part in content:
        if part.get("type") in ("image", "image_url"):
            raise ChatFormatError(
                f"messages[{index}] carries an image, and this container is "
                f"served from its chat.json, which cannot place one",
                param=f"messages[{index}].content")
        out.append(Segment(part["text"]))
    return out


class PlainParser:
    """Read a chat.json reply, including Kimi K2 native tool calls.

    Structure is recognized only from tokenizer marker ids. Marker-looking
    text carried by an ordinary token remains ordinary model content.
    """

    def __init__(self, *, markers: Optional[dict[int, str]] = None):
        self._markers = dict(markers or {})
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

        self.content += piece
        delta.content = piece
        return delta

    def finish(self) -> Delta:
        """Flush any in-progress Kimi tool argument block."""
        if self._current_tool is not None:
            self._current_tool.json_block = self._tool_arguments
        return Delta()

    def openai_message(self) -> dict:
        message = {
            "role": "assistant",
            "content": self.content if self.content else None,
        }

        if self.tool_calls:
            message["tool_calls"] = [
                call.to_openai() for call in self.tool_calls
            ]

        return message
