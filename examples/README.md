# Examples

## chat.json — the conversation format

`waste chat` addresses an instruct model in the format it was trained on.
That format is read from `chat.json` **inside the container**, next to
`manifest.json`. Without one the CLI says so and continues raw, which is
deliberate: a guessed format is worse than a visible absence.

[chat.json](chat.json) here is the ChatML layout, which is what a large
part of the instruct ecosystem uses. Copy it into a container and edit the
strings:

```bash
cp examples/chat.json ~/models/some-model.waste/chat.json
```

Every field is optional. Each role is a `[prefix, suffix]` pair, and
`open` is what is appended after the last user turn to hand the floor to
the model:

```json
{"system":    ["<prefix>", "<suffix>"],
 "user":      ["<prefix>", "<suffix>"],
 "assistant": ["<prefix>", "<suffix>"],
 "open":      "<what starts the model's turn>"}
```

`\n` and `\t` are the escapes the reader understands. Whatever markup you
put in these strings has to exist in the tokenizer as a *single* token, or
it will be split into ordinary text and the model will not recognize it —
`waste tokenize MODEL "<|im_start|>"` is the check, and the container's
`specials.json` is the list of what is available.

## chat-k3.json — Kimi K3

[chat-k3.json](chat-k3.json) is K3's own format, transcribed from
`encoding_k3.py` in the release. Copy it in and `waste run` answers
questions instead of continuing text:

```bash
cp examples/chat-k3.json ~/models/k3.waste/chat.json
```

**Neither Kimi release ships a Jinja template**, which is why the
converter has nothing to copy: K3 does not have one. It builds the prompt
with a Python program (`encoding_k3.py`) that emits a token sequence
directly, in **XTML** — an XML-like markup where the angle brackets are
three reserved special tokens, with a fourth as the stop marker:

| in the report | token | id |
|---|---|---|
| `[open]` | `<\|open\|>` | 163587 |
| `[sep]` | `<\|sep\|>` | 163589 |
| `[close]` | `<\|close\|>` | 163588 |
| `[end_of_msg]` | `<\|end_of_msg\|>` | 163586 |

A turn is a `message` element with a `role` attribute:

```
<|open|>message role="user"<|sep|> …content… <|close|>message<|sep|><|end_of_msg|>
```

and the model is handed the floor with an unclosed assistant message:

```
<|open|>message role="assistant"<|sep|><|open|>response<|sep|>
```

Everything that is not a control token — `message`, `role="user"`, the
tag names — is ordinary text, which is what lets a format this structured
fit four prefix/suffix strings at all.

### What this template leaves out

`encoding_k3.py` is 800 lines and this file covers the text conversation.
Not covered: **tool definitions and tool results**, which are their own
XTML elements with typed `argument` children; `response_format` and JSON
schemas, which the program injects as synthetic system messages; and the
**think channel**. K3 opens `<|open|>think<|sep|>` before the response
when thinking is enabled, and the technical report measures reasoning at
up to 73% of the tokens in a request — at this engine's speeds that is
hours before an answer starts, so this template asks for the `response`
channel directly. Set the `open` field to `…<|open|>think<|sep|>` if you
want the reasoning, and expect to wait for it.

Parsing the reply back into reasoning / answer / tool-call regions is the
other half of that program, and it belongs to whatever host wants those
regions. The engine's job ends at tokens.
