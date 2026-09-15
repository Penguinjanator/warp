# Vendored Kimi K2 chat template

`chat_template.jinja` in this directory is a byte-for-byte copy of the chat
template shipped with the upstream Kimi-K2-Instruct release. It exists so the
tool-protocol oracle in `tests/serve/test_chatfmt_upstream.py` — and the CI
job that runs it — never depends on a model download or a machine-local
`~/models` checkout.

It matters more here than the name suggests: **Kimi-Linear's tokenizer carries
K2's five tool-call tokens and its own release ships no chat_template at all**,
so the vocabulary is stated and the grammar is not. The grammar is K2's, and
only K2 publishes it. Without this file that rendering is checked only against
a parser that reads back what the renderer wrote, which agrees with itself
whatever the format is.

- Source:
  <https://huggingface.co/moonshotai/Kimi-K2-Instruct/resolve/fd1984e2b7a3350dbf7305fe73a4ede25c14de50/chat_template.jinja>
- Upstream revision: `fd1984e2b7a3350dbf7305fe73a4ede25c14de50`
- SHA-256 of the vendored file:
  `39e8c195b474c1a4148046f41fedd177b2ff8f420fa376e52fcdddb829e8f0c5`
- License: **Modified MIT** (see LICENCE), copyright notice:
  Copyright (c) 2025 Moonshot AI. The Hugging Face model repository is tagged
  `license:other` with `license_name: modified-mit`. The modification is an
  attribution clause: a commercial product or service built on the Software
  with more than 100 million monthly active users, or more than 20 million US
  dollars in monthly revenue, must prominently display "Kimi K2" in its user
  interface. WARP is neither, and vendoring 2 KB of template for a test does
  not make it one — the clause is recorded here so nobody has to go and find
  out what "modified" meant.

Used in CI oracle tests with FakeEngine; no real Kimi K2 model or weights are
used.
