# Generation settings

## Summary

Every request carries a set of generation settings — how many tokens may be generated, whether to
sample, temperature, and the rest. They are assembled once at launch (defaults, then an optional
config file, then command-line flags), changed at any prompt with `!set`, inspected with
`!status`, and sent with every turn. They are session state: `!clear` does not touch them.

## The layering

From bottom to top, later layers winning:

1. **Library defaults** for every knob.
2. **`--generation-config`** — a local JSON file or a Hub repo's `generation_config.json`.
3. **Session defaults** — the product turns on sampling and caps replies at 256 new tokens
   (`do_sample=True`, `max_new_tokens=256`), the cap being why the
   [continue? flow](../streaming/length-limit-continuation.md) is common.
4. **Command-line flags** — trailing `flag=value` arguments at launch, same grammar as `!set`.
5. **`!set flag=value …`** — at any prompt, merged over everything, in effect until the session
   ends or a later `!set` overrides.

The grammar, everywhere it appears: space-separated `flag=value` pairs; numbers bare, booleans
`True`/`False` in any case, `None` for null, lists of integers as `[1,2]`, everything else a
string. A pair missing `=` is rejected with a red format error (the whole line is ignored).

## Where the user sees them

- **`!status`** — the full dump: model name, then every setting on its own line, defaults
  included. The only view there is; there is no diff-from-default view.
- **The stats line and the continue? flow** — `max_new_tokens` shows up as behavior: reply length
  and how often continuation is offered.
- **Reply character** — sampling and temperature change the text, invisible as UI.

> Technical note: the merged settings are serialized and sent inside each request
> (`extra_body.generation_config`), so the server generates with exactly the session's settings;
> nothing is negotiated per turn.

## Interactions with the rest of the product

- **Turns** — settings are read at the moment a request is sent; a `!set` between turns affects
  the next turn, never a stream in flight.
- **Chat history** — settings are not messages; changing them mid-conversation silently changes
  the character of later replies within the same history.
- **`!save`** — the saved file records the settings alongside the messages, so a saved
  conversation carries the knobs that produced it.
- **`!clear` / `!example`** — reset the conversation, not the settings; a `!set` survives both.
- **The server** — unknown or malformed values may only fail at the server, mid-turn
  ([connection and errors](../session/connection-and-errors.md)).

## Edge cases

- `!set` accepts flags it has never heard of; they ride along to the server (whether they are
  ignored or rejected there is SC-09).
- Setting `max_new_tokens` very low (e.g. `!set max_new_tokens=8`) makes every reply end in the
  continue? flow — a good way to *see* that flow on demand.
- String values cannot contain spaces (the grammar splits on them); there is no quoting.
- `!set do_sample=False` with a temperature set is legal; the temperature is simply sent along
  and ignored.

## Open questions and verification

Verified against `huggingface/transformers` commit `4b27c4c7915b5672ab4e25349c5c2e209d25956c`
(grammar and layering from `parse_generate_flags` and its tests in `tests/cli/test_chat.py`;
`!status` shape from the code path).

- Server-side handling of unknown flags (SC-09).
- Whether `--generation-config` pointing at a gated Hub repo prompts for auth or fails — untried.
