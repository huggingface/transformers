# Checklist: session and commands

Covers `foundations/`, `session/`, and `cross-cutting/generation-settings.md`. Setup as in
[README.md](README.md); stub server unless a real one is named.

Commit for recorded results: `4b27c4c7915b5672ab4e25349c5c2e209d25956c`.

| ID | Claim | Steps | Expected | Pri | Terminal | Result |
| --- | --- | --- | --- | --- | --- | --- |
| SC-01 | Startup sequence | Launch against a warm stub | Screen clears → help card → load flash → italic notice → `<username>:` prompt | P1 | any | pass (machine, 4b27c4c) |
| SC-02 | Prompt echo and submit | Type `hi`, Enter | Line echoes while typing; blank line after submit; reply header next | P1 | any | pass (machine, 4b27c4c) |
| SC-03 | `!exit` ends the session cleanly | `!exit` at the prompt | Process exits 0; no traceback; transcript intact | P1 | any | pass (machine, 4b27c4c) |
| SC-04 | Ctrl+C at the prompt is deferred (documented trap) | Ctrl+C at prompt; type `hi`; Enter | Nothing on Ctrl+C; on Enter the message is not sent and the session dies with a traceback (BT-09) | P1 | any | pass (machine, 4b27c4c) — behavior confirmed and filed |
| SC-05 | Ctrl+D at the prompt | Ctrl+D | Session ends with `EOFError` traceback (BT-10) | P2 | any | pass (machine, 4b27c4c) — behavior confirmed and filed |
| SC-06 | Ctrl+C mid-stream | Interrupt a long reply | Partial reply committed and readable; session dies with traceback, exit −2 (BT-08) | P1 | any | pass (machine, 4b27c4c) — behavior confirmed and filed |
| SC-07 | `!help` and unknown commands | `!help`; then `!bogus` | Full reference renders; `!bogus` prints red error + full help | P2 | any | — |
| SC-08 | `!clear` resets conversation, keeps settings and scrollback | `!set max_new_tokens=64`, chat, `!clear`, `!status`, scroll up | Fresh conversation; `max_new_tokens=64` still set; old turns visible in scrollback | P2 | any | — |
| SC-09 | `!set` grammar and bad values | `!set max_new_tokens=64`, `!set foo` (no `=`), `!set nonsense_flag=5`, then chat | First applies; second red format error; third's fate at the server recorded here | P2 | any | — |
| SC-10 | `!example` behavior incl. unknown name | `--system-prompt "…"`, then `!example code`; new session: `!example nope` | Valid: clear + echo + immediate reply, system prompt gone (BT-17). Unknown: red list of names, then a request is still sent (BT-18) | P2 | any | — |
| SC-11 | `!save` default and named | Chat once; `!save`; `!save mychat` | Default: green path `./chat_history/<model>/chat_….json`, file exists with messages+settings. Named: unknown-command error (BT-13) | P2 | any | — |
| SC-12 | `!status` dump | `!status` | Model line + full settings dump, a screenful | P3 | any | — |
| SC-13 | Warm-load notice | Launch against warm stub | Progress flashes, erases itself, italic `was already loaded.` remains | P2 | any | pass (machine, 4b27c4c) |
| SC-14 | Cold-load progress detail | Real server, undownloaded small model | Stages in order; download shows bytes, speed, ETA; wide terminal prefixes the model name | P2 | *hand*, wide + narrow | — |
| SC-15 | Another session switches the server's model | Two sessions, different models, alternate turns | Document what the waiting user sees (delay? error?) — currently an open question | P3 | any | — |
| SC-16 | Unreachable default server message | No server; `transformers chat m` (default endpoint) | Immediate failure whose text includes "please run `transformers serve`" — inside a traceback (BT-11) | P2 | any | — |
| SC-17 | Custom endpoint skips the health check | No server; `transformers chat m http://127.0.0.1:9/v1` | No pre-flight message; failure surfaces at model load as a connection traceback (BT-11) | P3 | any | — |
| SC-18 | Empty input line | Press Enter alone at the prompt | An empty message is sent and a reply streams (BT-15) | P3 | any | — |
