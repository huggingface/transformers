# The terminal medium

Everything the product shows, it shows by printing lines into a terminal. This document is the
shared model of what a terminal gives us and what it refuses us; every other document leans on it,
especially [response streaming](../streaming/response-streaming.md).

## Screen, scrollback, viewport

A terminal is a grid of rows (the **screen**) sitting at the bottom of a long history of rows that
have scrolled off the top (**scrollback**). The user's scroll wheel moves a **viewport** over
scrollback plus screen. Two properties matter everywhere:

- **Printed lines are immutable history.** Once a line scrolls into scrollback, the product cannot
  edit, restyle, or remove it. The only rows a program can repaint are rows still on the screen.
- **The terminal follows the bottom on its own.** When the viewport is at the bottom, new printed
  lines slide everything up and the newest line stays visible — the terminal's native autoscroll.
  When the user scrolls up, virtually every emulator pins the viewport where it is, and new output
  accumulates below without yanking them back down.

This is why the product's design principle is **the terminal is the scrollbar**: anything printed
as ordinary lines gets correct scrolling, scroll-up-while-busy, search, and copy behavior for free,
in every emulator, because the emulator is doing it.

## The live region

The one exception to "printed lines are history" is a **live region**: an area at the bottom of the
screen that a program repaints in place by moving the cursor up and rewriting rows. The product
uses live regions in exactly two places: the [model-loading](../session/model-loading.md) progress
bar and the [tail of a streaming reply](../streaming/response-streaming.md).

A live region has hard physical limits, and they shape the whole streaming design:

- It cannot be taller than the screen. There is nowhere to draw the extra rows.
- Its contents are not in scrollback. If the user scrolls up while a live region is active, the
  live region's rows stay where they were on the physical screen area; only printed lines above it
  are in history.
- Every repaint rewrites all of its rows. Large live regions mean visible flicker on some
  emulators and a large volume of bytes written per second.

> Technical note: live regions are rich's `Live`. Its built-in overflow policies can only crop a
> too-tall renderable at the bottom or mark it with a `...` line — there is no "show the bottom"
> policy. The pre-rework display put the entire reply in one live region and hit exactly this
> wall: replies taller than the screen froze at their first screenful. The rework keeps the live
> region small on purpose; see [bug-triage.md](../bug-triage.md) BT-01.

## What a web page has that a terminal does not

Chat interfaces on the web scroll a pane they own: they can measure the scroll position, stick it
to the bottom while streaming, release it when the user scrolls up, and re-stick when the user
returns. A terminal program can do none of that — it cannot read the viewport position, cannot
detect the user scrolling, and cannot restyle what it already printed. The equivalent experience
must therefore be assembled from the two primitives above: print finished content as ordinary
lines (the terminal supplies scrolling and scroll-freedom), and keep only the still-changing part
in a bounded live region that shows its newest lines (the product supplies "follow").

## Resizing

The terminal may be resized at any moment. Printed lines are re-flowed (or not) by the emulator
according to its own rules; the product neither knows nor reacts. Live regions are repainted at
the new width on their next refresh. Consequence: making the window narrower mid-reply can leave
already-printed lines wrapped for the old width while newer output wraps for the new width — the
same thing that happens with any terminal program, and described per feature in each document's
Modifiers table.

## Color and width

Output uses the terminal's advertised colors; the reply's code blocks use a dark syntax theme
regardless of the terminal's background. All output wraps to the terminal's width at the moment it
is printed. When output is not a terminal at all (piped to a file), there are no live regions and
no colors: content is printed plainly, in full, as it becomes final.

## Open questions and verification

Verified against `huggingface/transformers` commit `4b27c4c7915b5672ab4e25349c5c2e209d25956c`
(scripted 24×80 pty; see `verification/README.md`).

- Scroll-up-while-streaming pinning is emulator behavior; confirmed in the pty rig's emulated
  history and by design in the code, but a hand pass across popular emulators (iTerm2, Windows
  Terminal, kitty, GNOME Terminal) has not been run — tracked as checklist items in
  `verification/streaming-output.md`.
