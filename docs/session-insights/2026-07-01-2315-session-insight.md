# Session Insight — 2026-07-01 23:15

## Task / Problem Summary

Ran the `pre-match-briefing` skill for tonight's Belgium vs Senegal Round of
32 tie (2026-07-01, Match 82). `scripts/pre_match_briefing.py --match BEL SEN`
(from the earlier session today) could not resolve a stable match_id for the
pairing — its R32-lookup (`knockout._resolve_r32`) came back empty for
`{BEL, SEN}` even though this is a real, confirmed fixture happening tonight.

## Root Cause

`knockout._match_thirds` assigns the 8 qualifying third-placed teams to the 8
R32 third-slots via constrained bipartite matching — it finds **a** valid
assignment (each slot only takes a group from its official Annex-C candidate
list), not necessarily **the** assignment FIFA's actual draw produced. When
more than one valid matching exists for a given 8-of-12 qualifying
combination — which is common, not an edge case — our backtracking
(`_match_thirds`'s `bt()`, most-constrained-slot-first) can legitimately land
on a different one than reality.

Concretely, for slot M82 (`R32[82] = (("W","G"), ("3", frozenset("AEHIJ")))`
— Belgium, group G's winner, vs a third-placed team from A/E/H/I/J):
Senegal (3rd in group I, one of the valid candidate groups) is who FIFA's
real draw actually placed there. Our `_match_thirds` instead placed Algeria
(3rd in group J) there and pushed Senegal to M87 against Colombia — both
technically legal per Annex C, but only one is what actually happened. This
was confirmed against multiple sources (Sky Sports, ESPN, Ticketmaster/FIFA's
own "Match 82" listing) — Algeria's real opponent is Switzerland, Colombia's
is Ghana, neither is our model's derived pairing.

**Important distinction**: the match *number* (M82) is structurally correct
and unaffected — "group G winner enters M82" is a fixed FIFA-table fact, not
something our matching algorithm decides. Only the *identity of the
third-place opponent* filling that slot is wrong when multiple valid
matchings exist. So `knockout.match_id_for(82)` = `"M82"` is still the right,
resolvable id for this fixture once you know it from an outside source (as
this briefing did) — the gap is that `pre_match_briefing.py`'s own
R32-lookup couldn't find it *automatically*, because it trusted our
possibly-wrong `_match_thirds` output as if it were the confirmed draw.

## Relationship to the earlier gap (2026-07-01 22:30 session)

This is a sharper, now-concretely-observed instance of the same theme
already flagged as a Follow-up Action in
`2026-07-01-2230-session-insight.md`: *"knockout.run() has no way to treat
an already-decided knockout match/pairing as fixed."* That note focused on
locking in **results**; this shows the **draw itself** (who plays whom) also
needs a real-world override once it's known, not just derived from our own
matching heuristic. The R32 draw is now public and stable for every slot —
this is knowable, static ground truth exactly like a finished group score.

## What Was NOT Done (deliberately, this session)

Did not patch `_match_thirds` or add a real-draw override table. That's the
same larger, separate fix already deferred in the 22:30 insight
(`known_results`-style locking), and doing it well means hardcoding the 8
actual third-place-to-slot assignments from the real draw somewhere durable
(a small `data/r32_thirds_override.csv`? a dict literal in `knockout.py`
alongside `R32`/`TREE`?) plus a test that `_resolve_r32` prefers the override
when present. Flagging precisely, not improvising a patch mid-briefing.

## How the briefing proceeded despite the gap

Used the real match number (M82, confirmed via FIFA/ticketing sources) as an
explicit `--match-id` override to `pre_match_briefing.py`
(`--match-id M82 --match BEL SEN`), which the script already supported
(explicit `--match-id` + `--match` bypasses the auto-lookup). No adjustment
was silently attached to a match_id the engine can't resolve — M82 is fully
resolvable by `DataStore.knockout_rating_deltas`/`active_adjustments` exactly
like any other knockout match_id from `2026-07-01-2230-session-insight.md`'s
fix.

## Follow-up Actions

- Build the real-draw override for R32 third-place slots (all 8 are now
  public/known) so `_resolve_r32`/`pre_match_briefing.py`'s auto-lookup
  matches reality instead of an arbitrary-but-valid matching. Natural
  companion to the "lock in known knockout results" follow-up already on file.
- Until that exists, treat `pre_match_briefing.py`'s auto-resolved match_id
  for any third-place-involving R32 tie as unverified — cross-check against
  a live source (as done here) before trusting it, the same caution rule #1
  in CLAUDE.md already asks for with `matches.csv`.
