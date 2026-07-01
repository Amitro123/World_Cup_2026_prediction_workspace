# Session Insight — 2026-07-01 22:30

## Task / Problem Summary

A manual chat session (Claude, no code tools) worked through the July 2026
knockout matches with the person and surfaced three things to fix here:

1. `data/matches.csv` had one stale row: `D5` (Paraguay vs Australia,
   kickoff 2026-06-26) was still `status=scheduled` with no goals, even
   though the group stage finished days ago and Round of 32 is live as of
   today (2026-07-01).
2. A real infrastructure gap: `news_adjustments.csv` /
   `models.py:_adjusted_inputs()` only resolve a `match_id` that exists as a
   static row in `matches.csv` — i.e. group-stage matches. Knockout ties
   (R32 onward) are built dynamically at simulation time from
   `knockout.R32`/`TREE` and have no such row, so a `rating_delta` attached
   to a knockout match_id silently did nothing.
3. No repeatable way to assemble "the model's numbers + a news check" for a
   specific fixture without re-deriving it by hand in chat each time.

---

## Root Causes

### `matches.csv` staleness (item 1)

`status` is not self-updating — it only changes when something calls
`update_match_state` or a human edits the CSV. D5 was missed in the earlier
manual batch-update (12 of the 13 flagged matches were fixed; D5 was not).
Because `knockout.run()`/`_simulate_group_fast` treat any non-`finished` game
as **unplayed and therefore simulated**, this didn't crash or error — it
just silently treated a real, known 0-0 draw as a coin-toss-ish unplayed
fixture on every Monte-Carlo iteration. The model math was correct throughout;
the *input* was wrong. This is exactly the failure mode rule #1 in this
repo's CLAUDE.md warns about, and it reproduced again despite the warning
already existing — the earlier fix just missed one row.

### The `news_adjustments` / knockout gap (item 2)

`DataStore._adjusted_inputs(match_id)` starts with `m = self.match(match_id)`,
which does `self.matches.loc[self.matches.match_id == match_id].iloc[0]` —
a hard lookup into the *static* `matches.csv` table. That table only has 72
rows (the group stage; see `README.md`'s own note: `stage: group (knockout
slots live in knockout.py)`). Knockout ties are never rows in that table —
`src/knockout.py` resolves R32 pairings from group results at simulation
time (`_resolve_r32`) and later rounds from simulated winners
(`TREE`/`TREE_ORDER`), entirely in memory, separately for every one of the
`N_SIMS` Monte-Carlo iterations. There was consequently no stable identifier
a human (or Hermes) could attach a `rating_delta` to for, say, "Belgium's
captain is injured for the R16 match" — and even if one invented an ad-hoc
id, nothing in `knockout.py` ever queried `news_adjustments.csv` at all; the
Monte-Carlo hot loop reads raw `teams.fifa_points` only (see `_prepare`'s
`ratings = dict(zip(ds.teams.team_id, ds.teams.fifa_points))`). This is why
`predict_bracket.py` and `sim_r32_news.py` both hand-roll a flat
team-level `NEWS = {...}` dict instead of using the official pipeline — the
official pipeline structurally could not reach them.

---

## What Changed

### 1. `data/matches.csv`
- `D5` (PAR vs AUS): `scheduled` → `finished`, `0-0` (confirmed via FIFA.com /
  ESPN — Paraguay 0-0 Australia, 2026-06-25, San Francisco Bay Area Stadium).
  All 72 group-stage rows are now `finished`; `pytest`/`--leakage` still pass.

**Champion-odds impact** (n=8000, seed=2026, before vs. after — `knockout.run`):

| team | qualify% before | qualify% after |
|------|-----------------:|----------------:|
| PAR  | 58.4 | **100.0** |
| AUS  | 82.7 | **100.0** |

Both had already really qualified; the stale row was making the sim
*re-decide* an already-known outcome on every iteration, which also nudges
the whole bracket (title% for the top 8 shifted by up to ±0.8pp — small here
because D5 doesn't touch a title contender directly, but the mechanism is
exactly what would produce "a materially wrong champion odds table" for a
row that *does* touch a contender).

### 2. Stable match_ids for knockout ties + news_adjustments wiring

- `src/knockout.py`: added `ALL_MATCH_NOS` (every simulated match number,
  M73–M102 + M104 — M103, the third-place play-off, is deliberately excluded
  since it isn't simulated) and `match_id_for(match_no) -> "M73"` etc. Added
  `build_knockout_news(ds)`, which precomputes `match_no -> {team_id: delta}`
  once per `run()` call (same pattern as the existing `build_h2h`/`build_form`
  precompute step — the hot loop still never touches pandas). Wired the
  result into `_prepare`'s context and applied it at every point a knockout
  tie is resolved (`simulate_once`'s R32 loop and `TREE_ORDER` loop,
  `simulate_detail`'s per-round loop) via a new `_ko_rating(ctx, match_no,
  team_id)` helper.
- `src/models.py`: added `DataStore.knockout_rating_deltas(match_id)`, which
  reads `active_adjustments(match_id)` directly — it never calls
  `self.match(match_id)`, so it does not require a `matches.csv` row. Only
  `rating_delta` is wired for knockout ties; `lambda_mult` is accepted by
  `add_news_adjustment` (no validation against a match_id) but has no effect
  in the knockout sim yet — see Follow-up Actions.
- **Scope boundary, deliberately**: group-stage news adjustments already only
  affected the single-match dashboard view (`pre_match_probs` /
  `match_briefing`), never the whole-tournament Monte-Carlo (`knockout.run`
  reads raw `fifa_points` for group games too, un-adjusted). This session
  only closes the knockout-specific gap that was explicitly reported; it does
  **not** retrofit group-stage news into `knockout.run()`. That is a
  separate, pre-existing design choice, not a regression introduced here.

### 3. `scripts/pre_match_briefing.py` (new)

`python scripts/pre_match_briefing.py --match BEL SEN` (or `--match-id C4`
for a group-stage row). Resolves the fixture (group row lookup, else a
deterministic R32-match-number lookup via `knockout._resolve_r32` — reliable
now that the group stage is fully `finished` — else a clearly-flagged
unresolvable placeholder for a not-yet-known pairing), prints FIFA
rating/h2h_sup/form_sup, xG, 1X2, and (for knockout ties) an
analytic advance-probability through ET/pens (mirrors `predict_bracket.py`'s
`ko_prob` so the two tools agree). Never calls `add_news_adjustment` unless
you pass `--apply`; by default a `--news-file findings.json` only prints a
"PROPOSED ADJUSTMENTS" table plus the exact `ds.add_news_adjustment(...)`
call for each finding. Also prints bookmaker-anchor / GOLAZO app-odds EV when
available for the fixture (group stage only — `app_odds.csv` and
`market_odds.csv` don't currently carry knockout-stage rows).

---

## What Went Well

- The knockout gap was closeable without touching `engine.py` at all —
  `resolve_knockout`/`knockout_winner` already take a rating directly, so
  applying a delta before the call (rather than threading a multiplier
  through the engine's signature) needed no engine changes. Kept the diff
  small and the blast radius contained to `knockout.py` + `models.py`.
- Reusing the `build_h2h`/`build_form`-style once-per-run precompute pattern
  for `build_knockout_news` meant zero pandas access inside the hot
  Monte-Carlo loop — consistent with the perf constraint already documented
  in `knockout.py`'s `_prepare` docstring.
- Writing the integration test (inject a −300 rating_delta on the real R32
  tie `M73` = South Africa vs Canada) immediately validated the whole chain
  end-to-end: r16% for the affected side dropped from ~32.8% to ~8.2% at
  n=6000 — a large, unambiguous, reproducible signal.

## What Went Poorly

- **Wrote real test pollution into `data/news_adjustments.csv` once.** An
  early manual smoke-test called `ds.add_news_adjustment(...)` directly
  (which persists via `save_news()`) against the real `data/` directory
  instead of injecting a row in memory. It landed in the tracked CSV
  (`git diff` caught it immediately) and was reverted with
  `git checkout -- data/news_adjustments.csv` before any test ran against it.
  The actual test suite avoids this entirely via an in-memory
  `_inject_news_row` helper. **Lesson: never call a `DataStore` mutator that
  persists (`add_news_adjustment`, `save_*`) against the real `data/`
  directory during exploration — always mutate `ds.news`/`ds.matches`
  in-memory, the same pattern `tests/test_integrity.py` already uses.**
- `scripts/pre_match_briefing.py`'s first draft mis-oriented `app_odds.csv`
  odds (that file's home/away order is independent of `matches.csv`'s), which
  briefly showed Brazil (a heavy favourite) at 6.5 decimal odds. Caught by
  manually running the script against a real fixture and eyeballing the
  numbers — `predict_value.py` already had the correct `flipped` re-orientation
  pattern; the fix just mirrors it.

---

## Tradeoffs / Alternatives Considered

- **Match-id scheme for knockout ties**: `"M<official-number>"` (e.g. `M78`)
  vs. a team-pair id like `"R32-BEL-SEN"`. Went with match-number, per the
  task's own steer, because it's the only scheme that survives past R32 — a
  team-pair id is meaningless for an R16+ tie whose participants aren't known
  until earlier rounds are simulated, while `TREE`'s match numbers are fixed
  bracket-tree positions regardless of who reaches them.
- **`lambda_mult` on knockout ties**: not wired this session. Doing it
  properly means threading an optional multiplier through
  `engine.expected_goals`/`resolve_knockout`/`knockout_winner` (currently
  rating-only). `rating_delta` covers the concrete ask (the explicit test
  requirement) and is already the dominant kind used elsewhere in this repo
  (`predict_bracket.py`'s flat `NEWS` dict is effectively rating_delta-only).
  Left as a documented follow-up rather than expanding the diff.
- **`pre_match_briefing.py`'s live news step**: the task allowed a TODO
  placeholder if runtime web access isn't available, which is the case for a
  plain `python` invocation. Implemented a `--news-file` JSON contract instead
  of guessing at a search API, so the workflow is: run the script for the
  base numbers → do the browser-connector news check (per the
  `pre-match-briefing` skill) → save findings as JSON → re-run with
  `--news-file` to see the adjusted numbers → `--apply` only after a human
  reviews the printed table.

---

## Tests Added / Updated

| File | Change |
|------|--------|
| `tests/test_knockout.py` | `test_match_id_for_covers_every_simulated_match_no` — **new**, locks in the M73-M104 numbering (excl. M103) |
| `tests/test_knockout.py` | `test_build_knockout_news_empty_by_default` — **new** |
| `tests/test_knockout.py` | `test_build_knockout_news_reads_rating_delta_by_match_id` — **new**, also confirms `lambda_mult` is correctly ignored (documented limitation) and same-team deltas sum |
| `tests/test_knockout.py` | `test_knockout_rating_delta_shifts_run` — **new**, end-to-end: injects a rating_delta on a real R32 match_id and asserts `knockout.run()`'s r16% for the affected team drops by >15pp |

Full suite: **214/214 passed** (210 pre-existing + 4 new).

---

## Lessons Learned

1. **A documented failure mode can still recur if the fix is manual and
   partial.** Rule #1 in this repo's CLAUDE.md already exists because of a
   past stale-`matches.csv` incident; this session found one more row that a
   prior manual pass missed. A partial manual fix does not fully retire the
   risk — worth considering a small `scripts/check_stale_matches.py` that
   flags any `scheduled` row whose `kickoff` is more than N hours in the past
   (see Follow-up Actions).
2. **"Works for the group stage" is a load-bearing assumption throughout this
   codebase**, not just in `news_adjustments`. `market_odds.csv`/`app_odds.csv`
   also have zero knockout-stage rows today, and `knockout.run()`'s
   Monte-Carlo never locks in a real (already-decided) knockout result the
   way it locks in a finished group game — it *always* simulates every
   knockout tie probabilistically, even ones the real world has already
   played. That's a bigger, separate gap from the one this session fixed (see
   Follow-up Actions) but it rhymes with the exact same root cause: this
   project was originally built group-stage-first, and the tournament has now
   moved past that phase.
3. **Never call a persisting `DataStore` method against the real `data/`
   directory while exploring/testing** — mutate the in-memory frame instead.
   Caught immediately via `git diff`/`git status --porcelain` here, but it's
   an easy, silent way to corrupt shipped data if you're not checking after
   every exploratory script.
4. **Rule #2 (never hand-build a bracket) held up well as a design
   constraint**: resolving an R32 match number for a given team pair
   (`r32_match_no` in the new script) was implemented by calling
   `knockout._resolve_r32` — reusing the real official-bracket logic — rather
   than re-deriving pairings by hand.

---

## Follow-up Actions

- **Knockout results are never locked into the Monte-Carlo sim.** Now that
  R32 is genuinely underway (confirmed live as of 2026-07-01 — e.g. Brazil
  2-1 Japan, already played), `knockout.run()` has no way to treat an
  already-decided knockout match as fixed the way `_simulate_group_fast`
  does for a `finished` group game — it always re-simulates every knockout
  tie from ratings. Fixing this properly would mean extending
  `_resolve_r32`/`simulate_once`/`simulate_detail` to accept a
  `known_results: dict[match_no, winner]` (or reuse the same stable
  `match_id_for` ids against a small results table) and skip simulation for
  those ties. This is materially larger than the news_adjustments fix in
  this session and deserves its own design/test pass rather than being
  folded in here.
- **Wire `lambda_mult` for knockout ties** once a concrete adjustment needs
  it (today only `rating_delta` is used anywhere in this repo).
- **`scripts/check_stale_matches.py`** (not built this session): flag any
  `matches.csv` row still `scheduled`/`live` whose `kickoff` is more than a
  few hours in the past, so the D5-style miss surfaces automatically instead
  of depending on a manual re-read of the whole file.
- Rule #1 in CLAUDE.md should probably gain a one-line pointer to this
  session: *"a manual pass already missed one row once (2026-07-01, D5) —
  prefer a scripted staleness check over eyeballing the CSV."*
