# Session Insight — 2026-06-11 15:30

## Task / Problem Summary

Implemented two items from the Genspark code review (GensparkCR.md):

1. **Stoppage time bias** (CR §3, item 3): `engine.in_play()` computed
   `remaining = max(0, (90 − minute) / 90)`, giving a trailing team 0%
   win probability at minute 90, even though ~5 minutes of stoppage time follow.

2. **MAX_GOALS truncation gap** (CR "Low-risk code quality wins"):
   The CR asked for a unit test verifying `P(goals > MAX_GOALS) < 1e-4` for
   realistic λ. Writing that test revealed that `MAX_GOALS = 8` was actually
   inadequate — losing up to 0.86% of probability mass for strong-favourite
   matchups (λ ≈ 2.5).

---

## Root Causes

### Stoppage time
`(90 - minute) / 90` hits 0 exactly at minute 90 and cannot represent stoppage
time. This is a known modelling assumption ("90 minutes of regular time;
stoppage time ignored") but is documented as a flaw in the code comments and
now fixed.

### MAX_GOALS = 8
The Poisson tail `P(X > 8)` for the strongest realistic matchup (France 1877 vs
minnow 1282, λ_home ≈ 2.54) is **7.1 × 10⁻⁵** — just under the 1e-4 threshold.
However, with expert blending (EXPERT_W=0.85, expert scoreline 3-0) λ_home
reaches **2.61**, giving tail **8.97 × 10⁻⁵** — still under 1e-4, but with
little margin.

The earlier `MAX_GOALS = 8` gave tails of:
- λ = 1.8 → tail = 1.10 × 10⁻⁴ (exceeds 1e-4)
- λ = 2.5 → tail = 1.14 × 10⁻³ (11× the threshold)

The `_grid_probs` function normalises (divides by `p_home + p_draw + p_away`),
so sum-to-1 is preserved. But the normalisation inflates all probabilities by
`1 / (1 − tail)`, which is a systematic bias of ~0.11% per outcome for λ=2.5.
Tiny but real.

---

## What Went Well

- Stoppage time fix was literally one line in the engine plus two constant-driven
  test updates. No callers needed changing.
- The MAX_GOALS test immediately surfaced a real discrepancy and quantified it.
- Running `python -c "..."` to compute exact Poisson tails was faster than
  hand-computing and gave precise answers to set the correct ceiling for the test.

---

## What Went Poorly

- The initial test lambda list included 4.0 and 3.0, which are not achievable
  in this model without absurdly extreme expert inputs. That caused two rounds
  of test-threshold chasing before computing the actual max lambda from the
  real `teams.csv` (2.610 with expert blending).
- The CR said "Should be fine but worth a unit test" for MAX_GOALS=8 — it was
  not actually fine. The test found a real bug on first run.

---

## How It Was Solved

1. Added `STOPPAGE_MIN = 5` constant to `engine.py` (after ET/SHOOTOUT constants,
   with calibration note for 2022 WC data).
2. Changed `in_play` remaining fraction:
   ```python
   # Before
   remaining = max(0.0, (90 - minute) / 90.0)
   # After
   remaining = max(0.0, (90 + STOPPAGE_MIN - minute) / 90.0)
   ```
3. Raised `MAX_GOALS` from 8 to 10, with a comment explaining the old tail mass
   problem and confirming the new tail is < 1e-4 for all achievable λ.
4. Updated two "minute 90 = certain" tests to use `90 + STOPPAGE_MIN` (≥95) as
   the reference. Added `test_stoppage_time_nonzero_remaining_at_90` to lock
   in the new behaviour.
5. Added `test_max_goals_truncation_mass_negligible` with lambda ceiling computed
   from real `teams.csv` data (2.61).

---

## Tradeoffs / Alternatives Considered

- **STOPPAGE_MIN = 3 vs 5 vs 7**: FIFA 2022 group average was ~5.2 min. 5 is
  the natural round number. The constant is documented and easy to change.
- **MAX_GOALS = 10 vs 12**: 12 would give < 1.6e-5 tail even at λ=3.0, but
  λ=3.0 is unachievable in this model. 10 gives adequate headroom (< 9e-5)
  for the actual max λ=2.61 while keeping the grid loop cost at 121 rather
  than 169 iterations. 10 chosen.
- **Test threshold 1e-4 vs 2e-4**: kept 1e-4 (tight) after confirming the real
  data ceiling of λ=2.61 comfortably passes. A looser threshold would hide
  future regressions if MAX_GOALS were accidentally lowered.

---

## Tests Added / Updated

| File | Change |
|------|--------|
| `tests/test_engine_invariants.py` | `test_no_time_left_result_is_certain` — now uses minute `90 + STOPPAGE_MIN` |
| `tests/test_engine_invariants.py` | `test_red_card_noop_at_fulltime` — same |
| `tests/test_engine_invariants.py` | `test_stoppage_time_nonzero_remaining_at_90` — **new**, verifies trailing team has nonzero probability at minute 90 |
| `tests/test_engine_invariants.py` | `test_max_goals_truncation_mass_negligible` — **new**, verifies `P(X > MAX_GOALS) < 1e-4` for all realistic λ |
| `tests/test_inplay.py` | `test_in_play_red_noop_at_full_time` — now uses minute `90 + STOPPAGE_MIN` |

Full suite: **210/210 passed**.

---

## Lessons Learned

1. **Don't assume the CR's "should be fine" without measuring.** The MAX_GOALS
   note was flagged as "worth a unit test" with the expectation it would pass.
   The test immediately found an 11× threshold exceedance for λ=2.5. Always
   measure.
2. **Compute the true model ceiling before writing test bounds.** The test
   initially used λ=4.0 (unreachable). Running `python -c` against the real
   CSV data gave the actual max (2.61) in 5 seconds and made the test both
   correct and tight.
3. **`_grid_probs` normalises, so truncation is a bias not a crash.** The
   sum-to-1 invariant is preserved regardless of MAX_GOALS. But the normalisation
   silently inflates all outcomes by `1 / (1 − tail)`. For the old MAX_GOALS=8
   this was a 0.11%–0.86% bias per outcome — invisible to casual inspection,
   visible to a careful test.
4. **Stoppage time tests must be updated when you change the effective end
   time.** Any test checking "minute 90 = zero remaining" breaks with this
   change. Grep for such tests before merging.

---

## Follow-up Actions

- Consider adding a CI step that recomputes the max-lambda bound whenever
  `teams.csv` changes, so the test ceiling stays honest after a rating refresh.
- The other CR items (joint logistic fit for H2H/form signal ablation, market
  calibration layer, ModelConfig dataclass) remain open.
- `HOME_SUP = 0.35` goals for host-nation advantage: the CR suggests backtesting
  on 2002/2010/2014/2018/2022 host games specifically. This is a data-collection
  task, not a code change.
