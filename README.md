# Toy AI-control demo: from red-teaming to a solved game

A small, real demonstration that stochastic-game/POMDP-planning machinery can compute a
*worst-case* guarantee for an AI-control monitoring protocol, rather than just measuring an
empirical attack rate against one sampled adversary. Built end to end: a small open model
attempting real backdoors on a handful of coding problems, a cheap automated monitor, and an
actual solved game value compared against every possible fixed alternative -- then extended
into a genuinely two-sided version where the attacker's view of the defender is imperfect
too.

**Start here:** [`visualization/monitors_dilemma.html`](visualization/monitors_dilemma.html)
is the playable version of the base game; its nav bar links through the rest of the pages in
order -- the solver's convergence, the reconstructed Monitor and attacker policies, a
safety/usefulness breakdown, and the two-sided concealment finding.

## What's in this repo, and what isn't

This repo tracks the project's own scripts, the generated game files they produce, and the
visualization pages built from the solved results.

- `visualization/` -- static HTML+D3 pages, no backend, each self-contained with its own
  data and explanatory text.
- `scripts/` -- every generator, baseline, and verification script, revision-numbered
  (`_v1`, `_v2`, ...) rather than kept as loose `_corrected`/`_old` copies: a script gets a
  new suffixed file each time its logic changes, so the previous revision stays around
  unmodified for diffing rather than only living in git history. **The highest-numbered
  suffix for a given base name is always the current one** -- e.g.
  `build_toy_control_game_v2.py` (not `_v1.py`) is what every current visualization and doc
  is built from. Scripts are meant to be run from the repo root (`python3
  scripts/<name>.py`) since they resolve `games/...` paths relative to the working
  directory, not the script's own location. Current revisions as of this writing:
  - `build_toy_control_game_v2.py`, `build_toy_control_game_twosided_v1.py` (+
    `_fullinfo_v1`, `_retuned_v1`) -- generate the game description files the solvers read.
  - `adaptive_rule_baseline_v2.py`, `naive_baseline_v2.py`, `build_safety_usefulness_v2.py`,
    `build_convergence_tree_v1.py` -- the baseline comparisons and data-processing steps
    behind several of the visualizations.
  - `toy_two_sided_brute_force_v1.py`, `toy_two_sided_build_and_verify_v1.py`,
    `toy_control_twosided_verify_v1.py` -- independent ground-truth and
    constraint-verification scripts for the two-sided extension (an exact LP-duality solve,
    checked against the solver's own result).
- `games/` -- the one-sided game files (`.posg`) the generator scripts produce.
- `twosided_games/` -- the two-sided game files (`.txt`), including the toy and full-scale
  versions used to test whether concealing a defender's mixed strategy from an attacker has
  measurable value (it does, confirmed at both scales).

## Headline results

The solved, belief-adaptive Monitor policy beats every possible fixed audit schedule with a
provable guarantee, not just an empirical one. Extending to a two-sided model (the attacker
no longer has perfect information about the defender) showed a genuine, independently-
confirmed effect: concealing a real mixed strategy from an adversary has measurable
defensive value -- something a one-sided model can't express at all, since its attacker
already sees everything.
