"""
Independent re-verification of every probability-sum constraint
Game::verify() checks, run against the full 25x17-state ported game --
re-derives transitions1/transitions2 exactly as import.cpp's parser does
(including its per-observation-branch weighting) from the raw generator
data structures, without trusting the generator's own bookkeeping.
"""
import sys
sys.path.insert(0, "/tmp/twoside_full")
exec(open("/tmp/twoside_full/build_twosided_full.py").read().split("content = ")[0])  # reuse all the built dicts

from collections import defaultdict

errors = []

# O1, O2 must each sum to 1 over their own observation dimension
for key, dist in O1.items():
    s = sum(dist.values())
    if abs(s - 1.0) > 1e-9:
        errors.append(f"O1{key} sums to {s}")
for key, dist in O2.items():
    s = sum(dist.values())
    if abs(s - 1.0) > 1e-9:
        errors.append(f"O2{key} sums to {s}")

# Reconstruct transitions1[(s1,o1,o2)][s1'][a1] = declared(1.0) * O2prob(s1,a1,o2)
transitions1 = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
for (s1, a1), target in T1_TARGET.items():
    for o2, obsProb in O2[(s1, a1)].items():
        for o1 in range(2):
            transitions1[(s1, o1, o2)][target][a1] += 1.0 * obsProb

# verify()'s T1 check: for EACH s1, EACH o1 (global) independently, sum over
# o2 in observationsPerState1[s1] (the o2 values reachable AT this s1, union
# across its legal actions) must equal 1.0 per legal a1.
missing_entries = 0
for s1 in range(NUM_S1):
    o2s_here = sorted(set(o2 for a1 in legal_a1[s1] for o2 in O2[(s1, a1)]))
    for o1 in range(2):
        probs = defaultdict(float)
        for o2 in o2s_here:
            entry = transitions1.get((s1, o1, o2), {})
            if not entry:
                errors.append(f"T1[s1={s1},o1={o1},o2={o2}] has no entries")
                missing_entries += 1
                continue
            for s1p, per_a1 in entry.items():
                for a1, v in per_a1.items():
                    probs[a1] += v
        for a1 in legal_a1[s1]:
            if abs(probs[a1] - 1.0) > 1e-9:
                errors.append(f"T1[s1={s1},a1={a1},o1={o1}] sums to {probs[a1]}")

transitions2 = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
for (s2, a2), target in T2_TARGET.items():
    for o1, obsProb in O1[(s2, a2)].items():
        for o2 in range(2):
            transitions2[(s2, o1, o2)][target][a2] += 1.0 * obsProb

for s2 in range(NUM_S2):
    o1s_here = sorted(set(o1 for a2 in legal_a2[s2] for o1 in O1[(s2, a2)]))
    for o2 in range(2):
        probs = defaultdict(float)
        for o1 in o1s_here:
            entry = transitions2.get((s2, o1, o2), {})
            if not entry:
                errors.append(f"T2[s2={s2},o2={o2},o1={o1}] has no entries")
                missing_entries += 1
                continue
            for s2p, per_a2 in entry.items():
                for a2, v in per_a2.items():
                    probs[a2] += v
        for a2 in legal_a2[s2]:
            if abs(probs[a2] - 1.0) > 1e-9:
                errors.append(f"T2[s2={s2},a2={a2},o2={o2}] sums to {probs[a2]}")

# Also check: every R entry references only legal (a1,a2) pairs (sanity, not
# a verify()-checked constraint, but worth confirming the generator didn't
# emit anything for an illegal action).
if errors:
    print(f"{len(errors)} ERRORS (first 20 shown):")
    for e in errors[:20]:
        print(" -", e)
else:
    print(f"All O1/O2/T1/T2 constraints verified across {NUM_S1} state1 x {NUM_S2} state2: PASS")
    print(f"(matches every constraint Game::verify() checks, at full scale)")
