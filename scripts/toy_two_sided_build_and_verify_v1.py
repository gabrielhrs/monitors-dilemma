"""
Builds the minimal two-sided toy game and independently verifies every
probability-sum constraint that Game::verify() will check in the C++ solver,
before handing the file off to be compiled/run.
"""
import itertools

# ---- design constants ----
C_PASS, C_AUDIT, C_CATASTROPHE = 0.0, -1.0, -10.0
P_SCHEMER_PRIOR = 0.25
P_LEAK = 0.3          # O1: prob a real attack passively leaks to the Monitor (flag), independent of a1
P_WATCHED_IF_AUDIT = 0.8   # O2: prob Schemer notices being audited
P_PARANOIA_IF_PASS = 0.1   # O2: prob Schemer falsely feels watched when Monitor passed

# state1 (Monitor private): 0=(r0,b1) 1=(r1,b1) 2=(r1,b0) 3=terminal
# state2 (Schemer private): 0=(r0,Benign) 1=(r0,Schemer) 2=(r1,Benign) 3=(r1,Schemer) 4=terminal
NUM_S1, NUM_S2 = 4, 5
NUM_A1, NUM_A2 = 2, 2   # 0=Pass/Honest, 1=Audit/Attack
NUM_O1, NUM_O2 = 2, 2   # o1: 0=clear,1=flag ; o2: 0=no-signal,1=felt-watched

lines = []
lines.append(f"discount: 0.95")
lines.append(f"states: {NUM_S1} {NUM_S2}")
lines.append(f"actions: {NUM_A1} {NUM_A2}")
lines.append(f"observations: {NUM_O1} {NUM_O2}")

# initial beliefs: Schemer knows the true starting budget (s1=0) with certainty;
# Monitor's belief over Schemer's type is the persistent-type prior.
lines.append("init1: 0 1.0 -1")
lines.append(f"init2: 0 {1-P_SCHEMER_PRIOR} 1 {P_SCHEMER_PRIOR} -1")

# legal actions per state
lines.append("actions1: 0   0 1 -1")   # round0, budget1: Pass or Audit
lines.append("actions1: 1   0 1 -1")   # round1, budget1: Pass or Audit
lines.append("actions1: 2   0 -1")     # round1, budget0: Pass only
lines.append("actions1: 3   0 -1")     # terminal: dummy

lines.append("actions2: 0   0 1 -1")   # r0 Benign: Honest or Attack(no-op)
lines.append("actions2: 1   0 1 -1")   # r0 Schemer: Honest or Attack
lines.append("actions2: 2   0 1 -1")   # r1 Benign
lines.append("actions2: 3   0 1 -1")   # r1 Schemer
lines.append("actions2: 4   0 -1")     # terminal: dummy

# --- O1: [s2] [a2] [o1] [prob] -- Monitor's passive, audit-independent signal ---
O1 = {}  # (s2,a2) -> {o1: prob}
for s2 in [0, 2]:          # Benign states (any round): always clear
    for a2 in [0, 1]:
        O1[(s2, a2)] = {0: 1.0}
for s2 in [1, 3]:          # Schemer states
    O1[(s2, 0)] = {0: 1.0}                       # Honest: always clear
    O1[(s2, 1)] = {0: 1 - P_LEAK, 1: P_LEAK}      # Attack: leaks w.p. P_LEAK
O1[(4, 0)] = {0: 1.0}      # terminal dummy

for (s2, a2), dist in O1.items():
    for o1, p in dist.items():
        lines.append(f"O1: {s2} {a2} {o1} {p}")

# --- O2: [s1] [a1] [o2] [prob] -- Schemer's noisy signal about the Monitor's realized action ---
O2 = {}
for s1 in [0, 1]:          # budget-1 states: both Pass and Audit legal
    O2[(s1, 0)] = {0: 1 - P_PARANOIA_IF_PASS, 1: P_PARANOIA_IF_PASS}
    O2[(s1, 1)] = {0: 1 - P_WATCHED_IF_AUDIT, 1: P_WATCHED_IF_AUDIT}
O2[(2, 0)] = {0: 1 - P_PARANOIA_IF_PASS, 1: P_PARANOIA_IF_PASS}   # budget0: only Pass legal
O2[(3, 0)] = {0: 1.0}      # terminal dummy

for (s1, a1), dist in O2.items():
    for o2, p in dist.items():
        lines.append(f"O2: {s1} {a1} {o2} {p}")

# --- T1: [s1] [a1] [o1] [o2] [s1'] [prob] --
# state1 evolution never depends on o1 -> split 0.5/0.5 across o1, identical at every o2.
T1_TARGET = {  # (s1,a1) -> s1'
    (0, 0): 1, (0, 1): 2,
    (1, 0): 3, (1, 1): 3,
    (2, 0): 3,
    (3, 0): 3,
}
# o2 values reachable at each s1 (union over legal actions there)
O2_REACHABLE_AT_S1 = {
    0: sorted(set(o2 for (s1, a1), d in O2.items() if s1 == 0 for o2 in d)),
    1: sorted(set(o2 for (s1, a1), d in O2.items() if s1 == 1 for o2 in d)),
    2: sorted(set(o2 for (s1, a1), d in O2.items() if s1 == 2 for o2 in d)),
    3: sorted(set(o2 for (s1, a1), d in O2.items() if s1 == 3 for o2 in d)),
}
for (s1, a1), s1p in T1_TARGET.items():
    for o2 in O2[(s1, a1)]:               # only o2 values THIS action can actually produce
        for o1 in range(NUM_O1):
            lines.append(f"T1: {s1} {a1} {o1} {o2} {s1p} 1.0")

# --- T2: [s2] [a2] [o2] [o1] [s2'] [prob] (o2 before o1 in the file!) --
# state2 evolution never depends on o2 -> split 0.5/0.5 across o2, identical at every o1.
T2_TARGET = {  # (s2,a2) -> s2'
    (0, 0): 2, (0, 1): 2,
    (1, 0): 3, (1, 1): 3,
    (2, 0): 4, (2, 1): 4,
    (3, 0): 4, (3, 1): 4,
    (4, 0): 4,
}
O1_REACHABLE_AT_S2 = {s2: sorted(set(o1 for (ss2, a2), d in O1.items() if ss2 == s2 for o1 in d))
                       for s2 in range(NUM_S2)}
for (s2, a2), s2p in T2_TARGET.items():
    for o1 in O1[(s2, a2)]:               # only o1 values THIS action can actually produce
        for o2 in range(NUM_O2):
            lines.append(f"T2: {s2} {a2} {o2} {o1} {s2p} 1.0")

# --- R: [s1] [s2] [a1] [a2] [reward] -- only round0 (s1=0) and round1 (s1 in {1,2}) pairs need entries ---
def catch_prob(a1):
    return 1.0 if a1 == 1 else 0.0

def cost(a1):
    return C_AUDIT if a1 == 1 else C_PASS

def reward(is_schemer, real_attack, a1):
    r = cost(a1)
    if real_attack:
        r += (1 - catch_prob(a1)) * C_CATASTROPHE
    return r

R = {}
# round0: s1=0 paired with s2 in {0(Benign),1(Schemer)}
for a1 in [0, 1]:
    for a2 in [0, 1]:
        R[(0, 0, a1, a2)] = reward(False, False, a1)                      # Benign, Attack is a no-op
        real_attack = (a2 == 1)
        R[(0, 1, a1, a2)] = reward(True, real_attack, a1)
# round1: s1=1 (budget1, both actions) and s1=2 (budget0, Pass only), paired with s2 in {2(Benign),3(Schemer)}
for s1 in [1, 2]:
    legal_a1 = [0, 1] if s1 == 1 else [0]
    for a1 in legal_a1:
        for a2 in [0, 1]:
            R[(s1, 2, a1, a2)] = reward(False, False, a1)
            real_attack = (a2 == 1)
            R[(s1, 3, a1, a2)] = reward(True, real_attack, a1)

for (s1, s2, a1, a2), r in R.items():
    lines.append(f"R: {s1} {s2} {a1} {a2} {r}")

content = "\n".join(lines) + "\n"
with open("/tmp/twoside_build/toy_two_sided.txt", "w") as f:
    f.write(content)

print(f"Wrote {len(lines)} lines.")

# ============ INDEPENDENT VERIFICATION of everything Game::verify() will check ============
errors = []

# O1, O2 each must sum to 1 over their own observation dimension, per (s2,a2) / (s1,a1)
for key, dist in O1.items():
    s = sum(dist.values())
    if abs(s - 1.0) > 1e-9:
        errors.append(f"O1{key} sums to {s}, not 1.0")
for key, dist in O2.items():
    s = sum(dist.values())
    if abs(s - 1.0) > 1e-9:
        errors.append(f"O2{key} sums to {s}, not 1.0")

# Reconstruct transitions1[soo_map(s1,o1,o2)][s1'][a1] = declared_prob * O2prob(s1,a1,o2), exactly as import.cpp does
from collections import defaultdict
transitions1 = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # [(s1,o1,o2)][s1'][a1]
for (s1, a1), s1p in T1_TARGET.items():
    for o2, obsProb in O2[(s1, a1)].items():
        for o1 in range(NUM_O1):
            declared = 1.0
            transitions1[(s1, o1, o2)][s1p][a1] += declared * obsProb

legal_actions1 = {0: [0, 1], 1: [0, 1], 2: [0], 3: [0]}
for s1 in range(NUM_S1):
    o1s_global = range(NUM_O1)
    o2s_here = O2_REACHABLE_AT_S1[s1]
    for o1 in o1s_global:                      # verify() checks EACH o1 independently
        probs = defaultdict(float)
        for o2 in o2s_here:
            entry = transitions1.get((s1, o1, o2), {})
            if not entry:
                errors.append(f"T1[s1={s1},o1={o1},o2={o2}] has no entries (verify() ERROR)")
                continue
            for s1p, per_a1 in entry.items():
                for a1, v in per_a1.items():
                    probs[a1] += v
        for a1 in legal_actions1[s1]:
            if abs(probs[a1] - 1.0) > 1e-9:
                errors.append(f"T1[s1={s1},a1={a1},o1={o1}] sums to {probs[a1]}, not 1.0")

transitions2 = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # [(s2,o1,o2)][s2'][a2]... wait soo_map uses (s2,o1,o2) via state1-style map but for state2 too
for (s2, a2), s2p in T2_TARGET.items():
    for o1, obsProb in O1[(s2, a2)].items():
        for o2 in range(NUM_O2):
            declared = 1.0
            transitions2[(s2, o1, o2)][s2p][a2] += declared * obsProb

legal_actions2 = {0: [0, 1], 1: [0, 1], 2: [0, 1], 3: [0, 1], 4: [0]}
for s2 in range(NUM_S2):
    o2s_global = range(NUM_O2)
    o1s_here = O1_REACHABLE_AT_S2[s2]
    for o2 in o2s_global:                      # verify() checks EACH o2 independently
        probs = defaultdict(float)
        for o1 in o1s_here:
            entry = transitions2.get((s2, o1, o2), {})
            if not entry:
                errors.append(f"T2[s2={s2},o2={o2},o1={o1}] has no entries (verify() ERROR)")
                continue
            for s2p, per_a2 in entry.items():
                for a2, v in per_a2.items():
                    probs[a2] += v
        for a2 in legal_actions2[s2]:
            if abs(probs[a2] - 1.0) > 1e-9:
                errors.append(f"T2[s2={s2},a2={a2},o2={o2}] sums to {probs[a2]}, not 1.0")

if errors:
    print(f"{len(errors)} ERRORS FOUND:")
    for e in errors:
        print(" -", e)
else:
    print("All O1/O2/T1/T2 probability-sum constraints verified: PASS (matches what Game::verify() checks)")
