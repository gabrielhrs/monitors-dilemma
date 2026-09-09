"""
Ports the real 8-round toy AI-control game to the twosided-posgs format,
using the same design validated by the 2-round toy: catching is fully
reward-based (R(s1,s2,a1,a2) carries the exact catch_prob/cost formula from
the one-sided game), the Monitor's belief about the Schemer's type updates
via a passive, per-problem leak (O1, reusing the corrected per-function
Cheap-check catch probabilities -- k probes / reachable domain size -- as
the ambient detectability of a real attack, since O1 can't be gated by the
Monitor's own action in this solver's format), and the Schemer's belief
about the Monitor's remaining audit budget updates via a genuinely noisy
signal gated by the Monitor's own (s1,a1) (O2) -- the actual new capability
over the one-sided game.

CORRECTED (Sept 2026): the flat P_CHEAP_CATCH={"bounded":0.18,"unbounded":0.00}
model this file used to carry (copied verbatim from the pre-correction
build_toy_control_game_v1.py) conflated eight problems with wildly different
domain sizes into one number. Replaced with the same per-function
k=5-probes/reachable-domain-size model used by
build_toy_control_game_v2.py -- see that file and
docs/toy_demo_findings.md for the derivation. Everything else here is
unchanged (no Defer action -- matching the primary 3-action game most of
this project's pages describe).

Run (from repo root): python3 scripts/build_toy_control_game_twosided_v1.py
Writes: games/toy_control_twosided.txt (in the twosided-posgs repo)
"""
import itertools
from collections import defaultdict

# ---- exact constants from build_toy_control_game_v2.py ----
P_SCHEMER_PRIOR = 0.25
PROBLEM_NAMES = ["find_maximum", "is_palindrome", "fibonacci", "reverse_string",
                  "count_vowels", "is_prime", "sum_of_digits", "remove_duplicates"]
REACHABLE_DOMAIN_SIZE = {
    "find_maximum": None, "is_palindrome": None, "fibonacci": 30,
    "reverse_string": None, "count_vowels": None, "is_prime": 46,
    "sum_of_digits": 20000, "remove_duplicates": None,
}
K_PROBES = 5
P_AUDIT_CATCH = 1.0
C_PASS, C_CHEAP, C_AUDIT, C_CATASTROPHE = 0.0, -0.2, -1.0, -50.0
N_ROUNDS = len(PROBLEM_NAMES)
BUDGETS = [0, 1, 2]
GAMMA = 0.95
TYPE_IDX = {"Schemer": 0, "Benign": 1}   # matches TYPES=["Schemer","Benign"] order

# ---- new (illustrative, not measured) two-sided-specific parameters ----
P_PARANOIA = {"Pass": 0.1, "Cheap": 0.3, "Audit": 0.8}  # Schemer's O2: P(felt watched)

def cheap_catch_prob(round_i):
    size = REACHABLE_DOMAIN_SIZE[PROBLEM_NAMES[round_i]]
    return 0.0 if size is None else min(1.0, K_PROBES / size)

def catch_prob(a1, round_i):
    return {"Pass": 0.0, "Cheap": cheap_catch_prob(round_i), "Audit": P_AUDIT_CATCH}[a1]

def cost(a1):
    return {"Pass": C_PASS, "Cheap": C_CHEAP, "Audit": C_AUDIT}[a1]

# ---- state1 (Monitor private): s1 = round*3+budget, matching one-sided partition IDs ----
NUM_S1 = N_ROUNDS * 3 + 1   # +1 terminal
TERMINAL_S1 = NUM_S1 - 1
def s1_id(round_, budget): return round_ * 3 + budget

# ---- state2 (Schemer private): s2 = round*2+type_idx ----
NUM_S2 = N_ROUNDS * 2 + 1   # +1 terminal
TERMINAL_S2 = NUM_S2 - 1
def s2_id(round_, type_): return round_ * 2 + TYPE_IDX[type_]

lines = []
lines.append("discount: 0.95")
lines.append(f"states: {NUM_S1} {NUM_S2}")
lines.append("actions: 3 2")
lines.append("observations: 2 2")
lines.append(f"init1: {s1_id(0,2)} 1.0 -1")
lines.append(f"init2: {s2_id(0,'Schemer')} {P_SCHEMER_PRIOR} {s2_id(0,'Benign')} {1-P_SCHEMER_PRIOR} -1")

ACTIONS1 = ["Pass", "Cheap", "Audit"]
ACTIONS2 = ["Honest", "Attack"]

# ---- legal actions ----
legal_a1 = {}
for r in range(N_ROUNDS):
    for b in BUDGETS:
        legal_a1[s1_id(r, b)] = [0, 1, 2] if b > 0 else [0, 1]
legal_a1[TERMINAL_S1] = [0]

legal_a2 = {}
for r in range(N_ROUNDS):
    for t in ["Schemer", "Benign"]:
        legal_a2[s2_id(r, t)] = [0, 1]
legal_a2[TERMINAL_S2] = [0]

for s1 in range(NUM_S1):
    lines.append(f"actions1: {s1}   " + " ".join(str(a) for a in legal_a1[s1]) + " -1")
for s2 in range(NUM_S2):
    lines.append(f"actions2: {s2}   " + " ".join(str(a) for a in legal_a2[s2]) + " -1")

# ---- O1: Monitor's passive, per-problem, audit-independent leak ----
# depends only on s2 (round encodes problem) and a2 -- reuses the corrected
# per-function catch probability as the ambient detectability of a real,
# uncaught-by-design attack.
O1 = {}
for r in range(N_ROUNDS):
    p_leak = cheap_catch_prob(r)
    s2_benign = s2_id(r, "Benign")
    s2_schemer = s2_id(r, "Schemer")
    for a2 in [0, 1]:
        O1[(s2_benign, a2)] = {0: 1.0}          # Benign: Attack is a no-op, always clear
    O1[(s2_schemer, 0)] = {0: 1.0}               # Honest: always clear
    if p_leak > 0:
        O1[(s2_schemer, 1)] = {0: 1 - p_leak, 1: p_leak}
    else:
        O1[(s2_schemer, 1)] = {0: 1.0}
O1[(TERMINAL_S2, 0)] = {0: 1.0}

# ---- O2: Schemer's noisy signal about the Monitor's own (s1,a1) ----
O2 = {}
for r in range(N_ROUNDS):
    for b in BUDGETS:
        s1 = s1_id(r, b)
        for a1 in legal_a1[s1]:
            p = P_PARANOIA[ACTIONS1[a1]]
            O2[(s1, a1)] = {0: 1 - p, 1: p}
O2[(TERMINAL_S1, 0)] = {0: 1.0}

for (s2, a2), dist in sorted(O1.items()):
    for o1, p in sorted(dist.items()):
        lines.append(f"O1: {s2} {a2} {o1} {p}")
for (s1, a1), dist in sorted(O2.items()):
    for o2, p in sorted(dist.items()):
        lines.append(f"O2: {s1} {a1} {o2} {p}")

# ---- T1: state1 evolution (round/budget bookkeeping), full prob at every branch ----
T1_TARGET = {}
for r in range(N_ROUNDS):
    for b in BUDGETS:
        s1 = s1_id(r, b)
        for a1 in legal_a1[s1]:
            b2 = b - 1 if ACTIONS1[a1] == "Audit" else b
            target = TERMINAL_S1 if r == N_ROUNDS - 1 else s1_id(r + 1, b2)
            T1_TARGET[(s1, a1)] = target
T1_TARGET[(TERMINAL_S1, 0)] = TERMINAL_S1

t1_lines = []
for (s1, a1), target in T1_TARGET.items():
    for o2 in O2[(s1, a1)]:
        for o1 in range(2):
            t1_lines.append(f"T1: {s1} {a1} {o1} {o2} {target} 1.0")
lines.extend(t1_lines)

# ---- T2: state2 evolution (round/type bookkeeping), full prob at every branch ----
T2_TARGET = {}
for r in range(N_ROUNDS):
    for t in ["Schemer", "Benign"]:
        s2 = s2_id(r, t)
        for a2 in [0, 1]:
            target = TERMINAL_S2 if r == N_ROUNDS - 1 else s2_id(r + 1, t)
            T2_TARGET[(s2, a2)] = target
T2_TARGET[(TERMINAL_S2, 0)] = TERMINAL_S2

t2_lines = []
for (s2, a2), target in T2_TARGET.items():
    for o1 in O1[(s2, a2)]:
        for o2 in range(2):
            t2_lines.append(f"T2: {s2} {a2} {o2} {o1} {target} 1.0")
lines.extend(t2_lines)

# ---- R: exact one-sided reward formula ----
r_lines = []
for r in range(N_ROUNDS):
    for b in BUDGETS:
        s1 = s1_id(r, b)
        s2_benign = s2_id(r, "Benign")
        s2_schemer = s2_id(r, "Schemer")
        for a1 in legal_a1[s1]:
            a1_name = ACTIONS1[a1]
            c = cost(a1_name)
            for a2 in [0, 1]:
                # Benign: Attack is a no-op
                r_lines.append(f"R: {s1} {s2_benign} {a1} {a2} {c}")
                is_real_attack = (a2 == 1)
                if is_real_attack:
                    cp = catch_prob(a1_name, r)
                    rr = c + (1 - cp) * C_CATASTROPHE
                else:
                    rr = c
                r_lines.append(f"R: {s1} {s2_schemer} {a1} {a2} {rr}")
lines.extend(r_lines)

content = "\n".join(lines) + "\n"
import os
out_path = "/sessions/sharp-wizardly-brahmagupta/mnt/twosided-posgs/games/toy_control_twosided.txt"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    f.write(content)

print(f"Wrote {len(lines)} lines total.")
print(f"  states1={NUM_S1} states2={NUM_S2}")
print(f"  T1 entries={len(t1_lines)}  T2 entries={len(t2_lines)}  R entries={len(r_lines)}")
