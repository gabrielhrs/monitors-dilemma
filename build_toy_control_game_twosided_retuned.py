"""
Retuned full 8-round two-sided game: adds the same C_CATCH_BONUS mechanism
that broke the toy's degenerate indifference under Audit
(catch_prob(Audit)=1.0 means, under the original formula, catching a real
attack costs the Monitor exactly the audit fee either way -- the Schemer
was indifferent, so nothing needed hedging against, and the noisy-vs-full-
info comparison came back null both in the toy and in this game's own
unretuned full-scale run).

reward = cost(a1) + is_real_attack * [catch_prob(a1)*C_CATCH_BONUS +
(1-catch_prob(a1))*C_CATASTROPHE] -- identical form to the toy's fix, now
applied at this game's own scale (C_CATASTROPHE=-50 here vs the toy's -10,
so C_CATCH_BONUS is scaled to the same bonus/catastrophe ratio the toy
used: 2.0/10.0 = 0.2, i.e. C_CATCH_BONUS = 0.2 * 50 = 10.0).

Writes BOTH the noisy and full-info O2 variants (only O2/T1 differ between
them, everything else -- including this new reward formula -- is identical),
so the same before/after comparison the toy used can be run here too.

Run: python3 build_toy_control_game_twosided_retuned.py
Writes: games/toy_control_twosided_retuned.txt (noisy O2)
        games/toy_control_twosided_retuned_fullinfo.txt (full-info O2)
"""
import os

# ---- exact constants from build_toy_control_game.py, except the new bonus ----
P_SCHEMER_PRIOR = 0.25
P_CHEAP_CATCH = {"bounded": 0.18, "unbounded": 0.00}
P_AUDIT_CATCH = 1.0
C_PASS, C_CHEAP, C_AUDIT, C_CATASTROPHE = 0.0, -0.2, -1.0, -50.0
C_CATCH_BONUS = 10.0   # 0.2 * |C_CATASTROPHE|, same ratio the toy used
DOMAIN_TYPES = ["unbounded", "unbounded", "bounded", "unbounded",
                "unbounded", "bounded", "bounded", "unbounded"]
N_ROUNDS = len(DOMAIN_TYPES)
BUDGETS = [0, 1, 2]
GAMMA = 0.95
TYPE_IDX = {"Schemer": 0, "Benign": 1}
P_PARANOIA = {"Pass": 0.1, "Cheap": 0.3, "Audit": 0.8}

def catch_prob(a1, domain):
    return {"Pass": 0.0, "Cheap": P_CHEAP_CATCH[domain], "Audit": P_AUDIT_CATCH}[a1]

def cost(a1):
    return {"Pass": C_PASS, "Cheap": C_CHEAP, "Audit": C_AUDIT}[a1]

NUM_S1 = N_ROUNDS * 3 + 1
TERMINAL_S1 = NUM_S1 - 1
def s1_id(round_, budget): return round_ * 3 + budget

NUM_S2 = N_ROUNDS * 2 + 1
TERMINAL_S2 = NUM_S2 - 1
def s2_id(round_, type_): return round_ * 2 + TYPE_IDX[type_]

ACTIONS1 = ["Pass", "Cheap", "Audit"]
ACTIONS2 = ["Honest", "Attack"]

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

O1 = {}
for r in range(N_ROUNDS):
    domain = DOMAIN_TYPES[r]
    p_leak = P_CHEAP_CATCH[domain]
    s2_benign = s2_id(r, "Benign")
    s2_schemer = s2_id(r, "Schemer")
    for a2 in [0, 1]:
        O1[(s2_benign, a2)] = {0: 1.0}
    O1[(s2_schemer, 0)] = {0: 1.0}
    if p_leak > 0:
        O1[(s2_schemer, 1)] = {0: 1 - p_leak, 1: p_leak}
    else:
        O1[(s2_schemer, 1)] = {0: 1.0}
O1[(TERMINAL_S2, 0)] = {0: 1.0}

T2_TARGET = {}
for r in range(N_ROUNDS):
    for t in ["Schemer", "Benign"]:
        s2 = s2_id(r, t)
        for a2 in [0, 1]:
            target = TERMINAL_S2 if r == N_ROUNDS - 1 else s2_id(r + 1, t)
            T2_TARGET[(s2, a2)] = target
T2_TARGET[(TERMINAL_S2, 0)] = TERMINAL_S2

T1_TARGET = {}
for r in range(N_ROUNDS):
    for b in BUDGETS:
        s1 = s1_id(r, b)
        for a1 in legal_a1[s1]:
            b2 = b - 1 if ACTIONS1[a1] == "Audit" else b
            target = TERMINAL_S1 if r == N_ROUNDS - 1 else s1_id(r + 1, b2)
            T1_TARGET[(s1, a1)] = target
T1_TARGET[(TERMINAL_S1, 0)] = TERMINAL_S1

def build(fullinfo, out_path):
    lines = []
    lines.append("discount: 0.95")
    lines.append(f"states: {NUM_S1} {NUM_S2}")
    lines.append("actions: 3 2")
    lines.append("observations: 2 2")
    lines.append(f"init1: {s1_id(0,2)} 1.0 -1")
    lines.append(f"init2: {s2_id(0,'Schemer')} {P_SCHEMER_PRIOR} {s2_id(0,'Benign')} {1-P_SCHEMER_PRIOR} -1")
    for s1 in range(NUM_S1):
        lines.append(f"actions1: {s1}   " + " ".join(str(a) for a in legal_a1[s1]) + " -1")
    for s2 in range(NUM_S2):
        lines.append(f"actions2: {s2}   " + " ".join(str(a) for a in legal_a2[s2]) + " -1")

    O2 = {}
    for r in range(N_ROUNDS):
        for b in BUDGETS:
            s1 = s1_id(r, b)
            for a1 in legal_a1[s1]:
                if fullinfo:
                    o2 = 1 if ACTIONS1[a1] == "Audit" else 0
                    O2[(s1, a1)] = {o2: 1.0}
                else:
                    p = P_PARANOIA[ACTIONS1[a1]]
                    O2[(s1, a1)] = {0: 1 - p, 1: p}
    O2[(TERMINAL_S1, 0)] = {0: 1.0}

    for (s2, a2), dist in sorted(O1.items()):
        for o1, p in sorted(dist.items()):
            lines.append(f"O1: {s2} {a2} {o1} {p}")
    for (s1, a1), dist in sorted(O2.items()):
        for o2, p in sorted(dist.items()):
            lines.append(f"O2: {s1} {a1} {o2} {p}")

    t1_lines = []
    for (s1, a1), target in T1_TARGET.items():
        for o2 in O2[(s1, a1)]:
            for o1 in range(2):
                t1_lines.append(f"T1: {s1} {a1} {o1} {o2} {target} 1.0")
    lines.extend(t1_lines)

    t2_lines = []
    for (s2, a2), target in T2_TARGET.items():
        for o1 in O1[(s2, a2)]:
            for o2 in range(2):
                t2_lines.append(f"T2: {s2} {a2} {o2} {o1} {target} 1.0")
    lines.extend(t2_lines)

    r_lines = []
    for r in range(N_ROUNDS):
        domain = DOMAIN_TYPES[r]
        for b in BUDGETS:
            s1 = s1_id(r, b)
            s2_benign = s2_id(r, "Benign")
            s2_schemer = s2_id(r, "Schemer")
            for a1 in legal_a1[s1]:
                a1_name = ACTIONS1[a1]
                c = cost(a1_name)
                for a2 in [0, 1]:
                    r_lines.append(f"R: {s1} {s2_benign} {a1} {a2} {c}")
                    is_real_attack = (a2 == 1)
                    if is_real_attack:
                        cp = catch_prob(a1_name, domain)
                        rr = c + cp * C_CATCH_BONUS + (1 - cp) * C_CATASTROPHE
                    else:
                        rr = c
                    r_lines.append(f"R: {s1} {s2_schemer} {a1} {a2} {rr}")
    lines.extend(r_lines)

    content = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(content)
    print(f"Wrote {len(lines)} lines to {out_path}  (O2 entries={sum(len(d) for d in O2.values())}, T1={len(t1_lines)}, T2={len(t2_lines)}, R={len(r_lines)})")

BASE = "/sessions/sharp-wizardly-brahmagupta/mnt/twosided-posgs/games"
build(fullinfo=False, out_path=f"{BASE}/toy_control_twosided_retuned.txt")
build(fullinfo=True, out_path=f"{BASE}/toy_control_twosided_retuned_fullinfo.txt")
