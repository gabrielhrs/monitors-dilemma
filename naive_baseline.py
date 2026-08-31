"""
Computes the value of every possible FIXED (non-adaptive) audit schedule --
i.e. Monitor commits in advance to which 2 of the 8 rounds get Audited
(Cheap-check on the rest), never reacting to observations -- and reports the
best and worst such schedules, to compare against the solved adaptive value
from the actual POSG solver.

For a fixed, non-reactive schedule, the value has a simple closed form
(shown analytically to be independent of persistent- vs per-round-type
framing under non-adaptive play): a Schemer (present w.p. P_SCHEMER_PRIOR)
always attacks except when directly audited (indifferent there), so

  round reward = cost(a1) + P_SCHEMER_PRIOR * (1 - catch_prob(a1, domain)) * C_CATASTROPHE

Run: python3 naive_baseline.py
"""
from itertools import combinations

P_SCHEMER_PRIOR = 0.25
P_CHEAP_CATCH = {"bounded": 0.18, "unbounded": 0.00}
P_AUDIT_CATCH = 1.0

C_PASS = 0.0
C_CHEAP = -0.2
C_AUDIT = -1.0
C_CATASTROPHE = -50.0

DOMAIN_TYPES = ["unbounded", "unbounded", "bounded", "unbounded",
                "unbounded", "bounded", "bounded", "unbounded"]
N_ROUNDS = len(DOMAIN_TYPES)
GAMMA = 0.95


def catch_prob(a1_name, domain):
    if a1_name == "Cheap":
        return P_CHEAP_CATCH[domain]
    if a1_name == "Audit":
        return P_AUDIT_CATCH
    return 0.0


def cost(a1_name):
    return {"Pass": C_PASS, "Cheap": C_CHEAP, "Audit": C_AUDIT}[a1_name]


def schedule_value(audit_rounds):
    total = 0.0
    for i in range(N_ROUNDS):
        a1 = "Audit" if i in audit_rounds else "Cheap"
        cp = catch_prob(a1, DOMAIN_TYPES[i])
        r = cost(a1) + P_SCHEMER_PRIOR * (1 - cp) * C_CATASTROPHE
        total += (GAMMA ** i) * r
    return total


def main():
    results = []
    for audit_rounds in combinations(range(N_ROUNDS), 2):
        v = schedule_value(set(audit_rounds))
        results.append((v, audit_rounds))
    results.sort(reverse=True)  # higher (less negative) is better for Monitor

    best_v, best_rounds = results[0]
    worst_v, worst_rounds = results[-1]

    print("Best fixed schedule:  audit rounds", best_rounds,
          "-> value", round(best_v, 5),
          "(domains:", [DOMAIN_TYPES[i] for i in best_rounds], ")")
    print("Worst fixed schedule: audit rounds", worst_rounds,
          "-> value", round(worst_v, 5),
          "(domains:", [DOMAIN_TYPES[i] for i in worst_rounds], ")")
    print("\nAll-Cheap, no audits at all:", round(schedule_value(set()), 5))


if __name__ == "__main__":
    main()
