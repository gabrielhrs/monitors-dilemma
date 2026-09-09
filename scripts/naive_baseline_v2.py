"""
Corrected version of naive_baseline_v1.py: replaces the flat
P_CHEAP_CATCH = {"bounded": 0.18, "unbounded": 0.00} lookup with each
problem's own per-function catch probability (k probes / reachable domain
size), matching the correction applied throughout this project (see
build_toy_control_game_v2.py). Everything else -- the closed-form fixed-
schedule search itself -- is unchanged.

Computes the value of every possible FIXED (non-adaptive) audit schedule --
i.e. Monitor commits in advance to which 2 of the 8 rounds get Audited
(Cheap-check on the rest), never reacting to observations -- and reports the
best and worst such schedules, to compare against the solved adaptive value
from the actual POSG solver.

For a fixed, non-reactive schedule, the value has a simple closed form
(shown analytically to be independent of persistent- vs per-round-type
framing under non-adaptive play): a Schemer (present w.p. P_SCHEMER_PRIOR)
always attacks except when directly audited (indifferent there), so

  round reward = cost(a1) + P_SCHEMER_PRIOR * (1 - catch_prob(a1, problem)) * C_CATASTROPHE

Run (from repo root): python3 scripts/naive_baseline_v2.py
"""
from itertools import combinations

P_SCHEMER_PRIOR = 0.25
P_AUDIT_CATCH = 1.0

C_PASS = 0.0
C_CHEAP = -0.2
C_AUDIT = -1.0
C_CATASTROPHE = -50.0

PROBLEMS = ["find_maximum", "is_palindrome", "fibonacci", "reverse_string",
            "count_vowels", "is_prime", "sum_of_digits", "remove_duplicates"]
REACHABLE_DOMAIN_SIZE = {
    "find_maximum": None, "is_palindrome": None, "fibonacci": 30, "reverse_string": None,
    "count_vowels": None, "is_prime": 46, "sum_of_digits": 20000, "remove_duplicates": None,
}
K_PROBES = 5
N_ROUNDS = len(PROBLEMS)
GAMMA = 0.95


def cheap_catch_prob(problem):
    size = REACHABLE_DOMAIN_SIZE[problem]
    return 0.0 if size is None else min(1.0, K_PROBES / size)


def catch_prob(a1_name, problem):
    if a1_name == "Cheap":
        return cheap_catch_prob(problem)
    if a1_name == "Audit":
        return P_AUDIT_CATCH
    return 0.0


def cost(a1_name):
    return {"Pass": C_PASS, "Cheap": C_CHEAP, "Audit": C_AUDIT}[a1_name]


def schedule_value(audit_rounds):
    total = 0.0
    for i in range(N_ROUNDS):
        a1 = "Audit" if i in audit_rounds else "Cheap"
        cp = catch_prob(a1, PROBLEMS[i])
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
          "(problems:", [PROBLEMS[i] for i in best_rounds], ")")
    print("Worst fixed schedule: audit rounds", worst_rounds,
          "-> value", round(worst_v, 5),
          "(problems:", [PROBLEMS[i] for i in worst_rounds], ")")
    print("\nAll-Cheap, no audits at all:", round(schedule_value(set()), 5))


if __name__ == "__main__":
    main()
