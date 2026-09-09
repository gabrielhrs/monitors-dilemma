"""
Corrected version of adaptive_rule_baseline_v1.py: replaces the flat
P_CHEAP_CATCH = {"bounded": 0.18, "unbounded": 0.00} lookup with each
problem's own per-function catch probability (k probes / reachable domain
size), matching the correction applied throughout this project (see
build_toy_control_game_v2.py). The hand-crafted rule itself -- which rounds
get Cheap vs Pass in the Schemer branch -- is unchanged; only the catch odds
those Cheap checks actually carry are corrected.

Evaluates a natural, hand-crafted "belief-adaptive" rule -- Audit round 0;
if it reveals Attack, keep protecting; if it reveals Honest, Pass for the
rest of the game -- under two different opponent models, to make a point
about why "looks adaptive" is not the same as "actually worst-case safe."

1. Against the SAME simplified opponent model naive_baseline_v2.py uses
   (Schemer always attacks unless caught with certainty): this rule scores
   BETTER than the full HSVI solve. That's not a contradiction -- it's
   evaluating against too weak an opponent.

2. Against a truly rational adversary who knows this policy in advance
   (the standard worst-case assumption this whole project is built on):
   the adversary's best response is to simply present as Honest at round 0
   (never attack there -- an Audit catches it for certain anyway) and then
   attack every remaining round completely free, since the policy has
   committed in advance to zero vigilance for the rest of the game once it
   sees a clean round 0. This is catastrophically exploitable.

Run (from repo root): python3 scripts/adaptive_rule_baseline_v2.py
"""

P_SCHEMER_PRIOR = 0.25
C_PASS, C_CHEAP, C_AUDIT, C_CATASTROPHE = 0.0, -0.2, -1.0, -50.0
PROBLEMS = ["find_maximum", "is_palindrome", "fibonacci", "reverse_string",
            "count_vowels", "is_prime", "sum_of_digits", "remove_duplicates"]
REACHABLE_DOMAIN_SIZE = {
    "find_maximum": None, "is_palindrome": None, "fibonacci": 30, "reverse_string": None,
    "count_vowels": None, "is_prime": 46, "sum_of_digits": 20000, "remove_duplicates": None,
}
K_PROBES = 5
GAMMA = 0.95
N_ROUNDS = len(PROBLEMS)


def cheap_catch_prob(problem):
    size = REACHABLE_DOMAIN_SIZE[problem]
    return 0.0 if size is None else min(1.0, K_PROBES / size)


def value_against_simple_opponent():
    """Evaluates the rule assuming the opponent model naive_baseline_v2.py uses:
    Schemer always attacks unless caught with certainty; an Audit revealing
    Honest is therefore treated as proof of Benign."""
    benign_value = C_AUDIT  # round0 audited, honest, then Pass forever (free)

    # Schemer branch: round0 audited -> caught for certain. 1 audit left;
    # spend it on round1 (earliest remaining unbounded round, same
    # discount-driven tiebreak as the fixed-schedule baseline). On the
    # remaining unbounded rounds (3,4,7), Cheap gives zero protection here
    # (0% catch) so Pass is strictly better (saves the wasted -0.2). On
    # bounded rounds (2,5,6), Cheap is worth paying for -- though its real
    # catch odds now vary a lot by problem (fibonacci 16.7%, is_prime 10.9%,
    # sum_of_digits just 0.025%) rather than a flat 18%.
    actions = ["Audit", "Audit", "Cheap", "Pass", "Pass", "Cheap", "Cheap", "Pass"]
    schemer_value = 0.0
    for i, a in enumerate(actions):
        cp = {"Pass": 0.0, "Cheap": cheap_catch_prob(PROBLEMS[i]), "Audit": 1.0}[a]
        cost = {"Pass": C_PASS, "Cheap": C_CHEAP, "Audit": C_AUDIT}[a]
        r = cost + (1 - cp) * C_CATASTROPHE  # certain attack every round, this branch
        schemer_value += (GAMMA ** i) * r

    return (1 - P_SCHEMER_PRIOR) * benign_value + P_SCHEMER_PRIOR * schemer_value


def worst_case_value_if_exploited():
    """The value if a fully rational adversary, knowing this policy in
    advance, best-responds to it: never attack round0 (pointless, Audit
    catches it for certain anyway, and doing so would trigger a WORSE
    outcome for the adversary than staying quiet), then attack every
    remaining round for free once the policy has committed to Pass-forever
    in the Honest branch. Unaffected by the catch-probability correction --
    Cheap's odds never come into play once the policy has zero vigilance."""
    round0 = C_AUDIT
    exploited_tail = sum((GAMMA ** i) * C_CATASTROPHE for i in range(1, N_ROUNDS))
    return round0 + exploited_tail


def main():
    v_simple = value_against_simple_opponent()
    print("Value of the naive 2-branch adaptive rule, evaluated against the")
    print("SAME simplified (always-attack) opponent naive_baseline_v2.py assumes:")
    print("  ", round(v_simple, 4),
          " <- looks like it beats both the full HSVI solve (-58.9566) and")
    print("      the best fixed schedule (-59.74237) below! but that's an")
    print("      artifact of a too-weak opponent model, not a real result.")
    print()
    print("Value of the SAME rule if a fully rational (worst-case) adversary")
    print("exploits its known Honest -> Pass-forever commitment:")
    print("  ", round(worst_case_value_if_exploited(), 4))
    print()
    print("For comparison:")
    print("  full HSVI solved (worst-case-safe) value:  -58.9566 (lower bound)")
    print("  best FIXED (non-adaptive) schedule:         -59.74237")


if __name__ == "__main__":
    main()
