"""
Evaluates a natural, hand-crafted "belief-adaptive" rule -- Audit round 0;
if it reveals Attack, keep protecting; if it reveals Honest, Pass for the
rest of the game -- under two different opponent models, to make a point
about why "looks adaptive" is not the same as "actually worst-case safe."

1. Against the SAME simplified opponent model naive_baseline.py uses
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

Run: python3 adaptive_rule_baseline.py
"""

P_SCHEMER_PRIOR = 0.25
P_CHEAP_CATCH = {"bounded": 0.18, "unbounded": 0.00}
C_PASS, C_CHEAP, C_AUDIT, C_CATASTROPHE = 0.0, -0.2, -1.0, -50.0
DOMAIN_TYPES = ["unbounded", "unbounded", "bounded", "unbounded",
                "unbounded", "bounded", "bounded", "unbounded"]
GAMMA = 0.95
N_ROUNDS = len(DOMAIN_TYPES)


def value_against_simple_opponent():
    """Evaluates the rule assuming the opponent model naive_baseline.py uses:
    Schemer always attacks unless caught with certainty; an Audit revealing
    Honest is therefore treated as proof of Benign."""
    benign_value = C_AUDIT  # round0 audited, honest, then Pass forever (free)

    # Schemer branch: round0 audited -> caught for certain. 1 audit left;
    # spend it on round1 (earliest remaining unbounded round, same
    # discount-driven tiebreak as the fixed-schedule baseline). On the
    # remaining unbounded rounds (3,4,7), Cheap gives zero protection here
    # (0% catch) so Pass is strictly better (saves the wasted -0.2). On
    # bounded rounds (2,5,6), Cheap's 18% catch chance is worth paying for.
    actions = ["Audit", "Audit", "Cheap", "Pass", "Pass", "Cheap", "Cheap", "Pass"]
    schemer_value = 0.0
    for i, a in enumerate(actions):
        d = DOMAIN_TYPES[i]
        cp = {"Pass": 0.0, "Cheap": P_CHEAP_CATCH[d], "Audit": 1.0}[a]
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
    in the Honest branch."""
    round0 = C_AUDIT
    exploited_tail = sum((GAMMA ** i) * C_CATASTROPHE for i in range(1, N_ROUNDS))
    return round0 + exploited_tail


def main():
    print("Value of the naive 2-branch adaptive rule, evaluated against the")
    print("SAME simplified (always-attack) opponent naive_baseline.py assumes:")
    print("  ", round(value_against_simple_opponent(), 4),
          " <- looks like it beats the full HSVI solve (-56.8119)! but that's")
    print("      an artifact of a too-weak opponent model, not a real result.")
    print()
    print("Value of the SAME rule if a fully rational (worst-case) adversary")
    print("exploits its known Honest -> Pass-forever commitment:")
    print("  ", round(worst_case_value_if_exploited(), 4))
    print()
    print("For comparison:")
    print("  full HSVI solved (worst-case-safe) value:  -56.8119")
    print("  best FIXED (non-adaptive) schedule:         -57.25062")


if __name__ == "__main__":
    main()
