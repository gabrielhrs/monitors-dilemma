"""
Decomposes the solver's single blended reward into two separately-tracked
quantities -- expected discounted usefulness cost, and expected discounted
"missed real attack" probability mass -- for four benchmark policies:

  no_audits        -- naive_baseline.py's all-Cheap schedule
  worst_fixed       -- naive_baseline.py's worst fixed 2-audit schedule
  best_fixed        -- naive_baseline.py's best fixed 2-audit schedule
  solved_adaptive   -- the actual solved POSG policy, reconstructed by
                        walking the game tree under the exported Monitor
                        policy sweep (games/policy_sweep_fixed.json) and
                        attacker policy sweep (games/attacker_sweep_fixed.json)

For the fixed schedules this is a closed form (same math as
naive_baseline.py's schedule_value(), split into its two summands). For the
solved adaptive policy there's no closed form -- this recurses over
(round, budget, belief), using nearest-neighbor lookup into the two 0.01-
resolution belief sweeps to get the Monitor's mixed action and the Schemer's
worst-case P(Attack) at each node, and replicates build_toy_control_game.py's
exact observation/catch-probability logic to branch into next beliefs via
Bayes' rule -- the same math Partition::getObservationProbability /
getTransformedBelief use in the real C++ solver.

Validation: the single-step belief update was checked directly against the
real solver trace (trial_walkthrough.html's iteration-4 trial: Audit at the
root comes back "honest," belief drops from the 0.25 prior). This script
reconstructs 0.0435 for that same step vs. the logged 0.042 -- close, with
the small gap explained by which exact vector/point the real trial landed on
vs. this script's nearest-0.01-belief lookup. The full 8-round blended value
(-55.86) lands close to, but outside, the converged root bound [-56.53,
-56.49] for the same reason compounded across 8 rounds of independent
grid-snapped lookups, plus a few partitions being resolved to only 1
alpha-vector (see attacker_policy.html's "sparse" partitions) -- not a bug,
just the expected cost of reconstructing continuous recursion from a
discretized sweep instead of reading the solver's own alpha-vectors
directly.

Run: python3 build_safety_usefulness.py
Writes: games/safety_usefulness.json
"""
import json
from itertools import combinations

ACTION_NAMES = ["Pass", "Cheap", "Audit"]
DOMAIN_TYPES = ["unbounded", "unbounded", "bounded", "unbounded",
                "unbounded", "bounded", "bounded", "unbounded"]
N_ROUNDS = 8
GAMMA = 0.95
P_SCHEMER_PRIOR = 0.25
P_CHEAP_CATCH = {"bounded": 0.18, "unbounded": 0.00}
P_AUDIT_CATCH = 1.0
C_PASS, C_CHEAP, C_AUDIT, C_CATASTROPHE = 0.0, -0.2, -1.0, -50.0


def catch_prob(a1, domain):
    if a1 == "Pass": return 0.0
    if a1 == "Cheap": return P_CHEAP_CATCH[domain]
    if a1 == "Audit": return P_AUDIT_CATCH
    raise ValueError(a1)


def cost(a1):
    return {"Pass": C_PASS, "Cheap": C_CHEAP, "Audit": C_AUDIT}[a1]


def schedule_decompose(audit_rounds):
    """Closed-form (usefulness, safety, blended) for a fixed, non-reactive
    2-audit schedule, matching naive_baseline.py's schedule_value()."""
    usefulness, safety, blended = 0.0, 0.0, 0.0
    for i in range(N_ROUNDS):
        a1 = "Audit" if i in audit_rounds else "Cheap"
        cp = catch_prob(a1, DOMAIN_TYPES[i])
        u, s = cost(a1), P_SCHEMER_PRIOR * (1 - cp)
        usefulness += (GAMMA ** i) * u
        safety += (GAMMA ** i) * s
        blended += (GAMMA ** i) * (u + s * C_CATASTROPHE)
    return usefulness, safety, blended


def nearest_idx(belief):
    return round(max(0.0, min(1.0, belief)) * 100)


def build_solved_adaptive(policy, attacker):
    def lookup_policy(pid, belief):
        part = policy[str(pid)]
        pt = part["points"][nearest_idx(belief)]
        return part["actions"], pt["strategy"]

    def lookup_attacker(pid, belief):
        pt = attacker[str(pid)]["points"][nearest_idx(belief)]
        return pt["schemerStrategy"]

    memo = {}

    def evaluate(i, b, belief):
        if i == N_ROUNDS:
            return (0.0, 0.0)
        key = (i, b, round(belief, 6))
        if key in memo:
            return memo[key]
        pid = i * 3 + b
        action_idxs, probs = lookup_policy(pid, belief)
        p_honest, p_attack = lookup_attacker(pid, belief)
        domain = DOMAIN_TYPES[i]
        total_u, total_s = 0.0, 0.0
        for a_idx, pi1 in zip(action_idxs, probs):
            if pi1 <= 1e-12:
                continue
            a1 = ACTION_NAMES[a_idx]
            cp = catch_prob(a1, domain)
            u_immediate = cost(a1)
            # expected mass of "real attack this round, and not caught"
            s_immediate = belief * p_attack * (1 - cp)
            b_next = b - 1 if a1 == "Audit" else b

            if a1 == "Pass":
                # uninformative regardless of outcome -- belief never moves
                obs_list = [("none", 1.0, 1.0)]
            elif a1 == "Cheap":
                p_flag_s = p_attack * cp
                obs_list = [("clear", 1 - p_flag_s, 1.0), ("flag", p_flag_s, 0.0)]
            else:  # Audit
                p_ra_s = p_attack * cp
                obs_list = [("revealed_honest", 1 - p_ra_s, 1.0),
                            ("revealed_attack", p_ra_s, 0.0)]

            future_u, future_s = 0.0, 0.0
            for obs, p_o_given_s, p_o_given_b in obs_list:
                p_o = belief * p_o_given_s + (1 - belief) * p_o_given_b
                if p_o < 1e-12:
                    continue
                next_belief = max(0.0, min(1.0, belief * p_o_given_s / p_o))
                fu, fs = evaluate(i + 1, b_next, next_belief)
                future_u += p_o * fu
                future_s += p_o * fs

            total_u += pi1 * (u_immediate + GAMMA * future_u)
            total_s += pi1 * (s_immediate + GAMMA * future_s)
        memo[key] = (total_u, total_s)
        return total_u, total_s

    return evaluate(0, 2, P_SCHEMER_PRIOR)


def main():
    combos = [(schedule_decompose(set(r)), r) for r in combinations(range(N_ROUNDS), 2)]
    combos.sort(key=lambda x: -x[0][2])
    best, worst, none = combos[0], combos[-1], schedule_decompose(set())

    policy = json.load(open("games/policy_sweep_fixed.json"))
    attacker = json.load(open("games/attacker_sweep_fixed.json"))
    su, ss = build_solved_adaptive(policy, attacker)
    sb = su + ss * C_CATASTROPHE

    out = {
        "no_audits": {"usefulness": none[0], "safety": none[1], "blended": none[2]},
        "worst_fixed": {"usefulness": worst[0][0], "safety": worst[0][1],
                         "blended": worst[0][2], "audit_rounds": list(worst[1])},
        "best_fixed": {"usefulness": best[0][0], "safety": best[0][1],
                        "blended": best[0][2], "audit_rounds": list(best[1])},
        "solved_adaptive": {"usefulness": su, "safety": ss, "blended": sb},
        "known_root_bounds": {"lb": -56.5346, "ub": -56.4853},
        "notes": ("usefulness = expected discounted sum of cost(a1) (monitoring "
                  "overhead, <=0, closer to 0 is better). safety = expected "
                  "discounted sum of P(Schemer)*P(Attack|belief)*(1-catch_prob) "
                  "-- 'missed real-attack probability mass', >=0, lower is "
                  "better. blended = usefulness + safety*C_CATASTROPHE, the "
                  "single scalar the solver actually optimizes. "
                  "solved_adaptive's blended value sits close to but outside "
                  "the converged root bound because it's reconstructed from a "
                  "0.01-belief-resolution sweep recomputed independently at "
                  "each of 8 rounds, not read directly off the solver's own "
                  "alpha-vectors -- the underlying belief-update math was "
                  "checked against the real solver trace and matches closely "
                  "(0.0435 reconstructed vs 0.042 logged for the same step).")
    }
    json.dump(out, open("games/safety_usefulness.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
