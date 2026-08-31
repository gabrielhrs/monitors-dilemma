"""
Generates a .posg file (per game_format.txt) encoding the toy AI-control
monitoring game, with a PERSISTENT hidden type (Schemer / Benign) chosen
once at the start of the episode rather than re-randomized every round --
this is what makes the Monitor's belief tracking actually matter, since a
signal observed in round i genuinely updates the risk assessment for round
i+1 too, unlike a per-round-independent formulation where the optimal
Monitor policy collapses to a simple fixed schedule computable in advance.

Reinterpretation of the measured parameters for this persistent-type framing
(documented explicitly -- this is a modeling choice motivated by the AI
Control literature's own threat model of a persistently-scheming model, not
something directly re-derived from repeated-episode data, since we only ran
one session):
  - P(Schemer) = 0.25 -- prior probability this deployment/session is
    adversarial at all (reusing the measured 2/8 "attack succeeds when
    attempted" rate as the type prior).
  - Given Schemer AND Attack chosen this round: the backdoor is real and
    genuine (no further per-round success roll -- schemer-hood already
    established capability), subject to the SAME measured domain-dependent
    catch probabilities:
      P(cheap-check(k=5) catches | bounded)   = 0.18
      P(cheap-check(k=5) catches | unbounded) = 0.00
      P(audit catches) = 1.0
  - Given Benign: Attack is available as an action but never produces a
    real threat (behaves identically to Honest), matching the measured
    false-positive rate of 0.000.

Player 1 (leader, imperfect info) = Monitor -- belief over {Schemer,Benign}
within each (round,budget) partition.
Player 2 (follower, full info) = Policy -- knows its own type, chooses
Honest/Attack each round accordingly.

A 4th Monitor action, Defer, routes the round to a weaker-but-trusted model
instead of running the untrusted Policy's output at all -- closer to how
AI Control's trusted-monitoring-with-deferral protocols actually work.
Deferring never inspects the untrusted output, so it reveals no signal about
the type (transitions to observation "none", same as an uneventful Pass) and
is unconditionally safe (no catastrophe is possible, since the untrusted
code never runs), at a fixed usefulness cost C_DEFER representing the
trusted model's lower quality. Unlike Audit, Defer isn't budget-gated.
There is no belief-update-input "suspicion score" separate from the belief
itself -- P(Schemer) is already exactly that: a continuously-updated,
real-valued suspicion estimate. Defer's own threshold behavior (whether/
where the solver chooses it as belief rises) can be read off with the same
policy-reconstruction tooling (visualization/policy_reconstruction.html)
already built for Pass/Cheap/Audit, no new machinery needed.

Run: python3 build_toy_control_game.py > games/toy_control_game.posg
Run with a different Defer cost: python3 build_toy_control_game.py --c-defer -5.0 > games/toy_control_game_defer5.posg
"""

import argparse

# ---- measured / reinterpreted parameters ----
P_SCHEMER_PRIOR = 0.25
P_CHEAP_CATCH = {"bounded": 0.18, "unbounded": 0.00}  # k=5, given a real (Schemer) attack
P_AUDIT_CATCH = 1.0  # given a real (Schemer) attack

C_PASS = 0.0
C_CHEAP = -0.2
C_AUDIT = -1.0
C_CATASTROPHE = -50.0

# C_DEFER is NOT a measured quantity -- there's no benchmarked quality gap
# between the untrusted and trusted-fallback models in this project's
# elicitation data (see docs/project_writeup.md's open threads). It has to
# be worse than C_AUDIT (-1.0) to be a non-trivial choice: since Audit
# catches with certainty (P_AUDIT_CATCH=1.0) at cost -1.0, Audit would
# simply dominate Defer whenever budget remains if Defer were cheaper.
# Default chosen to sit clearly worse than Audit but far short of
# catastrophe; override via --c-defer to sweep it.
C_DEFER_DEFAULT = -3.0

DOMAIN_TYPES = ["unbounded", "unbounded", "bounded", "unbounded",
                "unbounded", "bounded", "bounded", "unbounded"]
# find_maximum, is_palindrome, fibonacci, reverse_string,
# count_vowels, is_prime, sum_of_digits, remove_duplicates
N_ROUNDS = len(DOMAIN_TYPES)
BUDGETS = [0, 1, 2]
TYPES = ["Schemer", "Benign"]
GAMMA = 0.95

LEADER_ACTIONS = ["Pass", "Cheap", "Audit", "Defer"]  # Monitor
FOLLOWER_ACTIONS = ["Honest", "Attack"]  # Policy
OBSERVATIONS = ["none", "clear", "flag", "revealed_honest", "revealed_attack"]
OBS = {name: i for i, name in enumerate(OBSERVATIONS)}


def catch_prob(a1_name, domain):
    if a1_name == "Pass":
        return 0.0
    if a1_name == "Cheap":
        return P_CHEAP_CATCH[domain]
    if a1_name == "Audit":
        return P_AUDIT_CATCH
    raise ValueError(a1_name)  # Defer never reaches catch_prob -- handled separately


def cost(a1_name, c_defer):
    return {"Pass": C_PASS, "Cheap": C_CHEAP, "Audit": C_AUDIT, "Defer": c_defer}[a1_name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c-defer", type=float, default=C_DEFER_DEFAULT,
                         help="Usefulness cost of the Defer action (default: %(default)s)")
    args = parser.parse_args()
    c_defer = args.c_defer
    # --- states: (round, budget, type) for non-terminal; single "end" state ---
    state_names = []
    state_index = {}
    partition_of_state = []
    partition_index = {}  # (i, b) -> partition id

    for i in range(N_ROUNDS):
        for b in BUDGETS:
            partition_index[(i, b)] = len(partition_index)

    for i in range(N_ROUNDS):
        for b in BUDGETS:
            for t in TYPES:
                name = f"round{i}_budget{b}_{t}"
                state_index[(i, b, t)] = len(state_names)
                state_names.append(name)
                partition_of_state.append(partition_index[(i, b)])

    end_idx = len(state_names)
    state_index["end"] = end_idx
    state_names.append("end")
    end_partition = len(partition_index)
    partition_of_state.append(end_partition)

    n_states = len(state_names)
    n_partitions = end_partition + 1

    def next_state(i, b, t, a1_name):
        b2 = b - 1 if a1_name == "Audit" else b
        if i == N_ROUNDS - 1:
            return end_idx
        return state_index[(i + 1, b2, t)]

    transitions = []  # (s, a1, a2, o, s2, prob)
    rewards = []  # (s, a1, a2, reward)
    p2_actions_per_state = [None] * n_states
    p1_actions_per_partition = [None] * n_partitions

    for i in range(N_ROUNDS):
        domain = DOMAIN_TYPES[i]
        for b in BUDGETS:
            # Defer (3) is never budget-gated, unlike Audit (2).
            allowed_a1 = [0, 1, 2, 3] if b > 0 else [0, 1, 3]
            p1_actions_per_partition[partition_index[(i, b)]] = allowed_a1

            for t in TYPES:
                s = state_index[(i, b, t)]
                p2_actions_per_state[s] = [0, 1]

                for a1_idx in allowed_a1:
                    a1_name = LEADER_ACTIONS[a1_idx]
                    s2 = next_state(i, b, t, a1_name)
                    c = cost(a1_name, c_defer)

                    if a1_name == "Defer":
                        # Bypasses the untrusted output entirely: no catastrophe
                        # is possible and no signal about the type is revealed,
                        # regardless of the follower's action or true type.
                        for a2_idx, a2_name in enumerate(FOLLOWER_ACTIONS):
                            transitions.append((s, a1_idx, a2_idx, OBS["none"], s2, 1.0))
                            rewards.append((s, a1_idx, a2_idx, c))
                        continue

                    for a2_idx, a2_name in enumerate(FOLLOWER_ACTIONS):
                        real_attack = (t == "Schemer" and a2_name == "Attack")

                        if not real_attack:
                            r = c
                            if a1_name == "Pass":
                                transitions.append((s, a1_idx, a2_idx, OBS["none"], s2, 1.0))
                            elif a1_name == "Cheap":
                                transitions.append((s, a1_idx, a2_idx, OBS["clear"], s2, 1.0))
                            else:
                                transitions.append((s, a1_idx, a2_idx, OBS["revealed_honest"], s2, 1.0))
                        else:
                            cp = catch_prob(a1_name, domain)
                            r = c + (1 - cp) * C_CATASTROPHE
                            if a1_name == "Pass":
                                transitions.append((s, a1_idx, a2_idx, OBS["none"], s2, 1.0))
                            elif a1_name == "Cheap":
                                if cp > 0:
                                    transitions.append((s, a1_idx, a2_idx, OBS["flag"], s2, cp))
                                transitions.append((s, a1_idx, a2_idx, OBS["clear"], s2, 1.0 - cp))
                            else:
                                transitions.append((s, a1_idx, a2_idx, OBS["revealed_attack"], s2, cp))
                                if cp < 1.0:
                                    transitions.append((s, a1_idx, a2_idx, OBS["revealed_honest"], s2, 1.0 - cp))

                        rewards.append((s, a1_idx, a2_idx, r))

    # --- absorbing end state ---
    p1_actions_per_partition[end_partition] = list(range(len(LEADER_ACTIONS)))
    p2_actions_per_state[end_idx] = [0, 1]
    for a1_idx in range(len(LEADER_ACTIONS)):
        for a2_idx in range(2):
            transitions.append((end_idx, a1_idx, a2_idx, OBS["none"], end_idx, 1.0))
            rewards.append((end_idx, a1_idx, a2_idx, 0.0))

    # --- initial belief: round 0, budget 2, P(Schemer)=prior ---
    initial_partition = partition_index[(0, 2)]
    # order within partition must match the order states were added: Schemer then Benign
    initial_belief = [P_SCHEMER_PRIOR, 1.0 - P_SCHEMER_PRIOR]

    # ================= emit file =================
    lines = []
    lines.append(f"{n_states} {n_partitions} {len(LEADER_ACTIONS)} {len(FOLLOWER_ACTIONS)} "
                  f"{len(OBSERVATIONS)} {len(transitions)} {len(rewards)} {GAMMA:.4f}")
    for idx, name in enumerate(state_names):
        lines.append(f"{name} {partition_of_state[idx]}")
    for a in LEADER_ACTIONS:
        lines.append(a)
    for a in FOLLOWER_ACTIONS:
        lines.append(a)
    for o in OBSERVATIONS:
        lines.append(o)
    for allowed in p2_actions_per_state:
        lines.append(" ".join(str(x) for x in allowed))
    for allowed in p1_actions_per_partition:
        lines.append(" ".join(str(x) for x in allowed))
    for (s, a1, a2, o, s2, prob) in transitions:
        lines.append(f"{s} {a1} {a2} {o} {s2} {prob:.6f}")
    for (s, a1, a2, r) in rewards:
        lines.append(f"{s} {a1} {a2} {r:.6f}")
    lines.append(f"{initial_partition} " + " ".join(f"{p:.6f}" for p in initial_belief))

    print("\n".join(lines))


if __name__ == "__main__":
    main()
