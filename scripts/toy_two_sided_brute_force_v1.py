"""
Independent ground-truth computation for the toy two-sided game's minimax
value, completely separate from the C++ solver -- enumerates every pure
strategy for both players (small enough: 8x8), builds the exact zero-sum
payoff matrix by simulating every trajectory exactly against the game's own
transition/observation/reward definitions, and solves the resulting matrix
game via LP duality (minimax theorem) for the true value.
"""
import itertools
import numpy as np
from scipy.optimize import linprog

GAMMA = 0.95
P_SCHEMER_PRIOR = 0.25
C_PASS, C_AUDIT, C_CATASTROPHE = 0.0, -1.0, -10.0
P_LEAK = 0.3
P_WATCHED_IF_AUDIT = 0.8
P_PARANOIA_IF_PASS = 0.1

def catch_prob(a1): return 1.0 if a1 == 1 else 0.0
def cost(a1): return C_AUDIT if a1 == 1 else C_PASS
def reward(is_real_attack, a1):
    r = cost(a1)
    if is_real_attack:
        r += (1 - catch_prob(a1)) * C_CATASTROPHE
    return r

def O1_dist(is_schemer, a2):
    # Monitor's passive signal, from (schemer-type-truth, a2)
    if not is_schemer or a2 == 0:
        return {0: 1.0}
    return {0: 1 - P_LEAK, 1: P_LEAK}

def O2_dist(a1):
    # Schemer's noisy signal about the Monitor's realized action
    if a1 == 0:
        return {0: 1 - P_PARANOIA_IF_PASS, 1: P_PARANOIA_IF_PASS}
    return {0: 1 - P_WATCHED_IF_AUDIT, 1: P_WATCHED_IF_AUDIT}

# Monitor pure strategy: (a1_round0, a1_round1_if_s1eq1_o1eq0, a1_round1_if_s1eq1_o1eq1)
# (s1==2 branch is forced to Pass, no real choice)
monitor_strats = list(itertools.product([0, 1], repeat=3))

# Schemer pure strategy (Schemer-type branch only; Benign's action is a no-op, fixed to Honest):
# (a2_round0, a2_round1_if_o2eq0, a2_round1_if_o2eq1)
schemer_strats = list(itertools.product([0, 1], repeat=3))

def trajectory_value(pi_m, pi_s, is_schemer):
    a1_r0 = pi_m[0]
    a2_r0 = pi_s[0] if is_schemer else 0
    r0 = reward(is_schemer and a2_r0 == 1, a1_r0)

    s1_r1 = 1 if a1_r0 == 0 else 2   # Pass keeps budget, Audit spends it

    total = 0.0
    for o1, p_o1 in O1_dist(is_schemer, a2_r0).items():
        for o2, p_o2 in O2_dist(a1_r0).items():
            if s1_r1 == 1:
                a1_r1 = pi_m[1] if o1 == 0 else pi_m[2]
            else:
                a1_r1 = 0  # forced Pass, no budget left
            a2_r1 = (pi_s[1] if o2 == 0 else pi_s[2]) if is_schemer else 0
            r1 = reward(is_schemer and a2_r1 == 1, a1_r1)
            total += p_o1 * p_o2 * (r0 + GAMMA * r1)
    return total

# Build payoff matrix: rows = Monitor (maximizer) strategies, cols = Schemer (minimizer) strategies
M = np.zeros((len(monitor_strats), len(schemer_strats)))
for i, pi_m in enumerate(monitor_strats):
    for j, pi_s in enumerate(schemer_strats):
        v_benign = trajectory_value(pi_m, pi_s, is_schemer=False)
        v_schemer = trajectory_value(pi_m, pi_s, is_schemer=True)
        M[i, j] = (1 - P_SCHEMER_PRIOR) * v_benign + P_SCHEMER_PRIOR * v_schemer

# Solve as a zero-sum matrix game: Monitor (max) picks a mixed strategy over rows
# maximizing the worst-case (min over columns) expected payoff. Standard LP:
#   maximize v  s.t.  sum_i x_i * M[i,j] >= v  for all j,  sum x_i = 1,  x_i >= 0
n_rows, n_cols = M.shape
# variables: x_1..x_n, v  -> minimize -v
c = np.zeros(n_rows + 1); c[-1] = -1.0
A_ub = np.zeros((n_cols, n_rows + 1))
for j in range(n_cols):
    A_ub[j, :n_rows] = -M[:, j]
    A_ub[j, -1] = 1.0
b_ub = np.zeros(n_cols)
A_eq = np.zeros((1, n_rows + 1)); A_eq[0, :n_rows] = 1.0
b_eq = [1.0]
bounds = [(0, None)] * n_rows + [(None, None)]

res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
value = -res.fun
x = res.x[:n_rows]

print(f"Matrix game size: {n_rows} x {n_cols}")
print(f"Independent brute-force minimax value: {value:.6f}")
print(f"Monitor's optimal mixed strategy (nonzero support):")
for i, p in enumerate(x):
    if p > 1e-6:
        print(f"  strat {monitor_strats[i]} (a1_r0, a1_r1|o1=0, a1_r1|o1=1): weight {p:.4f}")

# sanity: also solve from Schemer's side (min player) and confirm matching value
c2 = np.zeros(n_cols + 1); c2[-1] = 1.0
A_ub2 = np.zeros((n_rows, n_cols + 1))
for i in range(n_rows):
    A_ub2[i, :n_cols] = M[i, :]
    A_ub2[i, -1] = -1.0
b_ub2 = np.zeros(n_rows)
A_eq2 = np.zeros((1, n_cols + 1)); A_eq2[0, :n_cols] = 1.0
b_eq2 = [1.0]
bounds2 = [(0, None)] * n_cols + [(None, None)]
res2 = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2, bounds=bounds2, method="highs")
print(f"Cross-check from Schemer's (min) side: value = {res2.fun:.6f} (should match)")
