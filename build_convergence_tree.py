"""
Merges the raw HSVI convergence trace (games/hsvi_trace.jsonl, produced by the
instrumented solver) into a single tree structure for the D3 viewer:
each distinct (depth, partition, belief) node visited across all trials is
merged into one node, carrying a history of every peek/backup bound update
it received, in iteration order. Nodes are linked by the (action, observation)
edge that was actually taken between them in at least one trial.

Depth attribution for "bound" records relies on the exact structure of
explore() in hsvi.cpp: a branched visit at depth D is immediately followed by
its own 2 peek bounds (lb, ub), then the ENTIRE recursive subtree for depth
D+1, then finally its own 2 backup bounds. A leaf visit (branched=false) is
immediately followed by its own 2 backup bounds, and then -- with no further
visit records -- the backup bounds for depth D-1, D-2, ..., 0 follow in that
order, two records each. This lets us track depth exactly with a simple
pointer instead of a naive call-stack simulation (which mis-attributes the
unwind, since the trace has no explicit "return" marker).

Run: python3 build_convergence_tree.py [trace_file] > games/convergence_tree.json
"""
import json
import sys

PROBLEMS = ["find_maximum", "is_palindrome", "fibonacci", "reverse_string",
            "count_vowels", "is_prime", "sum_of_digits", "remove_duplicates"]
DOMAIN_TYPES = ["unbounded", "unbounded", "bounded", "unbounded",
                "unbounded", "bounded", "bounded", "unbounded"]
N_ROUNDS = len(PROBLEMS)
BUDGETS = [0, 1, 2]
LEADER_ACTIONS = ["Pass", "Cheap", "Audit"]
OBSERVATIONS = ["none", "clear", "flag", "revealed_honest", "revealed_attack"]


def partition_label(pid):
    end_partition = N_ROUNDS * len(BUDGETS)
    if pid == end_partition:
        return {"partition": pid, "round": None, "budget": None,
                "problem": None, "domain": None, "label": "end"}
    i, b = divmod(pid, len(BUDGETS))
    return {
        "partition": pid, "round": i, "budget": b,
        "problem": PROBLEMS[i], "domain": DOMAIN_TYPES[i],
        "label": f"R{i+1} {PROBLEMS[i]} (budget={b})",
    }


def belief_key(belief, ndigits=4):
    return ",".join(f"{x:.{ndigits}f}" for x in belief)


def main():
    trace_path = sys.argv[1] if len(sys.argv) > 1 else "games/hsvi_trace.jsonl"

    with open(trace_path) as f:
        raw = [json.loads(l) for l in f if l.strip()]

    nodes = {}
    root_key = None
    iters = []
    trials = []       # per-iteration ordered list of visit steps (the trial's
                       # actual forward walk from root to frontier), for the
                       # "trial spotlight" panel: belief trajectory + the
                       # action/observation that produced each step.
    current_trial = None

    def get_or_create_node(depth, pid, belief):
        key = f"{depth}|{pid}|{belief_key(belief)}"
        if key not in nodes:
            info = partition_label(pid)
            nodes[key] = {
                "key": key, "depth": depth, **info,
                "belief": belief, "history": [], "children": {},
            }
        return key

    path = []            # path[d] = node key at depth d, for the current trial
    peek_depth = None     # depth the next 1-2 "peek" bound records belong to
    backup_depth = None   # depth the next "backup" bound record belongs to
    backup_count = 0      # how many backup records consumed at backup_depth so far (0 or 1)

    for rec in raw:
        t = rec["type"]

        if t == "iter":
            iters.append({k: rec[k] for k in
                           ("iter", "elapsed_ms", "lb", "ub", "width", "numVectors", "numPoints")})
            if current_trial is not None:
                trials.append(current_trial)
            current_trial = {"iter": rec["iter"], "steps": []}
            path = []
            peek_depth = None
            backup_depth = None
            backup_count = 0
            continue

        if t == "visit":
            d = rec["depth"]
            key = get_or_create_node(d, rec["partition"], rec["belief"])
            path = path[:d] + [key]
            if d == 0:
                root_key = key

            info = partition_label(rec["partition"])
            step = {
                "depth": d, "partition": rec["partition"],
                "round": info["round"], "budget": info["budget"], "problem": info["problem"],
                "belief": rec["belief"], "action": None, "observation": None,
            }

            if rec["branched"]:
                a, o = rec["action"], rec["observation"]
                step["action"] = LEADER_ACTIONS[a]
                step["observation"] = OBSERVATIONS[o]
                child_key = get_or_create_node(d + 1, rec["childPartition"], rec["childBelief"])
                nodes[key]["children"][child_key] = {
                    "childKey": child_key,
                    "action": LEADER_ACTIONS[a], "observation": OBSERVATIONS[o],
                }
                peek_depth = d
            else:
                backup_depth = d
                backup_count = 0
            current_trial["steps"].append(step)
            continue

        if t == "bound":
            if rec["phase"] == "peek":
                d = peek_depth
            else:
                d = backup_depth
                backup_count += 1
                if backup_count == 2:
                    backup_count = 0
                    backup_depth = d - 1
            key = path[d]
            nodes[key]["history"].append({
                "iter": rec["iter"], "bound": rec["bound"],
                "phase": rec["phase"], "value": rec["value"],
            })
            continue

    # collapse per-node history: pair up lb/ub entries sharing the same
    # (iter, phase) into single {iter, phase, lb, ub} events, in original order
    for node in nodes.values():
        merged = []
        pending = {}
        order = []
        for h in node["history"]:
            pk = (h["iter"], h["phase"])
            if pk not in pending:
                pending[pk] = {"iter": h["iter"], "phase": h["phase"]}
                order.append(pk)
            pending[pk][h["bound"]] = h["value"]
        node["history"] = [pending[pk] for pk in order]
        node["children"] = list(node["children"].values())

    if current_trial is not None:
        trials.append(current_trial)

    out = {"iterations": iters, "rootKey": root_key, "nodes": nodes, "trials": trials}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
