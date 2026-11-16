"""Search endpoint sequences for the original selected raw-GMM observation."""
import argparse
import json
from pathlib import Path
import numpy as np
from centered_raw_probe import update


def control_sequences(length):
    return list(dict.fromkeys([
        "0"*length,
        "1"*length,
        ("01"*length)[:length],
        ("10"*length)[:length],
    ]))


def retain_beam(states, sequences, target, width):
    scores = np.sum((states-target)**2, axis=1)
    sequence_indices = {sequence: index for index, sequence in enumerate(sequences)}
    required = control_sequences(len(sequences[0]))
    selected = [sequence_indices[sequence] for sequence in required]
    chosen = set(selected)
    for index in np.argsort(scores, kind="stable"):
        index = int(index)
        if index not in chosen:
            selected.append(index)
            chosen.add(index)
        if len(selected) >= min(width, len(states)):
            break
    return states[selected], [sequences[i] for i in selected], scores[selected]


def active_prefix_length(sequence):
    return sequence.rfind("1")+1


def run_pair(source, sample, target, terminal_center, sigma_high, width, horizon, total):
    means, pi, operator = [source[key] for key in ["means", "pi", "operator"]]
    tau2 = float(source["tau2"])
    gram = operator@operator.T
    b = source["b"][sample:sample+1]
    states = b.copy()
    sequences = [""]
    history = []
    for depth in range(1, horizon+1):
        low = update(states, b, means, pi, gram, tau2, .02)
        high = update(states, b, means, pi, gram, tau2, sigma_high)
        candidates = np.concatenate([low, high])
        candidate_sequences = [sequence+"0" for sequence in sequences]
        candidate_sequences += [sequence+"1" for sequence in sequences]
        states, sequences, scores = retain_beam(
            candidates, candidate_sequences, target, width)
        assert np.isfinite(states).all()
        history.append({
            "calls": depth,
            "retained": len(states),
            "best_distance_to_target": float(np.sqrt(scores.min())),
            "retained_control_prefixes": control_sequences(depth),
        })
    switch_states = states.copy()
    for _ in range(total-horizon):
        states = update(states, b, means, pi, gram, tau2, .02)
    assert np.isfinite(states).all()
    distances = np.linalg.norm(states-target, axis=1)
    baseline_distances = np.linalg.norm(states-terminal_center, axis=1)
    rows = []
    controls = set(control_sequences(horizon))
    for index, sequence in enumerate(sequences):
        rows.append({
            "sequence": sequence,
            "high_calls": sequence.count("1"),
            "active_prefix_length": active_prefix_length(sequence),
            "switches": sum(sequence[k] != sequence[k-1]
                            for k in range(1, len(sequence))),
            "fixed_control": sequence in controls,
            "distance_to_middle_center": float(distances[index]),
            "distance_to_terminal_center": float(baseline_distances[index]),
            "middle_branch_hit": bool(distances[index] < 1e-3),
        })
    hits = [row for row in rows if row["middle_branch_hit"]]
    preferred = min(
        hits,
        key=lambda row: (row["active_prefix_length"], row["high_calls"],
                         row["switches"], row["sequence"]),
        default=None,
    )
    label = "infinity_limit" if np.isinf(sigma_high) else float(sigma_high)
    return {
        "high_sigma": label,
        "beam_width": width,
        "final_candidates": len(rows),
        "middle_branch_hits": len(hits),
        "first_hit_in_final_beam": next(
            (row for row in rows if row["middle_branch_hit"]), None),
        "shortest_active_prefix_among_retained_hits": preferred,
        "best_endpoint_distance": float(distances.min()),
        "history": history,
        "rows": rows,
    }, switch_states, states, sequences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or args.data/"raw_endpoint_search"
    out.mkdir(exist_ok=False)
    sample, width, horizon, total = 37, 64, 25, 5000
    plan = {
        "role": "Post-hoc endpoint reachability counterexample search",
        "observation": "Original saved development observation 37 only",
        "sample": sample,
        "target": "Saved raw_certification intermediate fixed-point center",
        "selection_inputs": "Model, saved observation and fixed-point centers only",
        "truth_access": False,
        "low_sigma": .02,
        "high_sigmas": [.3, 1., "infinity_limit"],
        "horizon": horizon,
        "beam_width": width,
        "total_calls": total,
        "score": "Current squared Euclidean distance to target center",
        "retention": "All-low, all-high and two alternating prefixes are mandatory",
        "hit": "Final distance to the saved middle center below 1e-3",
        "scope": "A hit gives an explicit original-observation replacement sequence. "
                 "No hit does not prove endpoint unreachability. No new noise "
                 "or risk-tuning evaluation is performed.",
    }
    (out/"plan.json").write_text(json.dumps(plan, indent=2))
    source = np.load(args.data/"raw_reference_detail/absolute_.02_states.npz")
    balls = np.load(args.data/"raw_certification/candidates.npz")
    checks = json.loads((args.data/"raw_certification/checks.json").read_text())
    assert checks["source_sample"] == sample
    policy_indices = {row["policy"]: row["sample"] for row in checks["rows"]}
    target = balls["centers"][policy_indices["intermediate"]]
    terminal_center = balls["centers"][policy_indices["terminal"]]
    rows, switches, endpoints, sequence_sets = [], [], [], []
    for sigma_high in [.3, 1., np.inf]:
        result, switched, final, sequences = run_pair(
            source, sample, target, terminal_center, sigma_high,
            width, horizon, total)
        rows.append(result)
        switches.append(switched)
        endpoints.append(final)
        sequence_sets.append(sequences)
        print(json.dumps({
            "high_sigma": result["high_sigma"],
            "hits": result["middle_branch_hits"],
            "best_distance": result["best_endpoint_distance"],
            "preferred": result["shortest_active_prefix_among_retained_hits"],
        }), flush=True)
    summary = {"plan": plan, "pairs": rows}
    (out/"summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(out/"endpoints.npz", target=target,
                        terminal_center=terminal_center,
                        switch_states=np.array(switches),
                        final_states=np.array(endpoints),
                        sequences=np.array(sequence_sets))


if __name__ == "__main__":
    main()
