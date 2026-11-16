"""Budget-specific candidate families, each with a shared terminal tail."""
import math

import numpy as np


FAMILIES = ("terminal", "endpoints", "platform", "geometric")


def schedule(candidate, budget, terminal):
    kind, m = candidate["kind"], int(candidate.get("prefix", 0))
    result = np.full(budget, terminal, dtype=np.float64)
    if kind == "terminal":
        return result
    high = float(candidate["sigma_255"]) / 255
    if kind == "platform":
        result[:m] = high
    elif kind == "geometric":
        result[:m] = np.geomspace(high, terminal, m)
    elif kind == "endpoints":
        pattern = candidate["pattern"]
        if pattern == "high_low":
            result[:m] = high
        elif pattern == "low_high":
            result[m//2:m] = high
        else:
            block = int(candidate["block"])
            phase = 0 if pattern == "high_first" else 1
            high_steps = (np.arange(m) // block + phase) % 2 == 0
            result[:m] = np.where(high_steps, high, terminal)
    else:
        raise ValueError(f"Unknown schedule kind: {kind}")
    return result


def candidate_families(settings, budget):
    search, terminal = settings["search"], settings["solver"]["terminal_sigma_255"] / 255
    tail = max(1, math.ceil(budget * settings["minimum_terminal_fraction"]))
    lengths = sorted({min(budget-tail, max(1, round(budget*f)))
                      for f in search["prefix_fractions"]})
    groups = {family: [{"kind": "terminal", "prefix": 0}] for family in FAMILIES}
    for level in search["sigma_255"]:
        for m in lengths:
            if m < 1:
                continue
            base = {"prefix": m, "sigma_255": level}
            groups["platform"].append({**base, "kind": "platform"})
            if m >= 2:
                groups["geometric"].append({**base, "kind": "geometric"})
            for pattern in ("high_low", "low_high"):
                groups["endpoints"].append({**base, "kind": "endpoints", "pattern": pattern})
            for block in search["block_lengths"]:
                for pattern in ("high_first", "low_first"):
                    groups["endpoints"].append({**base, "kind": "endpoints",
                                                "pattern": pattern, "block": block})
    # Remove duplicate trajectories, including terminal-equivalent short prefixes.
    for family, items in groups.items():
        unique = {}
        for item in items:
            unique.setdefault(tuple(schedule(item, budget, terminal)), item)
        groups[family] = list(unique.values())
    count = min(int(search["max_candidates"]), *(len(groups[f]) for f in FAMILIES[1:]))
    for index, family in enumerate(FAMILIES[1:]):
        items = groups[family]
        # Always include terminal, high->low, low->high and alternation when the cap permits.
        preferred = []
        if family == "endpoints":
            for pattern in ("high_low", "low_high", "high_first", "low_first"):
                matches = [i for i in range(1, len(items)) if items[i].get("pattern") == pattern]
                if matches:
                    preferred.append(matches[len(matches)//2])
        preferred = list(dict.fromkeys(preferred))[:count-1]
        rest = [i for i in range(1, len(items)) if i not in preferred]
        rng = np.random.default_rng(np.random.SeedSequence([search["seed"], budget, index]))
        rng.shuffle(rest)
        chosen = preferred + rest[:count-1-len(preferred)]
        groups[family] = [items[0]] + [items[i] for i in sorted(chosen)]
    return groups


def tie_key(candidate):
    return (candidate.get("prefix", 0), candidate.get("sigma_255", 0),
            candidate.get("pattern", ""), candidate.get("block", 0))
