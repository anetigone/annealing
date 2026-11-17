"""Run the fixed three-scale path probe and frozen BSD68 evaluation serially."""
import argparse
from pathlib import Path

from common import read_json, write_json
from probe_path import prepare as prepare_path, run as run_path
from validate_repair import prepare as prepare_quality, run as run_quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=("both", "path", "quality"), default="both")
    parser.add_argument("--repair", type=Path, default=Path("experiments/results/drunet_repair_run"))
    parser.add_argument("--development-plan", type=Path,
                        default=Path("experiments/results/drunet_fast_run/calibration/plan.json"))
    parser.add_argument("--terminal-source", type=Path, default=Path("experiments/results/drunet_terminal_long"))
    parser.add_argument("--images", type=Path, default=Path("experiments/assets/dncnn/testsets/BSD68"))
    parser.add_argument("--max-calls", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    tasks = []
    if args.phase in ("both", "path"):
        target = args.out / "path_probe"
        spec = prepare_path(args.terminal_source, target, args.max_calls)
        tasks.append(("path_probe", target, spec, run_path))
    if args.phase in ("both", "quality"):
        target = args.out / "bsd68"
        spec = prepare_quality(args.repair, args.development_plan, target, [2026111715], args.images, "bsd68")
        tasks.append(("bsd68", target, spec, run_quality))
    total = sum(spec["expected_network_forwards"] for _, _, spec, _ in tasks)
    write_json(args.out / "plan.json", {
        "phase": args.phase, "execution_order": [name for name, _, _, _ in tasks],
        "expected_network_forwards": total, "inference_minutes_at_28_ms": total * .028 / 60,
        "scope": "All child plans frozen before inference. Serial execution, path first. "
                 "No tuning, downloads, training or rewriting previous results. New output "
                 "directory required. Plan-only previews use a different output directory.",
    })
    print(f"Directed work: {total} forwards; about {total * .028 / 60:.2f} inference minutes", flush=True)
    if args.plan_only:
        return
    completed = {}
    for name, target, spec, execute in tasks:
        print(f"Starting {name}", flush=True)
        execute(spec, target, args.device)
        completed[name] = read_json(target / "completion.json")
    write_json(args.out / "completion.json", {
        "phases": completed,
        "network_forwards": sum(c["network_forwards"] for c in completed.values()),
    })


if __name__ == "__main__":
    main()
