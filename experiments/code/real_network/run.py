"""Local-only nonblind deblurring experiments with fixed RED updates."""
import argparse

from calibrate import calibrate, plan
from evaluate import evaluate
from common import normalize_settings, read_json
from workflow import estimate, workflow
from history import prepare_history, history_workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    make = sub.add_parser("plan", help="Freeze local development images and candidate sets")
    make.add_argument("--config", required=True)
    make.add_argument("--images", required=True, help="Separate local development image folder")
    make.add_argument("--out", required=True, help="New calibration directory")
    fit = sub.add_parser("calibrate", help="Budget-specific development selection, no evaluation data")
    fit.add_argument("--out", required=True)
    fit.add_argument("--device", default="cuda")
    test = sub.add_parser("evaluate", help="Read frozen choices and evaluate local images")
    test.add_argument("--calibration", required=True)
    test.add_argument("--images", required=True)
    test.add_argument("--out", required=True, help="New evaluation directory")
    test.add_argument("--dataset", choices=("bsd68", "set12", "custom"), required=True)
    test.add_argument("--device", default="cuda")
    test.add_argument("--save-states", action="store_true", help="Save float endpoint arrays; can use substantial disk space")
    cost = sub.add_parser("estimate", help="Estimate forward count without loading weights or images")
    cost.add_argument("--config", required=True)
    cost.add_argument("--development-count", type=int, default=12)
    cost.add_argument("--evaluation-count", type=int, default=68)
    cost.add_argument("--milliseconds-per-call", type=float, default=28)
    whole = sub.add_parser("workflow", help="Validate local inputs, plan, calibrate, evaluate and summarize")
    whole.add_argument("--config", default="experiments/configs/drunet_fast.json")
    whole.add_argument("--development-images", required=True)
    whole.add_argument("--evaluation-images", required=True)
    whole.add_argument("--dataset", choices=("bsd68", "set12", "custom"), default="bsd68")
    whole.add_argument("--out", required=True, help="New output directory")
    whole.add_argument("--device", default="cuda")
    whole.add_argument("--milliseconds-per-call", type=float, default=28)
    whole.add_argument("--save-states", action="store_true")
    history = sub.add_parser("history", help="Saved-development best nonzero prefixes and common terminal extension")
    history.add_argument("--source-calibration", required=True)
    history.add_argument("--out", required=True, help="New output directory; source results are read-only")
    history.add_argument("--images", help="Local benchmark folder; unnecessary with --plan-only")
    history.add_argument("--dataset", choices=("bsd68", "set12", "custom"), default="bsd68")
    history.add_argument("--device", default="cuda")
    history.add_argument("--budgets", nargs="+", type=int, help="Subset of source calibrated budgets; default all")
    history.add_argument("--anchor", type=int, default=64)
    history.add_argument("--extra-calls", nargs="+", type=int, default=[32, 64])
    history.add_argument("--plan-only", action="store_true", help="Freeze choices and cost estimate without running a network")
    history.add_argument("--evaluation-count", type=int, default=68, help="Image count for --plan-only cost estimate")
    history.add_argument("--milliseconds-per-call", type=float, default=28)
    history.add_argument("--save-states", action="store_true")
    args = parser.parse_args()
    if args.phase == "plan":
        plan(args.config, args.images, args.out)
    elif args.phase == "calibrate":
        calibrate(args.out, args.device)
    elif args.phase == "evaluate":
        evaluate(args.calibration, args.images, args.out, args.dataset, args.device, args.save_states)
    elif args.phase == "estimate":
        import json
        print(json.dumps(estimate(normalize_settings(read_json(args.config)), args.development_count,
                                  args.evaluation_count, args.milliseconds_per_call), indent=2))
    elif args.phase == "workflow":
        workflow(args.config, args.development_images, args.evaluation_images, args.out,
                 args.dataset, args.device, args.milliseconds_per_call, args.save_states)
    else:
        options = {"budgets": args.budgets, "anchor": args.anchor, "extra_calls": args.extra_calls,
                   "milliseconds_per_call": args.milliseconds_per_call}
        if args.plan_only:
            prepare_history(args.source_calibration, args.out,
                            evaluation_count=args.evaluation_count, **options)
        else:
            if not args.images:
                parser.error("history requires --images unless --plan-only is used")
            history_workflow(args.source_calibration, args.images, args.out, args.dataset,
                             args.device, args.save_states, **options)


if __name__ == "__main__":
    main()
