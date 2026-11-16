"""Local-only nonblind deblurring experiments with fixed RED updates."""
import argparse

from calibrate import calibrate, plan
from evaluate import evaluate
from common import normalize_settings, read_json
from workflow import estimate, workflow


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
    else:
        workflow(args.config, args.development_images, args.evaluation_images, args.out,
                 args.dataset, args.device, args.milliseconds_per_call, args.save_states)


if __name__ == "__main__":
    main()
