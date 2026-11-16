"""Local-only nonblind deblurring experiments with fixed RED updates."""
import argparse

from calibrate import calibrate, plan
from evaluate import evaluate


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
    args = parser.parse_args()
    if args.phase == "plan":
        plan(args.config, args.images, args.out)
    elif args.phase == "calibrate":
        calibrate(args.out, args.device)
    else:
        evaluate(args.calibration, args.images, args.out, args.dataset, args.device, args.save_states)


if __name__ == "__main__":
    main()
