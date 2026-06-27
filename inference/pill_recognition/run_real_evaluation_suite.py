from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


SCOPE_MODES = ["none", "annotation", "ground-truth"]


@dataclass(frozen=True)
class SuiteCommand:
    name: str
    output: Path
    args: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run unscoped, annotation-scoped, and oracle-scoped real-smartphone "
            "evaluations plus comparison reports."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("../datasets/evaluation/real-smartphone"),
    )
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--annotations-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--prefix", default="real-smartphone")
    parser.add_argument("--pattern", default="*.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not rerun a step when its output JSON already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the suite plan without executing commands.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands = build_suite_commands(args)
    summary_path = args.output_dir / f"{args.prefix}-suite-summary.json"
    executed = []
    for command in commands:
        skipped = args.skip_existing and command.output.exists()
        row = {
            "name": command.name,
            "output": str(command.output),
            "command": command.args,
            "skipped": skipped,
        }
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if not args.dry_run and not skipped:
            run_command(command.args)
        executed.append(row)

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps({"commands": executed}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps({"summary": str(summary_path)}, ensure_ascii=False), flush=True)


def build_suite_commands(args: argparse.Namespace) -> list[SuiteCommand]:
    output_dir = args.output_dir
    eval_outputs = {
        "none": output_dir / f"{args.prefix}-unscoped.json",
        "annotation": output_dir / f"{args.prefix}-annotation-scope.json",
        "ground-truth": output_dir / f"{args.prefix}-oracle-scope.json",
    }

    commands = [
        SuiteCommand(
            name=f"evaluate:{scope_mode}",
            output=eval_outputs[scope_mode],
            args=evaluate_command(args, scope_mode, eval_outputs[scope_mode]),
        )
        for scope_mode in SCOPE_MODES
    ]
    commands.extend(
        [
            SuiteCommand(
                name="compare:annotation-vs-unscoped",
                output=output_dir / f"{args.prefix}-annotation-vs-unscoped.json",
                args=compare_command(
                    baseline=eval_outputs["none"],
                    candidate=eval_outputs["annotation"],
                    output=output_dir / f"{args.prefix}-annotation-vs-unscoped.json",
                    baseline_name="unscoped",
                    candidate_name="annotation-scope",
                ),
            ),
            SuiteCommand(
                name="compare:oracle-vs-unscoped",
                output=output_dir / f"{args.prefix}-oracle-vs-unscoped.json",
                args=compare_command(
                    baseline=eval_outputs["none"],
                    candidate=eval_outputs["ground-truth"],
                    output=output_dir / f"{args.prefix}-oracle-vs-unscoped.json",
                    baseline_name="unscoped",
                    candidate_name="oracle-scope",
                ),
            ),
            SuiteCommand(
                name="compare:oracle-vs-annotation",
                output=output_dir / f"{args.prefix}-oracle-vs-annotation.json",
                args=compare_command(
                    baseline=eval_outputs["annotation"],
                    candidate=eval_outputs["ground-truth"],
                    output=output_dir / f"{args.prefix}-oracle-vs-annotation.json",
                    baseline_name="annotation-scope",
                    candidate_name="oracle-scope",
                ),
            ),
        ]
    )
    return commands


def evaluate_command(args: argparse.Namespace, scope_mode: str, output: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "pill_recognition.evaluate_real_dataset",
        "--dataset-root",
        str(args.dataset_root),
        "--output",
        str(output),
        "--pattern",
        args.pattern,
        "--iou-threshold",
        str(args.iou_threshold),
        "--top-k",
        str(args.top_k),
        "--scope-mode",
        scope_mode,
    ]
    if args.images_dir:
        command.extend(["--images-dir", str(args.images_dir)])
    if args.annotations_dir:
        command.extend(["--annotations-dir", str(args.annotations_dir)])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    return command


def compare_command(
    baseline: Path,
    candidate: Path,
    output: Path,
    baseline_name: str,
    candidate_name: str,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pill_recognition.compare_real_evaluations",
        "--baseline",
        str(baseline),
        "--candidate",
        str(candidate),
        "--output",
        str(output),
        "--name-baseline",
        baseline_name,
        "--name-candidate",
        candidate_name,
    ]


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
