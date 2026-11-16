"""CLI entry point for agent evaluation and regression runs."""

import argparse
import json
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from agent_eval.models import AgentTrajectory
from agent_eval.evaluator import TrajectoryEvaluator

console = Console()


def load_trajectory(path: str) -> AgentTrajectory:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return AgentTrajectory(**data)


def run_eval(args):
    """Run single trajectory evaluation against constraints."""
    traj = load_trajectory(args.trajectory)
    evaluator = TrajectoryEvaluator()
    result = evaluator.evaluate(
        traj,
        required_tools=args.required_tools.split(",") if args.required_tools else None,
        prohibited_tools=args.prohibited_tools.split(",") if args.prohibited_tools else None,
        max_steps=args.max_steps,
        max_cost_usd=args.max_cost,
    )

    console.print(
        Panel.fit(
            f"[bold cyan]Scenario: {result.scenario_id}[/bold cyan]\n"
            f"Status: [{'green' if result.passed else 'red'}]{'PASSED' if result.passed else 'FAILED'}[/]\n"
            f"Score: {result.score} | Est. Cost: ${result.metrics.estimated_cost_usd} | Tokens: {result.metrics.total_tokens}",
            title="Agent Trajectory Evaluation",
        )
    )

    if result.violations:
        table = Table(title="Assertion Violations", style="red")
        table.add_column("Type", style="bold")
        table.add_column("Step", style="dim")
        table.add_column("Details")
        for v in result.violations:
            table.add_row(v.assertion_type, str(v.step_index or "-"), v.message)
        console.print(table)
        sys.exit(1)
    else:
        console.print("[green]✓ All trajectory assertions satisfied.[/green]")


def compare_runs(args):
    """Compare candidate trajectory against golden baseline to detect regression."""
    baseline = load_trajectory(args.baseline)
    candidate = load_trajectory(args.candidate)
    evaluator = TrajectoryEvaluator()
    diff = evaluator.compare_regression(baseline, candidate)

    table = Table(title=f"Regression Analysis: {diff.scenario_id}")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", style="dim")
    table.add_column("Candidate", style="dim")
    table.add_column("Delta")

    cost_style = "green" if diff.cost_delta_usd <= 0 else "yellow"
    table.add_row(
        "Estimated Cost",
        f"${baseline.metrics.estimated_cost_usd:.4f}",
        f"${candidate.metrics.estimated_cost_usd:.4f}",
        f"[{cost_style}]${diff.cost_delta_usd:+.4f}[/{cost_style}]",
    )
    table.add_row(
        "Total Tokens",
        str(baseline.metrics.total_tokens),
        str(candidate.metrics.total_tokens),
        f"{diff.token_delta:+d}",
    )
    table.add_row(
        "Latency",
        f"{baseline.metrics.latency_ms:.1f}ms",
        f"{candidate.metrics.latency_ms:.1f}ms",
        f"{diff.latency_delta_ms:+.1f}ms",
    )
    table.add_row(
        "Tool Sequence Match",
        "Golden Path",
        "Candidate Path",
        "[green]MATCH[/green]" if diff.tool_sequence_matched else "[yellow]DIVERGED[/yellow]",
    )

    console.print(table)

    if diff.new_violations:
        console.print(f"[bold red]❌ Regressions Introduced ({len(diff.new_violations)}):[/bold red]")
        for v in diff.new_violations:
            console.print(f" - [{v.assertion_type}] {v.message}")
        sys.exit(1)
    else:
        console.print("[bold green]✓ Candidate passed regression check without invariant failures.[/bold green]")


def main():
    parser = argparse.ArgumentParser(description="Agent Evaluation & Regression Harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # eval command
    p_eval = subparsers.add_parser("eval", help="Evaluate single trajectory against rules")
    p_eval.add_argument("--trajectory", "-t", required=True, help="Path to trajectory JSON")
    p_eval.add_argument("--required-tools", help="Comma-separated required tools")
    p_eval.add_argument("--prohibited-tools", help="Comma-separated prohibited tools")
    p_eval.add_argument("--max-steps", type=int, help="Max allowed steps")
    p_eval.add_argument("--max-cost", type=float, help="Max allowed cost in USD")
    p_eval.set_defaults(func=run_eval)

    # compare command
    p_compare = subparsers.add_parser("compare", help="Compare candidate against golden baseline")
    p_compare.add_argument("--baseline", "-b", required=True, help="Path to golden baseline JSON")
    p_compare.add_argument("--candidate", "-c", required=True, help="Path to candidate JSON")
    p_compare.set_defaults(func=compare_runs)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
