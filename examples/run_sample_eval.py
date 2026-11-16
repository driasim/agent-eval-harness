"""Executable demo script demonstrating trajectory evaluation and regression detection."""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_eval.cli import load_trajectory
from agent_eval.evaluator import TrajectoryEvaluator
from rich.console import Console
from rich.table import Table

console = Console()

def run_demo():
    curr_dir = Path(__file__).parent
    baseline_path = curr_dir / "baseline_crm.json"
    candidate_path = curr_dir / "candidate_regressed.json"

    baseline = load_trajectory(str(baseline_path))
    candidate = load_trajectory(str(candidate_path))

    evaluator = TrajectoryEvaluator()

    console.print("\n[bold blue]=== 1. Evaluating Candidate Trajectory Safety Invariants ===[/bold blue]")
    candidate_eval = evaluator.evaluate(
        candidate,
        required_tools=["erp_lookup_vendor", "match_po_line_items"],
        prohibited_tools=["raw_sql_query"],
        max_cost_usd=0.010,
    )

    console.print(f"Passed: [bold {'green' if candidate_eval.passed else 'red'}]{candidate_eval.passed}[/]")
    console.print(f"Total Violations: [bold red]{len(candidate_eval.violations)}[/bold red]")
    for v in candidate_eval.violations:
        console.print(f" - [{v.assertion_type}] {v.message}")

    console.print("\n[bold blue]=== 2. Running Regression Diff (Baseline vs. Candidate) ===[/bold blue]")
    diff = evaluator.compare_regression(baseline, candidate)

    table = Table(title=f"Regression Diff: {diff.scenario_id}")
    table.add_column("Dimension", style="bold")
    table.add_column("Baseline (v1.4)", style="green")
    table.add_column("Candidate (v1.5-exp)", style="cyan")
    table.add_column("Delta / Impact", style="red")

    table.add_row(
        "Estimated Cost",
        f"${baseline.metrics.estimated_cost_usd:.5f}",
        f"${candidate.metrics.estimated_cost_usd:.5f}",
        f"+${diff.cost_delta_usd:.5f} ({((diff.cost_delta_usd/baseline.metrics.estimated_cost_usd)*100):.1f}% drift)"
    )
    table.add_row(
        "Total Tokens",
        str(baseline.metrics.total_tokens),
        str(candidate.metrics.total_tokens),
        f"+{diff.token_delta} tokens"
    )
    table.add_row(
        "Execution Latency",
        f"{baseline.metrics.latency_ms}ms",
        f"{candidate.metrics.latency_ms}ms",
        f"+{diff.latency_delta_ms}ms"
    )
    table.add_row(
        "Tool Execution Graph",
        "Deterministic Golden Path",
        "Diverged to Raw SQL",
        "REGRESSION DETECTED"
    )
    console.print(table)
    console.print("\n[bold yellow]Result: Deployment blocked by CI assertion harness.[/bold yellow]\n")

if __name__ == "__main__":
    run_demo()
