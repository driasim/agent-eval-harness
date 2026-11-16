"""Trajectory evaluator and regression comparison engine."""

from typing import List, Optional
from agent_eval.models import (
    AgentTrajectory,
    EvalResult,
    RegressionDiff,
    AssertionViolation,
)
from agent_eval.assertions import TrajectoryAssertions
from agent_eval.cost import CostCalculator


class TrajectoryEvaluator:
    """Evaluates agent runs and compares candidate runs against golden baselines."""

    def __init__(self, cost_calculator: Optional[CostCalculator] = None):
        self.cost_calculator = cost_calculator or CostCalculator()

    def evaluate(
        self,
        trajectory: AgentTrajectory,
        required_tools: Optional[List[str]] = None,
        prohibited_tools: Optional[List[str]] = None,
        max_steps: Optional[int] = None,
        max_cost_usd: Optional[float] = None,
    ) -> EvalResult:
        violations: List[AssertionViolation] = []

        # 1. Runaway loop assertion (Safety circuit)
        violations.extend(TrajectoryAssertions.assert_no_runaway_loop(trajectory))

        # 2. Required tools check
        if required_tools:
            for tool in required_tools:
                violations.extend(
                    TrajectoryAssertions.assert_tool_called(trajectory, tool)
                )

        # 3. Prohibited tools check
        if prohibited_tools:
            for tool in prohibited_tools:
                violations.extend(
                    TrajectoryAssertions.assert_tool_not_called(trajectory, tool)
                )

        # 4. Turn budget
        if max_steps:
            violations.extend(
                TrajectoryAssertions.assert_max_turns(trajectory, max_steps)
            )

        # 5. Cost budget
        if max_cost_usd and trajectory.metrics.estimated_cost_usd > max_cost_usd:
            violations.append(
                AssertionViolation(
                    assertion_type="cost_budget_exceeded",
                    message=f"Cost ${trajectory.metrics.estimated_cost_usd} exceeded budget of ${max_cost_usd}",
                    expected=max_cost_usd,
                    actual=trajectory.metrics.estimated_cost_usd,
                )
            )

        passed = len(violations) == 0
        score = max(0.0, 1.0 - (len(violations) * 0.25))

        return EvalResult(
            scenario_id=trajectory.scenario_id,
            passed=passed,
            violations=violations,
            score=round(score, 2),
            metrics=trajectory.metrics,
        )

    def compare_regression(
        self, baseline: AgentTrajectory, candidate: AgentTrajectory
    ) -> RegressionDiff:
        """Determines if a candidate agent version regressed compared to baseline."""
        # 1. Compare tool invocation sequence
        base_tools = [
            call.tool_name
            for step in baseline.steps
            for call in step.tool_calls
        ]
        cand_tools = [
            call.tool_name
            for step in candidate.steps
            for call in step.tool_calls
        ]
        tools_matched = base_tools == cand_tools

        # 2. Delta metrics
        cost_delta = round(
            candidate.metrics.estimated_cost_usd - baseline.metrics.estimated_cost_usd,
            6,
        )
        token_delta = candidate.metrics.total_tokens - baseline.metrics.total_tokens
        latency_delta = round(
            candidate.metrics.latency_ms - baseline.metrics.latency_ms, 2
        )

        # 3. Violations check
        cand_eval = self.evaluate(candidate)
        base_eval = self.evaluate(baseline)

        # Candidate is considered regressed if it introduced violations that baseline did not have
        new_violations = [
            v for v in cand_eval.violations if v not in base_eval.violations
        ]
        passed_regression = len(new_violations) == 0 and (
            cost_delta <= (baseline.metrics.estimated_cost_usd * 0.25)
        )

        return RegressionDiff(
            scenario_id=candidate.scenario_id,
            baseline_run_id=baseline.run_id,
            candidate_run_id=candidate.run_id,
            passed_regression=passed_regression,
            cost_delta_usd=cost_delta,
            token_delta=token_delta,
            latency_delta_ms=latency_delta,
            tool_sequence_matched=tools_matched,
            new_violations=new_violations,
        )
