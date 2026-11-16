"""Unit tests for agent-eval-harness assertions and regression engine."""

import pytest
from agent_eval.models import (
    AgentTrajectory,
    TrajectoryStep,
    ToolCallRecord,
    CostMetrics,
)
from agent_eval.assertions import TrajectoryAssertions
from agent_eval.evaluator import TrajectoryEvaluator
from agent_eval.cost import CostCalculator


@pytest.fixture
def clean_trajectory() -> AgentTrajectory:
    return AgentTrajectory(
        run_id="run-101",
        agent_id="enterprise-crm-bot",
        scenario_id="customer-balance-lookup",
        model="gpt-4o",
        system_prompt_version="v2.1",
        steps=[
            TrajectoryStep(
                step_index=1,
                role="assistant",
                content="Looking up user account...",
                tool_calls=[
                    ToolCallRecord(
                        tool_name="get_user_id",
                        arguments={"email": "alice@corp.com"},
                        output={"user_id": "usr_9981"},
                    )
                ],
                prompt_tokens=450,
                completion_tokens=45,
            ),
            TrajectoryStep(
                step_index=2,
                role="assistant",
                content="Fetching account balance...",
                tool_calls=[
                    ToolCallRecord(
                        tool_name="get_account_balance",
                        arguments={"user_id": "usr_9981"},
                        output={"balance": 4500.50, "currency": "USD"},
                    )
                ],
                prompt_tokens=620,
                completion_tokens=60,
            ),
        ],
        metrics=CostMetrics(
            total_prompt_tokens=1070,
            total_completion_tokens=105,
            total_tokens=1175,
            estimated_cost_usd=0.0037,
            latency_ms=840.0,
        ),
    )


@pytest.fixture
def looping_trajectory() -> AgentTrajectory:
    """Agent stuck in recursive tool retry loop with identical arguments."""
    steps = []
    for i in range(1, 5):
        steps.append(
            TrajectoryStep(
                step_index=i,
                role="assistant",
                content="Retrying database query...",
                tool_calls=[
                    ToolCallRecord(
                        tool_name="query_db",
                        arguments={"sql": "SELECT balance FROM accounts WHERE id = 1"},
                        output=[],
                    )
                ],
                prompt_tokens=500,
                completion_tokens=30,
            )
        )
    return AgentTrajectory(
        run_id="run-broken-loop",
        agent_id="enterprise-crm-bot",
        scenario_id="customer-balance-lookup",
        model="gpt-4o",
        system_prompt_version="v2.2-broken",
        steps=steps,
        metrics=CostMetrics(
            total_prompt_tokens=2000,
            total_completion_tokens=120,
            total_tokens=2120,
            estimated_cost_usd=0.0062,
            latency_ms=2100.0,
        ),
    )


def test_required_tools_pass(clean_trajectory):
    evaluator = TrajectoryEvaluator()
    res = evaluator.evaluate(
        clean_trajectory,
        required_tools=["get_user_id", "get_account_balance"],
    )
    assert res.passed is True
    assert len(res.violations) == 0


def test_prohibited_tools_caught(clean_trajectory):
    evaluator = TrajectoryEvaluator()
    res = evaluator.evaluate(
        clean_trajectory,
        prohibited_tools=["get_user_id"],
    )
    assert res.passed is False
    assert any(v.assertion_type == "prohibited_tool_called" for v in res.violations)


def test_runaway_loop_detection(looping_trajectory):
    evaluator = TrajectoryEvaluator()
    res = evaluator.evaluate(looping_trajectory)
    assert res.passed is False
    assert any(v.assertion_type == "runaway_tool_loop" for v in res.violations)


def test_regression_diff(clean_trajectory, looping_trajectory):
    evaluator = TrajectoryEvaluator()
    diff = evaluator.compare_regression(
        baseline=clean_trajectory,
        candidate=looping_trajectory,
    )
    assert diff.passed_regression is False
    assert diff.tool_sequence_matched is False
    assert len(diff.new_violations) > 0
    assert diff.cost_delta_usd > 0
