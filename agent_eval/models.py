"""Data models for agent trajectories, tool invocations, and eval outcomes."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ToolCallRecord(BaseModel):
    """Record of a single tool/MCP call made by an agent."""
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    output: Optional[Any] = None
    execution_time_ms: float = 0.0
    error: Optional[str] = None


class TrajectoryStep(BaseModel):
    """Single turn/step in an agent's execution history."""
    step_index: int
    role: str  # "system", "user", "assistant", "tool"
    thought: Optional[str] = None
    content: Optional[str] = None
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class CostMetrics(BaseModel):
    """Cost and token usage metrics for a trajectory."""
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    latency_ms: float = 0.0


class AgentTrajectory(BaseModel):
    """Full execution trajectory of an agent run."""
    run_id: str
    agent_id: str
    scenario_id: str
    model: str
    system_prompt_version: str
    steps: List[TrajectoryStep] = Field(default_factory=list)
    final_output: Optional[str] = None
    metrics: CostMetrics = Field(default_factory=CostMetrics)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AssertionViolation(BaseModel):
    """Details of an evaluation assertion failure."""
    assertion_type: str
    message: str
    expected: Any = None
    actual: Any = None
    step_index: Optional[int] = None


class EvalResult(BaseModel):
    """Result of evaluating a trajectory against constraints or a baseline."""
    scenario_id: str
    passed: bool
    violations: List[AssertionViolation] = Field(default_factory=list)
    score: float = 1.0  # 0.0 to 1.0
    metrics: CostMetrics
    notes: List[str] = Field(default_factory=list)


class RegressionDiff(BaseModel):
    """Comparison between a candidate trajectory and a golden baseline."""
    scenario_id: str
    baseline_run_id: str
    candidate_run_id: str
    passed_regression: bool
    cost_delta_usd: float
    token_delta: int
    latency_delta_ms: float
    tool_sequence_matched: bool
    new_violations: List[AssertionViolation] = Field(default_factory=list)
