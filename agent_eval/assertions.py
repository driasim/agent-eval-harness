"""Deterministic assertions for agent trajectories and tool-call safety."""

from typing import Any, Callable, List, Optional
from agent_eval.models import AgentTrajectory, AssertionViolation


class TrajectoryAssertions:
    """Evaluates safety invariants and deterministic expectations on agent runs."""

    @staticmethod
    def assert_tool_called(
        trajectory: AgentTrajectory,
        tool_name: str,
        min_calls: int = 1,
        max_calls: Optional[int] = None,
    ) -> List[AssertionViolation]:
        violations = []
        actual_calls = 0
        for step in trajectory.steps:
            for call in step.tool_calls:
                if call.tool_name == tool_name:
                    actual_calls += 1

        if actual_calls < min_calls:
            violations.append(
                AssertionViolation(
                    assertion_type="tool_called_min",
                    message=f"Tool '{tool_name}' was expected at least {min_calls} time(s), but called {actual_calls} time(s).",
                    expected=min_calls,
                    actual=actual_calls,
                )
            )
        if max_calls is not None and actual_calls > max_calls:
            violations.append(
                AssertionViolation(
                    assertion_type="tool_called_max",
                    message=f"Tool '{tool_name}' was called {actual_calls} time(s), exceeding max allowed {max_calls}.",
                    expected=max_calls,
                    actual=actual_calls,
                )
            )
        return violations

    @staticmethod
    def assert_tool_not_called(
        trajectory: AgentTrajectory, tool_name: str
    ) -> List[AssertionViolation]:
        violations = []
        for step in trajectory.steps:
            for call in step.tool_calls:
                if call.tool_name == tool_name:
                    violations.append(
                        AssertionViolation(
                            assertion_type="prohibited_tool_called",
                            message=f"Prohibited tool '{tool_name}' was called during step {step.step_index}.",
                            expected=None,
                            actual=tool_name,
                            step_index=step.step_index,
                        )
                    )
        return violations

    @staticmethod
    def assert_argument_equals(
        trajectory: AgentTrajectory,
        tool_name: str,
        arg_name: str,
        expected_value: Any,
    ) -> List[AssertionViolation]:
        violations = []
        matched_any = False
        for step in trajectory.steps:
            for call in step.tool_calls:
                if call.tool_name == tool_name:
                    actual_val = call.arguments.get(arg_name)
                    if actual_val == expected_value:
                        matched_any = True
                    else:
                        violations.append(
                            AssertionViolation(
                                assertion_type="argument_mismatch",
                                message=f"Tool '{tool_name}' argument '{arg_name}' mismatch.",
                                expected=expected_value,
                                actual=actual_val,
                                step_index=step.step_index,
                            )
                        )
        if not matched_any and not violations:
            violations.append(
                AssertionViolation(
                    assertion_type="tool_missing_for_argument_check",
                    message=f"Tool '{tool_name}' was never called to verify argument '{arg_name}'.",
                    expected=expected_value,
                    actual=None,
                )
            )
        return violations

    @staticmethod
    def assert_no_runaway_loop(
        trajectory: AgentTrajectory, max_consecutive_identical: int = 3
    ) -> List[AssertionViolation]:
        """Detects tool invocation entropy collapse (agent calling same tool with same args repeatedly)."""
        violations = []
        consecutive_count = 0
        last_signature = None

        for step in trajectory.steps:
            for call in step.tool_calls:
                # Signature is tool name + sorted args
                sig = (call.tool_name, str(sorted(call.arguments.items())))
                if sig == last_signature:
                    consecutive_count += 1
                else:
                    consecutive_count = 1
                    last_signature = sig

                if consecutive_count >= max_consecutive_identical:
                    violations.append(
                        AssertionViolation(
                            assertion_type="runaway_tool_loop",
                            message=f"Detected runaway tool loop on '{call.tool_name}' ({consecutive_count} consecutive calls).",
                            expected=f"<{max_consecutive_identical} consecutive identical calls",
                            actual=consecutive_count,
                            step_index=step.step_index,
                        )
                    )
                    return violations
        return violations

    @staticmethod
    def assert_max_turns(
        trajectory: AgentTrajectory, max_steps: int
    ) -> List[AssertionViolation]:
        if len(trajectory.steps) > max_steps:
            return [
                AssertionViolation(
                    assertion_type="max_steps_exceeded",
                    message=f"Agent exceeded step budget of {max_steps} (took {len(trajectory.steps)} steps).",
                    expected=max_steps,
                    actual=len(trajectory.steps),
                )
            ]
        return []
