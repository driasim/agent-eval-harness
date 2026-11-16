"""Token accounting and cost calculation across agent providers."""

from typing import Dict, Tuple
from agent_eval.models import CostMetrics, TrajectoryStep

# Pricing per million tokens: (prompt_cost, completion_cost) in USD
DEFAULT_PRICING: Dict[str, Tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
    "default": (2.00, 8.00),
}


class CostCalculator:
    """Calculates token costs and detects economic drift across agent runs."""

    def __init__(self, custom_pricing: Dict[str, Tuple[float, float]] = None):
        self.pricing = {**DEFAULT_PRICING, **(custom_pricing or {})}

    def compute_trajectory_cost(self, steps: list[TrajectoryStep], model: str, latency_ms: float = 0.0) -> CostMetrics:
        total_prompt = sum(step.prompt_tokens for step in steps)
        total_completion = sum(step.completion_tokens for step in steps)
        total_tokens = total_prompt + total_completion

        # Match model prefix or fallback
        matched_pricing = self.pricing.get("default")
        model_lower = model.lower()
        for key, price in self.pricing.items():
            if key in model_lower:
                matched_pricing = price
                break

        prompt_price_per_m, completion_price_per_m = matched_pricing
        cost_usd = (total_prompt / 1_000_000 * prompt_price_per_m) + (
            total_completion / 1_000_000 * completion_price_per_m
        )

        return CostMetrics(
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            total_tokens=total_tokens,
            estimated_cost_usd=round(cost_usd, 6),
            latency_ms=round(latency_ms, 2),
        )
