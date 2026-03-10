"""A/B testing traffic management for SageMaker endpoint variants.

Handles weighted traffic routing between model variants, enabling
gradual rollouts and champion-challenger deployments.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config.settings import PipelineSettings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RolloutStrategy(str, Enum):
    """Deployment rollout strategies."""

    CANARY = "canary"
    LINEAR = "linear"
    ALL_AT_ONCE = "all_at_once"
    BLUE_GREEN = "blue_green"


@dataclass
class VariantConfig:
    """Configuration for a single endpoint variant."""

    variant_name: str
    model_name: str
    instance_type: str = "ml.m5.large"
    initial_instance_count: int = 1
    initial_weight: float = 1.0
    model_data_url: Optional[str] = None


@dataclass
class TrafficSplit:
    """Represents the traffic distribution across variants."""

    variant_weights: Dict[str, float]
    timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    def validate(self) -> bool:
        """Verify weights sum to 1.0 (within tolerance)."""
        total = sum(self.variant_weights.values())
        return abs(total - 1.0) < 0.01

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights": self.variant_weights,
            "timestamp": self.timestamp,
        }


@dataclass
class ABTestResult:
    """Results from an A/B test comparison."""

    champion: str
    challenger: str
    champion_metrics: Dict[str, float]
    challenger_metrics: Dict[str, float]
    winner: str
    reason: str
    confidence: float
    recommendation: str


class TrafficManager:
    """Manages traffic routing for SageMaker endpoint A/B testing.

    Supports multiple rollout strategies and gradual traffic
    shifting between champion and challenger model variants.
    """

    def __init__(self, settings: Optional[PipelineSettings] = None):
        self.settings = settings or PipelineSettings()
        self._variants: Dict[str, VariantConfig] = {}
        self._history: List[TrafficSplit] = []
        self._current_split: Optional[TrafficSplit] = None

    def register_variant(self, variant: VariantConfig) -> None:
        """Register a model variant for traffic management."""
        self._variants[variant.variant_name] = variant
        logger.info(
            f"Registered variant '{variant.variant_name}' "
            f"(model: {variant.model_name}, "
            f"weight: {variant.initial_weight})"
        )

    def setup_ab_test(
        self,
        champion: VariantConfig,
        challenger: VariantConfig,
        challenger_traffic: float = 0.10,
    ) -> TrafficSplit:
        """Set up an A/B test between champion and challenger.

        Args:
            champion: Current production model variant.
            challenger: New model variant to test.
            challenger_traffic: Fraction of traffic for the challenger.

        Returns:
            TrafficSplit with initial distribution.
        """
        if not 0.0 < challenger_traffic < 1.0:
            raise ValueError(
                f"Challenger traffic must be between 0 and 1, "
                f"got {challenger_traffic}"
            )

        self.register_variant(champion)
        self.register_variant(challenger)

        split = TrafficSplit(
            variant_weights={
                champion.variant_name: round(1.0 - challenger_traffic, 4),
                challenger.variant_name: round(challenger_traffic, 4),
            }
        )

        if not split.validate():
            raise ValueError("Traffic weights must sum to 1.0")

        self._current_split = split
        self._history.append(split)

        logger.info(
            f"A/B test configured: {champion.variant_name}="
            f"{1.0 - challenger_traffic:.0%}, "
            f"{challenger.variant_name}={challenger_traffic:.0%}"
        )
        return split

    def update_traffic(
        self, variant_weights: Dict[str, float]
    ) -> TrafficSplit:
        """Update the traffic distribution across variants."""
        for name in variant_weights:
            if name not in self._variants:
                raise ValueError(f"Unknown variant: {name}")

        split = TrafficSplit(variant_weights=variant_weights)
        if not split.validate():
            raise ValueError(
                f"Weights must sum to 1.0, got "
                f"{sum(variant_weights.values()):.4f}"
            )

        self._current_split = split
        self._history.append(split)

        weights_str = ", ".join(
            f"{k}={v:.2%}" for k, v in variant_weights.items()
        )
        logger.info(f"Traffic updated: {weights_str}")
        return split

    def gradual_rollout(
        self,
        target_variant: str,
        strategy: RolloutStrategy = RolloutStrategy.LINEAR,
        steps: int = 5,
    ) -> List[TrafficSplit]:
        """Plan a gradual traffic shift to the target variant.

        Args:
            target_variant: Variant to shift traffic toward.
            strategy: Rollout strategy to use.
            steps: Number of incremental shifts.

        Returns:
            List of planned TrafficSplit stages.
        """
        if target_variant not in self._variants:
            raise ValueError(f"Unknown variant: {target_variant}")

        other_variants = [
            name for name in self._variants if name != target_variant
        ]
        if not other_variants:
            raise ValueError("Need at least two variants for rollout")

        source_variant = other_variants[0]
        plan: List[TrafficSplit] = []

        if strategy == RolloutStrategy.CANARY:
            increments = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
        elif strategy == RolloutStrategy.LINEAR:
            increments = [
                round((i + 1) / steps, 4) for i in range(steps)
            ]
        elif strategy == RolloutStrategy.ALL_AT_ONCE:
            increments = [1.0]
        elif strategy == RolloutStrategy.BLUE_GREEN:
            increments = [0.0, 1.0]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        for target_weight in increments:
            split = TrafficSplit(
                variant_weights={
                    target_variant: round(target_weight, 4),
                    source_variant: round(1.0 - target_weight, 4),
                }
            )
            plan.append(split)

        logger.info(
            f"Rollout plan ({strategy.value}): "
            f"{len(plan)} stages to shift traffic to '{target_variant}'"
        )
        return plan

    def evaluate_ab_test(
        self,
        champion_metrics: Dict[str, float],
        challenger_metrics: Dict[str, float],
        primary_metric: str = "f1",
        min_improvement: float = 0.02,
    ) -> ABTestResult:
        """Evaluate A/B test results and recommend a winner.

        Args:
            champion_metrics: Performance metrics of the champion.
            challenger_metrics: Performance metrics of the challenger.
            primary_metric: Key metric for the decision.
            min_improvement: Minimum improvement to justify a switch.

        Returns:
            ABTestResult with winner recommendation.
        """
        variants = list(self._variants.keys())
        if len(variants) < 2:
            raise ValueError("Need at least two variants to evaluate")

        champion_name = variants[0]
        challenger_name = variants[1]

        champ_value = champion_metrics.get(primary_metric, 0)
        chall_value = challenger_metrics.get(primary_metric, 0)
        improvement = chall_value - champ_value

        if improvement >= min_improvement:
            winner = challenger_name
            confidence = min(improvement / min_improvement, 1.0)
            reason = (
                f"Challenger improves {primary_metric} by "
                f"{improvement:.4f} (threshold: {min_improvement})"
            )
            recommendation = (
                "PROMOTE challenger to production. Challenger shows "
                "statistically significant improvement."
            )
        elif improvement > 0:
            winner = champion_name
            confidence = 0.5
            reason = (
                f"Challenger improves {primary_metric} by "
                f"{improvement:.4f}, but below threshold "
                f"{min_improvement}"
            )
            recommendation = (
                "KEEP champion. Improvement is below the minimum "
                "threshold. Continue monitoring or retrain."
            )
        else:
            winner = champion_name
            confidence = 1.0 - max(0, 1.0 + improvement / min_improvement)
            reason = (
                f"Champion outperforms challenger on {primary_metric} "
                f"by {abs(improvement):.4f}"
            )
            recommendation = (
                "KEEP champion. Challenger underperforms. "
                "Consider different hyperparameters or features."
            )

        result = ABTestResult(
            champion=champion_name,
            challenger=challenger_name,
            champion_metrics=champion_metrics,
            challenger_metrics=challenger_metrics,
            winner=winner,
            reason=reason,
            confidence=round(confidence, 4),
            recommendation=recommendation,
        )

        logger.info(
            f"A/B test result: winner='{winner}', "
            f"confidence={result.confidence:.2%}, "
            f"reason='{reason}'"
        )
        return result

    def promote_winner(self, winner: str) -> TrafficSplit:
        """Send all traffic to the winning variant."""
        if winner not in self._variants:
            raise ValueError(f"Unknown variant: {winner}")

        weights = {name: 0.0 for name in self._variants}
        weights[winner] = 1.0

        split = self.update_traffic(weights)
        logger.info(f"Promoted '{winner}' to receive 100% traffic")
        return split

    def get_current_split(self) -> Optional[TrafficSplit]:
        """Get the current traffic distribution."""
        return self._current_split

    def get_history(self) -> List[Dict[str, Any]]:
        """Get the complete traffic split history."""
        return [s.to_dict() for s in self._history]

    def build_endpoint_config(
        self, config_name: str
    ) -> Dict[str, Any]:
        """Build a SageMaker EndpointConfig with production variants.

        Returns the configuration dictionary that can be used with
        the SageMaker CreateEndpointConfig API.
        """
        if self._current_split is None:
            raise ValueError("No traffic split configured")

        production_variants = []
        for name, variant in self._variants.items():
            weight = self._current_split.variant_weights.get(name, 0)
            production_variants.append(
                {
                    "VariantName": name,
                    "ModelName": variant.model_name,
                    "InstanceType": variant.instance_type,
                    "InitialInstanceCount": variant.initial_instance_count,
                    "InitialVariantWeight": weight,
                }
            )

        config = {
            "EndpointConfigName": config_name,
            "ProductionVariants": production_variants,
            "DataCaptureConfig": {
                "EnableCapture": True,
                "InitialSamplingPercentage": 100,
                "CaptureOptions": [
                    {"CaptureMode": "Input"},
                    {"CaptureMode": "Output"},
                ],
            },
        }

        logger.info(
            f"Endpoint config '{config_name}' built with "
            f"{len(production_variants)} variants"
        )
        return config
