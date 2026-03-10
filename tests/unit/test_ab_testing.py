"""Unit tests for the A/B testing traffic manager."""

import pytest

from src.ab_testing.traffic_manager import (
    ABTestResult,
    RolloutStrategy,
    TrafficManager,
    TrafficSplit,
    VariantConfig,
)


class TestVariantConfig:
    """Tests for VariantConfig."""

    def test_defaults(self):
        variant = VariantConfig(
            variant_name="champion", model_name="model-v1"
        )
        assert variant.instance_type == "ml.m5.large"
        assert variant.initial_weight == 1.0
        assert variant.initial_instance_count == 1


class TestTrafficSplit:
    """Tests for TrafficSplit validation."""

    def test_valid_split(self):
        split = TrafficSplit(
            variant_weights={"a": 0.7, "b": 0.3}
        )
        assert split.validate() is True

    def test_invalid_split(self):
        split = TrafficSplit(
            variant_weights={"a": 0.5, "b": 0.3}
        )
        assert split.validate() is False

    def test_to_dict(self):
        split = TrafficSplit(
            variant_weights={"a": 0.8, "b": 0.2}
        )
        result = split.to_dict()
        assert "weights" in result
        assert "timestamp" in result


class TestTrafficManager:
    """Tests for TrafficManager."""

    def _make_variants(self):
        champion = VariantConfig(
            variant_name="champion",
            model_name="model-v1",
        )
        challenger = VariantConfig(
            variant_name="challenger",
            model_name="model-v2",
        )
        return champion, challenger

    def test_register_variant(self):
        manager = TrafficManager()
        variant = VariantConfig(
            variant_name="test", model_name="m1"
        )
        manager.register_variant(variant)
        assert "test" in manager._variants

    def test_setup_ab_test(self):
        manager = TrafficManager()
        champion, challenger = self._make_variants()
        split = manager.setup_ab_test(
            champion, challenger, challenger_traffic=0.10
        )

        assert split.variant_weights["champion"] == 0.90
        assert split.variant_weights["challenger"] == 0.10
        assert split.validate()

    def test_setup_invalid_traffic(self):
        manager = TrafficManager()
        champion, challenger = self._make_variants()
        with pytest.raises(ValueError):
            manager.setup_ab_test(
                champion, challenger, challenger_traffic=1.5
            )

    def test_update_traffic(self):
        manager = TrafficManager()
        champion, challenger = self._make_variants()
        manager.setup_ab_test(champion, challenger)
        split = manager.update_traffic(
            {"champion": 0.5, "challenger": 0.5}
        )

        assert split.variant_weights["champion"] == 0.5
        assert split.validate()

    def test_update_unknown_variant(self):
        manager = TrafficManager()
        champion, challenger = self._make_variants()
        manager.setup_ab_test(champion, challenger)
        with pytest.raises(ValueError, match="Unknown variant"):
            manager.update_traffic({"unknown": 1.0})

    def test_gradual_rollout_linear(self):
        manager = TrafficManager()
        champion, challenger = self._make_variants()
        manager.setup_ab_test(champion, challenger)

        plan = manager.gradual_rollout(
            "challenger", RolloutStrategy.LINEAR, steps=4
        )
        assert len(plan) == 4
        assert plan[-1].variant_weights["challenger"] == 1.0

    def test_gradual_rollout_canary(self):
        manager = TrafficManager()
        champion, challenger = self._make_variants()
        manager.setup_ab_test(champion, challenger)

        plan = manager.gradual_rollout(
            "challenger", RolloutStrategy.CANARY
        )
        assert plan[0].variant_weights["challenger"] == 0.05
        assert plan[-1].variant_weights["challenger"] == 1.0

    def test_gradual_rollout_all_at_once(self):
        manager = TrafficManager()
        champion, challenger = self._make_variants()
        manager.setup_ab_test(champion, challenger)

        plan = manager.gradual_rollout(
            "challenger", RolloutStrategy.ALL_AT_ONCE
        )
        assert len(plan) == 1
        assert plan[0].variant_weights["challenger"] == 1.0

    def test_evaluate_challenger_wins(self):
        manager = TrafficManager()
        champion, challenger = self._make_variants()
        manager.setup_ab_test(champion, challenger)

        result = manager.evaluate_ab_test(
            champion_metrics={"f1": 0.80},
            challenger_metrics={"f1": 0.85},
            primary_metric="f1",
            min_improvement=0.02,
        )

        assert result.winner == "challenger"
        assert "PROMOTE" in result.recommendation

    def test_evaluate_champion_wins(self):
        manager = TrafficManager()
        champion, challenger = self._make_variants()
        manager.setup_ab_test(champion, challenger)

        result = manager.evaluate_ab_test(
            champion_metrics={"f1": 0.85},
            challenger_metrics={"f1": 0.80},
        )

        assert result.winner == "champion"
        assert "KEEP" in result.recommendation

    def test_evaluate_insufficient_improvement(self):
        manager = TrafficManager()
        champion, challenger = self._make_variants()
        manager.setup_ab_test(champion, challenger)

        result = manager.evaluate_ab_test(
            champion_metrics={"f1": 0.80},
            challenger_metrics={"f1": 0.81},
            min_improvement=0.05,
        )

        assert result.winner == "champion"
        assert "below" in result.reason.lower()

    def test_promote_winner(self):
        manager = TrafficManager()
        champion, challenger = self._make_variants()
        manager.setup_ab_test(champion, challenger)

        split = manager.promote_winner("challenger")
        assert split.variant_weights["challenger"] == 1.0
        assert split.variant_weights["champion"] == 0.0

    def test_build_endpoint_config(self):
        manager = TrafficManager()
        champion, challenger = self._make_variants()
        manager.setup_ab_test(
            champion, challenger, challenger_traffic=0.20
        )

        config = manager.build_endpoint_config("test-config")
        assert config["EndpointConfigName"] == "test-config"
        assert len(config["ProductionVariants"]) == 2
        assert config["DataCaptureConfig"]["EnableCapture"] is True

    def test_history_tracking(self):
        manager = TrafficManager()
        champion, challenger = self._make_variants()
        manager.setup_ab_test(champion, challenger)
        manager.update_traffic(
            {"champion": 0.5, "challenger": 0.5}
        )

        history = manager.get_history()
        assert len(history) == 2
