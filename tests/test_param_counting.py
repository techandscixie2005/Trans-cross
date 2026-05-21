"""Tests for model parameter counting utilities."""

import torch.nn as nn

from src.transcross.model_utils import (
    count_trainable_parameters,
    count_parameters_by_module,
    format_parameter_table,
    compare_models,
)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 5)

    def forward(self, x):
        return self.linear2(self.linear1(x))


class TestCountTrainableParameters:
    def test_simple_model(self):
        model = SimpleModel()
        n = count_trainable_parameters(model)
        # 10*20 + 20 (bias) + 20*5 + 5 (bias) = 200 + 20 + 100 + 5 = 325
        assert n == 325

    def test_frozen_params_not_counted(self):
        model = SimpleModel()
        model.linear2.weight.requires_grad = False
        n = count_trainable_parameters(model)
        # linear1: 10*20 + 20 = 220
        # linear2: 20*5 (weight frozen) + 5 (bias) = 100 + 5... but weight frozen
        # Actually: linear2.weight 20*5=100 frozen, linear2.bias 5 trainable
        # Total: 220 + 5 = 225
        assert n == 225


class TestCountParametersByModule:
    def test_returns_per_module_counts(self):
        model = SimpleModel()
        counts = count_parameters_by_module(model)
        assert "linear1" in counts
        assert "linear2" in counts
        assert counts["linear1"] == 220  # 10*20 + 20
        assert counts["linear2"] == 105  # 20*5 + 5

    def test_total_matches_count_trainable(self):
        model = SimpleModel()
        by_module = count_parameters_by_module(model)
        total = sum(by_module.values())
        assert total == count_trainable_parameters(model)


class TestFormatParameterTable:
    def test_returns_string(self):
        model = SimpleModel()
        table = format_parameter_table(model)
        assert isinstance(table, str)
        assert "linear1" in table
        assert "linear2" in table
        assert "TOTAL" in table


class TestCompareModels:
    def test_same_model_within_tolerance(self):
        m0 = SimpleModel()
        m1 = SimpleModel()
        result, within = compare_models(m0, m1)
        assert within
        assert result["abs_diff"] == 0
        assert result["rel_diff"] == 0.0

    def test_different_models_outside_tolerance(self):
        m0 = SimpleModel()
        m1 = nn.Linear(5, 3)  # much smaller
        result, within = compare_models(m0, m1, max_relative_diff=0.01)
        assert not within
        assert result["abs_diff"] > 0

    def test_strict_tolerance(self):
        m0 = SimpleModel()
        m1 = SimpleModel()
        # Same model should pass even with 0 tolerance
        result, within = compare_models(m0, m1, max_relative_diff=0.0)
        assert within
