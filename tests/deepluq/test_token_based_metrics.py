#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 21/08/2025 11:42
# @Author  : Chengjie
# @File    : test_token_based_metrics.py
# @Software: PyCharm

import torch
import pytest
from deepluq.metrics_vla import TokenMetrics   # adjust import if needed


@pytest.fixture
def tm():
    """Fixture to create a fresh TokenMetricsFast instance."""
    return TokenMetrics()


def test_calculate_metrics_output_shapes(tm):
    # Create logits for 3 samples, 4 classes
    logits = torch.tensor([[1.0, 2.0, 0.5, -0.5],
                           [2.0, 1.0, -1.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0]])
    metrics = tm.calculate_metrics(logits)

    # Expect 4 lists: [entropy, max_prob, pcs, deepgini]
    assert len(metrics) == 4
    for m in metrics:
        assert isinstance(m, list)
        assert len(m) == 3  # 3 samples


def test_calculate_metrics_properties(tm):
    logits = torch.tensor([[10.0, -10.0]])  # confident prediction
    entropy, max_prob, pcs, deepgini = tm.calculate_metrics(logits)

    # Confident → low entropy, max prob ~1, PCS ~1, DeepGini ~0
    assert entropy[0] < 0.1
    assert max_prob[0] > 0.9
    assert pcs[0] > 0.9
    assert deepgini[0] < 0.1


def test_compute_norm_inv_token_metrics_output_shapes(tm):
    logits = torch.tensor([[1.0, 2.0, 0.5],
                           [2.0, 1.0, -1.0]])
    metrics = tm.compute_norm_inv_token_metrics(logits)

    # Expect 4 lists: [entropy_norm, max_prob_inv, pcs_inv, deepgini_norm]
    assert len(metrics) == 4
    for m in metrics:
        assert isinstance(m, list)
        assert len(m) == 2  # 2 samples


def test_compute_norm_inv_token_metrics_ranges(tm):
    logits = torch.tensor([[0.0, 0.0, 0.0],
                           [5.0, -5.0, 0.0]])
    entropy, max_prob_inv, pcs_inv, deepgini = tm.compute_norm_inv_token_metrics(logits)

    # All metrics should be between 0 and 1
    for lst in [entropy, max_prob_inv, pcs_inv, deepgini]:
        assert all(0.0 <= v <= 1.0 for v in lst)


def test_clear_method(tm):
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    tm.calculate_metrics(logits)
    assert tm.shannon_entropy_list  # not empty

    tm.clear()
    assert tm.shannon_entropy_list == []
    assert tm.token_prob == []
    assert tm.pcs == []
    assert tm.deepgini == []
