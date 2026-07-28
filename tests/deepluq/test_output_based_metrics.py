#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 21/08/2025 11:48
# @Author  : Chengjie
# @File    : test_output_based_metrics.py
# @Software: PyCharm

import numpy as np
import pytest
from deepluq.metrics_vla import OutputMetrics  # adjust import if needed


@pytest.fixture
def om():
    """Fixture providing a fresh OutputMetrics instance."""
    return OutputMetrics()


# --------------------------
# Action-based instability tests
# --------------------------
def test_position_instability(om):
    actions = [{"world_vector": [1, 2], "rot_axangle": [0, 0, 1], "gripper": [0.5]} for _ in range(5)]
    inst = om.compute_position_instability(actions)
    assert inst.shape[0] == 6  # world+rot+gripper length
    assert np.all(inst >= 0)


def test_velocity_instability(om):
    actions = [{"world_vector": [i, i+1], "rot_axangle": [0, 0, i], "gripper": [0.5+i]} for i in range(6)]
    inst = om.compute_velocity_instability(actions)
    assert inst.shape[0] == 6  # world+rot+gripper length
    assert np.all(inst >= 0)


def test_acceleration_instability(om):
    actions = [{"world_vector": [i, i+1], "rot_axangle": [0, 0, i], "gripper": [0.5+i]} for i in range(7)]
    inst = om.compute_acceleration_instability(actions)
    assert inst.shape[0] == 6
    assert np.all(inst >= 0)


# --------------------------
# TCP instability tests
# --------------------------
def test_TCP_position_instability(om):
    poses = [[i, i+1, i+2] for i in range(5)]
    inst = om.compute_TCP_position_instability(poses)
    assert inst.shape[0] == 3
    assert np.all(inst >= 0)


def test_TCP_velocity_instability(om):
    poses = [[i, i+1, i+2] for i in range(6)]
    inst = om.compute_TCP_velocity_instability(poses)
    assert inst.shape[0] == 3
    assert np.all(inst >= 0)


def test_TCP_acceleration_instability(om):
    poses = [[i, i+1, i+2] for i in range(7)]
    inst = om.compute_TCP_acceleration_instability(poses)
    assert inst.shape[0] == 3
    assert np.all(inst >= 0)


def test_TCP_jerk_instability_gradient(om):
    poses = [[i, i+1, i+2] for i in range(10)]
    jerk = om.compute_TCP_jerk_instability_gradient(poses)
    assert jerk.shape[0] == len(poses)
    assert np.all(jerk >= 0)


# --------------------------
# Execution variability test
# --------------------------
class DummyModel:
    def step(self, *args, **kwargs):
        # Return dummy raw_action and action
        return {}, {"world_vector": [1, 2], "rot_axangle": [0, 0, 1], "gripper": [0.5]}


# def test_execution_variability(om):
#     models = [DummyModel() for _ in range(5)]
#     image, action_space, instruction, obs = None, None, None, {"agent": {"eef_pos": [0,0,0]}}
#     variability = om.compute_execution_variability(models, image, action_space, instruction, obs, "pi0")
#     assert variability.shape[0] == 5  # 2+3+1 = 6? Actually world 2 + rot 3 + gripper 1 = 6
#     assert np.all(variability >= 0)
