#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 21/08/2025 11:21
# @Author  : Chengjie
# @File    : metrics_vla.py
# @Software: PyCharm


import numpy as np
import torch
from torch.nn.functional import softmax
import deepluq.utils as uncerUtils
from typing import List, Dict, Any


class TokenMetrics:
    def __init__(self):
        self.shannon_entropy_list = []
        self.token_prob = []
        self.pcs = []
        self.token_prob_inv = []
        self.pcs_inv = []
        self.deepgini = []

    def calculate_metrics(self, logits):
        self.clear()

        probs = softmax(logits, dim=1).detach().to(torch.float32).cpu().numpy()

        eps = 1e-15
        log_base = np.log(2)

        # Shannon Entropy
        entropy = -np.sum(probs * np.log(probs + eps), axis=1) / log_base
        self.shannon_entropy_list = [float(f"{v:.5f}") for v in entropy]

        # Max token probability
        max_probs = np.max(probs, axis=1)
        self.token_prob = [float(f"{v:.5f}") for v in max_probs]

        # PCS (max - second max)
        sorted_probs = -np.sort(-probs, axis=1)
        pcs = sorted_probs[:, 0] - sorted_probs[:, 1]
        self.pcs = [float(f"{v:.5f}") for v in pcs]

        # DeepGini
        deepgini = 1 - np.sum(probs ** 2, axis=1)
        self.deepgini = [float(f"{v:.5f}") for v in deepgini]

        return [self.shannon_entropy_list, self.token_prob, self.pcs, self.deepgini]

    def compute_norm_inv_token_metrics(self, logits):
        """
        Compute various token-level uncertainty and confidence metrics from model logits,
        normalize them to [0, 1], and invert selected metrics so that higher values
        consistently indicate greater uncertainty.

        Metrics computed:
        - Shannon Entropy (normalized): uncertainty measure normalized by log2(num_classes).
        - Max Token Probability (normalized and inverted): confidence of top predicted token,
          normalized and inverted so higher means less confidence.
        - PCS (Prediction Confidence Score) (inverted): difference between top two token probabilities,
          inverted so higher means more uncertainty.
        - DeepGini (normalized): uncertainty measure normalized by its max possible value.

        Args:
            logits (torch.Tensor): raw output logits from the model with shape (batch_size, num_classes).

        Returns:
            list: four lists of float values rounded to 5 decimals, corresponding to:
                  [shannon_entropy, max_token_prob_inverted, pcs_inverted, deepgini]
        """
        self.clear()

        probs = softmax(logits, dim=1).detach().to(torch.float32).cpu().numpy()

        eps = 1e-15
        log_base = np.log(2)
        num_classes = probs.shape[1]

        # ---------- Shannon Entropy ----------
        # Raw range: [0, log2(C)]
        # Normalized range: [0, 1] → 0: certain, 1: most uncertain (uniform distribution)
        entropy = -np.sum(probs * np.log(probs + eps), axis=1) / log_base
        entropy_norm = entropy / np.log2(num_classes)
        self.shannon_entropy_list = [float(f"{v:.5f}") for v in entropy_norm]

        # ---------- Max Token Probability ----------
        # Raw range: [1/C, 1] → 1: confident, 1/C: uncertain
        # Normalized range: [0, 1] → 0: confident, 1: uncertain (after inversion)
        max_probs = np.max(probs, axis=1)
        max_probs_norm = (max_probs - 1.0 / num_classes) / (1 - 1.0 / num_classes)
        max_probs_inv = 1.0 - max_probs_norm
        self.token_prob_inv = [float(f"{v:.5f}") for v in max_probs_inv]

        # ---------- PCS (Prediction Confidence Score) ----------
        # Raw range: [0, 1] → 1: confident, 0: ambiguous (top-2 equal)
        # Inverted range: [0, 1] → 0: confident, 1: ambiguous
        sorted_probs = -np.sort(-probs, axis=1)
        pcs = sorted_probs[:, 0] - sorted_probs[:, 1]
        pcs_inv = 1.0 - pcs
        self.pcs_inv = [float(f"{v:.5f}") for v in pcs_inv]

        # ---------- DeepGini ----------
        # Raw range: [0, 1 - 1/C] → 0: confident, max: uniform
        # Normalized range: [0, 1] → 0: confident, 1: most uncertain
        deepgini = 1 - np.sum(probs ** 2, axis=1)
        deepgini_norm = deepgini / (1 - 1.0 / num_classes)
        self.deepgini = [float(f"{v:.5f}") for v in deepgini_norm]

        return [self.shannon_entropy_list, self.token_prob_inv, self.pcs_inv, self.deepgini]

    def clear(self):
        self.shannon_entropy_list = []
        self.token_prob = []
        self.pcs = []
        self.deepgini = []


class OutputMetrics:
    """
    Compute various instability and variability metrics for robot actions and TCP positions.

    Author: Pablo Valle
    Time  : 05/22/2025
    """

    VARIABILITY = 4

    # --------------------------
    # Generic instability metrics
    # --------------------------
    @staticmethod
    def _action_array(actions: List[Dict[str, Any]]) -> np.ndarray:
        """
        Convert a list of action dicts to a NumPy array.

        Each action dict should contain:
        - "world_vector"
        - "rot_axangle"
        - "gripper"
        """
        return np.array([
            np.concatenate((
                np.array(d["world_vector"]),
                np.array(d["rot_axangle"]),
                np.array(d["gripper"])
            ))
            for d in actions
        ])

    @staticmethod
    def _compute_instability(arr: np.ndarray, order: int = 1, scale: float = 1.0) -> np.ndarray:
        """
        Compute instability metrics by taking successive differences.

        Args:
            arr (np.ndarray): Input array of shape (T, M).
            order (int): Number of differences to compute (1=position, 2=velocity, 3=acceleration).
            scale (float): Scaling factor for difference magnitude.

        Returns:
            np.ndarray: Instability per dimension (M,).
        """
        if arr.shape[0] < order + 1:
            raise ValueError(f"At least {order + 1} time steps required for order {order} instability.")

        delta = arr.copy()
        for _ in range(order):
            delta = np.diff(delta, axis=0)
        instability = np.sum(np.abs(delta) / scale, axis=0) / delta.shape[0]
        return instability

    # --------------------------
    # Action-based instability
    # --------------------------
    def compute_position_instability(self, actions: List[Dict[str, Any]]) -> np.ndarray:
        arr = self._action_array(actions)
        return self._compute_instability(arr, order=1, scale=1.0)

    def compute_velocity_instability(self, actions: List[Dict[str, Any]]) -> np.ndarray:
        arr = self._action_array(actions)
        return self._compute_instability(arr, order=2, scale=2.0)

    def compute_acceleration_instability(self, actions: List[Dict[str, Any]]) -> np.ndarray:
        arr = self._action_array(actions)
        return self._compute_instability(arr, order=3, scale=4.0)

    # --------------------------
    # TCP position instability
    # --------------------------
    @staticmethod
    def _tcp_array(poses: List[List[float]]) -> np.ndarray:
        """
        Extract TCP positions (x, y, z) from poses.
        """
        return np.array(poses)[:, :3]

    def compute_TCP_position_instability(self, poses: List[List[float]]) -> np.ndarray:
        arr = self._tcp_array(poses)
        return self._compute_instability(arr, order=1, scale=1.0)

    def compute_TCP_velocity_instability(self, poses: List[List[float]]) -> np.ndarray:
        arr = self._tcp_array(poses)
        return self._compute_instability(arr, order=2, scale=2.0)

    def compute_TCP_acceleration_instability(self, poses: List[List[float]]) -> np.ndarray:
        arr = self._tcp_array(poses)
        return self._compute_instability(arr, order=3, scale=4.0)

    def compute_TCP_jerk_instability_gradient(self, poses: List[List[float]]) -> np.ndarray:
        """
        Compute TCP jerk using numerical gradients and return jerk magnitude per time step.
        """
        pos_array = self._tcp_array(poses)
        dt = 1.0

        # Compute jerk along each axis
        jerk = np.gradient(np.gradient(np.gradient(pos_array, dt, axis=0), dt, axis=0), dt, axis=0)
        jerk_magnitude = np.linalg.norm(jerk, axis=1)
        return jerk_magnitude

    # --------------------------
    # Execution variability
    # --------------------------
    @staticmethod
    def compute_execution_variability(
        variability_models: List[Any],
        image: Any,
        action_space: Any,
        instruction: Any,
        obs: Dict[str, Any],
        model_name: str
    ) -> np.ndarray:
        """
        Compute variability across multiple models' actions.

        Returns:
            np.ndarray: Standard deviation of actions across models.
        """
        actions = []

        for model in variability_models:
            if "pi0" in model_name:
                raw_action, action = model.step(image, instruction, eef_pos=obs["agent"]["eef_pos"])
            elif "spatialvla" in model_name:
                raw_action, action = model.step(image, instruction)
            else:
                raw_action, action = model.step(image)

            action = uncerUtils.normalize_action(action, action_space)
            actions.append(action)

        world_vectors = np.array([d["world_vector"] for d in actions])
        rot_axangles = np.array([d["rot_axangle"] for d in actions])
        grippers = np.array([d["gripper"][0] for d in actions])

        # Compute per-dimension standard deviation
        variability = np.concatenate([
            np.std(world_vectors, axis=0),
            np.std(rot_axangles, axis=0),
            [np.std(grippers)]
        ])
        return variability
