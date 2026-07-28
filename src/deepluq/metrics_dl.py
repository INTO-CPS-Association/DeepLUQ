#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/03/2024 21:45
# @Author  : Chengjie
# @File    : metrics.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import torch
from scipy.spatial import ConvexHull


class DLMetrics:
    """
    A class to compute various Uncertainty Quantification (UQ) metrics for Deep Learning,
    including variation ratio, entropy, mutual information, total variance,
    and prediction surface using convex hulls.
    """

    def __init__(self):
        # Initialize metrics with default values
        self.variation_ratio = 0.0
        self.shannon_entropy = 0.0
        self.mutual_information = 0.0
        self.total_var_center_point = 0.0
        self.total_var_bounding_box = 0.0
        self.prediction_surface = -1.0

        # Store geometric results
        self.hull = []  # List of convex hulls
        self.box = []   # List of bounding boxes (if needed later)

    # -------------------------------
    # Classification uncertainty metrics
    # -------------------------------
    def cal_vr(self, events):
        """
        Compute the Variation Ratio (VR).
        Measures the proportion of non-modal class predictions.

        Parameters:
            events (array-like): Model outputs or predictions.

        Returns:
            float: Variation ratio.
        """
        y = torch.argmax(torch.Tensor(events), dim=1).numpy()
        _, counts = np.unique(y, return_counts=True)
        self.variation_ratio = 1 - np.max(counts) / np.sum(counts)
        return self.variation_ratio

    def calcu_entropy(self, events, eps=1e-15, base=2):
        """
        Compute Shannon entropy of probabilities.

        Parameters:
            events (array-like): Probability distribution.
            eps (float): Small constant to avoid log(0).
            base (int): Logarithm base.

        Returns:
            float: Shannon entropy (rounded to 5 decimals).
        """
        self.shannon_entropy = round(
            -np.sum([p * (np.log(p + eps) / np.log(base)) for p in events]), 5
        )
        return 0 if self.shannon_entropy < 0 else self.shannon_entropy

    def calcu_mi(self, events, eps=1e-15, base=2):
        """
        Compute Mutual Information (MI) between predictions.

        Parameters:
            events (array-like): Model probability outputs.
            eps (float): Small constant to avoid log(0).
            base (int): Logarithm base.

        Returns:
            float: Mutual information.
        """

        def entropy_component(e):
            return np.mean([
                np.sum([p * (np.log(p + eps) / np.log(base)) for p in s])
                for s in e
            ])

        avg_probs = np.mean(np.transpose(events), axis=1)
        self.mutual_information = self.calcu_entropy(avg_probs) + entropy_component(events)
        return 0 if self.mutual_information < 0 else self.mutual_information

    # -------------------------------
    # Total variance metrics
    # -------------------------------
    def calcu_tv(self, matrix, tag):
        """
        Compute total variance of a multi-dimensional matrix using covariance.

        Parameters:
            matrix (array-like): Input data matrix.
            tag (str): Either 'bounding_box' or 'center_point'.

        Returns:
            float: Total variance.
        """
        cov_matrix = np.cov(np.array(matrix).T)
        trace_val = np.trace(cov_matrix)

        if tag == "bounding_box":
            self.total_var_bounding_box = trace_val
            return self.total_var_bounding_box
        elif tag == "center_point":
            self.total_var_center_point = trace_val
            return self.total_var_center_point
        else:
            raise ValueError("tag must be either 'bounding_box' or 'center_point'")

    # -------------------------------
    # Mutual Information between variables
    # -------------------------------
    def calcu_mutual_information(self, X, Y, Z):
        """
        Compute mutual information between three discrete random variables X, Y, and Z.

        Reference:
            http://www.scholarpedia.org/article/Mutual_information

        Parameters:
            X, Y, Z (array-like): Discrete random variables of shape (n_samples,).

        Returns:
            float: Mutual information.
        """
        unique_X, unique_Y, unique_Z = np.unique(X), np.unique(Y), np.unique(Z)

        # Joint probability distribution
        joint_probs = np.zeros((len(unique_X), len(unique_Y), len(unique_Z)))
        for i, x in enumerate(unique_X):
            for j, y in enumerate(unique_Y):
                for k, z in enumerate(unique_Z):
                    joint_probs[i, j, k] = np.mean((X == x) & (Y == y) & (Z == z))

        # Marginals
        px = np.sum(joint_probs, axis=(1, 2))
        py = np.sum(joint_probs, axis=(0, 2))
        pz = np.sum(joint_probs, axis=(0, 1))

        # Mutual information
        mutual_info = 0.0
        for i, _ in enumerate(unique_X):
            for j, _ in enumerate(unique_Y):
                for k, _ in enumerate(unique_Z):
                    if joint_probs[i, j, k] > 0.0:
                        mutual_info += joint_probs[i, j, k] * np.log2(
                            joint_probs[i, j, k] / (px[i] * py[j] * pz[k])
                        )

        self.mutual_information = mutual_info
        return 0 if self.mutual_information < 0 else self.mutual_information

    # -------------------------------
    # Geometric uncertainty metrics
    # -------------------------------
    def calcu_prediction_surface(self, boxes):
        """
        Compute prediction surface by calculating convex hull areas
        from bounding box corners.

        Parameters:
            boxes (array-like): List of bounding boxes [x1, y1, x2, y2].

        Returns:
            float: Prediction surface area (sum of convex hulls).
        """
        self.prediction_surface = -1
        self.hull.clear()

        cluster_df = pd.DataFrame(boxes, columns=["x1", "y1", "x2", "y2"])

        if cluster_df.shape[0] > 2:
            print(cluster_df.shape[0])
            sf_tmp = 0
            try:
                for corner_set in [["x1", "y1"], ["x2", "y1"], ["x1", "y2"], ["x2", "y2"]]:
                    center_data = cluster_df[corner_set].values

                    # Deduplicate points
                    unique_points = np.unique(center_data, axis=0)

                    # Need at least 3 non-collinear points for a 2D hull
                    if len(unique_points) < 3:
                        sf_tmp += 0
                        continue

                    # Check for collinearity: all points on the same line
                    # Using the rank of the centered matrix — rank < 2 means collinear
                    centered = unique_points - unique_points.mean(axis=0)
                    if np.linalg.matrix_rank(centered) < 2:
                        sf_tmp += 0
                        continue

                    hull = ConvexHull(unique_points)
                    self.hull.append(hull)
                    sf_tmp += hull.area

                self.prediction_surface = sf_tmp

            except Exception:
                self.prediction_surface = 0

        return self.prediction_surface
