#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import binom
import pandas as pd


def kill_count_img(miss_s, ghost_s):
    kill_miss_c = sum(1 for miss in miss_s if miss)
    kill_ghost_c = sum(1 for ghost in ghost_s if ghost)
    kill_miss_ghost_c = sum(1 for miss, ghost in zip(miss_s, ghost_s) if miss or ghost)
    return kill_miss_c, kill_ghost_c, kill_miss_ghost_c


def kill_rate_obj(match_s, miss_s, ghost_s):
    miss_rate = np.array([len(miss) / (len(match) + len(miss)) for match, miss in zip(match_s, miss_s)])
    ghost_rate = np.array([len(ghost) / (len(match) + len(miss) + len(ghost))
                           for match, miss, ghost in zip(match_s, miss_s, ghost_s)])
    miss_ghost_rate = np.array([(len(miss) + len(ghost)) / (len(match) + len(miss) + len(ghost))
                                for match, miss, ghost in zip(match_s, miss_s, ghost_s)])

    return {'[obj_level] kill_rate_miss_mean': miss_rate.mean(),
            '[obj_level] kill_rate_ghost_mean': ghost_rate.mean(),
            '[obj_level] kill_rate_miss_ghost_mean': miss_ghost_rate.mean(),
            '[obj_level] kill_rate_miss_std': miss_rate.std(),
            '[obj_level] kill_rate_ghost_std': ghost_rate.std(),
            '[obj_level] kill_rate_miss_ghost_std': miss_ghost_rate.std()}


def un_ms_calcu(match_metrics, miss_metrics, ghost_metrics):
    # Keys that require |Original - Mutation| calculation
    diff_keys = ["vr", "ie", "mi", "var", "ps"]

    # 1. Process Match Metrics
    for m in match_metrics:
        # Calculate IoU separately (formula is unique)
        m["iou_ms"] = 1 - m["avg_iou"]

        # Calculate absolute differences for the rest
        for k in diff_keys:
            m[f"{k}_ms"] = abs(m[f"{k}_orig"] - m[f"{k}_mut"])

    # 2. Process Miss and Ghost Metrics (Logic is identical: 1 - value)
    # We loop through both lists to avoid writing the code twice
    for metrics_list, suffix in [(miss_metrics, "miss"), (ghost_metrics, "ghost")]:
        for m in metrics_list:
            for k in diff_keys:
                m[f"{k}_ms"] = 1 - m[f"{k}_{suffix}"]

    return match_metrics, miss_metrics, ghost_metrics


def check_kill_binomial_test(success, n, p_null, alpha=0.05):
    """
    Calculates Cohen's h, Power, and P-value for a one-sided (right-tailed) Binomial Test.

    Parameters:
    -----------
    n : int
        Sample size (e.g., 10)
    success : int
        Number of observed successes (e.g., 3)
    p_null : float
        The null hypothesis proportion/noise level (e.g., 0.05)
    alpha : float, optional
        Significance level, default is 0.05

    Returns:
    --------
    dict
        A dictionary containing the calculated statistics.
    """

    # 1. Calculate Observed Proportion (p1)
    p1 = success / n

    # 2. Calculate Cohen's h
    # Formula: h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p_null))
    # Note: The result includes the sign. Positive means p1 > p_null.
    h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p_null))

    # 3. Calculate P-value
    # For a right-tailed test (is p1 significantly greater than p_null?)
    # P-value = Probability of getting 'success' or more, given p_null.
    # binom.sf(k, n, p) is P(X > k), so we use success-1 to get P(X >= success)
    p_value = binom.sf(success - 1, n, p_null)

    # 4. Calculate Power
    # Step A: Find the Critical Value (k_crit)
    # The smallest number of successes needed to reject the null at level alpha.
    k_crit = binom.ppf(1 - alpha, n, p_null)

    # Adjust for discrete nature: ensure P(X >= k_crit) <= alpha
    if binom.sf(k_crit - 1, n, p_null) > alpha:
        k_crit += 1

    # Step B: Calculate Power
    # Power = Probability of falling in the rejection region (>= k_crit)
    # assuming the observed proportion (p1) is the True proportion.
    if k_crit <= n:
        power = binom.sf(k_crit - 1, n, p1)
    else:
        power = 0.0  # It's impossible to reject the null with this n and alpha

    return {
        "sample_size": n,
        "observed_success": success,
        "observed_p1": p1,
        "null_p0": p_null,
        "cohens_h": h,
        "p_value": p_value,
        "critical_threshold": int(k_crit),
        "power": power
    }


def iskill_img(miss_s, ghost_s):
    kill_miss_c, kill_ghost_c, kill_miss_ghost_c = kill_count_img(miss_s, ghost_s)

    iskill_miss_stats = check_kill_binomial_test(kill_miss_c, 10, 0.01)
    iskill_ghost_stats = check_kill_binomial_test(kill_ghost_c, 10, 0.01)
    iskill_miss_ghost_stats = check_kill_binomial_test(kill_miss_ghost_c, 10, 0.01)

    return iskill_miss_stats, iskill_ghost_stats, iskill_miss_ghost_stats


def calcu_mean(results, cols_to_mean):
    means_dict = {f"{metrics}": 0 for metrics in cols_to_mean}
    if results:
        df = pd.DataFrame(results)
        # Calculate mean for specific columns
        means = df[cols_to_mean].mean()
        # If you need it back as a dictionary:
        means_dict = means.to_dict()

    return means_dict


def compute_iou(boxA, boxB):
    """Standard IoU calculation with safe checks."""
    # Compute intersection
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    interArea = inter_width * inter_height

    # Compute area of each box
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    # Avoid division by zero
    denom = boxAArea + boxBArea - interArea
    if denom == 0:
        return 0.0

    return interArea / denom


def spatial_aware(rate, metric, penalty):
    return (rate * metric) + ((1.0 - rate) * penalty)


def identify_matches_misses_ghosts(orig_objs, mut_objs, iou_threshold=0.5):
    """
    Performs Hungarian Matching and filters by IoU threshold.
    """

    orig_objs = list(orig_objs.values())
    mut_objs = list(mut_objs.values())

    n_org = len(orig_objs)
    n_mut = len(mut_objs)

    matches = []
    misses = []
    ghosts = []

    # Edge cases
    if n_org == 0: return [], [], mut_objs
    if n_mut == 0: return [], orig_objs, []

    # Build Cost Matrix (Cost = 1 - IoU)
    cost_matrix = np.ones((n_org, n_mut))
    iou_matrix = np.zeros((n_org, n_mut))

    for i, o in enumerate(orig_objs):
        for j, m in enumerate(mut_objs):
            iou = compute_iou(o['box'], m['box'])
            iou_matrix[i, j] = iou
            cost_matrix[i, j] = 1.0 - iou

    # Hungarian Algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_org_indices = set()
    matched_mut_indices = set()

    # Filter by Threshold
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] > iou_threshold and orig_objs[r]["label"] == mut_objs[c]["label"]:
            matches.append((orig_objs[r], mut_objs[c]))
            matched_org_indices.add(r)
            matched_mut_indices.add(c)

    # Identify Leftovers
    for i in range(n_org):
        if i not in matched_org_indices: misses.append(orig_objs[i])

    for j in range(n_mut):
        if j not in matched_mut_indices: ghosts.append(mut_objs[j])

    return matches, misses, ghosts


def convert_detection_output(pred, default_score=1.0):
    """Convert a torchvision-style detection output (boxes/labels tensors) into
    the ``{"label_i": {"box", "label", "score", "logit"}}`` format expected by
    :func:`identify_matches_misses_ghosts`."""
    boxes = pred['boxes'].tolist()
    labels = pred['labels'].tolist()

    output = {}

    for i, (box, label) in enumerate(zip(boxes, labels)):
        # Build a one-hot-like logit placeholder
        num_classes = 3  # change if needed
        logit = [0.0] * num_classes
        logit[label] = default_score

        output[f"label_{i}"] = {
            "box": box,
            "label": label,
            "score": default_score,
            "logit": logit
        }

    return output


def yolo_to_absolute(yolo_labels, img_w=1280, img_h=736):
    """
    Convert YOLO normalized bounding boxes to absolute pixel coordinates and compute their areas.

    Parameters:
        yolo_labels (list of lists): Each element is [label, x_center, y_center, width, height],
                                     where coordinates are normalized between 0 and 1.
        img_w (int): Image width in pixels.
        img_h (int): Image height in pixels.

    Returns:
        dict:
        {
            'boxes': torch.Tensor of shape [N, 4] with absolute coordinates [xmin, ymin, xmax, ymax],
            'labels': torch.Tensor of shape [N] with class IDs,
            'areas': torch.Tensor of shape [N] with the area of each bounding box in pixels.
        }
    """
    absolute_boxes = []
    labels = []
    areas = []

    for l, x_c, y_c, w, h in yolo_labels:
        x_c_abs = x_c * img_w
        y_c_abs = y_c * img_h
        w_abs = w * img_w
        h_abs = h * img_h

        xmin = x_c_abs - w_abs / 2
        ymin = y_c_abs - h_abs / 2
        xmax = x_c_abs + w_abs / 2
        ymax = y_c_abs + h_abs / 2

        labels.append(l)
        absolute_boxes.append([xmin, ymin, xmax, ymax])
        areas.append((xmax - xmin) * (ymax - ymin))

    return {
        'boxes': torch.tensor(absolute_boxes, dtype=torch.float32),
        'labels': torch.tensor(labels, dtype=torch.long),
        'areas': torch.tensor(areas, dtype=torch.float32)
    }
