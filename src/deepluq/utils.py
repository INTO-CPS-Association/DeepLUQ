#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 07.05.2024 11:38
# @Author  : Chengjie
# @File    : utils.py
# @Software: PyCharm
import copy

from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from hdbscan import HDBSCAN
from ensemble_boxes import weighted_boxes_fusion


def compute_iou(box1, box2):
    """Calculates Intersection over Union (IoU) for two normalized boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def wbf_clustering(predictions_dict, iou_thr=0.5, skip_box_thr=0.01):
    """
    Applies WBF and groups the original input boxes into their respective clusters
    alongside the final merged detection.
    """
    orig_keys = list(predictions_dict.keys())
    orig_boxes_n = []
    orig_boxes_abs = []
    orig_scores = []
    orig_labels = []
    orig_logits = []

    img_width, img_height = None, None

    # 1. Parse dictionary
    for key in orig_keys:
        pred = predictions_dict[key]
        norm_box = pred['box_n']
        abs_box = pred['box']

        if img_width is None and img_height is None:
            img_width = abs_box[2] / norm_box[2]
            img_height = abs_box[3] / norm_box[3]

        # Clamp normalized coords
        x_min = max(0.0, min(1.0, norm_box[0]))
        y_min = max(0.0, min(1.0, norm_box[1]))
        x_max = max(0.0, min(1.0, norm_box[2]))
        y_max = max(0.0, min(1.0, norm_box[3]))

        orig_boxes_n.append([x_min, y_min, x_max, y_max])
        orig_boxes_abs.append(abs_box)
        orig_scores.append(pred['score'])
        orig_labels.append(pred['label'])
        orig_logits.append(pred.get('logit', []))

    # 2. Run Weighted Boxes Fusion
    merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
        [orig_boxes_n],
        [orig_scores],
        [orig_labels],
        weights=[1],
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr
    )

    # 3. Match original boxes to the fused boxes to build clusters
    cluster_assignments = {i: [] for i in range(len(merged_boxes))}

    # Assign each original box to the fused box with the highest IoU
    for j, o_box in enumerate(orig_boxes_n):
        best_iou = 0
        best_fused_idx = -1

        for i, f_box in enumerate(merged_boxes):
            if orig_labels[j] == int(merged_labels[i]):
                iou = compute_iou(f_box, o_box)
                if iou > best_iou:
                    best_iou = iou
                    best_fused_idx = i

        # If it overlaps enough, add it to that cluster
        if best_fused_idx != -1 and best_iou >= iou_thr:
            cluster_assignments[best_fused_idx].append(j)

    # 4. Format the final output
    final_output = {}

    for i, f_box in enumerate(merged_boxes):
        cluster_id = f"cluster_{i}"  # Using 'cluster_i' instead of 'label_i' to distinguish from class labels
        matched_indices = cluster_assignments[i]

        # Gather all original data for this cluster
        cluster_boxes_abs = [orig_boxes_abs[j] for j in matched_indices]
        cluster_boxes_n = [orig_boxes_n[j] for j in matched_indices]
        cluster_scores = [orig_scores[j] for j in matched_indices]
        cluster_logits = [orig_logits[j] for j in matched_indices]
        cluster_lbls = [orig_labels[j] for j in matched_indices]

        # Calculate average logit for the merged detection
        avg_logits = []
        if cluster_logits and all(len(l) > 0 for l in cluster_logits):
            avg_logits = [sum(col) / len(col) for col in zip(*cluster_logits)]

        fused_detection = {
            'box_n': [float(f_box[0]), float(f_box[1]), float(f_box[2]), float(f_box[3])],
            'box': [
                float(f_box[0] * img_width),
                float(f_box[1] * img_height),
                float(f_box[2] * img_width),
                float(f_box[3] * img_height)
            ],
            'score': float(merged_scores[i]),
            'label': int(merged_labels[i]),
            'logit': avg_logits
        }

        final_output[cluster_id] = {
            'box': cluster_boxes_abs,
            'box_n': cluster_boxes_n,
            'score': cluster_scores,
            'label': cluster_lbls,
            'logit': cluster_logits,
            'detection': fused_detection
        }

    return final_output


# https://stackoverflow.com/questions/15050389/estimating-choosing-optimal-hyperparameters-for-dbscan
# https://www.kaggle.com/discussions/questions-and-answers/166388


class DBSCANCluster:
    def __init__(self, x, eps=8.5, min_samples=8):
        # self.cluster = DBSCAN(eps=eps, min_samples=min_samples).fit(x)
        # try:
        self.cluster = HDBSCAN(min_cluster_size=3).fit(x)
        # except:
        #    print('except')
        #    self.cluster = HDBSCAN(min_cluster_size=2).fit(x)

        self.mc_locations = np.c_[x, self.cluster.labels_.ravel()]
        self.mc_locations_df = pd.DataFrame(
            data=self.mc_locations,
            columns=["x1", "y1", "x2", "y2", "center_x", "center_y", "label"],
        )
        self.cluster_labels = np.unique(
            self.mc_locations[:, len(self.mc_locations[0]) - 1]
        )

    def cluster_preds(self, preds):
        pred_id = 0
        preds_new = {}

        for cluster_label in self.cluster_labels:
            cluster_df = self.mc_locations_df.query("label == " + str(cluster_label))
            boxs = cluster_df[["x1", "y1", "x2", "y2"]].values
            t = 0
            for key in preds.keys():
                if preds[key]["box"] in boxs and t == 0:
                    preds_new.update(
                        {
                            "label_{}".format(pred_id): {
                                "box": [preds[key]["box"]],
                                "label": [preds[key]["label"]],
                                "score": [preds[key]["score"]],
                                "logit": [preds[key]["logit"]],
                                # 'center_point': [preds[key]['center_point']]
                            }
                        }
                    )
                    t = 1
                elif preds[key]["box"] in boxs and t != 0:
                    preds_new["label_{}".format(pred_id)]["box"].append(
                        preds[key]["box"]
                    )
                    preds_new["label_{}".format(pred_id)]["label"].append(
                        preds[key]["label"]
                    )
                    preds_new["label_{}".format(pred_id)]["score"].append(
                        preds[key]["score"]
                    )
                    preds_new["label_{}".format(pred_id)]["logit"].append(
                        preds[key]["logit"]
                    )

            pred_id += 1

        return preds_new


def cluster(mc_locations):
    clustering = DBSCAN(eps=100, min_samples=2).fit(mc_locations)
    mc_locations = np.c_[mc_locations, clustering.labels_.ravel()]

    mc_locations_df = pd.DataFrame(
        data=mc_locations, columns=["x1", "y1", "x2", "y2", "label"]
    )

    cluster_labels = np.unique(mc_locations[:, len(mc_locations[0]) - 1])
    total_cluster_surface = 0.0
    for cluster_label in cluster_labels:
        sf_tmp = 0
        cluster_df = mc_locations_df.query("label == " + str(cluster_label))
        if cluster_df.shape[0] > 2:
            center_data = cluster_df[["x1", "y1"]].values
            hull = ConvexHull(center_data)
            total_cluster_surface += hull.area
            sf_tmp += hull.area

            center_data = cluster_df[["x2", "y1"]].values
            hull = ConvexHull(center_data)
            total_cluster_surface += hull.area
            sf_tmp += hull.area

            center_data = cluster_df[["x1", "y2"]].values
            hull = ConvexHull(center_data)
            total_cluster_surface += hull.area

            center_data = cluster_df[["x2", "y2"]].values
            hull = ConvexHull(center_data)
            total_cluster_surface += hull.area
            sf_tmp += hull.area
        # print(sf_tmp)
        total_cluster_surface / mc_locations.shape[0]

    # print(total_cluster_surface, avg_surface)


# def dbscan_cluster(mc_locations):


def get_kdist_plot(X=None, k=None, radius_nbrs=1.0):
    nbrs = NearestNeighbors(n_neighbors=k, radius=radius_nbrs).fit(X)

    # For each point, compute distances to its k-nearest neighbors
    distances, indices = nbrs.kneighbors(X)

    distances = np.sort(distances, axis=0)
    distances = distances[:, k - 1]

    # Plot the sorted K-nearest neighbor distance for each point in the dataset
    plt.figure(figsize=(8, 8))
    plt.plot(distances)
    plt.xlabel("Points/Objects in the dataset", fontsize=12)
    plt.ylabel("Sorted {}-nearest neighbor distance".format(k), fontsize=12)
    plt.grid(True, linestyle="--", color="black", alpha=0.4)
    plt.show()
    plt.close()


def normalize_action(action, normalization_values):
    new_action = copy.deepcopy(action)
    low = np.array(normalization_values.low)
    high = np.array(normalization_values.high)

    action_raw = np.concatenate([
        np.array(action["world_vector"]),
        np.array(action["rot_axangle"]),
        np.array(action["gripper"])
    ])
    length_world = len(action["world_vector"])
    length_angles = len(action["world_vector"]) + len(action["rot_axangle"])
    lenth_gripper = len(action["world_vector"]) + len(action["rot_axangle"]) + len(action["gripper"])

    normalized_action = (action_raw - low) / (high - low)
    normalized_action = np.clip(normalized_action, 0.0, 1.0)

    new_action['world_vector'] = normalized_action[:length_world]
    new_action["rot_axangle"] = normalized_action[length_world:length_angles]
    new_action["gripper"] = normalized_action[length_angles:lenth_gripper]  # assuming 1-dim gripper

    return new_action


def action_uncertainty(action, mutated_action):
    world_vectors = np.array([
        action['world_vector'],
        mutated_action['world_vector']
    ])

    rot_axangles = np.array([
        action['rot_axangle'],
        mutated_action['rot_axangle']
    ])

    grippers = np.array([
        action['gripper'],
        mutated_action['gripper']
    ])

    metamorphic = np.concatenate([
        np.std(world_vectors, axis=0),
        np.std(rot_axangles, axis=0),
        [np.std(grippers)]
    ])

    return metamorphic
