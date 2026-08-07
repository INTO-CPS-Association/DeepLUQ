#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Uncertainty-Aware Mutation Score (UA-MS) calculation.

Consumes the per-repetition detection outputs (``prediction_{t}.json``) produced
by running a model under MC-Dropout / MC-DropBlock mutation (see
``docs/uq4ma.md`` for how those mutants are generated) and computes the
Uncertainty-Aware Mutation Score between an original model and each of its
mutants, at both the image level and the object level.
"""
import pickle
from pathlib import Path

import numpy as np
import json

import pandas as pd
from deepluq import metrics_dl
from sklearn.cluster import DBSCAN

from .ms import compute_iou, identify_matches_misses_ghosts, spatial_aware, un_ms_calcu, \
    iskill_img, kill_rate_obj, calcu_mean
from .helper import init_metrics, update_metrics

uq = metrics_dl.DLMetrics()


def process_match_metrics(repetitions):
    """
    Calculates metrics for Spatial (Box), Classification (Label), and Probabilistic (Entropy) stability.
    """

    # --- Step 1: Registry Building (Anchoring) ---
    unique_orig_map = {}
    next_id = 0
    orig_data_reference = {}

    for run in repetitions:
        for pair in run:
            orig_obj, mut_obj = pair
            # Use Box coordinates as unique hash
            box_sig = tuple(np.round(orig_obj['box'], 4))

            if box_sig not in unique_orig_map:
                unique_orig_map[box_sig] = next_id
                orig_data_reference[next_id] = orig_obj
                next_id += 1

    # --- Step 2: Aggregation ---
    aggregated = {
        i: {
            'mut_logits': [],
            'mut_boxes': [],
            'ious': [],
        }
        for i in range(next_id)
    }

    for run in repetitions:
        for pair in run:
            orig_obj, mut_obj = pair

            box_sig = tuple(np.round(orig_obj['box'], 4))
            obj_id = unique_orig_map[box_sig]

            # 1. Store Logits & Boxes
            aggregated[obj_id]['mut_logits'].append(mut_obj['logit'])
            aggregated[obj_id]['mut_boxes'].append(mut_obj['box'])

            # 2. Store IoU
            iou = compute_iou(orig_obj['box'], mut_obj['box'])
            aggregated[obj_id]['ious'].append(iou)

    # --- Step 3: Calculation ---
    results = []
    total_runs = len(repetitions)

    for obj_id, data in aggregated.items():
        orig_ref = orig_data_reference[obj_id]

        # A. Detection Rate
        count_detected = len(data['mut_logits'])
        detection_rate = count_detected / total_runs

        # B. Original Uncertainty
        orig_logits = np.array(orig_ref['logit'])
        orig_probs = orig_logits / orig_logits.sum()

        ie_orig = uq.calcu_entropy(np.mean(np.array([orig_probs for _ in range(10)]), axis=0))
        vr_orig = uq.cal_vr(np.array([orig_probs for _ in range(10)]))
        mi_orig = uq.calcu_mi(np.array([orig_probs for _ in range(10)]))
        ps_orig, var_orig = 0.0, 0.0

        ie_orig = ie_orig / (ie_orig + 1)
        vr_orig = vr_orig / (vr_orig + 1)
        mi_orig = mi_orig / (mi_orig + 1)

        # C. Calculate Found Stats
        if count_detected > 0:
            # 1. Entropy
            mut_stack = np.array(data['mut_logits'])
            mut_probs = mut_stack / mut_stack.sum(axis=1, keepdims=True)

            vr_mut = uq.cal_vr(mut_probs)
            ie_mut = uq.calcu_entropy(np.mean(mut_probs, axis=0))
            mi_mut = uq.calcu_mi(mut_probs)

            # 2. Box Variance
            box_stack = np.array(data['mut_boxes'])

            var_mut = uq.calcu_tv(box_stack, tag='bounding_box') if count_detected > 1 else 0.0

            ps_mut = uq.calcu_prediction_surface(box_stack) if count_detected > 2 else 0.0

            # 3. Spatial Costs (IoU)
            iou_list = np.array(data['ious'])
            avg_iou = np.mean(iou_list)

        else:
            # Defaults for Total Miss
            vr_mut, ie_mut, mi_mut = 1.0, 1.0, 1.0
            var_mut, ps_mut = 1.0, 1.0
            avg_iou = 0.0

        vr_mut, ie_mut, mi_mut = spatial_aware(detection_rate, vr_mut / (vr_mut + 1), 1), \
            spatial_aware(detection_rate, ie_mut / (ie_mut + 1), 1), \
            spatial_aware(detection_rate, mi_mut / (mi_mut + 1), 1)
        var_mut, ps_mut = spatial_aware(detection_rate, var_mut / (var_mut + 1), 1), \
            spatial_aware(detection_rate, ps_mut / (ps_mut + 1), 1)

        avg_iou = spatial_aware(detection_rate, avg_iou, 0)

        results.append({
            "id": obj_id,
            "label": orig_ref['label'],
            "match_rate": detection_rate,

            "vr_orig": vr_orig,
            "ie_orig": ie_orig,
            "mi_orig": mi_orig,
            'var_orig': var_orig,
            'ps_orig': ps_orig,

            "vr_mut": vr_mut,
            "ie_mut": ie_mut,
            "mi_mut": mi_mut,
            'var_mut': var_mut,
            'ps_mut': ps_mut,

            # Raw Stats
            "avg_iou": avg_iou,
        })

    return results


def process_missing_set(repetitions):
    """
    Analyzes the 'Missing Set' to determine if misses are due to
    instability (flickering) or blindness, weighted by original confidence.
    """

    total_runs = len(repetitions)

    # --- Step 1: Registry Building ---
    # We define a "Unique Missed Object" by its Original Bounding Box
    unique_miss_map = {}
    next_id = 0

    # We need to track:
    # 1. The original object data (reference)
    # 2. How many times it was missed
    miss_tracker = {}

    for run_idx, run_list in enumerate(repetitions):
        for orig_obj in run_list:

            # Anchor Key
            box_sig = tuple(np.round(orig_obj['box'], 4))

            if box_sig not in unique_miss_map:
                unique_miss_map[box_sig] = next_id
                miss_tracker[next_id] = {
                    'orig_obj': orig_obj,
                    'miss_count': 0
                }
                next_id += 1

            # Increment count for this ID
            obj_id = unique_miss_map[box_sig]
            miss_tracker[obj_id]['miss_count'] += 1

    # --- Step 2: Calculation ---
    results = []

    for obj_id, data in miss_tracker.items():
        orig_obj = data['orig_obj']
        count = data['miss_count']

        # A. Miss Rate (Frequency)
        miss_rate = count / total_runs

        # B. Original Uncertainty
        orig_logits = np.array(orig_obj['logit'])
        orig_probs = orig_logits / orig_logits.sum()

        ie_miss = uq.calcu_entropy(np.mean(np.array([orig_probs for _ in range(10)]), axis=0))
        vr_miss = uq.cal_vr(np.array([orig_probs for _ in range(10)]))
        mi_miss = uq.calcu_mi(np.array([orig_probs for _ in range(10)]))
        ps_miss, var_miss = 0.0, 0.0

        ie_miss, vr_miss, mi_miss = spatial_aware(miss_rate, ie_miss / (ie_miss + 1), 1), \
            spatial_aware(miss_rate, vr_miss / (vr_miss + 1), 1), \
            spatial_aware(miss_rate, mi_miss / (mi_miss + 1), 1)
        ps_miss, var_miss = spatial_aware(miss_rate, ps_miss / (ps_miss + 1), 1), \
            spatial_aware(miss_rate, var_miss / (var_miss + 1), 1)

        results.append({
            "id": obj_id,
            "label": orig_obj['label'],
            "miss_rate": miss_rate,

            "ie_miss": ie_miss,
            "vr_miss": vr_miss,
            "mi_miss": mi_miss,
            "ps_miss": ps_miss,
            "var_miss": var_miss
        })

    return results


def process_ghost_set_dbscan(repetitions, iou_threshold=0.5):
    """
    Clusters ghost detections using DBSCAN with (1 - IoU) distance metric.
    """
    total_runs = len(repetitions)

    # --- Step 1: Flatten Data ---
    # We need a flat list of ALL ghost boxes from ALL runs to feed into clustering
    all_ghosts = []

    # We maintain a mapping to know which run each ghost came from
    for run_idx, run_list in enumerate(repetitions):
        for ghost in run_list:
            # Add metadata for tracking
            ghost_entry = ghost.copy()
            ghost_entry['run_idx'] = run_idx
            all_ghosts.append(ghost_entry)

    n_ghosts = len(all_ghosts)

    if n_ghosts == 0:
        return []

    # --- Step 2: Precompute Distance Matrix ---
    # Distance = 1.0 - IoU
    # 0.0 distance means Perfect Overlap
    # 1.0 distance means No Overlap

    distance_matrix = np.zeros((n_ghosts, n_ghosts))

    for i in range(n_ghosts):
        for j in range(i, n_ghosts):  # Optimization: Fill upper triangle
            if i == j:
                dist = 0.0
            else:
                iou = compute_iou(all_ghosts[i]['box'], all_ghosts[j]['box'])
                dist = 1.0 - iou

            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist  # Symmetric

    # --- Step 3: Run DBSCAN ---
    # eps = 1.0 - iou_threshold
    # If threshold is 0.5 IoU, then distance must be < 0.5.
    eps_val = 1.0 - iou_threshold

    # min_samples=1 ensures that even a ghost appearing in only 1 run
    # is considered a "cluster" (of size 1) and not discarded as noise.
    clustering = DBSCAN(eps=eps_val, min_samples=1, metric='precomputed')
    labels = clustering.fit_predict(distance_matrix)

    # --- Step 4: Group by Cluster ID ---
    unique_labels = set(labels)
    results = []

    for label in unique_labels:
        # Get indices of all ghosts in this cluster
        indices = [i for i, l in enumerate(labels) if l == label]

        # Aggregate data for this cluster
        cluster_occurrences = [all_ghosts[i] for i in indices]

        logits = [g['logit'] for g in cluster_occurrences]
        boxes = [g['box'] for g in cluster_occurrences]

        logits = np.array(logits)
        probs = logits / logits.sum(axis=1, keepdims=True)

        # A. Metrics Calculation
        ghost_rate = len(indices) / total_runs

        vr_ghost = uq.cal_vr(probs)
        ie_ghost = uq.calcu_entropy(probs)
        mi_ghost = uq.calcu_mi(probs)

        var_ghost = uq.calcu_tv(boxes, tag='bounding_box') if len(indices) > 1 else 0.0
        ps_ghost = uq.calcu_prediction_surface(boxes) if len(indices) > 2 else 0.0

        ie_ghost, vr_ghost, mi_ghost = spatial_aware(ghost_rate, ie_ghost / (ie_ghost + 1), 1), \
            spatial_aware(ghost_rate, vr_ghost / (vr_ghost + 1), 1), \
            spatial_aware(ghost_rate, mi_ghost / (mi_ghost + 1), 1)
        ps_ghost, var_ghost = spatial_aware(ghost_rate, ps_ghost / (ps_ghost + 1), 1), \
            spatial_aware(ghost_rate, var_ghost / (var_ghost + 1), 1)

        results.append({
            "id": int(label),
            "label": cluster_occurrences[0]['label'],
            "ghost_rate": ghost_rate,

            "ie_ghost": ie_ghost,
            "vr_ghost": vr_ghost,
            "mi_ghost": mi_ghost,
            "ps_ghost": ps_ghost,
            "var_ghost": var_ghost
        })

    return results


def ms_per_test_case_mutant(test_case, org_model, mutation_operator, mutation_rate, case_study, T=10):
    """
    Computes the UA-MS components (match/miss/ghost metrics, image- and
    object-level kill rates) for a single test case, comparing the original
    model's predictions against one mutant (identified by ``mutation_rate``).

    Expects ``T`` repetitions of ``prediction_{t}.json`` files under:
        ``{case_study}/experiment_results_{mutation_operator}/{org_model}/dataset/orig/``
    for both the original run (rate/size ``0``) and the mutant run.
    """
    file_path = test_case.stem
    matches_set, misses_set, ghosts_set = [], [], []

    if mutation_operator == 'mc_dropout':
        org = Path(f"{case_study}/experiment_results_{mutation_operator}/{org_model}/"
                   f"dataset/orig/{mutation_operator}_0_{file_path}")
        mutant = Path(f"{case_study}/experiment_results_{mutation_operator}/{org_model}/"
                      f"dataset/orig/{mutation_operator}_{mutation_rate}_{file_path}")
    else:
        org = Path(f"{case_study}/experiment_results_{mutation_operator}/{org_model}/"
                   f"dataset/orig/{mutation_operator}_0_1_{file_path}")
        mutant = Path(f"{case_study}/experiment_results_{mutation_operator}/{org_model}/"
                      f"dataset/orig/{mutation_operator}_{mutation_rate[0]}_{mutation_rate[1]}_{file_path}")

    for t in range(T):
        orig_path = org / f"prediction_{t}.json"
        mut_path = mutant / f"prediction_{t}.json"

        with open(orig_path, 'r') as f:
            orig_json = json.load(f)
        with open(mut_path, 'r') as f:
            mut_json = json.load(f)

        matches, misses, ghosts = identify_matches_misses_ghosts(orig_json, mut_json)
        matches_set.append(matches)
        misses_set.append(misses)
        ghosts_set.append(ghosts)

    match_metrics = process_match_metrics(matches_set)
    miss_metrics = process_missing_set(misses_set)
    ghost_metrics = process_ghost_set_dbscan(ghosts_set)

    iskill_miss, iskill_ghost, iskill_miss_ghost = iskill_img(misses_set, ghosts_set)
    ms_obj_level = kill_rate_obj(matches_set, misses_set, ghosts_set)
    match_metrics, miss_metrics, ghost_metrics = un_ms_calcu(match_metrics, miss_metrics, ghost_metrics)

    metrics = ["vr_ms", "ie_ms", "mi_ms", "var_ms", "ps_ms"]
    match_metrics_avg = calcu_mean(match_metrics, metrics + ["iou_ms", "match_rate"])
    miss_metrics_avg = calcu_mean(miss_metrics, metrics + ["miss_rate"])
    ghost_metrics_avg = calcu_mean(ghost_metrics, metrics + ["ghost_rate"])

    return iskill_miss, iskill_ghost, iskill_miss_ghost, ms_obj_level, \
        match_metrics_avg, miss_metrics_avg, ghost_metrics_avg


def calcu_mutation_score(test_set, org_model, mutation_operator, case_study_p, case_study_n, save_folder,
                         mutation_rates=None):
    """
    Computes the UA-MS for every mutant of ``mutation_operator`` (MC-Dropout
    dropout rates, or MC-DropBlock (dropout_rate, block_size) pairs) across a
    whole test set, and writes one CSV per mutant under
    ``{save_folder}/{case_study_n}/{org_model}/{mutation_operator}/``.
    """
    if mutation_rates is None:
        if mutation_operator == "mc_dropout":
            mutation_rates = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        else:
            d_rates = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
            b_sizes = [1, 3, 5, 7, 9]
            mutation_rates = [(d, b) for d in d_rates for b in b_sizes]

    for mutation_rate in mutation_rates:
        # calculate mutation scores for each mutant
        metrics = init_metrics()

        folder = Path(f'./{save_folder}/{case_study_n}/{org_model}/{mutation_operator}')
        folder.mkdir(parents=True, exist_ok=True)
        file_n = f'mutant_{mutation_rate}.csv' if mutation_operator == 'mc_dropout' else \
            f'mutant_{mutation_rate[0]}_{mutation_rate[1]}.csv'

        file_name = folder / file_n
        for test_i in test_set:
            iskill_miss, iskill_ghost, iskill_miss_ghost, \
                ms_obj_level, \
                match_metrics, miss_metrics, ghost_metrics = \
                ms_per_test_case_mutant(test_i, org_model, mutation_operator,
                                        mutation_rate=mutation_rate, case_study=case_study_p)

            metrics = update_metrics(metrics, iskill_miss, iskill_ghost, iskill_miss_ghost,
                                     ms_obj_level, match_metrics, miss_metrics, ghost_metrics)

        suite_list = [test.stem for test in test_set]
        metrics = {'test_suite': suite_list, **metrics}
        pd.DataFrame(metrics).to_csv(file_name, header=True, index=False)


def ms_calcu_exec(case_study_name, case_study, save_folder):
    """
    Runs :func:`calcu_mutation_score` for both the MC-DropBlock and MC-Dropout
    mutants of every model registered under ``case_study[case_study_name]``.

    ``case_study`` is a dict of the form::

        {
            "<case_study_name>": {
                "dataset": "<path to a folder with common_images_{model}_filtered.pkl per model>",
                "raw_result": "<path to the experiment_results_{mutation_operator} folders>",
                "models": ["<model_1>", ...],
            },
            ...
        }
    """
    case_study_path = case_study[case_study_name]['raw_result']
    models = case_study[case_study_name]['models']

    for model in models:
        dataset_path = Path(case_study[case_study_name]['dataset']) / f'common_images_{model}_filtered.pkl'
        with open(dataset_path, 'rb') as f_n:
            loaded_imgs = pickle.load(f_n)

        print(f"Loaded {len(loaded_imgs)} images for evaluating {model}.")

        calcu_mutation_score(test_set=loaded_imgs, org_model=model,
                             mutation_operator='mc_dropblock', case_study_p=case_study_path,
                             case_study_n=case_study_name, save_folder=save_folder)

        calcu_mutation_score(test_set=loaded_imgs, org_model=model,
                             mutation_operator='mc_dropout', case_study_p=case_study_path,
                             case_study_n=case_study_name, save_folder=save_folder)
