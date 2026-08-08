# Reference API

Full reference of the public classes and functions in `deepluq` package
(`src/deepluq/`).

## `deepluq.metrics_dl`

### `class DLMetrics`

A class to compute Uncertainty Quantification (UQ) metrics for Deep
Learning, including variation ratio, entropy, mutual information, total variance,
and prediction surface using convex hulls.

Attributes: `variation_ratio`,
`shannon_entropy`, `mutual_information`, `total_var_center_point`,
`total_var_bounding_box`, `prediction_surface`, `hull`, `box`.

#### `cal_vr(events)`

Compute the Variation Ratio (VR): the proportion of non-modal class predictions.

| Parameter | Type | Description |
|---|---|---|
| `events` | array-like | Model outputs or predictions, shape `(N, num_classes)`. |

**Returns:** `float`: variation ratio, in `[0, 1]`.

#### `calcu_entropy(events, eps=1e-15, base=2)`

Compute Shannon entropy of a probability distribution.

| Parameter | Type | Description |
|---|---|---|
| `events` | array-like | Probability distribution. |
| `eps` | `float` | Small constant to avoid `log(0)`. Default `1e-15`. |
| `base` | `int` | Logarithm base. Default `2`. |

**Returns:** `float`: Shannon entropy, rounded to 5 decimals (clamped to `>= 0`).

#### `calcu_mi(events, eps=1e-15, base=2)`

Compute Mutual Information (MI) between repeated predictions, combining the entropy
of the mean prediction with the average per-sample entropy.

| Parameter | Type | Description |
|---|---|---|
| `events` | array-like | Model probability outputs, shape `(N, num_classes)`. |
| `eps` | `float` | Small constant to avoid `log(0)`. Default `1e-15`. |
| `base` | `int` | Logarithm base. Default `2`. |

**Returns:** `float`: mutual information (clamped to `>= 0`).

#### `calcu_tv(matrix, tag)`

Compute total variance of a multi-dimensional matrix using the trace of its
covariance matrix.

| Parameter | Type | Description |
|---|---|---|
| `matrix` | array-like | Input data matrix. |
| `tag` | `str` | Either `"bounding_box"` or `"center_point"`. |

**Returns:** `float`: total variance.

**Raises:** `ValueError` if `tag` is not `"bounding_box"` or `"center_point"`.

#### `calcu_mutual_information(X, Y, Z)`

Compute mutual information between three discrete random variables.
Reference: [scholarpedia.org/article/Mutual_information](http://www.scholarpedia.org/article/Mutual_information).

| Parameter | Type | Description |
|---|---|---|
| `X`, `Y`, `Z` | array-like | Discrete random variables of shape `(n_samples,)`. |

**Returns:** `float`: mutual information (clamped to `>= 0`).

#### `calcu_prediction_surface(boxes)`

Compute prediction surface by summing convex-hull areas over each corner
(top-left, top-right, bottom-left, bottom-right) of repeated bounding-box
predictions for the same object.

| Parameter | Type | Description |
|---|---|---|
| `boxes` | array-like | List of bounding boxes `[x1, y1, x2, y2]`. |

**Returns:** `float`: prediction surface area (sum of convex hulls). `-1` if fewer
than 3 boxes are given; `0` for a corner set with fewer than 3 unique or collinear
points.

## `deepluq.metrics_vla`

### `class TokenMetrics`

Computes token-level uncertainty metrics from a Vision-Language-Action (VLA) model's output
logits. 

Attributes: `shannon_entropy_list`, `token_prob`, `pcs`,
`token_prob_inv`, `pcs_inv`, `deepgini`.

#### `calculate_metrics(logits)`

Compute Shannon entropy, max token probability, prediction confidence score (PCS), and DeepGini from raw logits.

| Parameter | Type | Description |
|---|---|---|
| `logits` | `torch.Tensor` | Raw output logits, shape `(batch_size, num_classes)`. |

**Returns:** `list` of 4 lists (one value per sample):
`[shannon_entropy, token_prob, pcs, deepgini]`.

#### `compute_norm_inv_token_metrics(logits)`

Compute the same four metrics, normalized to `[0, 1]` and inverted where needed so
that higher values always mean greater uncertainty.

| Parameter | Type | Description |
|---|---|---|
| `logits` | `torch.Tensor` | Raw output logits, shape `(batch_size, num_classes)`. |

**Returns:** `list` of 4 lists, rounded to 5 decimals:
`[shannon_entropy_norm, max_token_prob_inv, pcs_inv, deepgini_norm]`.

#### `clear()`

Reset `shannon_entropy_list`, `token_prob`, `pcs`, and `deepgini` to empty lists.

**Returns:** `None`.

### `class OutputMetrics`

Compute instability and variability metrics for robot actions and TCP positions.
Class attribute: `VARIABILITY = 4`.

#### `compute_position_instability(actions)`

| Parameter | Type | Description |
|---|---|---|
| `actions` | `List[Dict[str, Any]]` | Each dict has `"world_vector"`, `"rot_axangle"`, `"gripper"` keys. Requires >= 2 steps. |

**Returns:** `np.ndarray`: mean absolute 1st-order (position) difference per action
dimension.

#### `compute_velocity_instability(actions)`

Same `actions` input as above (requires >= 3 steps).

**Returns:** `np.ndarray`: mean absolute 2nd-order (velocity) difference per
dimension, scaled by `2.0`.

#### `compute_acceleration_instability(actions)`

Same `actions` input as above (requires >= 4 steps).

**Returns:** `np.ndarray`: mean absolute 3rd-order (acceleration) difference per
dimension, scaled by `4.0`.

#### `compute_TCP_position_instability(poses)`

| Parameter | Type | Description |
|---|---|---|
| `poses` | `List[List[float]]` | Poses; only the first 3 coordinates `(x, y, z)` are used. Requires >= 2 steps. |

**Returns:** `np.ndarray`: 1st-order instability per coordinate (shape `(3,)`).

#### `compute_TCP_velocity_instability(poses)`

Same `poses` input (requires >= 3 steps).

**Returns:** `np.ndarray`: 2nd-order instability per coordinate (shape `(3,)`).

#### `compute_TCP_acceleration_instability(poses)`

Same `poses` input (requires >= 4 steps).

**Returns:** `np.ndarray`: 3rd-order instability per coordinate (shape `(3,)`).

#### `compute_TCP_jerk_instability_gradient(poses)`

Compute TCP jerk via numerical gradients (`np.gradient`, applied 3 times).

| Parameter | Type | Description |
|---|---|---|
| `poses` | `List[List[float]]` | Poses; only the first 3 coordinates are used. |

**Returns:** `np.ndarray`: jerk magnitude per time step, shape `(len(poses),)`.

#### `compute_execution_variability(variability_models, image, action_space, instruction, obs, model_name)`

Static method. Runs each model's `step(...)` for one observation and computes the
standard deviation of the resulting (normalized) actions across models.

| Parameter | Type | Description |
|---|---|---|
| `variability_models` | `List[Any]` | Models exposing a `.step(...)` method. |
| `image` | `Any` | Observation image passed to `model.step(...)`. |
| `action_space` | `Any` | Action space with `.low`/`.high`, used to normalize actions. |
| `instruction` | `Any` | Task instruction passed to `model.step(...)`. |
| `obs` | `Dict[str, Any]` | Must contain `obs["agent"]["eef_pos"]` for `"pi0"` models. |
| `model_name` | `str` | Selects the calling convention: contains `"pi0"`, `"spatialvla"`, or falls back to `model.step(image)`. |

**Returns:** `np.ndarray`: per-dimension standard deviation of
`world_vector` + `rot_axangle` + `gripper` across models.

## `deepluq.metrics_mut`

Computes the Uncertainty-Aware Mutation Score (UA-MS) for MC-Dropout /
MC-DropBlock mutants of an object detection model. See [uq4ma.md](uq4ma.md) for usage examples.

#### `ms_calcu.process_match_metrics(repetitions)`

Given `T` repetitions of `(orig, mut)` match pairs (as returned per-repetition by
`ms.identify_matches_misses_ghosts`), anchors matched objects across
repetitions by original box, and for each unique object computes the detection
rate plus original- and mutant-side VR/entropy/MI/total-variance/prediction-surface,
weighted by detection rate (`ms.spatial_aware`).

| Parameter | Type | Description |
|---|---|---|
| `repetitions` | `list` of `list` of `(dict, dict)` | Per-repetition lists of `(orig_obj, mut_obj)` match pairs. |

**Returns:** `list[dict]`, one entry per unique matched object with `id`, `label`,
`match_rate`, `{vr,ie,mi,var,ps}_{orig,mut}`, `avg_iou`.

#### `ms_calcu.process_missing_set(repetitions)`

Given `T` repetitions of miss lists, anchors missed objects across repetitions
by original box and computes the miss rate plus the original detection's own
uncertainty metrics, weighted by miss rate.

| Parameter | Type | Description |
|---|---|---|
| `repetitions` | `list` of `list[dict]` | Per-repetition lists of missed original objects. |

**Returns:** `list[dict]`, one entry per unique missed object with `id`, `label`,
`miss_rate`, `{vr,ie,mi,var,ps}_miss`.

#### `ms_calcu.process_ghost_set_dbscan(repetitions, iou_threshold=0.5)`

Clusters ghost detections across all repetitions with DBSCAN on a
`(1 - IoU)` distance matrix (so recurring hallucinations at the same location
are grouped into one object), then computes the ghost rate and uncertainty
metrics per cluster, weighted by ghost rate.

| Parameter | Type | Description |
|---|---|---|
| `repetitions` | `list` of `list[dict]` | Per-repetition lists of ghost mutant objects. |
| `iou_threshold` | `float` | IoU threshold used to derive the DBSCAN `eps` (`1 - iou_threshold`). Default `0.5`. |

**Returns:** `list[dict]`, one entry per ghost cluster with `id`, `label`,
`ghost_rate`, `{vr,ie,mi,var,ps}_ghost`.

#### `ms_calcu.ms_per_test_case_mutant(test_case, org_model, mutation_operator, mutation_rate, case_study, T=10)`

Loads `T` repetitions of original/mutant predictions for one test case, matches
them per repetition, and runs `process_match_metrics` /
`process_missing_set` / `process_ghost_set_dbscan` plus the image- and
object-level kill-rate calculations on the results.

**Returns:** `tuple` of `(iskill_miss, iskill_ghost, iskill_miss_ghost, ms_obj_level, match_metrics, miss_metrics, ghost_metrics)`.

#### `ms_calcu.calcu_mutation_score(test_set, org_model, mutation_operator, case_study_p, case_study_n, save_folder, mutation_rates=None)`

Score a whole test set against every mutant of `mutation_operator`; writes one
CSV per mutant under `{save_folder}/{case_study_n}/{org_model}/{mutation_operator}/`.

**Returns:** `None`.

#### `ms_calcu.ms_calcu_exec(case_study_name, case_study, save_folder)`

Run `calcu_mutation_score` for both `mc_dropblock` and `mc_dropout` across every
model registered in `case_study[case_study_name]`.

**Returns:** `None`.

#### `ms.identify_matches_misses_ghosts(orig_objs, mut_objs, iou_threshold=0.5)`

Hungarian-match original vs. mutant detections by IoU and label.

**Returns:** `tuple` of `(matches, misses, ghosts)`.

#### `ms.check_kill_binomial_test(success, n, p_null, alpha=0.05)`

One-sided binomial test for whether an observed kill count is significantly
above the noise floor `p_null`.

**Returns:** `dict` with `p_value`, `cohens_h`, `power`, `critical_threshold`.

#### `ms.compute_iou(boxA, boxB)`, `ms.spatial_aware(rate, metric, penalty)`, `ms.calcu_mean(results, cols_to_mean)`, `ms.un_ms_calcu(match_metrics, miss_metrics, ghost_metrics)`, `ms.kill_rate_obj(match_s, miss_s, ghost_s)`, `ms.kill_count_img(miss_s, ghost_s)`, `ms.iskill_img(miss_s, ghost_s)`

Supporting helpers used by `ms_calcu`; see [uq4ma.md](uq4ma.md) for how their
outputs feed into the final UA-MS columns. `un_ms_calcu` turns the raw
`{vr,ie,mi,var,ps}_{orig,mut}` / `_miss` / `_ghost` values from
`process_match_metrics` / `process_missing_set` / `process_ghost_set_dbscan`
into the `_ms` (UA-MS) columns; `kill_rate_obj` computes the Obj-MS ratios;
`kill_count_img` / `iskill_img` compute the per-test-case binomial-test inputs
for Img-MS.

#### `ms.convert_detection_output(pred, default_score=1.0)`, `ms.yolo_to_absolute(yolo_labels, img_w=1280, img_h=736)`

Convert a torchvision-style detection output or YOLO-format ground-truth labels
into the `{"label_i": {"box", "label", "score", "logit"}}` format expected by
`identify_matches_misses_ghosts`.

#### `helper.init_metrics()`, `helper.update_metrics(...)`, `helper.print_metric(match_metrics, miss_metrics, ghost_metrics)`

Accumulate per-test-case results into the CSV-ready metrics dict, and
pretty-print raw match/miss/ghost metrics for debugging.


## `deepluq.utils`

#### `compute_iou(box1, box2)`

Calculate Intersection over Union (IoU) for two normalized `[x1, y1, x2, y2]` boxes.

**Returns:** `float`: IoU, `0` if the union area is `0`.

#### `wbf_clustering(predictions_dict, iou_thr=0.5, skip_box_thr=0.01)`

Apply Weighted Boxes Fusion (WBF) and group the original input boxes into their
respective clusters alongside the final merged detection.

| Parameter | Type | Description |
|---|---|---|
| `predictions_dict` | `dict` | Keyed by prediction id, each value a dict with `box`, `box_n`, `score`, `label`, and optional `logit`. |
| `iou_thr` | `float` | IoU threshold for WBF and for matching original boxes to fused clusters. Default `0.5`. |
| `skip_box_thr` | `float` | Score threshold below which boxes are skipped by WBF. Default `0.01`. |

**Returns:** `dict` keyed `cluster_0`, `cluster_1`, ... Each value contains the
member `box`, `box_n`, `score`, `label`, `logit` lists plus a `detection` entry with
the fused `box_n`, `box`, `score`, `label`, and averaged `logit`.

### `class DBSCANCluster`

#### `__init__(x, eps=8.5, min_samples=8)`

Cluster box-derived points with HDBSCAN (`min_cluster_size=3`).

| Parameter | Type | Description |
|---|---|---|
| `x` | array-like | Points to cluster, shape `(N, >=4)` (e.g. `x1, y1, x2, y2, center_x, center_y`). |
| `eps` | `float` | Unused by the current HDBSCAN implementation (kept for API compatibility). Default `8.5`. |
| `min_samples` | `int` | Unused by the current HDBSCAN implementation (kept for API compatibility). Default `8`. |

Populates `self.cluster`, `self.mc_locations`, `self.mc_locations_df`, and
`self.cluster_labels`.

#### `cluster_preds(preds)`

| Parameter | Type | Description |
|---|---|---|
| `preds` | `dict` | Predictions keyed by id, each with `box`, `label`, `score`, `logit`. |

**Returns:** `dict` keyed `label_0`, `label_1`, ... grouping the matching
`box`/`label`/`score`/`logit` lists per cluster found by `__init__`.

#### `cluster(mc_locations)`

Cluster MC-Dropout box predictions with `DBSCAN(eps=100, min_samples=2)` and
compute the convex-hull surface per cluster corner set.

| Parameter | Type | Description |
|---|---|---|
| `mc_locations` | array-like | Box corner/center points, shape `(N, 4)` (`x1, y1, x2, y2`). |

**Returns:** `None` (diagnostic/summary function; does not return the computed
surfaces).

#### `get_kdist_plot(X=None, k=None, radius_nbrs=1.0)`

Plot the sorted k-nearest-neighbor distance for each point, useful for choosing a
DBSCAN `eps` value.

| Parameter | Type | Description |
|---|---|---|
| `X` | array-like | Points to analyze. |
| `k` | `int` | Number of neighbors. |
| `radius_nbrs` | `float` | Neighborhood radius passed to `NearestNeighbors`. Default `1.0`. |

**Returns:** `None` (shows a matplotlib plot).

#### `normalize_action(action, normalization_values)`

Normalize an action dict's `world_vector`, `rot_axangle`, and `gripper` fields into
`[0, 1]` using `normalization_values.low` / `.high`.

| Parameter | Type | Description |
|---|---|---|
| `action` | `dict` | Action with `world_vector`, `rot_axangle`, `gripper` keys. |
| `normalization_values` | `Any` | Object with `.low` and `.high` bounds (e.g. a gym `Box` action space). |

**Returns:** `dict`: deep copy of `action` with normalized, clipped fields.

#### `action_uncertainty(action, mutated_action)`

Compute per-dimension standard deviation between an action and a mutated version
of it (metamorphic-testing style uncertainty).

| Parameter | Type | Description |
|---|---|---|
| `action`, `mutated_action` | `dict` | Actions with `world_vector`, `rot_axangle`, `gripper` keys. |

**Returns:** `np.ndarray`: standard deviation per dimension across the two
actions.

## `deepluq.version`

| Name | Description |
|---|---|
| `__version__` | Package version string, e.g. `"0.1.5"`. |
| `__version_info__` | Tuple of version components as strings, e.g. `("0", "1", "5")`. |
