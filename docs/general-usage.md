# General Usage

This page describes how to use the metrics provided by `deepluq`. Examples below are
adapted from the test suite in `tests/deepluq/`, which is the best place to see each
metric exercised with concrete inputs and expected output shapes/ranges.

Install the package first:

```bash
pip install deepluq
```

The library exposes three areas of functionality:

- `deepluq.metrics_dl.DLMetrics` — UQ metrics for general deep learning models
  (classification and object detection), see [`test_metrics.py`](../tests/deepluq/test_metrics.py).
- `deepluq.metrics_vla` — UQ metrics for Vision-Language-Action (VLA) models
  (`TokenMetrics`, `OutputMetrics`), see
  [`test_token_based_metrics.py`](../tests/deepluq/test_token_based_metrics.py) and
  [`test_output_based_metrics.py`](../tests/deepluq/test_output_based_metrics.py).
- `deepluq.utils` — clustering and box-fusion helpers used to prepare Monte-Carlo
  Dropout detections before UQ metrics are computed, see
  [`test_clustering.py`](../tests/deepluq/test_clustering.py) and
  [`test_iou.py`](../tests/deepluq/test_iou.py).

## Deep Learning UQ Metrics (`DLMetrics`)

`DLMetrics` computes classification-uncertainty metrics from repeated stochastic
predictions (e.g. multiple MC-Dropout forward passes), and geometric metrics from
repeated bounding-box predictions.

```python
from deepluq.metrics_dl import DLMetrics

uq = DLMetrics()
```

### Classification uncertainty metrics

**Variation Ratio (VR)** — proportion of predictions that disagree with the modal
(most frequent) class across `N` stochastic forward passes. Each row in `events` is a
per-pass probability distribution over classes.

```python
events = [[0.2, 0.8], [0.1, 0.9], [0.6, 0.4]]
vr = uq.cal_vr(events)  # -> float in [0, 1]
```

**Shannon Entropy** — entropy of a single probability distribution, using natural
log by default converted to the given `base` (default base 2).

```python
entropy = uq.calcu_entropy([0.5, 0.5])  # -> 1.0 for a uniform 2-class distribution
```

**Mutual Information (MI)** — combines the entropy of the mean prediction across
passes with the average per-pass entropy, giving a measure of predictive uncertainty
that accounts for disagreement between passes.

```python
import numpy as np

events = np.array([[0.7, 0.3], [0.6, 0.4], [0.5, 0.5]])
mi = uq.calcu_mi(events)  # -> float
```

### Total Variance (TV)

**Total variance** — trace of the covariance matrix of repeated predictions, either
for bounding-box corners (`"bounding_box"`) or box center points (`"center_point"`).
An invalid `tag` raises `ValueError`.

```python
import numpy as np

matrix = np.array([[1, 2], [3, 4], [5, 6]])
tv_center = uq.calcu_tv(matrix, "center_point")
tv_box = uq.calcu_tv(matrix, "bounding_box")
```

### Mutual Information between three variables

`calcu_mutual_information` computes MI between three discrete random variables
(e.g. class label, mutation type, and detection outcome across mutation-based
robustness testing).

```python
import numpy as np

X = np.array([0, 0, 1, 1])
Y = np.array([0, 1, 0, 1])
Z = np.array([1, 1, 0, 0])
mi = uq.calcu_mutual_information(X, Y, Z)  # -> float >= 0
```

### Geometric uncertainty: prediction surface

**Prediction surface** — sum of convex-hull areas formed by each corner (top-left,
top-right, bottom-left, bottom-right) of repeated bounding-box predictions for the
same object. Larger surfaces indicate greater localization uncertainty. At least 3
non-collinear boxes are required per corner set; otherwise the metric returns `-1`.

```python
boxes = [
    [0, 0, 1, 1],
    [1, 0, 2, 1],
    [0, 1, 1, 2],
    [1, 1, 2, 2],
]
surface = uq.calcu_prediction_surface(boxes)  # -> float >= 0

# Too few points to form a hull:
uq.calcu_prediction_surface([[0, 0, 1, 1]])  # -> -1
```

## VLA Metrics (`deepluq.metrics_vla`)

### Token-based metrics (`TokenMetrics`)

`TokenMetrics` computes per-token uncertainty/confidence metrics directly from a
VLA model's output logits (shape `(batch_size, num_classes)`).

```python
import torch
from deepluq.metrics_vla import TokenMetrics

tm = TokenMetrics()

logits = torch.tensor([
    [1.0, 2.0, 0.5, -0.5],
    [2.0, 1.0, -1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
])

entropy, max_prob, pcs, deepgini = tm.calculate_metrics(logits)
```

`calculate_metrics` returns four lists (one value per sample):

- **Shannon entropy** of the softmax distribution.
- **Max token probability** — confidence of the top predicted token.
- **PCS (Prediction Confidence Score)** — gap between the top-1 and top-2 probabilities.
- **DeepGini** — `1 - sum(p^2)`.

For a confident (peaked) prediction, entropy/DeepGini are close to 0 and
max-probability/PCS are close to 1:

```python
logits = torch.tensor([[10.0, -10.0]])
entropy, max_prob, pcs, deepgini = tm.calculate_metrics(logits)
# entropy[0] < 0.1, max_prob[0] > 0.9, pcs[0] > 0.9, deepgini[0] < 0.1
```

`compute_norm_inv_token_metrics` returns the same four metrics normalized to
`[0, 1]` and inverted where necessary, so that **higher values always mean higher
uncertainty**:

```python
logits = torch.tensor([[1.0, 2.0, 0.5], [2.0, 1.0, -1.0]])
entropy_norm, max_prob_inv, pcs_inv, deepgini_norm = tm.compute_norm_inv_token_metrics(logits)
```

Internally, `TokenMetrics` accumulates results in `shannon_entropy_list`,
`token_prob`/`token_prob_inv`, `pcs`/`pcs_inv`, and `deepgini`. Call `clear()` to
reset the accumulated state between batches:

```python
tm.calculate_metrics(torch.tensor([[1.0, 2.0, 3.0]]))
tm.clear()
assert tm.shannon_entropy_list == []
```

### Output-based instability and variability metrics (`OutputMetrics`)

`OutputMetrics` quantifies how unstable a VLA model's predicted actions or
tool-center-point (TCP) poses are over a rollout, and how much variability there is
across an ensemble of models given the same observation.

```python
from deepluq.metrics_vla import OutputMetrics

om = OutputMetrics()
```

**Action-based instability** — each action is a dict with `"world_vector"`,
`"rot_axangle"`, and `"gripper"` keys. Instability is the mean absolute successive
difference (position = 1st order, velocity = 2nd order, acceleration = 3rd order)
across the rollout, returned per action dimension.

```python
actions = [
    {"world_vector": [1, 2], "rot_axangle": [0, 0, 1], "gripper": [0.5]}
    for _ in range(5)
]
position_instability = om.compute_position_instability(actions)      # requires >= 2 steps
velocity_instability = om.compute_velocity_instability(actions)      # requires >= 3 steps
acceleration_instability = om.compute_acceleration_instability(actions)  # requires >= 4 steps
```

**TCP instability** — same idea, but computed from a list of `[x, y, z, ...]` poses
(only the first three coordinates are used).

```python
poses = [[i, i + 1, i + 2] for i in range(10)]

tcp_position = om.compute_TCP_position_instability(poses)
tcp_velocity = om.compute_TCP_velocity_instability(poses)
tcp_acceleration = om.compute_TCP_acceleration_instability(poses)

# Jerk instability via numerical gradients (one value per time step):
jerk = om.compute_TCP_jerk_instability_gradient(poses)
```

**Execution variability** — standard deviation of actions produced by an ensemble
of models (or repeated stochastic rollouts of the same model) given the same
observation, used to quantify epistemic uncertainty in action selection.

```python
variability = om.compute_execution_variability(
    variability_models=[model_1, model_2, model_3],
    image=image,
    action_space=action_space,
    instruction=instruction,
    obs=obs,               # dict with obs["agent"]["eef_pos"]
    model_name="pi0",      # selects the model.step(...) calling convention
)
```

## Detection Utilities (`deepluq.utils`)

These helpers support MC-Dropout–style UQ pipelines for object detection, where a
single object may receive several overlapping predictions across passes that need
to be clustered/fused before UQ metrics (e.g. `calcu_tv`, `calcu_prediction_surface`)
are computed on them.

### Weighted Boxes Fusion clustering

`wbf_clustering` groups raw per-pass detections into clusters using Weighted Boxes
Fusion (WBF), and returns, for each cluster, both the original member detections and
the fused/merged detection.

```python
from deepluq.utils import wbf_clustering

predictions = {
    "pred_0": {"box": [822.0, 301.6, 902.1, 340.7], "box_n": [0.642, 0.419, 0.704, 0.473],
               "label": 0, "score": 0.932, "logit": [0.932, 1.107e-08, 0.0002]},
    "pred_1": {"box": [865.9, 309.1, 892.9, 332.6], "box_n": [0.676, 0.429, 0.697, 0.462],
               "label": 2, "score": 0.877, "logit": [0.0005, 7.545e-06, 0.877]},
    "pred_2": {"box": [822.0, 301.6, 902.1, 340.7], "box_n": [0.642, 0.419, 0.704, 0.473],
               "label": 0, "score": 0.900, "logit": [0.900, 2.107e-08, 0.0003]},
}

clusters = wbf_clustering(predictions, iou_thr=0.5, skip_box_thr=0.01)
```

Each entry in the returned dict (keyed `cluster_0`, `cluster_1`, ...) contains the
member `box`/`box_n`/`score`/`label`/`logit` lists plus a `detection` entry with the
fused box, score, label, and averaged logit — ready to feed into `DLMetrics`.

`compute_iou(box1, box2)` is the plain IoU helper (normalized `[x1, y1, x2, y2]`
boxes) used internally by `wbf_clustering` to match original boxes back to their
fused cluster.

### Density-based clustering

For scenarios where WBF is not suitable, `deepluq.utils` also provides density-based
clustering on box corner points via `DBSCANCluster` (HDBSCAN-backed) and the
lower-level `cluster(mc_locations)` function, which clusters MC-Dropout box
predictions with DBSCAN and reports the convex-hull surface per cluster.

```python
from deepluq.utils import DBSCANCluster

# x: array of shape (N, >=4) — e.g. [x1, y1, x2, y2, center_x, center_y]
clustering = DBSCANCluster(x)
clustered_preds = clustering.cluster_preds(preds)
```

`get_kdist_plot(X, k)` is a diagnostic helper that plots the sorted k-nearest-neighbor
distances for a point set, useful for choosing a DBSCAN `eps` value.

### Intersection over Union (object detection quality)

For evaluating detection accuracy alongside UQ metrics, this project relies on
[`torchmetrics.detection`](https://lightning.ai/docs/torchmetrics/stable/detection/intersection_over_union.html)
rather than reimplementing IoU-based detection metrics. `test_iou.py` shows the
expected usage pattern with `preds`/`targets` dictionaries of `boxes`, `scores`, and
`labels`:

```python
import torch
from torchmetrics.detection import IntersectionOverUnion
from torchmetrics.functional.detection import intersection_over_union

metric = IntersectionOverUnion(class_metrics=True, respect_labels=True)
result = metric(preds, targets)

per_box_iou = intersection_over_union(preds[0]["boxes"], targets[0]["boxes"], aggregate=False)
```

## Running the tests

The examples above mirror the test suite. To run all metric tests locally:

```bash
pytest tests/deepluq
```