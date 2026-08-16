# General Usage

This page describes how to use the metrics provided by `deepluq`. Examples below are
adapted from the test suite in `tests/deepluq/`, which is the best place to see each
metric exercised with concrete inputs and expected output shapes/ranges.

Follow the installation and setup instructions in the [Installation Page](https://into-cps-association.github.io/DeepLUQ/docs/install.html).

## Key Concepts

`deepluq`'s metrics are built around Bayesian approximation approaches,
**MC-Dropout** ([Gal and Ghahramani, 2016](http://proceedings.mlr.press/v48/gal16.html))
and **MC-DropBlock** ([Yelleni and Kumari, 2024](https://doi.org/10.1016/j.patcog.2023.110003)),
for uncertainty quantification (UQ) in deep neural networks.
Instead of retraining a model, we inject dropout (or DropBlock, which drops
contiguous spatial regions of a feature map rather than individual units,
and is better suited to convolutional object-detection backbones) layers into
a pre-trained model and keep them *active at inference time* (normally
dropout is only active during training). Running the resulting model `T`
times on the same input produces `T` slightly different predictions, a
sample from the model's predictive distribution. A confident model produces near-identical
predictions across the `T` runs; an uncertain one produces predictions that
disagree, either in the predicted class or in the predicted location. The
metrics below turn that disagreement into a number.

For object detection, the `T` predictions for one image contain a variable
number of detected boxes that don't line up 1:1 across runs (an object may be
missed on some passes, or several boxes may correspond to the same object).
Before per-object uncertainty can be computed, the boxes must first be
grouped by the object they refer to, which is what the clustering helpers in
`deepluq.utils` (`wbf_clustering`, `DBSCANCluster`) are for. Each resulting
cluster represents one detected object, with a set of `W` (`1 <= W <= T`)
per-pass class scores and bounding boxes to compute uncertainty from.

Two families of metrics are computed per object/cluster:

- **Classification uncertainty**: how much the *class* predictions disagree
  across the `W` passes:
  - **Variation Ratio (VR)**: the fraction of passes that disagree with the
    most frequent (modal) class. `0` = every pass agrees, higher = more
    disagreement.
  - **Shannon Entropy (SE)**: the average information content of the mean
    class-probability distribution across passes. `0` = one class always gets
    probability 1, maximum = all classes equally likely.
  - **Mutual Information (MI)**: the gap between the entropy of the *mean*
    prediction and the *mean* of the per-pass entropies. Low when the model is
    just generally unsure about the class (but consistent across passes),
    high when the passes actively disagree with each other.
- **Geometric (localization) uncertainty**: how much the *bounding box*
  predictions disagree across the `W` passes:
  - **Total Variance (TV)**: sum of the per-coordinate variances of the box
    corners (or center point) across passes. `0` = identical boxes every
    pass, larger = more spread out.
  - **Prediction Surface (PS)**: area of the convex hull formed by each box
    corner across passes, averaged over the four corners. `0`-ish = corners
    barely move between passes, larger = more positional disagreement.

These correspond to `deepluq.metrics_dl.DLMetrics`'s `cal_vr`, `calcu_entropy`,
`calcu_mi`, `calcu_tv`, and `calcu_prediction_surface` methods used below.
`deepluq.metrics_vla` applies the same underlying idea (disagreement across
repeated/ensembled predictions) to VLA model outputs instead of detection
boxes, and `deepluq.metrics_mut` uses MC-Dropout/MC-DropBlock *mutants* of a
model to score how much a mutation affects both correctness and uncertainty
(see [Uncertainty-Aware Mutation Analysis](uq4ma.md) for those concepts).

For the full derivation, parameter choices (e.g. `T=20`), and an evaluation
on a real sticker-detection model, see the references below.

### References

1. Yarin Gal and Zoubin Ghahramani. "Dropout as a Bayesian Approximation:
   Representing Model Uncertainty in Deep Learning." In *International
   Conference on Machine Learning (ICML)*. PMLR, 2016.
2. Sai Harsha Yelleni and Deepshikha Kumari. "Monte Carlo DropBlock for
   Modeling Uncertainty in Object Detection." *Pattern Recognition* 146
   (2024): 110003.
3. Chengjie Lu, Jiahui Wu, Shaukat Ali, and Mikkel Labori Olsen. "Assessing
   the Uncertainty and Robustness of the Laptop Refurbishing Software." In
   *2025 IEEE Conference on Software Testing, Verification and Validation
   (ICST)*, Napoli, Italy, 2025, pp. 406-416.
   [doi:10.1109/ICST62969.2025.10988977](https://doi.org/10.1109/ICST62969.2025.10988977)

Main functionality:

- `deepluq.metrics_dl.DLMetrics`: UQ metrics for general deep learning models
  (classification and object detection), see [`test_metrics.py`](../tests/deepluq/test_metrics.py).
- `deepluq.metrics_vla`: UQ metrics for Vision-Language-Action (VLA) models
  (`TokenMetrics`, `OutputMetrics`), see
  [`test_token_based_metrics.py`](../tests/deepluq/test_token_based_metrics.py) and
  [`test_output_based_metrics.py`](../tests/deepluq/test_output_based_metrics.py).
- `deepluq.utils`: clustering and box-fusion helpers used to prepare Monte-Carlo
  Dropout detections before UQ metrics are computed, see
  [`test_clustering.py`](../tests/deepluq/test_clustering.py) and
  [`test_iou.py`](../tests/deepluq/test_iou.py).
- `deepluq.metrics_mut`: Uncertainty-Aware Mutation Score (UA-MS) computation for
  MC-Dropout / MC-DropBlock mutants of object detection models.

## Deep Learning UQ Metrics (`DLMetrics`)

`DLMetrics` computes classification-uncertainty metrics from repeated stochastic
predictions (e.g. multiple MC-Dropout forward passes), and geometric metrics from
repeated bounding-box predictions.

```python
from deepluq.metrics_dl import DLMetrics

uq = DLMetrics()
```

### Classification uncertainty metrics

**Variation Ratio (VR)**: proportion of predictions that disagree with the modal
(most frequent) class across `N` stochastic forward passes. Each row in `events` is a
per-pass probability distribution over classes.

```python
events = [[0.2, 0.8], [0.1, 0.9], [0.6, 0.4]]
vr = uq.cal_vr(events)  # -> float in [0, 1]
```

**Shannon Entropy**: entropy of a single probability distribution, using natural
log by default converted to the given `base` (default base 2).

```python
entropy = uq.calcu_entropy([0.5, 0.5])  # -> 1.0 for a uniform 2-class distribution
```

**Mutual Information (MI)**: combines the entropy of the mean prediction across
passes with the average per-pass entropy, giving a measure of predictive uncertainty
that accounts for disagreement between passes.

```python
import numpy as np

events = np.array([[0.7, 0.3], [0.6, 0.4], [0.5, 0.5]])
mi = uq.calcu_mi(events)  # -> float
```

### Total Variance (TV)

**Total variance**: trace of the covariance matrix of repeated predictions, either
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

**Prediction surface**: sum of convex-hull areas formed by each corner (top-left,
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

`TokenMetrics` computes token-level uncertainty metrics directly from a
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
- **Max token probability**: confidence of the top predicted token.
- **PCS (Prediction Confidence Score)**: gap between the top-1 and top-2 probabilities.
- **DeepGini**: `1 - sum(p^2)`.

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
tool-center-point (TCP) poses, and how much variability there is
across an ensemble of models given the same observation.

```python
from deepluq.metrics_vla import OutputMetrics

om = OutputMetrics()
```

**Action-based instability**: each action is a dict with `"world_vector"`,
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

**TCP instability**: same idea, but computed from a list of `[x, y, z, ...]` poses
(only the first three coordinates are used).

```python
poses = [[i, i + 1, i + 2] for i in range(10)]

tcp_position = om.compute_TCP_position_instability(poses)
tcp_velocity = om.compute_TCP_velocity_instability(poses)
tcp_acceleration = om.compute_TCP_acceleration_instability(poses)

# Jerk instability via numerical gradients (one value per time step):
jerk = om.compute_TCP_jerk_instability_gradient(poses)
```

**Execution variability**: standard deviation of actions produced by an ensemble
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
fused box, score, label, and averaged logit.

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

# x: array of shape (N, >=4), e.g. [x1, y1, x2, y2, center_x, center_y]
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

## Uncertainty-Aware Mutation Analysis (`deepluq.metrics_mut`)

`deepluq.metrics_mut` computes the Uncertainty-Aware Mutation Score (UA-MS) for
an object detection model: it compares the original model's predictions against
predictions from MC-Dropout / MC-DropBlock *mutants*, and scores how much each
mutant's outputs and predictive uncertainty diverge from the original.
Unlike the metrics above, this module reads prediction files from disk (the
output of running a model repeatedly under mutation) rather than working on
in-memory arrays, so see [Uncertainty-Aware Mutation Analysis](uq4ma.md) for the
required file layout and the full concepts (matches/misses/ghosts, Img-MS,
Obj-MS, MS_iou, UA-MS). The entry point for scoring one test case against one
mutant is:

```python
from pathlib import Path
from deepluq.metrics_mut.ms_calcu import ms_per_test_case_mutant

iskill_miss, iskill_ghost, iskill_miss_ghost, ms_obj_level, \
    match_metrics, miss_metrics, ghost_metrics = ms_per_test_case_mutant(
        test_case=Path("image_0001"),
        org_model="fasterrcnn_resnet50_fpn",
        mutation_operator="mc_dropout",   # or "mc_dropblock"
        mutation_rate=0.3,                # or (dropout_rate, block_size) for mc_dropblock
        case_study="/path/to/experiment_results_root",
        T=10,
    )
```

`calcu_mutation_score` scores a whole test set against every mutant of an
operator and writes one CSV per mutant, and `ms_calcu_exec` runs that for every
model in a case study.

## Running the tests

The examples above mirror the test suite. To run all metric tests locally:

```bash
pytest tests/deepluq
```