# Usage: Uncertainty-Aware Mutation Analysis for DL Models

Follow the installation and setup instructions in the [Installation Page](https://into-cps-association.github.io/DeepLUQ/docs/install.html).

`deepluq.metrics_mut` implements **Uncertainty-Aware Mutation Testing (UAMT)** for
object detection models: it compares a model's predictions against predictions
from *mutants* of that model produced by MC-Dropout and MC-DropBlock, and reports
how much each mutant's outputs (and predictive uncertainty) diverge from the
original — the Uncertainty-Aware Mutation Score (UA-MS).

## Concepts

- **Mutation operator** — the source of randomness injected into the model to
  produce mutants:
  - `mc_dropout` — Monte Carlo Dropout applied at a given dropout rate.
  - `mc_dropblock` — Monte Carlo DropBlock applied at a given
    `(dropout_rate, block_size)` pair.
- **Mutant** — one specific configuration of a mutation operator (e.g.
  `mc_dropout` at rate `0.3`, or `mc_dropblock` at rate `0.1` with block size `5`).
  The unmutated model corresponds to rate `0` (`mc_dropout_0_...` /
  `mc_dropblock_0_1_...`).
- **Repetition** — because MC-Dropout/MC-DropBlock are stochastic, each mutant is
  run `T` times (`T=10` by default) per test image to get a distribution of
  predictions rather than a single one.

Mutation score quantifies test-suite quality by measuring how many mutants a
test suite *kills*. Because an object detector's output is a *set* of detections
rather than a single vector, `deepluq.metrics_mut` first reduces each pair of
(original, mutant) outputs to three disjoint sets, then defines mutation scores
on top of them.

Given an object detection model $ODM$ and a test input $img$, its output is a
set of detections $O = ODM(img) = \{d_1, ..., d_n\}$, where each detection
$d_i = (b_i, c_i, p_i)$ has a bounding box $b_i$, a predicted class $c_i$, and a
class-probability vector $p_i$.

For an original model's output $O$ and a mutant's output $O'$ on the same
input:

- **Match** — a pair $(d_i, d'_j) \in O \times O'$ is a match if
  $\mathrm{IoU}(b_i, b'_j) > \theta_{\mathrm{IoU}} \land c_i = c'_j$ (deepluq
  uses $\theta_{\mathrm{IoU}} = 0.5$ and finds the matching via Hungarian
  assignment on the IoU cost). $\mathcal{S}_{match}$ is the set of all matched
  pairs.
- **Miss** — a detection $d_i \in O$ with no counterpart in
  $\mathcal{S}_{match}$: an object the original model found but the mutant did
  not. $\mathcal{S}_{miss}$ is the set of all misses.
- **Ghost** — a detection $d'_j \in O'$ with no counterpart in
  $\mathcal{S}_{match}$: a false positive introduced by the mutation.
  $\mathcal{S}_{ghost}$ is the set of all ghosts.

`deepluq.metrics_mut.ms.identify_matches_misses_ghosts` computes exactly these
three sets for one repetition of one test image. `deepluq.metrics_mut.ms_calcu`
then defines four mutation scores on top of $\mathcal{S}_{match}$,
$\mathcal{S}_{miss}$, $\mathcal{S}_{ghost}$, computed over `T` repetitions:

1. **Image-level Mutation Score (Img-MS)** — the traditional, output-agnostic
   notion of "killed": a mutant is killed if, for at least one test case, the
   presence of misses/ghosts/either across the `T` repetitions is
   *statistically significant* (not just noise), per a one-sided binomial test
   against a null miss/ghost rate. Img-MS is the fraction of mutants killed.
   Reported as `Img-MS_miss`, `Img-MS_ghost`, `Img-MS_mg`.
2. **Object-level Mutation Score (Obj-MS)** — a finer-grained score computed
   directly from set sizes, e.g.
   $MS_{obj}^{miss} = |\mathcal{S}_{miss}| / (|\mathcal{S}_{miss}| + |\mathcal{S}_{match}|)$,
   and analogously for `ghost` and the combined `mg` (miss-or-ghost) variant.
3. **IoU-based Mutation Score (MS_iou)** — for matched objects only, the mean
   spatial degradation $\frac{1}{|\mathcal{S}_{match}|}\sum (1 - \mathrm{IoU}(b_i, b'_j))$.
4. **Uncertainty-Aware Mutation Score (UA-MS)** — for each of
   $\mathcal{S}_{match}$, $\mathcal{S}_{miss}$, $\mathcal{S}_{ghost}$ and each
   of the 5 `deepluq.metrics_dl.DLMetrics` uncertainty metrics (VR, Shannon
   Entropy, MI, Total Variance, Prediction Surface), $UA\text{-}MS = |UM_{orig} - UM_{mut}|$:
   how much the mutant's predictive uncertainty diverges from the original's
   (15 scores in total: 3 object sets × 5 metrics). For miss/ghost objects
   there is no counterpart detection to diff against, so `deepluq.metrics_mut`
   instead scores the *original* detection's own uncertainty, weighted by how
   consistently it was missed/hallucinated across repetitions — a confidently
   missed or confidently hallucinated object contributes a larger score than an
   already-uncertain one.

## Setting up MC-Dropout / MC-DropBlock mutation operators

`deepluq.metrics_mut` does not itself run inference — it scores prediction
outputs that were already produced by executing a detection model repeatedly
under MC-Dropout / MC-DropBlock mutation (this is the same execution step
described in [Uncertainty Quantification for Perception Models](uq4dl.md), just
run once for the *original* model and once per *mutant* configuration).

For each mutant, generate `T` repetitions of predictions and save them as
`prediction_{t}.json` (`t` in `0..T-1`) under a folder tagged with the mutation
operator and its rate:

```text
{case_study}/experiment_results_{mutation_operator}/{model}/dataset/{test_dataset}/
    {mutation_operator}_{tag}_{image_stem}/
        prediction_0.json
        prediction_1.json
        ...
        prediction_9.json
```

- For `mc_dropout`, `{tag}` is the dropout rate as a string, e.g. `0`, `0.1`,
  `0.15`, ..., `0.5` (`0` is the unmutated original).
- For `mc_dropblock`, `{tag}` is `{dropout_rate}_{block_size}`, e.g. `0_1`
  (unmutated original), `0.1_3`, `0.3_9`, etc.

Each `prediction_{t}.json` holds one entry per detected object, keyed by an
arbitrary id, with the fields consumed by the matching/scoring functions:

```json
{
  "label_0": {
    "box": [822.0, 301.6, 902.1, 340.7],
    "label": 2,
    "score": 0.93,
    "logit": [0.01, 0.02, 0.93]
  }
}
```

`deepluq.metrics_mut.ms.convert_detection_output` and `yolo_to_absolute` are
provided to help build this format from a raw torchvision-style detection
output or YOLO-format ground-truth labels, respectively.

## Computing the Uncertainty-Aware Mutation Score

### Score a single test case against one mutant

```python
from deepluq.metrics_mut.ms_calcu import ms_per_test_case_mutant
from pathlib import Path

iskill_miss, iskill_ghost, iskill_miss_ghost, ms_obj_level, \
    match_metrics, miss_metrics, ghost_metrics = ms_per_test_case_mutant(
        test_case=Path("image_0001"),
        org_model="fasterrcnn_resnet50_fpn",
        mutation_operator="mc_dropout",
        mutation_rate=0.3,
        case_study="/path/to/experiment_results_root",
        T=10,
    )
```

This loads the `T` repetitions for the original (`rate=0`) and mutant
(`rate=0.3`) runs, matches detections per repetition, and returns:

- `iskill_miss` / `iskill_ghost` / `iskill_miss_ghost` — binomial-test results
  (`p_value`, `cohens_h`, `power`, `observed_success`) for whether the mutant's
  miss/ghost/either rate across repetitions is significantly above a `0.01`
  noise floor (`deepluq.metrics_mut.ms.check_kill_binomial_test`).
- `ms_obj_level` — mean/std object-level kill rates for miss, ghost, and
  miss-or-ghost, across all objects in the test case.
- `match_metrics`, `miss_metrics`, `ghost_metrics` — dicts averaged over all
  matched / missed / ghost objects, each with `vr_ms`, `ie_ms`, `mi_ms`,
  `var_ms`, `ps_ms` (absolute divergence in variation ratio, entropy, mutual
  information, total variance, and prediction surface between original and
  mutant), plus `iou_ms`/`match_rate` (match set), `miss_rate` (miss set), or
  `ghost_rate` (ghost set).

### Score a whole test set against every mutant of an operator

```python
from pathlib import Path
from deepluq.metrics_mut.ms_calcu import calcu_mutation_score

test_set = [Path("image_0001"), Path("image_0002"), ...]

calcu_mutation_score(
    test_set=test_set,
    org_model="fasterrcnn_resnet50_fpn",
    mutation_operator="mc_dropout",         # or "mc_dropblock"
    case_study_p="/path/to/experiment_results_root",
    case_study_n="sticker_detection",
    save_folder="ms_results",
    # mutation_rates=[0.1, 0.15, ...],      # optional; defaults to the standard sweep
)
```

This iterates every mutant of `mutation_operator` (by default the dropout-rate
sweep `0.1..0.5` for `mc_dropout`, or the full `dropout_rate x block_size` grid
for `mc_dropblock`), scores every test case in `test_set` against that mutant,
and writes one CSV per mutant to
`{save_folder}/{case_study_n}/{org_model}/{mutation_operator}/mutant_<tag>.csv`
— one row per test case, columns as described below.

### Score every model in a case study

```python
from deepluq.metrics_mut.ms_calcu import ms_calcu_exec

case_study = {
    "sticker_detection": {
        "dataset": "/path/to/datasets",       # holds common_images_{model}_filtered.pkl
        "raw_result": "/path/to/experiment_results_root",
        "models": ["fasterrcnn_resnet50_fpn", "retinanet_resnet50_fpn"],
    },
}

ms_calcu_exec(case_study_name="sticker_detection", case_study=case_study, save_folder="ms_results")
```

For each model, this loads `common_images_{model}_filtered.pkl` (a pickled list
of `Path`s naming the common test images to evaluate — e.g. images every mutant
successfully produced predictions for) and runs `calcu_mutation_score` for both
`mc_dropblock` and `mc_dropout`.

## Output columns

Each mutant CSV has one row per test case (one image tested against that one
mutant) with `test_suite` (image name) plus:

| Group | Columns | Corresponds to | Meaning |
|---|---|---|---|
| Image-level kill | `[img_level] iskill_{miss,ghost,miss_ghost}_{p_value,cohens_h,power,count}` | Img-MS | Binomial-test inputs for whether this test case's misses/ghosts/either are statistically significant over `T` repetitions (`p_value`, effect size `cohens_h`, `power`, and the raw success `count`). |
| Object-level kill | `[obj_level] kill_rate_{miss,ghost,miss_ghost}_{mean,std}` | Obj-MS | Mean/std, across this test case's objects, of $\lvert\mathcal{S}_{miss}\rvert/(\lvert\mathcal{S}_{miss}\rvert+\lvert\mathcal{S}_{match}\rvert)$ and analogous ratios for `ghost`/`miss_ghost`. |
| Match | `[match] iou_mean`, `[match] match_rate`, `[match] {vr,ie,mi,var,ps}_diff` | MS_iou, UA-MS | `iou_mean` is $1-\mathrm{IoU}$ averaged over $\mathcal{S}_{match}$ (MS_iou); the `{vr,ie,mi,var,ps}_diff` columns are $\lvert UM_{orig}-UM_{mut}\rvert$ for matched objects (UA-MS on $\mathcal{S}_{match}$). |
| Miss | `[miss] miss_rate`, `[miss] {vr,ie,mi,var,ps}_diff` | UA-MS | UA-MS on $\mathcal{S}_{miss}$: since there is no mutant detection to diff against, this is `1 -` the original detection's own (miss-rate-weighted) uncertainty metric. |
| Ghost | `[ghost] ghost_rate`, `[ghost] {vr,ie,mi,var,ps}_diff` | UA-MS | UA-MS on $\mathcal{S}_{ghost}$, defined analogously from the mutant's own uncertainty. |

Use `deepluq.metrics_mut.helper.print_metric` to pretty-print the raw
match/miss/ghost metric lists returned by `process_match_metrics`,
`process_missing_set`, and `process_ghost_set_dbscan` while debugging a single
test case.

```{note}
These CSVs give you, per test case, the *ingredients* for each score (the
binomial-test stats for Img-MS, the raw Obj-MS/MS_iou/UA-MS ratios), but not yet
the final numbers from the formal definitions above. Img-MS (Eq. 7) is a ratio
over the set of *mutants*: for each mutant, `isKilled` is True if **any** test
case's `p_value` is significant, and Img-MS is the fraction of mutants killed —
that reduction (per-mutant `isKilled`, then the ratio across all mutant CSVs
produced by `calcu_mutation_score`) isn't implemented in `metrics_mut` yet. If
you want it added as a `deepluq.metrics_mut.ms_calcu` function, let me know.
```
