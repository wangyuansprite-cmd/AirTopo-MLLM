# AirTopo-MLLM

AirTopo-MLLM is a topology-informed multimodal framework for
query-conditioned 3D airway segmentation. The framework combines structured
anatomical queries, query-aware residual FiLM modulation, a high-resolution
local CT pathway, branch-level topology semantics, and semantic-centerline
supervision.

This compact repository is intended to support academic review of the method.
It contains selected, inspectable implementation components; it is not an
end-to-end training or inference release. Clinical data, model checkpoints,
pretrained weights, patient-level outputs, and vendored third-party model code
are not included.

## Method overview

1. A cleaned airway graph is assigned anatomical root, carina, laterality,
   parent-child hierarchy, and generation attributes.
2. Branch generations and sides define central, peripheral, and compositional
   semantic targets.
3. A deterministic query registry represents target, laterality, extent, and
   generation range.
4. Query-aware residual FiLM modulates the dense global visual feature while
   preserving the pretrained pathway at zero initialization.
5. A high-resolution CT branch is fused with the segmentation decoder.
6. Training combines Dice, adaptive weighted binary cross-entropy, and a
   positive-only semantic-centerline recall objective.
7. Full-volume inference averages probabilities over overlapping 3D windows
   and applies a strict `probability > 0.5` decision rule.
8. Evaluation includes Dice, IoU, centerline recall (CLR), detected branch
   ratio (DBR), and detected length ratio (DLR).

See [docs/method_overview.md](docs/method_overview.md) for the implementation
scope and the interfaces between the released components.

## Repository structure

- `models/`: query-aware modulation and high-resolution decoder adaptations.
- `losses/`: segmentation and semantic-centerline supervision objectives.
- `semantic/`: structured anatomical query parsing.
- `preprocessing/`: selected topology stages for proximal anatomy,
  laterality, branch hierarchy, and generation-based branch semantics.
- `inference/`: data-independent sliding-window probability reconstruction.
- `evaluation/`: voxelwise and topology-aware metrics.
- `visualization/`: aligned 3D ground-truth and prediction rendering.
- `resources/`: non-clinical query vocabulary and canonical prompt registry.
- `configs/`: a path-neutral reference configuration.
- `tests/`: synthetic CPU-compatible checks.

## Installation and checks

Use an isolated Python environment and review the dependency versions before
installation:

```bash
python -m pip install -r requirements.txt
PYTHONPATH=. python -m pytest -q tests
```

Validate a structured query without loading a model:

```bash
python -m semantic.structured_query --text "segment the entire airway tree"
```

The included inference utility accepts a caller-provided patch prediction
function. The complete checkpoint loader, tokenizer/model orchestration,
distributed training entry point, and dataset preparation pipeline are not
included, so this repository does not claim complete experiment reproduction.

## Evaluation and visualization

`evaluation.metrics.compute_airway_metrics` computes Dice, IoU, precision,
recall, CLR, DBR, and centerline-length-based DLR from aligned NumPy arrays.
`visualization.visualize_3d.render_airway_comparison` renders aligned airway
masks without embedding patient data in the repository. Users are responsible
for supplying authorized inputs and checking their spatial alignment.

## Data and model availability

ATM22 volumes, derived patient-level targets, predictions, model checkpoints,
and pretrained weights are not distributed. They must be obtained from
authorized sources under the applicable data-use, institutional, and model
license terms.

## Third-party software and license status

This staging directory does not vendor M3D-LaMed, MONAI, Segment Anything, or
other third-party model implementations. The retained AirTopo-MLLM adapters
expect compatible external components in a complete deployment. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

No public software license for the original AirTopo-MLLM source was found in
the reviewed materials. The `LICENSE` file records that the license decision
is pending; it does not grant open-source rights. Rights-holder approval and a
final citation are required before public publication.

## Citation

Final citation metadata was not available in the reviewed materials. Add the
approved paper citation before publication.

## Code and Model Availability

This repository provides the code and supplementary materials for reviewing the proposed AirTopo-MLLM framework.

To support research transparency and facilitate technical evaluation, the repository includes selected implementation components, evaluation scripts, visualization tools, and experimental documentation.

The complete source code of AirTopo-MLLM, together with the trained model checkpoints and corresponding inference configurations, will be made publicly available upon publication of the paper, subject to applicable third-party licensing terms and data-use restrictions.

We hope that the full release will facilitate further research on topology-informed multimodal learning and compositional 3D medical image segmentation.
