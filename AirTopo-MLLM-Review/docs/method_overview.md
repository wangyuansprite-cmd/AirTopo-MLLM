# AirTopo-MLLM Method and Release Scope

## Released method components

The selected modules expose the parts of AirTopo-MLLM that can be inspected
without distributing clinical data, model weights, or incompletely attributed
third-party source.

### Topology semantics

The released preprocessing stages operate on an already extracted and cleaned
airway graph. They detect proximal tracheal anatomy, assign left/right
subtrees, construct a deterministic root-based hierarchy in the presence of
residual non-self cycles, and assign generation-based branch semantics.
Residual cycle edges are retained as explicit non-hierarchy records and do not
participate in generation recursion.

The upstream CT resampling, skeletonization, skeleton-to-graph extraction, and
graph-cleaning stages are not included. Their reviewed versions were coupled
to dataset-specific execution and internal diagnostics; publishing them
without a separate numerical regression review could change topology behavior.

### Structured anatomical queries

`semantic/structured_query.py` maps free-form English phrases to the frozen
target, laterality, extent, and generation-range schema. The parser is
deterministic and uses the non-clinical registries under
`resources/language_queries/`. It does not load a language model or produce an
embedding.

### Query-aware visual adaptation

`models/query_aware_modulation.py` implements identity-at-initialization
residual gating and residual FiLM over the dense visual feature. The original
semantic representation remains available to the base decoder. The complete
external multimodal backbone is deliberately not vendored.

`models/high_resolution_decoder.py` implements the local CT stem and additive
or gated fusion wrappers used to preserve fine airway detail. The wrapper
expects an externally provided compatible decoder/model object.

### Training objectives

`losses/segmentation_loss.py` combines the checkpoint model's Dice function
with adaptive weighted binary cross-entropy. The positive weight is
`min(sqrt(negative / positive), 20)` by default.

`losses/centerline_supervision.py` implements positive-only semantic-centerline
recall supervision on the final segmentation logits. It adds no prediction
head and performs no interpolation or resampling of the target.

### Inference and evaluation

`inference/volume_inference.py` provides complete sliding-window coverage,
uniform overlap averaging, and strict probability thresholding. The caller
must inject the model-specific patch function.

`inference/infer_full_case.py` is a sanitized reference driver for NIfTI input,
spatial validation, reconstruction, optional structural metrics, and output
export. Model construction, tokenizer setup, data-manifest resolution, and
checkpoint interpretation cross an explicit external integration boundary and
are not distributed.

`evaluation/metrics.py` provides Dice, IoU, precision, recall, CLR, DBR, and
DLR. DLR weights detected branches by their ground-truth centerline voxel
counts; it is not a foreground-volume ratio.

## Reproducibility boundary

This repository does not include end-to-end training, checkpoint loading,
tokenizer/model orchestration, raw data preprocessing, or private experiment
configurations. It therefore supports method inspection and synthetic unit
checks, not complete reproduction of reported experiments. No clinical volume,
patient-level result, or model weight is included.
