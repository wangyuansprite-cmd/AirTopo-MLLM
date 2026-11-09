# Synthetic Airway Format Example

This directory contains a procedural, non-clinical example that demonstrates
the array formats and spatial metadata expected by the released evaluation and
inference utilities. No patient image or source-dataset voxel was used.

Files:

- `ct.nii.gz`: synthetic normalized CT-like volume (`float32`).
- `whole_airway_mask.nii.gz`: binary airway target (`uint8`).
- `semantic_centerline.nii.gz`: binary target centerline (`uint8`).
- `branch_id_map.nii.gz`: positive branch ownership inside the airway and zero
  background (`int16`).
- `branch_metadata.json`: synthetic branch generation and side attributes.
- `sample_manifest.csv`: one whole-airway query row using relative paths.
- `metadata.json`: geometry, provenance, and integrity summary.

The example is intended for file-loading, geometry, metric, and visualization
tests. It is not a benchmark case and must not be used to report model quality.

Regenerate it from the repository root with:

```bash
python examples/generate_synthetic_case.py
```
