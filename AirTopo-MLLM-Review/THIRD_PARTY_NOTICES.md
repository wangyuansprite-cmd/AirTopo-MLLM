# Third-Party Notices

This file records dependency and provenance information available in the
reviewed materials. It is not legal advice and does not replace a release
license review.

## External model framework

AirTopo-MLLM was developed on an M3D-LaMed/SegVol-style multimodal and
volumetric segmentation stack. The exact authoritative upstream repository
revision was not recorded in the reviewed materials. No M3D-LaMed, MONAI,
Segment Anything, volumetric SAM, or pretrained-model source is vendored in
this staging directory because the complete provenance and notice chain could
not be verified.

The files in `models/` are AirTopo-MLLM adapter modules that accept compatible
external model objects. Their inclusion does not grant rights to any external
implementation or pretrained weight.

## Python dependencies

The selected code imports PyTorch, NumPy, SciPy, Matplotlib, and SimpleITK.
Each package remains subject to its own license. Dependency declarations in
`requirements.txt` are installation references, not bundled source.

## Required manual review

Before a public release, the maintainers should:

1. identify and record the authoritative upstream repository and commit for
   every external model component used by the complete implementation;
2. verify all pretrained-model terms and redistribution restrictions;
3. select a license for the original AirTopo-MLLM source; and
4. add any copyright and notice text required by the selected dependencies.
