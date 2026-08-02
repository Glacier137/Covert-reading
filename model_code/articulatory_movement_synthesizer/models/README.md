# VQGAN model implementation

This directory contains the VQGAN implementation used by the articulatory-movement synthesizer.

## Files

| File | Role |
| --- | --- |
| `training_vqgan.py` | first-stage training, validation, checkpointing, and experiment-mode entry point |
| `vqgan.py` | encoder-codebook-decoder wrapper and quantization projections |
| `encoder.py` | convolutional encoder adapted for the ECoG input representation |
| `decoder.py` | decoder adapted for articulatory-trajectory output |
| `codebook.py` | vector-quantization codebook |
| `discriminator.py` | PatchGAN-style discriminator |
| `helper.py` | residual, attention, sampling, normalization, and activation blocks |
| `utils.py` | prepared-DataLoader loading and weight initialization |
| `transformer.py` | optional upstream second-stage transformer helper; not used by `training_vqgan.py` |

Run `training_vqgan.py` through the commands documented in [`../README.md`](../README.md). The files use local imports, so the supplied entry-point path should be used rather than importing this directory as an installed Python package.

## Upstream attribution

Substantial portions of this implementation are adapted from:

- Project: [dome272/VQGAN-pytorch](https://github.com/dome272/VQGAN-pytorch)
- Original author: Dominic Rampas
- Original copyright: Copyright (c) 2022 Dominic Rampas
- Upstream license: MIT License

The complete upstream license is retained in [`LICENSE-VQGAN-MIT.txt`](LICENSE-VQGAN-MIT.txt). The original copyright and license must remain with redistributed copies or substantial portions of the upstream implementation.

Changes for this study include support for ECoG inputs and articulatory trajectories, configurable participant/condition/band experiments, fold and percentage modes, held-sound inference, prepared DataLoader inputs, checkpoint management, and portable workspace paths.

`discriminator.py` also retains the source link for the PatchGAN discriminator from the CycleGAN/pix2pix project.

## Project license

Original additions made for this repository are covered by the repository-level Apache License 2.0. The Apache license does not remove or replace the retained upstream MIT terms.

