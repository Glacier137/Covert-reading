# Articulatory-movement synthesizer

This directory contains the PyTorch/VQGAN route used to predict articulatory trajectories from overt and covert ECoG.

## Contents

```text
articulatory_movement_synthesizer/
  models/
    training_vqgan.py              # training entry point
    vqgan.py                       # VQGAN wrapper
    encoder.py, decoder.py         # encoder and decoder
    codebook.py                    # vector-quantization codebook
    discriminator.py               # PatchGAN-style discriminator
    helper.py, utils.py            # network blocks and data loading
    transformer.py                 # optional upstream transformer helper
    LICENSE-VQGAN-MIT.txt          # upstream VQGAN license
  scripts/
    data_prepare&get_results.ipynb # data preparation, inference, and result export
    train_vqgan.sh                 # standard fold-indexed experiment
    train_vqgan_infer.sh           # held-sound experiment
    train_vqgan_percentage.sh      # reduced-training-percentage experiment
```

## Software requirements

The first-stage VQGAN training route imports:

```text
torch
numpy
matplotlib
scikit-learn
dill
tqdm
tensorboard
```

The data-preparation notebook additionally uses `scipy`, `pandas`, and Jupyter. `transformer.py` is retained from the upstream VQGAN implementation but is not called by `training_vqgan.py`; using it separately requires a compatible `mingpt` implementation.

A CUDA-capable PyTorch installation is required for the supplied GPU launch commands. Select the PyTorch build that matches the CUDA driver on the target machine.

## Workspace layout

Inputs and outputs default to the repository-level `workspace/` directory:

```text
workspace/
  model_data/
    HSblockdata/                    # participant ECoG MATLAB files
    at_dir/                         # articulatory-trajectory inputs
    dataset_for_decoding_trace/
      last_dim_<band>.npy
      HS<id>_*_train_loader_*.pkl
      HS<id>_*_val_loader_*.pkl
    checkpoints/                    # saved VQGAN weights
    decoding_experiment/            # TensorBoard event directories
  private/
    electrode_lists.json
  model_results/
    movement_logs/                  # stdout/stderr from shell launchers
```

Set `COVERT_READING_WORKSPACE` to relocate the complete workspace, or `COVERT_READING_DATA_ROOT` to override `workspace/model_data/` only. The shell launchers also accept:

| Environment variable | Purpose | Default |
| --- | --- | --- |
| `COVERT_READING_GPU_IDS` | space-separated physical GPU IDs | `0` |
| `PYTHON_BIN` | Python executable | `python3` |
| `COVERT_READING_LOG_DIR` | launcher log directory | `workspace/model_results/movement_logs` |

## Prepare movement-model inputs

Copy `model_code/electrode_lists.example.json` to `workspace/private/electrode_lists.json` and populate the `movement` section. The notebook reads entries of the form:

```text
movement[participant][band][electrode_selection]
```

For example, participant keys are strings such as `"45"`, bands include `"high gamma"`, `"beta1"`, and `"hgb1"`, and the electrode-selection name must match the experiment argument.

Open the preparation notebook from the repository root:

```bash
jupyter notebook "model_code/articulatory_movement_synthesizer/scripts/data_prepare&get_results.ipynb"
```

The notebook loads the original ECoG and articulatory-trajectory data, creates fold-specific DataLoader pickle files, writes `last_dim_<band>.npy`, loads trained checkpoints for inference, and exports reconstruction results.

## Train one configuration

Run the training entry directly from the repository root:

```bash
python model_code/articulatory_movement_synthesizer/models/training_vqgan.py \
  --HS 54 \
  --reading_name covert \
  --elec_type downsample_covert_sig \
  --band "high gamma" \
  --fold_ind 1 \
  --epochs 80 \
  --device cuda:0
```

By default, prepared loaders are read from `workspace/model_data/dataset_for_decoding_trace/`, checkpoints are written to `workspace/model_data/checkpoints/`, and TensorBoard runs are written below `workspace/model_data/decoding_experiment/`.

Use `--help` for the complete argument list:

```bash
python model_code/articulatory_movement_synthesizer/models/training_vqgan.py --help
```

## Batch launchers

The launchers use Bash. On Windows, run them in Ubuntu WSL or another Linux environment:

```bash
bash model_code/articulatory_movement_synthesizer/scripts/train_vqgan.sh
bash model_code/articulatory_movement_synthesizer/scripts/train_vqgan_infer.sh
bash model_code/articulatory_movement_synthesizer/scripts/train_vqgan_percentage.sh
```

- `train_vqgan.sh` runs fold-indexed high-gamma models for matched overt/overt-electrode and covert/covert-electrode configurations.
- `train_vqgan_infer.sh` runs the covert SASI held-sound experiment with the `hgb1` input.
- `train_vqgan_percentage.sh` runs reduced-data experiments for `beta1` and `hgb1`.

The launchers distribute tasks across the GPU IDs supplied in `COVERT_READING_GPU_IDS` and return a non-zero exit status if any child process fails.

## Required non-public inputs

Numerical retraining requires the original participant ECoG MATLAB files, articulatory-trajectory arrays, prepared DataLoaders, `last_dim` metadata, private electrode selections, and any checkpoints used for downstream inference. These files are not included in the public code archive.

## Upstream code and license

The VQGAN implementation in `models/` is adapted from [dome272/VQGAN-pytorch](https://github.com/dome272/VQGAN-pytorch), Copyright (c) 2022 Dominic Rampas. The upstream code is distributed under the MIT License; the complete text is retained in [`models/LICENSE-VQGAN-MIT.txt`](models/LICENSE-VQGAN-MIT.txt).

The implementation was modified for ECoG-based articulatory-trajectory prediction, including changes to input and output dimensions, training and validation loops, data loading, experiment modes, checkpoint naming, and repository-relative paths. The PatchGAN-style discriminator retains its source reference to the CycleGAN/pix2pix implementation.

