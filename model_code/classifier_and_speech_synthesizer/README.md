# Syllable classifier and ECoG-to-log-mel model

This directory contains two related analysis routes:

1. TensorFlow/Keras classifiers for decoding syllable identity from overt and covert ECoG; and
2. a PyTorch notebook that predicts log-mel features and calculates mel-cepstral distortion (MCD).

The classifier environment does not install the separate PyTorch notebook dependencies.

## Contents

| File | Purpose |
| --- | --- |
| `train_model_new.py` | primary percentage-based classifier experiment |
| `train_model.py` | original classifier variant |
| `train_model_b1hg.py` | beta1/high-gamma combined-band variant |
| `train_model_full_half.py` | full-versus-half electrode experiment |
| `train_model_sig_half.py` | significant-electrode half experiment |
| `base_*.py` | data loading and TensorFlow/Keras model definitions |
| `launch_all_new.py` | batch launcher for percentage experiments |
| `launch_all_full_half.py` | batch launcher for full-versus-half experiments |
| `launch_all_sig_half.py` | batch launcher for significant-half experiments |
| `MCD_new_251006.ipynb` | ECoG-to-log-mel modelling and MCD calculation |
| `paths.py` | repository-relative path configuration |
| `environment.yml` | pinned TensorFlow classifier environment |

## Workspace layout

All paths default to the repository-level `workspace/` directory:

```text
workspace/
  model_data/
    HSblockdata/
      HS45_Block_overt_covert_12_24_zscore_100Hz.mat
      HS45_Block_overt_covert_70_150_zscore_100Hz.mat
      ...
  private/
    electrode_lists.json
  model_results/
    2468/
    full_half/
    sig_half/
    speech/
  speech_checkpoints/
```

The expected participant IDs in the supplied batch launchers are `45`, `47`, `48`, `50`, `54`, `71`, `73`, `76`, and `78`.

The default paths can be overridden with `COVERT_READING_WORKSPACE`, `COVERT_READING_DATA_ROOT`, `COVERT_READING_RESULTS_ROOT`, `COVERT_READING_SPEECH_CHECKPOINT_DIR`, and `COVERT_READING_ELECTRODE_LIST`.

## Electrode-list file

From the repository root, copy:

```text
model_code/electrode_lists.example.json
```

to:

```text
workspace/private/electrode_lists.json
```

Replace the empty arrays with the electrode-index lists used in the experiment. The scripts access the following sections directly:

- `classifier[state][band][participant]` for `train_model_new.py`, `train_model.py`, `train_model_b1hg.py`, and the MCD notebook;
- `sig_half[state][band][participant]` for `train_model_sig_half.py`; and
- `full_half[selection_name]` for `train_model_full_half.py`.

Here, `state` is `SA` for overt speech or `SI` for covert speech, and `band` is `hg` or `b1` where applicable.

## TensorFlow classifier

### Environment

Create the pinned classifier environment from the repository root:

```bash
conda env create -f model_code/classifier_and_speech_synthesizer/environment.yml
conda activate covert_reading_tf
```

The environment specifies Python 3.9, TensorFlow 2.15.1, Keras 2.15.0, NumPy 1.26.2, SciPy 1.11.4, scikit-learn 1.3.0, and h5py 3.10.0.

### Run one configuration

```bash
python model_code/classifier_and_speech_synthesizer/train_model_new.py \
  --HS 45 \
  --condition ECoG_overt \
  --percent 100% \
  --band hg
```

Arguments:

| Argument | Values used by the supplied experiments |
| --- | --- |
| `--HS` | participant ID |
| `--condition` | `ECoG_overt` or `ECoG_covert` |
| `--percent` | `20%`, `40%`, `60%`, `80%`, or `100%` |
| `--band` | `hg` or `b1` |

The main script performs 10-fold shuffled cross-validation and writes NumPy result dictionaries to `workspace/model_results/2468/` by default.

### Run the supplied batches

```bash
python model_code/classifier_and_speech_synthesizer/launch_all_new.py
python model_code/classifier_and_speech_synthesizer/launch_all_full_half.py
python model_code/classifier_and_speech_synthesizer/launch_all_sig_half.py
```

The percentage launcher runs 20%, 40%, 60%, and 80% experiments for both `hg` and `b1`. The other launchers run their corresponding 100% electrode-selection experiments. Existing result keys are skipped by the individual training scripts.

## PyTorch ECoG-to-log-mel and MCD notebook

Open the notebook from the repository root:

```bash
jupyter notebook model_code/classifier_and_speech_synthesizer/MCD_new_251006.ipynb
```

Its active external dependencies are:

```text
torch
numpy
pandas
scipy
scikit-learn
librosa
datasets
tqdm
torchinfo
mel_cepstral_distance
jupyter
```

The notebook reads ECoG MATLAB files from `workspace/model_data/HSblockdata/`, electrode selections from `workspace/private/electrode_lists.json`, and checkpoints from `workspace/speech_checkpoints/`. Its generated results are placed under `workspace/model_results/speech/`.

This notebook predicts log-mel representations and computes MCD. Waveform reconstruction additionally requires the original vocoder implementation, configuration, and checkpoint, which are not included in this code archive.

## Data and compute requirements

The public repository does not include participant ECoG arrays, private electrode indices, cached tensors, or trained weights. GPU memory and runtime depend on participant, band, selected electrodes, batch size, and software build. The original experiments used CUDA-capable GPUs; reproduce the reported software environment before comparing runtimes or numerical output.

