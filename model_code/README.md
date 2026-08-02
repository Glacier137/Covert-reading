# Neural decoding and synthesis models

This directory contains the model source code associated with the Covert Reading study. It is separate from the Source Data plotting notebooks in the repository root.

## Components

| Directory | Framework | Purpose |
| --- | --- | --- |
| [`classifier_and_speech_synthesizer/`](classifier_and_speech_synthesizer/README.md) | TensorFlow/Keras and PyTorch | Syllable classification and ECoG-to-log-mel/MCD analysis |
| [`articulatory_movement_synthesizer/`](articulatory_movement_synthesizer/README.md) | PyTorch | VQGAN-based prediction of articulatory trajectories from ECoG |

## Shared workspace

All default model inputs and outputs are kept below one repository-level directory:

```text
workspace/
  model_data/
    HSblockdata/                    # participant ECoG MATLAB files
    dataset_for_decoding_trace/     # prepared movement-model DataLoaders and metadata
    checkpoints/                    # movement-model checkpoints
    decoding_experiment/            # movement-model TensorBoard runs
  private/
    electrode_lists.json
  model_results/                    # classifier, speech, and launcher outputs
  speech_checkpoints/               # ECoG-to-log-mel/vocoder checkpoints
```

These data, checkpoint, and private-selection files are not included in the public code archive.

## Electrode-list configuration

Copy the template:

```bash
cp model_code/electrode_lists.example.json workspace/private/electrode_lists.json
```

On PowerShell:

```powershell
New-Item -ItemType Directory -Force workspace\private | Out-Null
Copy-Item model_code\electrode_lists.example.json workspace\private\electrode_lists.json
```

Replace the empty arrays with the required electrode indices. The JSON sections are used as follows:

| JSON section | Consumer |
| --- | --- |
| `classifier` | primary and percentage-based classifiers; ECoG-to-log-mel notebook |
| `sig_half` | significant-electrode half experiment |
| `full_half` | full-versus-half electrode experiment |
| `movement` | articulatory-movement data-preparation notebook |

Participant identifiers are strings, for example `"45"`. Extend the template to include every participant, condition, frequency band, and electrode-selection name required by the experiment.

## Path configuration

The default workspace is `<repository>/workspace`. It can be relocated without editing source code:

| Environment variable | Meaning | Default |
| --- | --- | --- |
| `COVERT_READING_WORKSPACE` | complete workspace root | `<repository>/workspace` |
| `COVERT_READING_DATA_ROOT` | model input root | `<workspace>/model_data` |
| `COVERT_READING_RESULTS_ROOT` | classifier/result root | `<workspace>/model_results` |
| `COVERT_READING_SPEECH_CHECKPOINT_DIR` | speech checkpoint root | `<workspace>/speech_checkpoints` |
| `COVERT_READING_ELECTRODE_LIST` | electrode-list JSON | `<workspace>/private/electrode_lists.json` |

Example on PowerShell:

```powershell
$env:COVERT_READING_WORKSPACE = (Resolve-Path .\workspace)
```

Example on Linux:

```bash
export COVERT_READING_WORKSPACE="$(pwd)/workspace"
```

The two component READMEs document their separate environments, commands, required files, and outputs.

## License

Original project code is distributed under the repository-level Apache License 2.0. The VQGAN-derived implementation retains its upstream MIT License in `articulatory_movement_synthesizer/models/LICENSE-VQGAN-MIT.txt`.

